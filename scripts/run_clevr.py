import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from typing import Union

import argparse
import torch as th

from ComposableDiff.composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict
)
from ComposableDiff.composable_diffusion.image_datasets import load_data
from ComposableDiff.classifier.eval import load_classifier
import baselines_clevr

from PIL import Image
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import wandb
from utils import plot_energy_histogram


class CLEVRPosDataset(Dataset):
    def __init__(
        self,
        data_path,
    ):
        self.data_path = data_path

        data = np.load(self.data_path)
        self.labels = data['coords_labels']

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        label = self.labels[index]
        return label.astype(np.float32), self.convert_caption(label)

    def convert_caption(self, label):
        paragraphs = []
        for j in range(label.shape[0]):
            x, y = label[j, :2]
            paragraphs.append(f'object at position {x}, {y}')
        return ' and '.join(paragraphs)


class CLEVRRelDataset(Dataset):
    def __init__(
        self,
        data_path,
    ):
        self.data_path = data_path

        data = np.load(self.data_path)
        self.labels = data['labels']

        self.description = {
            "left": ["to the left of"],
            "right": ["to the right of"],
            "behind": ["behind"],
            "front": ["in front of"],
            "above": ["above"],
            "below": ["below"]
        }

        self.shapes_to_idx = {"cube": 0, "sphere": 1, "cylinder": 2, 'none': 3}
        self.colors_to_idx = {"gray": 0, "red": 1, "blue": 2, "green": 3, "brown": 4, "purple": 5, "cyan": 6,
                              "yellow": 7, 'none': 8}
        self.materials_to_idx = {"rubber": 0, "metal": 1, 'none': 2}
        self.sizes_to_idx = {"small": 0, "large": 1, 'none': 2}
        self.relations_to_idx = {"left": 0, "right": 1, "front": 2, "behind": 3, 'below': 4, 'above': 5, 'none': 6}

        self.idx_to_colors = list(self.colors_to_idx.keys())
        self.idx_to_shapes = list(self.shapes_to_idx.keys())
        self.idx_to_materials = list(self.materials_to_idx.keys())
        self.idx_to_sizes = list(self.sizes_to_idx.keys())
        self.idx_to_relations = list(self.relations_to_idx.keys())

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        label = self.labels[index]
        return label, self.convert_caption(label)

    def convert_caption(self, label):
        paragraphs = []
        for j in range(label.shape[0]):
            text_label = []
            for k in range(2):
                shape, size, color, material, pos = label[j, k * 5:k * 5 + 5]
                obj = ' '.join([self.idx_to_sizes[size], self.idx_to_colors[color],
                                self.idx_to_materials[material], self.idx_to_shapes[shape]])
                text_label.append(obj.strip())

            relation = self.idx_to_relations[label[j, -1]]
            # single object
            if relation == 'none':
                paragraphs.append(text_label[0])
            else:
                paragraphs.append(f'{text_label[0]} {self.description[relation][0]} {text_label[1]}')
        return ' and '.join(paragraphs)


@th.no_grad()
def calculate_classification_score(classifier: th.nn.Module,
                                   samples: th.Tensor,
                                   labels: th.Tensor,
                                   device: Union[th.device, str]):
    """calculate the classification score

    Args:
        classifier (th.nn.Module): the classifier
        samples (th.Tensor): samples to be classified, range in [-1, 1], (N, C, H, W)
        labels (th.Tensor): labels for the samples, (M, D)
        device (Union[th.device, str]): device to run the calculation
    """
    assert samples.dim() == 4, f"Samples should be in shape (N, C, H, W), got {samples.shape}"
    assert labels.dim() == 2, f"Labels should be in shape (M, D), got {labels.shape}"
    samples = samples.to(device)
    labels = labels.to(device)
    # scale the samples from [-1, 1] to [0, 1]
    samples = ((samples + 1) * 127.5).round().clamp(0, 255).to(th.uint8) / 255.

    result = th.zeros((samples.shape[0],), dtype=th.long, device=device)
    for label in labels:
        label = label.unsqueeze(0).expand(samples.shape[0], -1).to(device)
        outputs = classifier(samples, label)
        result += (outputs[:,0] < outputs[:,1]).long()
    corrects = th.sum(result == len(labels))
    return corrects.clone().cpu().item()


@hydra.main(config_path="../conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    path_of_dataset = cfg.data_path
    name_of_dataset = path_of_dataset.split("/")[-1].split(".")[0]
    wandb.init(project="rejection_sampling",
               name=f"{name_of_dataset}_{cfg.support_interval_sample_number}x{cfg.n_sample_for_elbo}",
               # to dict
               config=OmegaConf.to_container(cfg, resolve=True))

    # Setup
    th.set_float32_matmul_precision('high')
    th.set_grad_enabled(False)
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')

    options = OmegaConf.to_container(cfg.model, resolve=True)

    options["use_fp16"] = th.cuda.is_available()

    model, diffusion = create_model_and_diffusion(**options)

    model.eval()
    if options['use_fp16']:
        model.convert_to_fp16()
    model.to(device)

    print(f'Loading checkpoint from {cfg.ckpt_path}')
    checkpoint = th.load(cfg.ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    print('Total base parameters', sum(x.numel() for x in model.parameters()))

    # Create output directory
    output_dir = Path(cfg.output_dir)
    experiment_name = cfg.data_path.split('/')[-1].split('.')[0]
    output_dir = output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset and dataloader
    if "clevr_pos" in cfg.data_path:
        dataset = CLEVRPosDataset(data_path=cfg.data_path)
    elif "clevr_rel" in cfg.data_path:
        dataset = CLEVRRelDataset(data_path=cfg.data_path)
    else:
        raise ValueError("Unknown dataset")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    # import classifier
    classifier = load_classifier(**cfg.classifier)
    classifier.eval()
    classifier.to(device)

    # Sampling function
    def sample_batch(global_step, model, diffusion, labels):
        # add zeros to the labels for unconditioned sampling
        labels = th.cat([labels, th.zeros_like(labels[:, :1, :])], dim=1).to(device)
        masks = th.ones_like(labels[:, :, 0], dtype=th.bool).to(device)
        masks[:, -1] = False
        num_relations_per_sample = labels.shape[1]

        def conditions_denoise_fn_factory(labels, batch_size=cfg.mini_batch):
            def create_condition_denoise_fn(rel_idx):
                def condition_denoise_fn(x_t, ts, use_cfg=False, batch_size=batch_size):
                    current_label = labels[:, rel_idx, :].to(device)
                    current_mask = masks[:, rel_idx].to(device)

                    num_samples = x_t.shape[0]
                    if use_cfg:
                        batch_size = batch_size // 2
                    results = []

                    for i in range(0, num_samples, batch_size):
                        # Create batch slices for current iteration
                        x_t_batch = x_t[i:i+batch_size]
                        ts_batch = ts[i:i+batch_size]
                        current_batch_size = x_t_batch.shape[0]

                        # Expand the current label and mask for the current batch size
                        expanded_label = current_label.expand(current_batch_size, -1)
                        expanded_mask = current_mask.expand(current_batch_size)

                        if use_cfg:
                            # Add the unconditioned label
                            x_t_batch = th.cat([x_t_batch, x_t_batch], dim=0)
                            ts_batch = th.cat([ts_batch, ts_batch], dim=0)
                            expanded_label = th.cat([expanded_label, th.zeros_like(expanded_label)], dim=0)
                            expanded_mask = th.cat([expanded_mask, th.zeros_like(expanded_mask)], dim=0)
                            result = model(x_t_batch, ts_batch, y=expanded_label, masks=expanded_mask)
                            eps, rest = result[:, :3], result[:, 3:]
                            cond_eps, uncond_eps = eps[expanded_mask], eps[~expanded_mask]
                            eps = uncond_eps + (cfg.cfg_weight * (cond_eps - uncond_eps))
                            result = th.cat([eps, rest[~expanded_mask]], dim=1)
                        else:
                            result = model(x_t_batch, ts_batch, y=expanded_label, masks=expanded_mask)
                        results.append(result)

                    # Concatenate the results from all batches
                    return th.cat(results, dim=0)
                return condition_denoise_fn

            return [create_condition_denoise_fn(rel_idx) for rel_idx in range(num_relations_per_sample)]

        def composed_model_fn(x_t, ts, batch_size=cfg.mini_batch // num_relations_per_sample):
            num_samples = x_t.shape[0]
            results_eps = []
            results_rest = []

            for i in range(0, num_samples, batch_size):
                # Create batch slices for current iteration
                x_t_batch = x_t[i:i+batch_size]
                ts_batch = ts[i:i+batch_size]

                current_batch_size = x_t_batch.shape[0]
                combined = th.repeat_interleave(x_t_batch, num_relations_per_sample, dim=0)
                ts_batch = th.repeat_interleave(ts_batch, num_relations_per_sample, dim=0)

                current_label = labels.expand(current_batch_size, -1, -1).to(device).flatten(0, 1)
                current_mask = masks.expand(current_batch_size, -1).to(device).flatten(0, 1)

                model_out = model(combined, ts_batch, y=current_label, masks=current_mask)
                eps, rest = model_out[:, :3], model_out[:, 3:]

                cond_eps, uncond_eps = eps[current_mask], eps[~current_mask]
                uncond_eps = uncond_eps.view(current_batch_size, -1, *uncond_eps.shape[1:])
                cond_eps = cond_eps.view(current_batch_size, -1, *cond_eps.shape[1:])

                eps_batch = uncond_eps + (cfg.cfg_weight * (cond_eps - uncond_eps)).sum(dim=1, keepdim=True)
                eps_batch = eps_batch.flatten(0, 1)
                rest_batch = rest[~current_mask]

                # Collect the results from this batch
                results_eps.append(eps_batch)
                results_rest.append(rest_batch)

            # Concatenate results from all batches
            eps = th.cat(results_eps, dim=0)
            rest = th.cat(results_rest, dim=0)

            return th.cat([eps, rest], dim=1)


        conditions_denoise_fn = conditions_denoise_fn_factory(labels)

        if cfg.method_name == "rejection":
            method = baselines_clevr.rejection_sampling_baseline_with_interval_calculation_elbo
        elif cfg.method_name == "best_of_n":
            method = baselines_clevr.best_of_n_sampling_baseline_with_interval_calculation_elbo
        elif cfg.method_name == "baseline":
            method = lambda **kwargs: baselines_clevr.diffusion_baseline(
                denoise_fn=kwargs["composed_denoise_fn"],
                diffusion=kwargs["noise_scheduler"],
                x_shape=kwargs["x_shape"],
                eval_batch_size=1,
            )

        samples, filter_ratios, intervals, samples_per_condition, unfiltered_samples, energies_per_condition, energies_composed = \
        method(
            composed_denoise_fn=composed_model_fn,
            unconditioned_denoise_fn=conditions_denoise_fn[-1],
            conditions_denoise_fn=conditions_denoise_fn[:-1],
            x_shape=(3, options["image_size"], options["image_size"]),
            algebras=["product"]*(num_relations_per_sample-1),
            bootstrap_cfg=cfg.bootstrap,
            rejection_scheduler_cfg=cfg.rejection_scheduler,
            noise_scheduler=diffusion,
            eval_batch_size=cfg.support_interval_sample_number,
            n_sample_for_elbo=cfg.n_sample_for_elbo,
            mini_batch=cfg.mini_batch,
        )

        info = {}
        # # plot the energy histograms
        # for t, energies_per_condition_at_t in energies_per_condition.items():
        #     for energy_per_condition, condition_idx in zip(energies_per_condition_at_t, range(len(energies_per_condition_at_t))):
        #         img = plot_energy_histogram(energy_per_condition.flatten())
        #         info.update({f"energy_{t:03d}/condition_{condition_idx}_at_individual": [wandb.Image(img, caption=f"Energy histogram for condition {condition_idx} at t={t}")]})

        # for t, energy_composed_at_t in energies_composed.items():
        #     for energy_composed, condition_idx in zip(energy_composed_at_t, range(len(energy_composed_at_t))):
        #         img = plot_energy_histogram(energy_composed.flatten())
        #         info.update({f"energy_{t:03d}/condition_{condition_idx}_at_composed": [wandb.Image(img, caption=f"Energy histogram for composed model for condition {condition_idx} at t={t}")]})

        for condition_idx, dataset_per_condition, label in zip(range(len(samples_per_condition)), samples_per_condition, labels[0, :-1]):
            corrects = calculate_classification_score(classifier, dataset_per_condition, label[None, ...], device)
            print(f"Corrects for condition {condition_idx}: {corrects}/{len(dataset_per_condition)}")
            info.update({f"acc/condition_{condition_idx}": corrects / len(dataset_per_condition),
                         "global_step": global_step})

        final_unfiltered_sample = unfiltered_samples[-1]
        if len(final_unfiltered_sample) == 0:
            final_unfiltered_sample = th.zeros((1, 3, options["image_size"], options["image_size"]), device=device)

        final_sample = samples[-1]
        if len(final_sample) == 0:
            final_sample = th.zeros((1, 3, options["image_size"], options["image_size"]), device=device)

        # Calculate the classification score
        corrects_unfiltered = calculate_classification_score(classifier, final_unfiltered_sample, labels[0, :-1], device)
        print(f"Corrects unfiltered: {corrects_unfiltered}/{len(final_unfiltered_sample)}")

        corrects = calculate_classification_score(classifier, final_sample, labels[0, :-1], device)
        print(f"Corrects: {corrects}/{len(final_sample)}")

        print(f"Filter ratios: {filter_ratios}")

        info.update({"acc/final_unfiltered": corrects_unfiltered / len(final_unfiltered_sample),
                     "acc/final_filtered": corrects / len(final_sample),
                     "filter_ratios/final": filter_ratios[-1],
                     "global_step": global_step})

        return final_sample, info

    # Sampling loop
    img_idx = 0
    list_acc_unfiltered = []
    list_acc_filtered = []
    for batch_labels, batch_captions in dataloader:
        # check if the img is already generated
        if (output_dir / f'sample_{img_idx:05d}.png').exists():
            img_idx += len(batch_labels)
            continue

        samples, info = sample_batch(img_idx, model, diffusion, batch_labels)

        list_acc_unfiltered.append(info["acc/final_unfiltered"])
        list_acc_filtered.append(info["acc/final_filtered"])

        samples = samples[th.randint(0, len(samples), size=(1,))]
        samples = ((samples + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu() / 255.

        if img_idx == 0:
            # Save grid of samples
            grid = make_grid(samples, nrow=int(np.sqrt(len(samples))), padding=2)
            save_image(grid, f'{experiment_name}.png')

        # Save individual images with captions
        grid = make_grid(samples, nrow=int(np.sqrt(len(samples))), padding=0)
        save_image(grid, output_dir / f'sample_{img_idx:05d}.png')
        with open(output_dir / f'sample_{img_idx:05d}.txt', 'w') as f:
            f.write(batch_captions[0])
        wandb.log({"sample": [wandb.Image(Image.open(output_dir / f'sample_{img_idx:05d}.png'), caption=batch_captions[0])],
                   "global_step": img_idx,
                   "avg_acc/final_unfiltered": np.mean(list_acc_unfiltered),
                   "avg_acc/final_filtered": np.mean(list_acc_filtered),
                   **info})
        img_idx += 1

        if img_idx >= cfg.max_samples_for_generation:
            break

    print(f"Generated {img_idx} samples. Saved to {output_dir}")


if __name__ == '__main__':
    main()
