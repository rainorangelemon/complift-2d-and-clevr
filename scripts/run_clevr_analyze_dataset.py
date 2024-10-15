import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from typing import Union
import os

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
from torchvision.io import read_image
from tqdm.auto import tqdm
import wandb
from utils import plot_energy_histogram, catchtime


def read_samples(output_dir, global_step, num_samples, num_conditions):
    output_dir = Path(output_dir)
    samples = []
    for i in range(num_samples):
        image_path = output_dir / f'composed_sample_{global_step:05d}_step_{i:02d}.png'
        # Read the image
        image = read_image(str(image_path))
        # Convert to float and scale to [0, 1]
        image = image.float() / 255.0
        # Reverse the normalization
        image = (image * 2 - 1)
        samples.append(image)
    # Stack the samples into a single tensor
    final_sample = th.stack(samples)

    condition_samples = []
    for i in range(num_conditions):
        contion_samples_per_condition = []
        for j in range(num_samples):
            image_path = output_dir / f'condition_sample_{global_step:05d}_condition_{i:02d}step_{j:02d}.png'
            # Read the image
            image = read_image(str(image_path))
            # Convert to float and scale to [0, 1]
            image = image.float() / 255.0
            # Reverse the normalization
            image = (image * 2 - 1)
            contion_samples_per_condition.append(image)

        # Stack the samples into a single tensor
        final_condition_sample = th.stack(contion_samples_per_condition)
        condition_samples.append(final_condition_sample)

    return final_sample, condition_samples


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
    corrects = (result == len(labels))
    return corrects, corrects.sum().clone().cpu().item()


@hydra.main(config_path="../conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    wandb.init(project="rejection_sampling",
               name=f"{cfg.experiment_name}",
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

    # Save the config
    config_save_path = os.path.join(cfg.output_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

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

        estimate_neg_logp = baselines_clevr.make_estimate_neg_logp(elbo_cfg=cfg.elbo,
                                                                   noise_scheduler=diffusion,
                                                                   unconditioned_denoise_fn=conditions_denoise_fn[-1],
                                                                   mini_batch=cfg.mini_batch,
                                                                   progress=True)

        # read samples from the output directory
        samples, condition_samples = read_samples("runs/10-12_16-47-08/test_clevr_rel_5000_2", global_step,
                                                  num_samples=100, num_conditions=num_relations_per_sample-1)
        samples = samples.to(device)
        condition_samples = [sample.to(device) for sample in condition_samples]

        # check whether exists the samples
        if os.path.exists(f"runs/10-13_13-00-11/test_clevr_rel_5000_2/energies_{global_step:05d}.pt"):
            energies = th.load(f"runs/10-13_13-00-11/test_clevr_rel_5000_2/energies_{global_step:05d}.pt")
            energies_per_condition = th.load(f"runs/10-13_13-00-11/test_clevr_rel_5000_2/energies_per_condition_{global_step:05d}.pt")
        else:
            # calculate the energies
            with catchtime('calculate energy'):
                energies = [estimate_neg_logp(denoise_fn, samples,
                                                t=th.full((len(samples),), 0, dtype=th.long, device=samples.device))
                                                for denoise_fn in conditions_denoise_fn[:-1]]

                energies_per_condition = [estimate_neg_logp(denoise_fn, condition_samples[i],
                                                            t=th.full((len(condition_samples[i]),), 0, dtype=th.long, device=condition_samples[i].device))
                                                            for i, denoise_fn in enumerate(conditions_denoise_fn[:-1])]

            # save the energies to pt files
            th.save(energies, output_dir / f"energies_{global_step:05d}.pt")
            th.save(energies_per_condition, output_dir / f"energies_per_condition_{global_step:05d}.pt")

        info = {}

        hot_condition_samples = []
        for condition_idx, condition_denoise_fn in enumerate(conditions_denoise_fn[:-1]):
            condition_samples_all_t = baselines_clevr.diffusion_baseline(
                denoise_fn=lambda x, t: condition_denoise_fn(x, t, use_cfg=True),
                diffusion=diffusion,
                x_shape=(3, options["image_size"], options["image_size"]),
                eval_batch_size=100,
            )
            hot_condition_samples.append(condition_samples_all_t[-1])  # Keep only the final samples

            # test the classifier on the condition samples
            correct_per_sample, corrects = calculate_classification_score(classifier, hot_condition_samples[-1], labels[None, 0, condition_idx], device)
            print(f"Corrects for condition {condition_idx} on hot samples: {corrects}/{len(condition_samples[-1])}")

        for condition_idx, condition_sample, label in zip(range(len(condition_samples)), condition_samples, labels[0, :-1]):
            # calculate the energy for the condition samples
            correct_per_sample, corrects = calculate_classification_score(classifier, condition_sample, label[None, ...], device)
            print(f"Corrects for condition {condition_idx} on individual samples: {corrects}/{len(condition_sample)}")

        # plot the energy histograms
        for condition_idx, composed_energies_per_condition, individual_energies_per_condition, label in \
            zip(range(len(energies)), energies, energies_per_condition, labels[0, :-1]):
            img = plot_energy_histogram(individual_energies_per_condition.flatten().cpu().numpy())
            info.update({f"energy_individual/condition_{condition_idx}_at_individual": [wandb.Image(img, caption=f"Energy histogram for condition {condition_idx}")]})

            img = plot_energy_histogram(composed_energies_per_condition.flatten().cpu().numpy())
            info.update({f"energy_composed/condition_{condition_idx}_at_individual": [wandb.Image(img, caption=f"Energy histogram for condition {condition_idx}")]})

            correct_per_sample, corrects = calculate_classification_score(classifier, samples, label[None, ...], device)
            print(f"Corrects for condition {condition_idx}: {corrects}/{len(samples)}")
            info.update({f"acc/condition_{condition_idx}": corrects / len(samples),
                         "global_step": global_step})

            correct_energy = composed_energies_per_condition[correct_per_sample]
            incorrect_energy = composed_energies_per_condition[~correct_per_sample]

            img_correct = plot_energy_histogram(correct_energy.flatten().cpu().numpy())
            img_incorrect = plot_energy_histogram(incorrect_energy.flatten().cpu().numpy())
            info.update({f"energy_correct/condition_{condition_idx}": [wandb.Image(img_correct, caption=f"Energy histogram for correct samples for condition {condition_idx}")],
                         f"energy_incorrect/condition_{condition_idx}": [wandb.Image(img_incorrect, caption=f"Energy histogram for incorrect samples for condition {condition_idx}")]})

        # Calculate the classification score
        _, corrects_unfiltered = calculate_classification_score(classifier, samples, labels[0, :-1], device)
        print(f"Corrects unfiltered: {corrects_unfiltered}/{len(samples)}")

        # filter the samples
        thresholds_per_condition = [th.quantile(energies_per_condition[i].flatten(), 0.1) for i in range(len(energies_per_condition))]
        # thresholds_per_condition = [th.quantile(energies[i].flatten(), 0.8) for i in range(len(energies))]
        out_of_threshold = [composed_energies_per_condition > threshold for composed_energies_per_condition, threshold in zip(energies, thresholds_per_condition)]
        out_of_threshold = th.stack(out_of_threshold, dim=1).any(dim=1)
        filtered_samples = samples[~out_of_threshold]

        # # choose the sample with the minimum energy
        # filtered_samples_idx = th.argmin(th.stack(energies).sum(dim=0))
        # filtered_samples = samples[filtered_samples_idx].unsqueeze(0)

        if len(filtered_samples) != 0:
            # Calculate the classification score for the filtered samples
            _, corrects_filtered = calculate_classification_score(classifier, filtered_samples, labels[0, :-1], device)
            print(f"Corrects filtered: {corrects_filtered}/{len(filtered_samples)}")
        else:
            corrects_filtered = 0
            filtered_samples = samples

        info.update({"acc/final_unfiltered": corrects_unfiltered / len(samples),
                     "acc/final_filtered": corrects_filtered / len(filtered_samples),
                     "global_step": global_step})

        return info

    # Sampling loop
    img_idx = 0
    list_acc_unfiltered = []
    list_acc_filtered = []
    for batch_labels, batch_captions in dataloader:
        # check if the img is already generated
        if (output_dir / f'sample_{img_idx:05d}.png').exists():
            img_idx += len(batch_labels)
            continue

        info = sample_batch(img_idx, model, diffusion, batch_labels)

        list_acc_unfiltered.append(info["acc/final_unfiltered"])
        list_acc_filtered.append(info["acc/final_filtered"])

        wandb.log({"global_step": img_idx,
                   "avg_acc/final_unfiltered": np.mean(list_acc_unfiltered),
                   "avg_acc/final_filtered": np.mean(list_acc_filtered),
                   **info})
        img_idx += 1

        if img_idx >= cfg.max_samples_for_generation:
            break

    print(f"Generated {img_idx} samples. Saved to {output_dir}")


if __name__ == '__main__':
    main()
