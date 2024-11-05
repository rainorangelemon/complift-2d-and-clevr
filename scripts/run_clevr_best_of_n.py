import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from typing import Union
import os
import json

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
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor


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
def calculate_classification_score_rel(classifier: th.nn.Module,
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


def show_points(coords, ax, marker_size=375):
    import copy
    coords = copy.deepcopy(coords)
    coords[:, 0] = coords[:, 0] * 128
    coords[:, 1] = (1 - coords[:, 1]) * 128
    ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    # label the points with the index
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), fontsize=12, color='white', ha='center', va='center')


# @th.no_grad()
# def calculate_classification_score_pos(model: th.nn.Module,
#         input_points = np.array(origin_points)
#         input_points[:, 0] = input_points[:, 0] * len(image)
#         input_points[:, 1] = (1 - input_points[:, 1]) * len(image[0])
#         input_labels = np.array([1] * len(input_points))
#         draw_labels =  np.array([1] * len(input_points))

#         masks_batch, _, _ = predictor.predict_batch(
#             point_coords_batch=[p[np.newaxis] for p in input_points],
#             point_labels_batch=[l[np.newaxis] for l in input_labels],
#             multimask_output=True,
#         )

#         success = True
#         for i, masks in zip(range(len(input_points)), masks_batch):
#             point = input_points[[i], :]
#             label = input_labels[[i]]
#             # print(f"Point {i+1}: {point} with label {label}")

#             success_per_point = ((masks.sum(axis=(1, 2)) > 100) & (masks.sum(axis=(1, 2)) < 1000)).any()
#             success = success and success_per_point

#             if not success_per_point:
#                 draw_labels[i] = 0

#             # update the positive and negative samples
#             if not success_per_point:
#                 negative_im_paths.append(f'sample_{example_idx}.png')
#                 negative_labels.append(origin_points[i])


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


            # half = x_t[::num_relations_per_sample]
            # combined = th.repeat_interleave(half, num_relations_per_sample, dim=0)
            # model_out = model(combined, ts, **kwargs)
            # eps, rest = model_out[:, :3], model_out[:, 3:]
            # masks = kwargs.get('masks')
            # cond_eps, uncond_eps = eps[masks], eps[~masks]
            # uncond_eps = uncond_eps.view(batch_size, -1, *uncond_eps.shape[1:])
            # cond_eps = cond_eps.view(batch_size, -1, *cond_eps.shape[1:])
            # half_eps = uncond_eps + (weights * (cond_eps - uncond_eps)).sum(dim=1, keepdim=True)
            # # -> (batch_size, 1, 3, H, W)
            # eps = th.repeat_interleave(half_eps, num_relations_per_sample, dim=1).view(-1, *half_eps.shape[2:])
            # return th.cat([eps, rest], dim=1)

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

        samples, original_samples, energies = \
        baselines_clevr.best_of_n_sampling(
            composed_denoise_fn=composed_model_fn,
            unconditioned_denoise_fn=conditions_denoise_fn[-1],
            conditions_denoise_fn=conditions_denoise_fn[:-1],
            x_shape=(3, options["image_size"], options["image_size"]),
            algebras=["product"]*(num_relations_per_sample-1),
            noise_scheduler=diffusion,
            best_of_n_scheduler_cfg=cfg.best_of_n_scheduler,
            elbo_cfg=cfg.elbo,
            **cfg.best_of_n,
        )

        final_samples = samples[-1]

        info = {}

        # info.update({"acc/final_unfiltered": corrects_unfiltered / len(final_unfiltered_sample),
        #              "acc/final_filtered": corrects / len(final_sample),
        #              "filter_ratios/final": filter_ratios[-1],
        #              "global_step": global_step})

        return final_samples, original_samples, energies, info

    # Sampling loop
    img_idx = 0
    list_acc_unfiltered = []
    list_acc_filtered = []
    for batch_labels, batch_captions in dataloader:
        # check if the img is already generated
        if (output_dir / f'sample_{img_idx:05d}.png').exists():
            img_idx += len(batch_labels)
            continue

        samples, original_samples, energies, info = sample_batch(img_idx, model, diffusion, batch_labels)

        # list_acc_unfiltered.append(info["acc/final_unfiltered"])
        # list_acc_filtered.append(info["acc/final_filtered"])

        samples = samples[th.randint(0, len(samples), size=(1,))]
        samples = ((samples + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu() / 255.

        final_original_samples = ((original_samples[0] + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu() / 255.

        # Save individual images with captions
        grid = make_grid(samples, nrow=int(np.sqrt(len(samples))), padding=0)
        save_image(grid, output_dir / f'sample_{img_idx:05d}.png')

        for i, sample in enumerate(final_original_samples):
            grid = make_grid(sample[None, :], nrow=1, padding=0)
            save_image(grid, output_dir / f'original_sample_{img_idx:05d}_{i:05d}.png')

            import matplotlib.pyplot as plt
            plt.close('all')
            plt.clf()
            plt.figure(figsize=(10, 10))
            plt.imshow(sample.permute(1, 2, 0))
            show_points(batch_labels[0].numpy(), plt.gca())
            plt.axis('on')
            plt.savefig(output_dir / f'labelled_sample_{img_idx:05d}_{i:05d}.png')
        # grid = make_grid(final_original_samples, nrow=int(np.sqrt(len(final_original_samples))), padding=0)
        # save_image(grid, output_dir / f'original_{img_idx:05d}.png')

        # save energies to txt
        with open(output_dir / f'energies_{img_idx:05d}.txt', 'w') as f:
            for energy in energies[0]:
                for e in energy:
                    # scientific notation
                    f.write(f'{e:.2e} ')
                f.write('\n')

        with open(output_dir / f'sample_{img_idx:05d}.txt', 'w') as f:
            f.write(batch_captions[0])
        wandb.log({"sample": [wandb.Image(Image.open(output_dir / f'sample_{img_idx:05d}.png'), caption=batch_captions[0])],
                   "global_step": img_idx,
                #    "avg_acc/final_unfiltered": np.mean(list_acc_unfiltered),
                #    "avg_acc/final_filtered": np.mean(list_acc_filtered),
                   **info})
        img_idx += 1

        if img_idx >= cfg.max_samples_for_generation:
            break

    print(f"Generated {img_idx} samples. Saved to {output_dir}")


if __name__ == '__main__':
    main()
