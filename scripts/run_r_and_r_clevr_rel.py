import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

import argparse
import torch as th

from ComposableDiff.composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict
)
from ComposableDiff.composable_diffusion.image_datasets import load_data
import baselines_clevr

from PIL import Image
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm


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


@hydra.main(config_path="../conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

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

    # Sampling function
    def sample_batch(model, diffusion, labels):
        # add zeros to the labels for unconditioned sampling
        labels = th.cat([labels, th.zeros_like(labels[:, :1, :])], dim=1)
        masks = th.ones_like(labels[:, :, 0], dtype=th.bool)
        masks[:, -1] = False
        num_relations_per_sample = labels.shape[1]

        def conditions_denoise_fn_factory(labels, batch_size=cfg.mini_batch):
            conditions_denoise_fns = []

            for rel_idx in range(num_relations_per_sample-1):  # ignore the unconditioned label
                current_label = labels[:, rel_idx, :].to(device)
                current_mask = masks[:, rel_idx].to(device)

                def condition_denoise_fn(x_t, ts, batch_size=batch_size):
                    num_samples = x_t.shape[0]
                    results = []

                    for i in range(0, num_samples, batch_size):
                        # Create batch slices for current iteration
                        x_t_batch = x_t[i:i+batch_size]
                        ts_batch = ts[i:i+batch_size]
                        current_batch_size = x_t_batch.shape[0]

                        # Expand the current label and mask for the current batch size
                        expanded_label = current_label.expand(current_batch_size, -1)
                        expanded_mask = current_mask.expand(current_batch_size)

                        # Run the model on the current batch and collect the result
                        result = model(x_t_batch, ts_batch, y=expanded_label, masks=expanded_mask)
                        results.append(result)

                    # Concatenate the results from all batches
                    return th.cat(results, dim=0)

                conditions_denoise_fns.append(condition_denoise_fn)

            return conditions_denoise_fns

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

        samples, filter_ratios, intervals = baselines_clevr.rejection_sampling_baseline_with_interval_calculation_elbo(
            composed_denoise_fn=composed_model_fn,
            conditions_denoise_fn=conditions_denoise_fn_factory(labels),
            x_shape=(3, options["image_size"], options["image_size"]),
            algebras=["product"]*(num_relations_per_sample-1),
            noise_scheduler=diffusion,
            eval_batch_size=cfg.support_interval_sample_number,
            n_sample_for_elbo=cfg.n_sample_for_elbo,
            mini_batch=cfg.mini_batch,
        )

        print(f"Filter ratios: {filter_ratios}")

        return samples[-1]

    # Sampling loop
    img_idx = 0
    for batch_labels, batch_captions in dataloader:
        # check if the img is already generated
        if (output_dir / f'sample_{img_idx:05d}.png').exists():
            img_idx += len(batch_labels)
            continue

        samples = sample_batch(model, diffusion, batch_labels)
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
        img_idx += 1

    print(f"Generated {img_idx} samples. Saved to {output_dir}")


if __name__ == '__main__':
    main()
