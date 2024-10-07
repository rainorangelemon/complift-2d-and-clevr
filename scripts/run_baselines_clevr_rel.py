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
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    # Sampling function
    def sample_batch(model, diffusion, labels):
        # add zeros to the labels for unconditioned sampling
        labels = th.cat([labels, th.zeros_like(labels[:, :1, :])], dim=1)
        masks = th.ones_like(labels[:, :, 0], dtype=th.bool)
        masks[:, -1] = False
        batch_size = labels.shape[0]
        num_relations_per_sample = labels.shape[1]

        model_kwargs = dict(
            y=labels.to(device).flatten(0, 1),
            masks=masks.to(device).flatten(0, 1),
        )

        def conditions_denoise_fn_factory(labels):
            conditions_denoise_fns = []
            for rel_idx in range(num_relations_per_sample-1):  # ignore the unconditioned label
                current_label = labels[:, rel_idx, :]
                current_mask = masks[:, rel_idx]
                model_kwargs = dict(
                    y=current_label.to(device),
                    masks=current_mask.to(device),
                )
                conditions_denoise_fns.append(lambda x_t, ts: model(x_t, ts, **model_kwargs))
            return conditions_denoise_fns

        def composed_model_fn(x_t, ts):
            combined = th.repeat_interleave(x_t, num_relations_per_sample, dim=0)
            ts = th.repeat_interleave(ts, num_relations_per_sample, dim=0)
            model_out = model(combined, ts, **model_kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            masks = model_kwargs.get('masks')
            cond_eps, uncond_eps = eps[masks], eps[~masks]
            uncond_eps = uncond_eps.view(batch_size, -1, *uncond_eps.shape[1:])
            cond_eps = cond_eps.view(batch_size, -1, *cond_eps.shape[1:])
            eps = uncond_eps + (cfg.cfg_weight * (cond_eps - uncond_eps)).sum(dim=1, keepdim=True)
            # -> (batch_size, 1, 3, H, W)
            eps = eps.flatten(0, 1)
            rest = rest[~masks]
            return th.cat([eps, rest], dim=1)

        samples = baselines_clevr.diffusion_baseline(
            denoise_fn=composed_model_fn,
            diffusion=diffusion,
            x_shape=(3, options["image_size"], options["image_size"]),
            eval_batch_size=cfg.batch_size,
            clip_denoised=True,
            progress=True,
            callback=None,
        )[-1]

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

        return samples

    # Sampling loop
    img_idx = 0
    for batch_labels, batch_captions in tqdm(dataloader, desc="Generating samples"):
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
        for i, (sample, caption) in enumerate(zip(samples, batch_captions)):
            save_image(sample, output_dir / f'sample_{img_idx:05d}.png')
            with open(output_dir / f'sample_{img_idx:05d}.txt', 'w') as f:
                f.write(caption)
            img_idx += 1

    print(f"Generated {img_idx} samples. Saved to {output_dir}")


if __name__ == '__main__':
    main()
