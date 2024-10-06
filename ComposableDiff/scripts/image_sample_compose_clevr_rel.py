import numpy as np

import argparse
import torch as th

from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict
)
from composable_diffusion.image_datasets import load_data

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
        return label, self.convert_caption(label)

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


def main():

    parser = argparse.ArgumentParser()
    defaults = model_and_diffusion_defaults()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--weights', type=float, nargs="+", default=[7.5])
    parser.add_argument('--data_path', type=str, default='./dataset/test_clevr_rel_5000_3.npz')

    args = parser.parse_args()

    # Setup
    th.set_float32_matmul_precision('high')
    th.set_grad_enabled(False)
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')

    options = args_to_dict(args, defaults.keys())

    # Dataset specific options
    if "clevr_rel" in args.data_path:
        options.update(dict(
            dataset='clevr_rel',
            use_fp16=th.cuda.is_available(),
            timestep_respacing='100',  # use 100 diffusion steps for fast sampling
            num_classes='4,3,9,3,3,7'
        ))
    elif "clevr_pos" in args.data_path:
        options.update(dict(
            dataset='clevr_pos',
            use_fp16=th.cuda.is_available(),
            timestep_respacing='100',  # use 100 diffusion steps for fast sampling
            num_classes='2'
        ))
    else:
        raise ValueError("Unknown dataset")

    model, diffusion = create_model_and_diffusion(**options)

    model.eval()
    if options['use_fp16']:
        model.convert_to_fp16()
    model.to(device)

    if "clevr_rel" in args.data_path:
        args.ckpt_path = "models/clevr_rel.pt"
    elif "clevr_pos" in args.data_path:
        args.ckpt_path = "models/clevr_pos.pt"

    print(f'Loading checkpoint from {args.ckpt_path}')
    checkpoint = th.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    print('Total base parameters', sum(x.numel() for x in model.parameters()))

    # Create output directory
    output_dir = Path(args.output_dir)
    experiment_name = args.data_path.split('/')[-1].split('.')[0]
    output_dir = output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset and dataloader
    if "clevr_pos" in args.data_path:
        dataset = CLEVRPosDataset(data_path=args.data_path)
    elif "clevr_rel" in args.data_path:
        dataset = CLEVRRelDataset(data_path=args.data_path)
    else:
        raise ValueError("Unknown dataset")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Sampling function
    def sample_batch(model, diffusion, labels):
        # add zeros to the labels for unconditioned sampling
        labels = th.cat([labels, th.zeros_like(labels[:, :1, :])], dim=1)
        masks = th.ones_like(labels[:, :, 0], dtype=th.bool)
        masks[:, -1] = False
        batch_size = labels.shape[0]
        num_relations_per_sample = labels.shape[1]
        full_batch_size = batch_size * num_relations_per_sample

        weights = th.tensor(args.weights).reshape(-1, 1, 1, 1).to(device)

        model_kwargs = dict(
            y=labels.float().to(device).flatten(0, 1),
            masks=masks.to(device).flatten(0, 1),
        )

        def model_fn(x_t, ts, **kwargs):
            half = x_t[::num_relations_per_sample]
            combined = th.repeat_interleave(half, num_relations_per_sample, dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            masks = kwargs.get('masks')
            cond_eps, uncond_eps = eps[masks], eps[~masks]
            uncond_eps = uncond_eps.view(batch_size, -1, *uncond_eps.shape[1:])
            cond_eps = cond_eps.view(batch_size, -1, *cond_eps.shape[1:])
            half_eps = uncond_eps + (weights * (cond_eps - uncond_eps)).sum(dim=1, keepdim=True)
            # -> (batch_size, 1, 3, H, W)
            eps = th.repeat_interleave(half_eps, num_relations_per_sample, dim=1).view(-1, *half_eps.shape[2:])
            return th.cat([eps, rest], dim=1)

        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )
        samples = samples[::labels.shape[1]]

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

        # Save individual images with captions
        for i, (sample, caption) in enumerate(zip(samples, batch_captions)):
            save_image(sample, output_dir / f'sample_{img_idx:05d}.png')
            with open(output_dir / f'sample_{img_idx:05d}.txt', 'w') as f:
                f.write(caption)
            img_idx += 1

    # Save grid of samples
    grid = make_grid(samples, nrow=int(np.sqrt(len(samples))), padding=2)
    save_image(grid, f'{experiment_name}.png')

    print(f"Generated {img_idx} samples. Saved to {output_dir}")


if __name__ == '__main__':
    main()
