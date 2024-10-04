import os
import numpy as np
import argparse
import torch as th

from composable_diffusion.download import load_checkpoint
from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
    add_dict_to_argparser,
    args_to_dict
)
from composable_diffusion.download import download_model

from torchvision.utils import make_grid, save_image
from PIL import Image
from pathlib import Path
from tqdm import tqdm

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100'  # use 100 diffusion steps for fast sampling
options['num_classes'] = '2'

parser = argparse.ArgumentParser()
add_dict_to_argparser(parser, options)
parser.add_argument('--weights', type=float, nargs="+", default=7.5)

args = parser.parse_args()

options = args_to_dict(args, model_and_diffusion_defaults().keys())
model, diffusion = create_model_and_diffusion(**options)

model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)

print(f'loading model')
checkpoint = th.load(download_model('clevr_pos'), map_location='cpu')
model.load_state_dict(checkpoint)

print('total base parameters', sum(x.numel() for x in model.parameters()))


def show_images(batch: th.Tensor, file_name: str = 'result.png'):
    """Display a batch of images inline."""
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    Image.fromarray(reshaped.numpy()).save(file_name)


batch_size = 1


# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
# upsample_temp = 0.997
upsample_temp = 0.980

##############################
# Sample from the base model #
##############################

# Create the position label
weights = args.weights
# [-1, -1] is unconditional score
positions = [[0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5], [0.9, 0.5], [-1, -1]]  # horizontal
if isinstance(weights, float):
    weights = [weights] * (len(positions) - 1)
assert len(weights) == 1 or len(weights) == len(positions) - 1, \
    "the number of weights should be the same as the number of prompts."
full_batch_size = batch_size * len(positions)
masks = [True] * (len(positions) - 1) + [False]
weights = th.tensor(weights).reshape(-1, 1, 1, 1).to(device)

model_kwargs = dict(
    y=th.tensor(positions, dtype=th.float, device=device),
    masks=th.tensor(masks, dtype=th.bool, device=device)
)


def model_fn(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs['y'].size(0), dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    # assume weights are equal to guidance scale
    half_eps = uncond_eps + (weights * (cond_eps - uncond_eps)).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)
    return th.cat([eps, rest], dim=1)


# Sample from the base model.
number_images = 32
all_samples = []
for i in range(number_images):
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]

    all_samples.append(samples)

samples = ((th.cat(all_samples, dim=0) + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu() / 255.
grid = make_grid(samples, nrow=int(samples.shape[0] ** 0.5), padding=0)
save_image(grid, f'clevr_pos_{options["image_size"]}.png')