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
from copy import deepcopy
import matplotlib.pyplot as plt
import ComposableDiff
from typing import Callable, Tuple


device = th.device('cuda' if th.cuda.is_available() else 'cpu')
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    th.autocast("cuda", dtype=th.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pyth.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if th.cuda.get_device_properties(0).major >= 8:
        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()


# utility function copied from segment anything model v2
def show_points(coords, ax, marker_size=375):
    ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_image(img, coord_label, path):
    plt.clf()
    plt.close('all')
    coord_label = coord_label.copy()
    coord_label[..., 0] = coord_label[..., 0] * 128
    coord_label[..., 1] = (1 - coord_label[..., 1]) * 128
    plt.imshow(img)
    if len(coord_label.shape) == 1:
        coord_label = coord_label[None, ...]
    show_points(coord_label, plt.gca())
    plt.savefig(path)


def conditions_denoise_fn_factory(model, labels, batch_size, cfg):
    # add zeros to the labels for unconditioned sampling
    labels = th.cat([labels, th.zeros_like(labels[:, :1, :])], dim=1).to(device)
    masks = th.ones_like(labels[:, :, 0], dtype=th.bool).to(device)
    masks[:, -1] = False
    num_relations_per_sample = labels.shape[1]
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


def to_tensor(data: Union[np.ndarray, th.Tensor]) -> th.Tensor:
    """convert data to th.Tensor

    Args:
        data (Union[np.ndarray, th.Tensor]): data

    Returns:
        th.Tensor: th.Tensor
    """
    if isinstance(data, np.ndarray):
        return th.from_numpy(data).to(device)
    elif isinstance(data, th.Tensor):
        return data.to(device)
    else:
        raise ValueError("data should be np.ndarray or torch.Tensor")


def calculate_denoising_matching_term(x_k: th.Tensor,
                                      true_noise_at_t: th.Tensor,
                                      pred_noise_at_k: th.Tensor,
                                      cumprod_alpha_t: th.Tensor,
                                      cumprod_alpha_k: th.Tensor) -> th.Tensor:
    """calculate the denoising matching term

    Args:
        x_k (th.Tensor): samples at timestep k (batch, n_features)
        true_noise_at_t (th.Tensor): true noise at timestep t (batch, n_features)
        pred_noise_at_k (th.Tensor): predicted noise at timestep k (batch, n_features)
        cumprod_alpha_t (th.Tensor): cumulative product of alpha at timestep t (batch,)
        cumprod_alpha_k (th.Tensor): cumulative product of alpha at timestep k (batch,)

    Returns:
        th.Tensor: denoising matching term (batch, n_features)
    """
    x_k_coeff = (cumprod_alpha_t - th.sqrt(cumprod_alpha_t)) * th.sqrt(1 - cumprod_alpha_k) / (cumprod_alpha_t - cumprod_alpha_k)
    # -> (batch,)
    noise_t_coeff = th.sqrt(1 - cumprod_alpha_k) / th.sqrt(cumprod_alpha_t - cumprod_alpha_k)
    # -> (batch,)

    if len(pred_noise_at_k.shape) > 2:  # an image
        pred_noise_at_k = pred_noise_at_k[:, :3]  # only use the first 3 channels, no learned sigma

    x_k_coeff = x_k_coeff.reshape(-1, *([1] * (len(x_k.shape) - 1)))
    noise_t_coeff = noise_t_coeff.reshape(-1, *([1] * (len(x_k.shape) - 1)))

    return x_k_coeff * x_k + noise_t_coeff * true_noise_at_t - pred_noise_at_k


def add_noise_at_t(noise_scheduler: ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                   x_t: th.Tensor,
                   x_noise: th.Tensor,
                   timesteps_t: th.Tensor,
                   timesteps_k: th.Tensor) -> th.Tensor:
    """add noise to the sample at time t

    Args:
        noise_scheduler (ComposableDiff.composable_diffusion.respace.SpacedDiffusion): noise scheduler
        x_t (th.Tensor): the sample at time t (batch_size, feature_size)
        x_noise (th.Tensor): add noise to the sample (batch_size, feature_size)
        timesteps_t (th.Tensor): the timesteps where x_t is at (batch_size,)
        timesteps_k (th.Tensor): the timesteps to add noise (batch_size,)

    Returns:
        th.Tensor: the noisy sample (batch_size, feature_size)
    """
    assert (timesteps_t <= timesteps_k).all(), "timesteps_t should be less than or equal to timesteps_k"

    device = x_t.device
    B, T = x_t.size(0), noise_scheduler.num_timesteps

    timesteps_t = timesteps_t.to(device)
    timesteps_k = timesteps_k.to(device)
    alphas = 1 - noise_scheduler.betas

    log_alphas = th.log(to_tensor(alphas).to(device))

    # -> (T,)
    log_alphas_batched = log_alphas[None, :].repeat(B, 1)
    log_alphas_batched = log_alphas_batched.to(device)
    # -> (batch_size, T)
    is_earlier_timesteps = (th.arange(T, device=device)[None, :] < timesteps_t[:, None])
    log_alphas_batched[is_earlier_timesteps] = 0
    log_alphas_cumsum = th.cumsum(log_alphas_batched, dim=-1)
    # -> (batch_size, T)
    alphas_cumprod = th.exp(log_alphas_cumsum)
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    # -> (batch_size, T)
    sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5
    # -> (batch_size, T)

    s1 = sqrt_alphas_cumprod[th.arange(B, device=device), timesteps_k]
    s2 = sqrt_one_minus_alphas_cumprod[th.arange(B, device=device), timesteps_k]

    s1 = s1.reshape(-1, *([1] * (len(x_t.shape) - 1)))
    s2 = s2.reshape(-1, *([1] * (len(x_t.shape) - 1)))

    s1 = s1.to(device).float()
    s2 = s2.to(device).float()

    return s1 * x_t + s2 * x_noise


def remove_noise_to_t(noise_scheduler: ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                      x_k: th.Tensor,
                      x_noise: th.Tensor,
                      timesteps_t: th.Tensor,
                      timesteps_k: th.Tensor) -> th.Tensor:
    """remove noise to the sample at time k

    Args:
        noise_scheduler (ComposableDiff.composable_diffusion.respace.SpacedDiffusion): noise scheduler
        x_t (th.Tensor): the sample at time t (batch_size, feature_size)
        x_noise (th.Tensor): add noise to the sample (batch_size, feature_size)
        timesteps_t (th.Tensor): the timesteps where x_t is at (batch_size,)
        timesteps_k (th.Tensor): the timesteps to add noise (batch_size,)

    Returns:
        th.Tensor: the noisy sample (batch_size, feature_size)
    """
    assert (timesteps_t <= timesteps_k).all(), "timesteps_t should be less than or equal to timesteps_k"

    device = x_k.device
    B, T = x_k.size(0), noise_scheduler.num_timesteps

    timesteps_t = timesteps_t.to(device)
    timesteps_k = timesteps_k.to(device)
    alphas = 1 - noise_scheduler.betas

    log_alphas = th.log(to_tensor(alphas).to(device))

    # -> (T,)
    log_alphas_batched = log_alphas[None, :].repeat(B, 1)
    log_alphas_batched = log_alphas_batched.to(device)
    # -> (batch_size, T)
    is_earlier_timesteps = (th.arange(T, device=device)[None, :] < timesteps_t[:, None])
    log_alphas_batched[is_earlier_timesteps] = 0
    log_alphas_cumsum = th.cumsum(log_alphas_batched, dim=-1)
    # -> (batch_size, T)
    alphas_cumprod = th.exp(log_alphas_cumsum)
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    # -> (batch_size, T)
    sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5
    # -> (batch_size, T)

    s1 = sqrt_alphas_cumprod[th.arange(B, device=device), timesteps_k]
    s2 = sqrt_one_minus_alphas_cumprod[th.arange(B, device=device), timesteps_k]

    s1 = s1.reshape(-1, *([1] * (len(x_k.shape) - 1)))
    s2 = s2.reshape(-1, *([1] * (len(x_k.shape) - 1)))

    s1 = s1.to(device).float()
    s2 = s2.to(device).float()

    return (x_k - s2 * x_noise) / s1


def get_s1_and_s2(noise_scheduler: ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                  t: int,
                  k: int) -> Tuple[th.Tensor, th.Tensor]:
    """get s1 and s2 for the diffusion process

    Args:
        noise_scheduler (ComposableDiff.composable_diffusion.respace.SpacedDiffusion): noise scheduler
        t (int): timestep t
        k (int): timestep k

    Returns:
        th.Tensor: s1, s2 (batch, n_features)
    """
    alphas = 1 - noise_scheduler.betas
    log_alphas = th.log(to_tensor(alphas).to(device))
    is_earlier_timesteps = (th.arange(noise_scheduler.num_timesteps, device=device) < t)
    log_alphas[is_earlier_timesteps] = 0

    log_alphas_cumsum = th.cumsum(log_alphas, dim=0)
    alphas_cumprod = th.exp(log_alphas_cumsum)
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5

    s1 = sqrt_alphas_cumprod[k]
    s2 = sqrt_one_minus_alphas_cumprod[k]
    return s1, s2


@th.no_grad()
def calculate_elbo(model: th.nn.Module,
                   noise_scheduler: Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                                          ComposableDiff.composable_diffusion.respace.SpacedDiffusion],
                   x_t: th.Tensor,
                   t: int,
                   n_samples: int,
                   seed: int,
                   mini_batch: int,
                   same_noise: bool,
                   sample_timesteps: str,
                   progress: bool=False) -> th.Tensor:
    """calculate the approximate ELBO

    Args:
        model (th.nn.Module): a diffusion model
        noise_scheduler (Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                               ComposableDiff.composable_diffusion.respace.SpacedDiffusion]): noise scheduler
        x_t (th.Tensor): samples (batch, n_features)
        t (int): timestep where x_t is at
        n_samples (int): number of samples used for Monte Carlo estimation
        cumprod_alpha (th.Tensor): cumulative product of alpha (T,)
        seed (int): random seed
        mini_batch (int): mini batch size
        same_noise (bool): whether to use the same noise for all samples
        sample_timesteps (str): how to sample the timesteps, "random" or "interleave"
        progress (bool, optional): whether to show the progress bar. Defaults to False.

    Returns:
        th.Tensor: approximate ELBO (batch,)
    """
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    if isinstance(noise_scheduler, ComposableDiff.composable_diffusion.respace.SpacedDiffusion):
        noise_scheduler = noise_scheduler.base_diffusion

    B, *D = x_t.shape

    if same_noise:
        # sample noise (batch, n_sample, n_features)
        # following https://arxiv.org/pdf/2305.15241, use the same noise for all samples
        noise = th.randn(1, 1, *D, device=x_t.device).expand(B, n_samples, *D)
    else:
        noise = th.randn(1, n_samples, *D, device=x_t.device).expand(B, n_samples, *D)

    T = noise_scheduler.num_timesteps
    if sample_timesteps == "interleave":
        # interleave the samples
        ts_k = th.linspace(t, T-1, n_samples, device=x_t.device).round().long().clamp(t, T-1)
        ts_k = ts_k[None, :].expand(B, n_samples)
    elif sample_timesteps == "random":
        # sample timestep randomly from [t, T): (batch, n_sample)
        ts_k = th.randint(t, T, (1, n_samples), device=x_t.device).expand(B, n_samples)
    else:
        raise ValueError("sample_timesteps should be 'random' or 'interleave'")

    # estimate the ELBO
    cumprod_alpha_prev = to_tensor(noise_scheduler.alphas_cumprod_prev).to(x_t.device).float()
    cumprod_alpha = to_tensor(noise_scheduler.alphas_cumprod).to(x_t.device).float()

    denoising_matching_terms = th.zeros(B * n_samples, device=x_t.device)

    # vectorized_x_k = x_k.flatten(0, 1)
    vectorized_ts_k = ts_k.flatten()
    vectorized_noise = noise.flatten(0, 1)
    vectorized_ts_t = th.full((B * n_samples,), t, device=x_t.device)
    vectorized_x_k_idx = th.arange(B, device=x_t.device)[:, None].expand(B, n_samples).flatten()

    if progress:
        iterator = tqdm(range(0, B * n_samples, mini_batch))
    else:
        iterator = range(0, B * n_samples, mini_batch)

    for i in iterator:
        # Prepare mini-batch
        batch_ts_k = vectorized_ts_k[i:i + mini_batch]
        batch_noise = vectorized_noise[i:i + mini_batch]
        batch_ts_t = vectorized_ts_t[i:i + mini_batch]
        batch_x_t = x_t[vectorized_x_k_idx[i:i + mini_batch]]
        batch_x_k = add_noise_at_t(noise_scheduler,
                                   batch_x_t,
                                   batch_noise,
                                   batch_ts_t,
                                   batch_ts_k)

        # Model prediction
        batch_noise_pred = model(batch_x_k, batch_ts_k)

        batch_term = calculate_denoising_matching_term(
            x_k=batch_x_k,
            true_noise_at_t=batch_noise,
            pred_noise_at_k=batch_noise_pred,
            cumprod_alpha_t=cumprod_alpha_prev[batch_ts_t],
            cumprod_alpha_k=cumprod_alpha[batch_ts_k]
        )
        # -> (batch * min(mini_batch, n_samples - i), n_features)
        batch_term = batch_term.pow(2)
        clipped_batch_term = batch_term[:, :, 25:65, 32:72]
        batch_term = clipped_batch_term.reshape(batch_term.shape[0], -1).sum(dim=1)
        # -> (batch * min(mini_batch, n_samples - i),)
        denoising_matching_terms[i:i + mini_batch] = batch_term
        # -> (batch, min(mini_batch, n_samples - i))

    denoising_matching_term = denoising_matching_terms.view(B, n_samples)
    assert denoising_matching_term.shape == (B, n_samples)
    # -> (batch, n_sample)

    # for i in range(n_samples):
    #     im_pos = plot_energy_histogram(denoising_matching_term[:10, i].cpu().numpy())
    #     im_neg = plot_energy_histogram(denoising_matching_term[10:, i].cpu().numpy())
    #     # log to wandb
    #     k = ts_k[0, i]
    #     wandb.log({"energy_histogram_pos": [wandb.Image(im_pos, caption=f"timestep {k}")],
    #                "energy_histogram_neg": [wandb.Image(im_neg, caption=f"timestep {k}")]})

    denoising_matching_term = denoising_matching_term.mean(dim=1)
    # -> (batch,)

    return -denoising_matching_term


def make_estimate_neg_logp(elbo_cfg: dict[str, Union[bool, str]],
                            noise_scheduler: Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                                                    ComposableDiff.composable_diffusion.respace.SpacedDiffusion],
                            unconditioned_denoise_fn: Callable[[th.Tensor, th.Tensor], th.Tensor],
                            mini_batch: int,
                            progress: bool=False) -> Callable[[Callable[[th.Tensor, th.Tensor], th.Tensor], th.Tensor, th.Tensor], th.Tensor]:
    """Creates a function to estimate the negative log probability using ELBO.

    Args:
        elbo_cfg (dict[str, Union[bool, str]]): Configuration for ELBO estimation, including whether to use CFG, number of samples, and other parameters.
        noise_scheduler (Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, ComposableDiff.composable_diffusion.respace.SpacedDiffusion]): Noise scheduler for the diffusion process.
        unconditioned_denoise_fn (Callable[[th.Tensor, th.Tensor], th.Tensor]): Denoising function for the unconditioned model.
        mini_batch (int): Mini batch size for ELBO estimation.
        progress (bool, optional): Whether to show the progress bar. Defaults to False.

    Returns:
        Callable[[Callable[[th.Tensor, th.Tensor], th.Tensor], th.Tensor, th.Tensor], th.Tensor]: Function to estimate the negative log probability.
    """
    def estimate_neg_logp(denoise_fn: Callable[[th.Tensor, th.Tensor], th.Tensor],
                            x: th.Tensor,
                            t: th.Tensor) -> th.Tensor:
        if x.shape[0] == 0:
            return th.zeros((0,), dtype=x.dtype, device=x.device)
        assert len(t.unique()) == 1, "t should be the same for all samples, but got {}".format(t.unique())
        if elbo_cfg["use_cfg"]:
            log_px_given_c = calculate_elbo(denoise_fn,
                                            noise_scheduler,
                                            x_t=x,
                                            t=t[0],
                                            seed=t[0],
                                            mini_batch=mini_batch,
                                            n_samples=elbo_cfg["n_samples"],
                                            same_noise=elbo_cfg["same_noise"],
                                            sample_timesteps=elbo_cfg["sample_timesteps"],
                                            progress=progress)
            log_px = calculate_elbo(unconditioned_denoise_fn,
                                    noise_scheduler,
                                    x_t=x,
                                    t=t[0],
                                    seed=t[0],
                                    mini_batch=mini_batch,
                                    n_samples=elbo_cfg["n_samples"],
                                    same_noise=elbo_cfg["same_noise"],
                                    sample_timesteps=elbo_cfg["sample_timesteps"],
                                    progress=progress)
            log_pc_given_x = log_px_given_c - log_px
            return -log_pc_given_x
        else:
            log_px_given_c = calculate_elbo(denoise_fn,
                                            noise_scheduler,
                                            x_t=x,
                                            t=t[0],
                                            seed=t[0],
                                            mini_batch=mini_batch,
                                            n_samples=elbo_cfg["n_samples"],
                                            same_noise=elbo_cfg["same_noise"],
                                            sample_timesteps=elbo_cfg["sample_timesteps"],
                                            progress=progress)
            return -log_px_given_c
    return estimate_neg_logp



@th.no_grad()
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

    dataset = CLEVRPosDataset(data_path=cfg.data_path)

    # Sampling loop
    condition_idx = 1
    test_idx = 1
    positive_ims = [np.array(Image.open(f"runs/10-21_17-39-55/test_clevr_pos_5000_5/original_sample_{test_idx:05d}_{i:05d}.png").convert("RGB")) for i in [3, 6, 20, 31, 45, 46, 69, 76, 77, 82]]
    negative_ims = [np.array(Image.open(f"runs/10-21_17-39-55/test_clevr_pos_5000_5/original_sample_{test_idx:05d}_{i:05d}.png").convert("RGB")) for i in [81, 83, 53, 29, 16, 13, 11, 00, 74, 47]]
    labels, _ = dataset[test_idx]

    conditions_denoise_fn = conditions_denoise_fn_factory(model, th.tensor(labels[np.newaxis], dtype=th.float32),
                                                          batch_size=cfg.mini_batch, cfg=cfg)
    estimate_neg_logp = make_estimate_neg_logp(elbo_cfg=cfg.elbo,
                                                            noise_scheduler=diffusion,
                                                            unconditioned_denoise_fn=conditions_denoise_fn[-1],
                                                            mini_batch=cfg.mini_batch,
                                                            progress=True)


    all_samples = th.from_numpy(np.stack(positive_ims + negative_ims)).to(device).permute(0, 3, 1, 2).float()
    # normalize the images
    all_samples = (all_samples / 255.0) * 2.0 - 1.0

    # set the region around label to 0
    coord = labels[condition_idx]
    coord = coord.copy()
    coord[0] = coord[0] * 128
    coord[1] = (1 - coord[1]) * 128
    coord = coord.astype(np.int32)

    # cropped_samples = th.zeros_like(all_samples)
    # slice_region_idx = slice(coord[1] - 20, coord[1] + 20), slice(coord[0] - 20, coord[0] + 20)
    # cropped_samples[:, :, slice_region_idx[0], slice_region_idx[1]] = all_samples[:, :, slice_region_idx[0], slice_region_idx[1]]
    # all_samples = cropped_samples

    # save the image
    unnormalized_all_samples = ((all_samples + 1.0) / 2.0).clamp(0, 1)
    grid = make_grid(unnormalized_all_samples, nrow=5)
    save_image(grid, output_dir / f"cropped_samples.png")

    packed_l2_distance = th.zeros((len(all_samples), 100, 1000), device=all_samples.device)

    # test add_noise_at_t
    for seed in tqdm(range(100)):
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        noise = th.randn_like(all_samples)
        timesteps_t = th.full((len(all_samples),), 0, dtype=th.long, device=all_samples.device)
        for k in range(1000):
            timesteps_k = th.full((len(all_samples),), k, dtype=th.long, device=all_samples.device)

            # load the L2 distance from pt file
            l2_distance = th.load(f"runs/10-24_01-54-35_100_seeds/test_clevr_pos_5000_5/energy_seed_{seed:05d}_timestep_{k:05d}.pt",
                                  weights_only=True)["l2_distance"]

            packed_l2_distance[:, seed, k] = l2_distance

            # # plot the histogram
            # im_pos = plot_energy_histogram(l2_distance[:len(positive_ims)].cpu().numpy())
            # im_neg = plot_energy_histogram(l2_distance[len(positive_ims):].cpu().numpy())

            # im_pos_clamp = plot_energy_histogram(l2_distance_clamp[:len(positive_ims)].cpu().numpy())
            # im_neg_clamp = plot_energy_histogram(l2_distance_clamp[len(positive_ims):].cpu().numpy())

            # # use scikit to calculate the auc
            # from sklearn.metrics import roc_auc_score
            # labels = [1] * len(positive_ims) + [0] * len(negative_ims)
            # auc = roc_auc_score(labels, -l2_distance.cpu().numpy())
            # auc_clamp = roc_auc_score(labels, -l2_distance_clamp.cpu().numpy())

            # # log to wandb
            # wandb.log({f"{seed}/l2_distance_pos": [wandb.Image(im_pos, caption=f"timestep {k}")],
            #            f"{seed}/l2_distance_neg": [wandb.Image(im_neg, caption=f"timestep {k}")],
            #            f"auc/{seed}": auc,
            #            f"{seed}/l2_distance_pos_clamp": [wandb.Image(im_pos_clamp, caption=f"timestep {k}")],
            #            f"{seed}/l2_distance_neg_clamp": [wandb.Image(im_neg_clamp, caption=f"timestep {k}")],
            #            f"auc_clamp/{seed}": auc_clamp})

    # log the auc across timesteps to wandb
    for k in range(1000):
        l2_distance = packed_l2_distance[:, :, k]
        l2_distance = l2_distance.mean(dim=1)
        labels = [1] * len(positive_ims) + [0] * len(negative_ims)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, -l2_distance.cpu().numpy())
        wandb.log({f"auc": auc,
                   "timestep": k})

    import pdb
    pdb.set_trace()

    # log the overall auc
    l2_distance = packed_l2_distance.mean(dim=(1, 2))
    labels = [1] * len(positive_ims) + [0] * len(negative_ims)
    overall_auc = roc_auc_score(labels, -l2_distance.cpu().numpy())
    wandb.log({"overall_auc": overall_auc})

    # rank_sample = th.argsort(packed_l2_distance, dim=0)
    # # plot the histogram of rank for each sample
    # for i in range(len(all_samples)):
    #     plt.clf()
    #     plt.close('all')
    #     plt.hist(rank_sample[i, :].flatten().cpu().numpy(), bins=len(all_samples))
    #     plt.savefig(output_dir / f"rank_sample_{i}.png")

    exit()

    all_energies = estimate_neg_logp(conditions_denoise_fn[condition_idx],
                                     all_samples,
                                     t=th.full((len(all_samples),), 0, dtype=th.long, device=all_samples.device))

    positive_energies = all_energies[:len(positive_ims)]
    negative_energies = all_energies[len(positive_ims):]

    print(f"Positive energies: {positive_energies}")
    print(f"Negative energies: {negative_energies}")


if __name__ == '__main__':
    main()
