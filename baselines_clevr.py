from tqdm.auto import tqdm
import numpy as np
import types
from copy import deepcopy
from typing import Union, Callable

import ComposableDiff.composable_diffusion
import ComposableDiff.composable_diffusion.gaussian_diffusion
import ComposableDiff.composable_diffusion.respace
import torch
from typing import Tuple, List, Dict
from elbo import calculate_elbo
import ComposableDiff
from anneal_samplers import AnnealedUHASampler, AnnealedULASampler
from utils_clevr import get_device


# for more precise elbo estimation
def disable_tf32(func):
    def wrapper(*args, **kwargs):
        old_tf32_cuda = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            return func(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_cuda
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
    return wrapper


@disable_tf32
@torch.autocast("cuda", enabled=False)
def diffusion_baseline(denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                       diffusion: ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                       x_shape: Tuple[int, ...],
                       eval_batch_size: int=256,
                       clip_denoised: bool=True,
                       progress: bool = True,
                       callback: Union[None, Callable[[torch.Tensor, int], torch.Tensor]] = None,
                       ) -> List[torch.Tensor]:
    """diffusion baseline

    Args:
        denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function, takes (x, t) as input and returns predicted noise
        diffusion (ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion): gaussian diffusion procedure, basically a noise scheduler
        x_shape (Tuple[int, ...]): shape of each sample
        eval_batch_size (int, optional): number of samples to generate. Defaults to 8000.
        clip_denoised (bool, optional): whether to clip the denoised samples. Defaults to False.
        progress (bool, optional): whether to show the progress bar. Defaults to True.
        callback (Union[None, Callable[[torch.Tensor, int], torch.Tensor]], optional): callback function to modify the samples. Defaults to None.

    Returns:
        List[torch.Tensor]: generated samples at each timestep, from the start timestep to the end timestep through reverse diffusion process,
                            all the samples are stored in CPU to save GPU memory.
    """

    device = get_device()
    sample = torch.randn((eval_batch_size,) + x_shape, device=device)
    indices = list(range(diffusion.num_timesteps))[::-1]

    if progress:
        indices = tqdm(indices)

    samples = [sample.cpu()]
    for i in indices:
        t = torch.full((len(sample),), i, dtype=torch.long, device=device)
        if len(sample):
            with torch.no_grad():
                out = diffusion.p_sample(
                    denoise_fn,
                    sample,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=None,
                )
                sample = out["sample"]
        if callback is not None:
            sample = callback(sample, t)
        samples.append(sample.cpu())
    return samples


@disable_tf32
@torch.autocast("cuda", enabled=False)
def ebm_baseline(composed_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 x_shape: Tuple[int, ...],
                 noise_scheduler: Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                                        ComposableDiff.composable_diffusion.respace.SpacedDiffusion],
                 num_samples_to_generate: int,
                 ebm_cfg: dict[str, Union[float, str]],
                 progress: bool,
                 ) -> List[torch.Tensor]:
    """ebm baseline

    Args:
        composed_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the composed model
        x_shape (Tuple[int, ...]): shape of each sample
        noise_scheduler (Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, ComposableDiff.composable_diffusion.respace.SpacedDiffusion]): noise scheduler
        num_samples_to_generate (int): number of samples to generate
        ebm_cfg (dict[str, Union[float, str]]): configuration for the ebm
        progress (bool): whether to show the progress bar

    Returns:
        List[torch.Tensor]: generated samples
    """

    # Hypeprparameters For Samplers : Need to be tuned carefully for generating good proposals
    num_steps = 100

    #ULA
    la_steps = 20
    la_step_sizes = noise_scheduler.betas * 2

    #UHMC SAMPLER
    ha_steps = 10
    num_leapfrog_steps = 3
    damping_coeff = 0.7
    mass_diag_sqrt = noise_scheduler.betas
    ha_step_sizes = noise_scheduler.betas * 0.1

    def gradient_function(x, t):
        scalar = 1 / noise_scheduler.sqrt_one_minus_alphas_cumprod
        eps = composed_denoise_fn(x, t).chunk(2, dim=1)[0]
        scale = scalar[t[0]]
        return -1*scale*eps

    if ebm_cfg.sampler_type == 'ULA':
        sampler = AnnealedULASampler(num_steps, la_steps, la_step_sizes, gradient_function)
    elif ebm_cfg.sampler_type == 'UHMC':
        sampler = AnnealedUHASampler(num_steps,
                    ha_steps,
                    ha_step_sizes,
                    damping_coeff,
                    mass_diag_sqrt,
                    num_leapfrog_steps,
                    gradient_function,
                )
    else:
        raise ValueError(f"Sampler type {ebm_cfg.sampler_type} not supported")

    def callback(x, t):
        if t[0] > 50:
            return sampler.sample_step(x, t[0], t, model_args={})
        else:
            return x

    samples = diffusion_baseline(composed_denoise_fn, noise_scheduler,
                                 x_shape=x_shape,
                                 eval_batch_size=num_samples_to_generate,
                                 callback=callback,
                                 progress=progress)
    return samples


def create_intervene_timesteps(method: str,
                               T: int,
                               timesteps_to_select: int):
    """method to create intervention timesteps

    Args:
        method (str): method to create intervention timesteps, should be one of ['uniform', 'latest']
        T (int): number of total timesteps

    Returns:
        List[int]: intervention timesteps
    """
    assert method in ["uniform", "latest"], "method should be one of ['uniform', 'latest'], but got {}".format(method)

    if method == "uniform":
        interval = T // timesteps_to_select
        return list(range(0, T, interval))
    elif method == "latest":
        return list(range(0, timesteps_to_select))


@disable_tf32
@torch.autocast("cuda", enabled=False)
def make_estimate_lift(elbo_cfg: dict[str, Union[bool, str]],
                            noise_scheduler: Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                                                    ComposableDiff.composable_diffusion.respace.SpacedDiffusion],
                            unconditioned_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            progress: bool=False) -> Callable[[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    """Creates a function to estimate the lift using ELBO.

    Args:
        elbo_cfg (dict[str, Union[bool, str]]): Configuration for ELBO estimation, including whether to use CFG, number of samples, mini batch size, etc.
        noise_scheduler (Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, ComposableDiff.composable_diffusion.respace.SpacedDiffusion]): Noise scheduler for the diffusion process.
        unconditioned_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Denoising function for the unconditioned model.
        progress (bool, optional): Whether to show the progress bar. Defaults to False.

    Returns:
        Callable[[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]: Function to estimate the lift.
    """
    def estimate_lift(denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            x: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.zeros((0,), dtype=x.dtype, device=x.device)
        assert len(t.unique()) == 1, "t should be the same for all samples, but got {}".format(t.unique())
        log_px_given_c = calculate_elbo(denoise_fn,
                                        noise_scheduler,
                                        x=x,
                                        seed=t[0],
                                        mini_batch=elbo_cfg["mini_batch"],
                                        n_samples=elbo_cfg["n_samples"],
                                        same_noise=elbo_cfg["same_noise"],
                                        sample_timesteps=elbo_cfg["sample_timesteps"],
                                        progress=progress)
        log_px = calculate_elbo(unconditioned_denoise_fn,
                                noise_scheduler,
                                x=x,
                                seed=t[0],
                                mini_batch=elbo_cfg["mini_batch"],
                                n_samples=elbo_cfg["n_samples"],
                                same_noise=elbo_cfg["same_noise"],
                                sample_timesteps=elbo_cfg["sample_timesteps"],
                                progress=progress)
        lift = log_px_given_c - log_px
        return lift
    return estimate_lift


@disable_tf32
@torch.autocast("cuda", enabled=False)
def rejection_baseline(composed_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                unconditioned_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                conditions_denoise_fn: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                                x_shape: Tuple[int, ...],
                                noise_scheduler: Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                                                       ComposableDiff.composable_diffusion.respace.SpacedDiffusion],
                                num_samples_to_generate: int,
                                rejection_scheduler_cfg: dict[str, Union[float, str]],
                                elbo_cfg: dict[str, Union[bool, str]],
                                progress: bool,
        ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[int, np.ndarray], Dict[int, List[np.ndarray]]]:
    """rejection sampling baseline

    Args:
        composed_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the composed model
        unconditioned_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the unconditioned model
        conditions_denoise_fn (List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): denoising function for multiple conditions
        x_shape (Tuple[int, ...]): shape of each sample
        noise_scheduler (ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, optional): noise scheduler. Defaults to None.
        num_samples_to_generate (int, optional): number of samples to generate for each trial.
        rejection_scheduler_cfg (dict[str, Union[float, str]]): configuration for the rejection scheduler
        elbo_cfg (dict[str, Union[bool, str]]): configuration for the elbo estimation
        progress (bool): whether to show the progress bar

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor], Dict[int, np.ndarray], Dict[int, List[np.ndarray]]]:
        final_samples, unfiltered_samples, need_to_remove_across_timesteps, energies_across_timesteps
    """

    rejection_timesteps = create_intervene_timesteps(**rejection_scheduler_cfg, T=noise_scheduler.num_timesteps)
    if isinstance(noise_scheduler, ComposableDiff.composable_diffusion.respace.SpacedDiffusion):
        rejection_timesteps_unspaced = [noise_scheduler.timestep_map[t] for t in rejection_timesteps]
    else:
        rejection_timesteps_unspaced = rejection_timesteps

    estimate_lift = make_estimate_lift(elbo_cfg, noise_scheduler, unconditioned_denoise_fn)

    need_to_remove_across_timesteps = {}
    energies_across_timesteps = {}

    def callback(x, t):
        if t[0] in rejection_timesteps:
            unspaced_t = rejection_timesteps_unspaced[rejection_timesteps.index(t[0])]
            energies = [estimate_lift(denoise_fn, x,
                                          t=torch.full((len(x),), unspaced_t, dtype=torch.long, device=x.device))
                                          for denoise_fn in conditions_denoise_fn]
            energies_across_timesteps[t[0].item()] = [e.cpu().numpy() for e in energies]
            need_to_remove = (torch.stack(energies, dim=0) <= 0).any(dim=0)
            need_to_remove_across_timesteps[t[0].item()] = need_to_remove.cpu().numpy()

        return x

    unfiltered_samples = diffusion_baseline(composed_denoise_fn,
                                            noise_scheduler,
                                            x_shape,
                                            eval_batch_size=num_samples_to_generate,
                                            callback=callback,
                                            progress=progress)

    need_to_remove = torch.from_numpy(np.array(list(need_to_remove_across_timesteps.values()))).any(dim=0)
    return unfiltered_samples[-1][~need_to_remove], unfiltered_samples, need_to_remove_across_timesteps, energies_across_timesteps


@disable_tf32
@torch.autocast("cuda", enabled=False)
def cache_rejection_baseline(composed_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                unconditioned_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                conditions_denoise_fn: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                                cache_callback: Callable[[], List[torch.Tensor]],
                                x_shape: Tuple[int, ...],
                                noise_scheduler: Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                                                       ComposableDiff.composable_diffusion.respace.SpacedDiffusion],
                                num_samples_to_generate: int,
                                elbo_cfg: dict[str, Union[bool, str]],
                                progress: bool) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cached rejection sampling baseline that stores intermediate scores for efficient filtering

    Args:
        composed_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Cached composition model
        unconditioned_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Denoising function for the unconditioned model
        conditions_denoise_fn (List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): Denoising function for multiple conditions
        cache_callback (Callable[[], List[torch.Tensor]]): Callback function to get cached scores and samples
        x_shape (Tuple[int, ...]): Shape of each sample
        noise_scheduler (Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, ComposableDiff.composable_diffusion.respace.SpacedDiffusion]): Noise scheduler
        num_samples_to_generate (int): Number of samples to generate
        elbo_cfg (dict[str, Union[bool, str]]): Configuration for ELBO estimation
        progress (bool): Whether to show progress bar

    Returns:
        Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]: Filtered samples, final samples, is valid, energies
    """

    device = get_device()
    def estimate_lift(cached_cond_scores: torch.Tensor,
                        cached_uncond_scores: torch.Tensor,
                        noise: torch.Tensor) -> torch.Tensor:
        if cached_cond_scores.shape[0] == 0:
            return torch.zeros((0,), dtype=cached_cond_scores.dtype, device=cached_cond_scores.device)

        B = cached_cond_scores.shape[1]
        log_px_given_c = torch.zeros(B, device=cached_cond_scores.device)
        log_px = torch.zeros(B, device=cached_cond_scores.device)
        mini_batch = 20
        for i in range(0, B, mini_batch):
            log_px_given_c[i:i+mini_batch] = -(cached_cond_scores[:, i:i+mini_batch] - noise[:, i:i+mini_batch]).pow(2).mean(dim=(0, 2, 3, 4))
            log_px[i:i+mini_batch] = -(cached_uncond_scores[:, i:i+mini_batch] - noise[:, i:i+mini_batch]).pow(2).mean(dim=(0, 2, 3, 4))
        lift = log_px_given_c - log_px
        return lift

    sample_timesteps = elbo_cfg["sample_timesteps"]
    if sample_timesteps in ["interleave", "random"]:
        log_timesteps = list(range(noise_scheduler.num_timesteps))
    elif sample_timesteps.startswith("specified"):
        # sample timestep from the specified timesteps
        specified_timesteps = [int(s) for s in sample_timesteps.split("specified")[1].split(",")]
        log_timesteps = [noise_scheduler.timestep_map.index(t) for t in specified_timesteps]

    # Store cached scores during diffusion
    cached_scores = [[] for _ in range(1 + len(conditions_denoise_fn))]  # the final cached score is the unconditioned score
    cached_xs = []
    cached_ts = []
    cached_mask = []
    def callback(x, t):
        cached_mask.append(t[0] in log_timesteps)
        if t[0] in log_timesteps:
            cached_ts.append(t[0])
            for i in range(len(cached_scores)):
                cached_scores[i].append(cache_callback()["cached_scores"][i].clone())
            cached_xs.append(cache_callback()["cached_xs"].clone())
        return x

    # Run diffusion process and collect samples
    generated_samples = diffusion_baseline(composed_denoise_fn,
                                         diffusion=noise_scheduler,
                                         x_shape=x_shape,
                                         eval_batch_size=num_samples_to_generate,
                                         callback=callback,
                                         progress=progress)

    final_samples = generated_samples[-1].to(device=device)
    cached_ts = torch.stack(cached_ts).to(device=device)
    cached_mask = torch.BoolTensor(cached_mask).to(device=device)
    cached_scores = torch.stack([torch.stack(cached_scores[i]) for i in range(len(cached_scores))])
    cached_xs = torch.stack(cached_xs)

    # Convert to torch tensors for filtering
    sqrt_alpha_bar = torch.from_numpy(noise_scheduler.sqrt_alphas_cumprod).to(device=device)[cached_ts]
    sqrt_one_minus_alpha_bar = torch.from_numpy(noise_scheduler.sqrt_one_minus_alphas_cumprod).to(device=device)[cached_ts]

    # Calculate noise
    noise = (cached_xs - sqrt_alpha_bar.view(-1, 1, *[1] * len(x_shape)) * final_samples) / sqrt_one_minus_alpha_bar.view(-1, 1, *[1] * len(x_shape))

    cached_cond_scores = cached_scores
    cached_uncond_scores = cached_scores[-1]

    # Filter samples
    energies = [estimate_lift(cached_cond_scores[i],
                              cached_uncond_scores,
                              noise) for i in range(len(conditions_denoise_fn))]
    energies = torch.stack(energies, dim=0)

    is_valid = (energies > 0).all(dim=0)
    is_valid = is_valid.cpu().numpy()
    filtered_samples = final_samples[is_valid]

    return filtered_samples, final_samples, is_valid, energies
