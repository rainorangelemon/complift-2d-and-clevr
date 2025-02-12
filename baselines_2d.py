from tqdm.auto import tqdm
import numpy as np
from typing import Union

import ddpm
import torch
from typing import Tuple, List, Callable, Dict
from elbo import calculate_elbo
from anneal_samplers import AnnealedMALASampler, AnnealedULASampler, AnnealedUHASampler, AnnealedCHASampler


def diffusion_baseline(denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                       diffusion: ddpm.NoiseScheduler,
                       x_shape: Tuple[int, ...] = (2,),
                       eval_batch_size: int=8000,
                       progress: bool = True,
                       callback: Union[None, Callable[[torch.Tensor, int], torch.Tensor]]=None) -> List[np.ndarray]:
    """
    Diffusion baseline

    Args:
        denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The denoising function to test
        diffusion (ddpm.NoiseScheduler): The noise scheduler
        x_shape (Tuple[int, ...], optional): Shape of each sample. Defaults to (2,).
        eval_batch_size (int, optional): Batch size for evaluation. Defaults to 8000.
        progress (bool, optional): Whether to show the progress bar. Defaults to True.
        callback (Union[None, Callable[[torch.Tensor, int], torch.Tensor]], optional): Optional callback function. Defaults to None.

    Returns:
        List[np.ndarray]: List of samples
    """
    device = ddpm.device
    denoise_fn = denoise_fn.to(device)
    sample = torch.randn((eval_batch_size,) + x_shape).to(device)
    timesteps = list(range(diffusion.num_timesteps))[::-1]

    samples = [sample.cpu().numpy()]
    for i, t in enumerate(tqdm(timesteps)):
        if len(sample) != 0:
            t_tensor = torch.from_numpy(np.repeat(t, len(sample))).long().to(device)
            with torch.no_grad():
                residual = denoise_fn(sample, t_tensor)
            sample = diffusion.step(residual, t_tensor[0], sample)
            if callback is not None:
                sample = callback(sample, t_tensor)
        samples.append(sample.cpu().numpy())
    return samples


def ebm_baseline(composed_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                         composed_energy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                         x_shape: Tuple[int, ...],
                         noise_scheduler: ddpm.NoiseScheduler,
                         num_samples_to_generate: int,
                         sampler_type: str = "ULA",  # Can be "ULA", "MALA", "UHMC", or "HMC"
                         progress: bool = True) -> np.ndarray:
    """PyTorch implementation of EBM baseline using annealed samplers

    Args:
        composed_denoise_fn: Denoising function for the composed model
        x_shape: Shape of each sample
        noise_scheduler: Noise scheduler
        num_samples_to_generate: Number of samples to generate
        sampler_type: Type of sampler to use ("ULA", "MALA", "UHMC", or "HMC")
        progress: Whether to show progress bar

    Returns:
        np.ndarray: Generated samples
    """
    # Common parameters
    num_steps = 50  # Total noise levels
    steps_per_level = 10  # MCMC steps per noise level

    # ULA/MALA parameters
    la_step_size_scale = 0.001  # Base step size
    la_step_sizes = torch.ones_like(noise_scheduler.betas) * la_step_size_scale

    # UHMC/HMC parameters
    num_leapfrog_steps = 3  # Standard choice for moderate dimensionality
    damping_coeff = 0.5  # Standard choice for good mixing
    mass_diag_sqrt = torch.ones_like(noise_scheduler.betas)  # Constant mass
    ha_step_size_scale = 0.03  # Base step size
    ha_step_sizes = torch.ones_like(noise_scheduler.betas) * ha_step_size_scale

    @torch.no_grad()
    def gradient(x, t):
        scalar = 1 / noise_scheduler.sqrt_one_minus_alphas_cumprod
        eps = composed_denoise_fn(x, t)
        scale = scalar[t[0]]
        return -1 * scale * eps

    @torch.no_grad()
    def gradient_cha(x, t):
        """Gradient function for MALA and HMC samplers"""
        scalar = 1 / noise_scheduler.sqrt_one_minus_alphas_cumprod
        energy = composed_energy_fn(x, t)
        eps = composed_denoise_fn(x, t)
        scale = scalar[t[0]]
        return -1 * scale * energy, -1 * scale * eps

    if sampler_type == "MALA":
        sampler = AnnealedMALASampler(num_steps, steps_per_level, la_step_sizes, gradient_cha)
    elif sampler_type == "ULA":
        sampler = AnnealedULASampler(num_steps, steps_per_level, la_step_sizes, gradient)
    elif sampler_type in ["UHMC", "UHA"]:
        sampler = AnnealedUHASampler(num_steps,
                                    steps_per_level,
                                    ha_step_sizes,
                                    damping_coeff,
                                    mass_diag_sqrt,
                                    num_leapfrog_steps,
                                    gradient)
    elif sampler_type in ["HMC", "MUHA"]:
        sampler = AnnealedCHASampler(num_steps,
                                    steps_per_level,
                                    ha_step_sizes,
                                    damping_coeff,
                                    mass_diag_sqrt,
                                    num_leapfrog_steps,
                                    gradient_cha)
    else:
        raise ValueError(f"Sampler type {sampler_type} not supported")

    device = ddpm.device
    sample = torch.randn((num_samples_to_generate,) + x_shape).to(device)
    timesteps = list(range(noise_scheduler.num_timesteps))[::-1]

    for i, t in enumerate(tqdm(timesteps)):
        t_tensor = torch.from_numpy(np.repeat(t, len(sample))).long().to(device)
        sample = sampler.sample_step(sample, t, t_tensor, model_args={})

    return sample.cpu().numpy()


def make_estimate_lift(elbo_cfg: dict[str, Union[bool, str]],
                           noise_scheduler: ddpm.NoiseScheduler,
                           progress: bool=False) -> Callable[[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    """Creates a function to estimate the log of lift

    Args:
        elbo_cfg (dict[str, Union[bool, str]]): Configuration for ELBO estimation, including number of samples, mini batch size, etc.
        noise_scheduler (ddpm.NoiseScheduler): Noise scheduler.
        progress (bool, optional): Whether to show the progress bar. Defaults to False.

    Returns:
        Callable[[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]: Function to estimate the negative log probability.
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
        log_px = calculate_elbo(lambda x, t: elbo_cfg["alpha"] * denoise_fn(x, t),
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


def rejection_baseline(composed_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                conditions_denoise_fn: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                                algebras: list[str],
                                x_shape: Tuple[int, ...],
                                noise_scheduler: ddpm.NoiseScheduler,
                                num_samples_to_generate: int,
                                elbo_cfg: dict[str, Union[bool, str]],
                                progress: bool,
        ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[int, np.ndarray], Dict[int, List[np.ndarray]]]:
    """rejection sampling baseline

    Args:
        composed_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the composed model
        conditions_denoise_fn (List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): denoising function for multiple conditions
        algebras (list[str]): algebra to combine the conditions
        x_shape (Tuple[int, ...]): shape of each sample
        noise_scheduler (ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, optional): noise scheduler. Defaults to None.
        num_samples_to_generate (int, optional): number of samples to generate for each trial.
        elbo_cfg (dict[str, Union[bool, str]]): configuration for the elbo estimation
        progress (bool): whether to show the progress bar

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor], Dict[int, np.ndarray], Dict[int, List[np.ndarray]]]:
        final_samples, unfiltered_samples, need_to_remove_across_timesteps, energies_across_timesteps
    """
    assert all([algebra in ['product', 'summation', 'negation'] for algebra in algebras])

    estimate_lift = make_estimate_lift(elbo_cfg, noise_scheduler)

    energies_across_timesteps = {}
    unfiltered_samples = []

    def callback(x, t):
        unfiltered_samples.append(x.clone().cpu().numpy())
        if t[0] == 0:
            energies = [estimate_lift(denoise_fn, x, t) for denoise_fn in conditions_denoise_fn]
            energies_across_timesteps[t[0].item()] = [e.cpu().numpy() for e in energies]
            is_valid = torch.ones(len(x), dtype=torch.bool, device=x.device)
            for algebra_idx, algebra in enumerate(algebras):
                if algebra == "negation":
                    is_valid = is_valid & (energies[algebra_idx] <= 0)
                elif algebra == "product":
                    is_valid = is_valid & (energies[algebra_idx] > 0)
                elif algebra == "summation":
                    is_valid = is_valid | (energies[algebra_idx] > 0)
            filtered_x = x[is_valid]
        else:
            filtered_x = x
        return filtered_x

    samples = diffusion_baseline(composed_denoise_fn,
                                 noise_scheduler,
                                 x_shape,
                                 eval_batch_size=num_samples_to_generate,
                                 callback=callback,
                                 progress=progress)

    return samples, unfiltered_samples, len(samples[-1]) / len(unfiltered_samples[-1]), energies_across_timesteps


def cache_rejection_baseline(composed_denoise_fn: ddpm.CachedCompositionEnergyMLP,
                           algebras: list[str],
                           x_shape: Tuple[int, ...],
                           noise_scheduler: ddpm.NoiseScheduler,
                           num_samples_to_generate: int,
                           elbo_cfg: dict[str, Union[bool, str]],
                           progress: bool = True) -> Tuple[np.ndarray, float]:
    """Cached rejection sampling baseline that stores intermediate scores for efficient filtering

    Args:
        composed_denoise_fn (ddpm.CachedCompositionEnergyMLP): Cached composition model
        algebras (list[str]): List of algebras to combine the conditions
        x_shape (Tuple[int, ...]): Shape of each sample
        noise_scheduler (ddpm.NoiseScheduler): Noise scheduler
        num_samples_to_generate (int): Number of samples to generate
        elbo_cfg (dict[str, Union[bool, str]]): Configuration for ELBO estimation
        progress (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        Tuple[np.ndarray, float]: Filtered samples and acceptance ratio
    """
    def make_estimate_lift(elbo_cfg):
        def estimate_lift(cached_scores: torch.Tensor,
                            noise: torch.Tensor) -> torch.Tensor:
            if cached_scores.shape[0] == 0:
                return torch.zeros((0,), dtype=cached_scores.dtype, device=cached_scores.device)
            T, B, D = cached_scores.shape[0], cached_scores.shape[1], cached_scores.shape[2]
            log_px_given_c = torch.zeros(B, device=cached_scores.device)
            log_px = torch.zeros(B, device=cached_scores.device)
            mini_batch = 100
            for i in range(0, B, mini_batch):
                log_px_given_c[i:i+mini_batch] = -(cached_scores[:, i:i+mini_batch] - noise[:, i:i+mini_batch]).pow(2).mean(dim=(0, 2))
                log_px[i:i+mini_batch] = -(elbo_cfg["alpha"] * cached_scores[:, i:i+mini_batch] - noise[:, i:i+mini_batch]).pow(2).mean(dim=(0, 2))
            lift = log_px_given_c - log_px
            return lift
        return estimate_lift

    # Store cached scores during diffusion
    cached_scores = [[], []]
    def callback(x, t):
        for i in range(len(cached_scores)):
            cached_scores[i].append(composed_denoise_fn.cached_scores[i].clone().cpu().numpy())
        return x

    # Run diffusion process and collect samples
    generated_samples = diffusion_baseline(composed_denoise_fn,
                                         diffusion=noise_scheduler,
                                         x_shape=x_shape,
                                         eval_batch_size=num_samples_to_generate,
                                         callback=callback,
                                         progress=progress)

    final_samples = generated_samples[-1]
    intermediate_samples = np.array(generated_samples[:-1][::-1])
    cached_scores = [np.array(cached_scores[i][::-1]) for i in range(len(cached_scores))]

    # Convert to torch tensors for filtering
    cached_scores = torch.from_numpy(np.array(cached_scores))
    sqrt_alpha_bar = noise_scheduler.sqrt_alphas_cumprod
    sqrt_one_minus_alpha_bar = noise_scheduler.sqrt_one_minus_alphas_cumprod
    intermediate_samples = torch.from_numpy(intermediate_samples)

    # Calculate noise
    noise = (intermediate_samples - sqrt_alpha_bar[:, None, None] * torch.from_numpy(final_samples)) / sqrt_one_minus_alpha_bar[:, None, None]

    # Filter samples
    estimate_lift = make_estimate_lift(elbo_cfg)
    cached_scores = cached_scores.to(device='cuda')
    noise = noise.to(device='cuda')
    energies = [estimate_lift(cached_scores[i], noise) for i in range(len(cached_scores))]

    is_valid = torch.ones(energies[0].shape[0], dtype=torch.bool, device=cached_scores.device)
    for algebra_idx, algebra in enumerate(algebras):
        if algebra == "negation":
            is_valid = is_valid & (energies[algebra_idx] <= 0)
        elif algebra == "product":
            is_valid = is_valid & (energies[algebra_idx] > 0)
        elif algebra == "summation":
            is_valid = is_valid | (energies[algebra_idx] > 0)

    is_valid = is_valid.cpu().numpy()
    filtered_samples = final_samples[is_valid]
    acceptance_ratio = len(filtered_samples) / len(final_samples)

    return filtered_samples, acceptance_ratio
