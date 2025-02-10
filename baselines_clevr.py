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
import os


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

    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
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


def ebm_baseline(composed_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 x_shape: Tuple[int, ...],
                 noise_scheduler: Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                                        ComposableDiff.composable_diffusion.respace.SpacedDiffusion],
                 num_samples_per_trial: int,
                 ebm_cfg: dict[str, Union[float, str]],
                 progress: bool,
                 ) -> List[torch.Tensor]:
    """ebm baseline

    Args:
        composed_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the composed model
        x_shape (Tuple[int, ...]): shape of each sample
        noise_scheduler (Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, ComposableDiff.composable_diffusion.respace.SpacedDiffusion]): noise scheduler
        num_samples_per_trial (int): number of samples to generate
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
                                 eval_batch_size=num_samples_per_trial,
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


def make_estimate_neg_logp(elbo_cfg: dict[str, Union[bool, str]],
                            noise_scheduler: Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                                                    ComposableDiff.composable_diffusion.respace.SpacedDiffusion],
                            unconditioned_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            progress: bool=False) -> Callable[[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    """Creates a function to estimate the negative log probability using ELBO.

    Args:
        elbo_cfg (dict[str, Union[bool, str]]): Configuration for ELBO estimation, including whether to use CFG, number of samples, mini batch size, etc.
        noise_scheduler (Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, ComposableDiff.composable_diffusion.respace.SpacedDiffusion]): Noise scheduler for the diffusion process.
        unconditioned_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Denoising function for the unconditioned model.
        progress (bool, optional): Whether to show the progress bar. Defaults to False.

    Returns:
        Callable[[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]: Function to estimate the negative log probability.
    """
    def estimate_neg_logp(denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            x: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.zeros((0,), dtype=x.dtype, device=x.device)
        assert len(t.unique()) == 1, "t should be the same for all samples, but got {}".format(t.unique())
        if elbo_cfg["use_cfg"]:
            log_px_given_c = calculate_elbo(denoise_fn,
                                            noise_scheduler,
                                            x_t=x,
                                            t=t[0],
                                            seed=t[0],
                                            mini_batch=elbo_cfg["mini_batch"],
                                            n_samples=elbo_cfg["n_samples"],
                                            same_noise=elbo_cfg["same_noise"],
                                            sample_timesteps=elbo_cfg["sample_timesteps"],
                                            progress=progress)
            log_px = calculate_elbo(unconditioned_denoise_fn,
                                    noise_scheduler,
                                    x_t=x,
                                    t=t[0],
                                    seed=t[0],
                                    mini_batch=elbo_cfg["mini_batch"],
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
                                            mini_batch=elbo_cfg["mini_batch"],
                                            n_samples=elbo_cfg["n_samples"],
                                            same_noise=elbo_cfg["same_noise"],
                                            sample_timesteps=elbo_cfg["sample_timesteps"],
                                            progress=progress)
            return -log_px_given_c
    return estimate_neg_logp


# declare a rejection error
class RejectionError(Exception):
    pass


def rejection_sampling_baseline(composed_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                unconditioned_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                conditions_denoise_fn: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                                x_shape: Tuple[int, ...],
                                noise_scheduler: Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                                                       ComposableDiff.composable_diffusion.respace.SpacedDiffusion],
                                num_samples_per_trial: int,
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
        num_samples_per_trial (int, optional): number of samples to generate for each trial.
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

    estimate_neg_logp = make_estimate_neg_logp(elbo_cfg, noise_scheduler, unconditioned_denoise_fn)

    need_to_remove_across_timesteps = {}
    energies_across_timesteps = {}

    def callback(x, t):
        if t[0] in rejection_timesteps:
            unspaced_t = rejection_timesteps_unspaced[rejection_timesteps.index(t[0])]
            energies = [estimate_neg_logp(denoise_fn, x,
                                          t=torch.full((len(x),), unspaced_t, dtype=torch.long, device=x.device))
                                          for denoise_fn in conditions_denoise_fn]
            energies_across_timesteps[t[0].item()] = [e.cpu().numpy() for e in energies]
            need_to_remove = (torch.stack(energies, dim=0) > 0).any(dim=0)
            need_to_remove_across_timesteps[t[0].item()] = need_to_remove.cpu().numpy()

        return x

    unfiltered_samples = diffusion_baseline(composed_denoise_fn,
                                            noise_scheduler,
                                            x_shape,
                                            eval_batch_size=num_samples_per_trial,
                                            callback=callback,
                                            progress=progress)

    need_to_remove = torch.from_numpy(np.array(list(need_to_remove_across_timesteps.values()))).any(dim=0)
    return unfiltered_samples[-1][~need_to_remove], unfiltered_samples, need_to_remove_across_timesteps, energies_across_timesteps
