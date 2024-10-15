from tqdm.auto import tqdm
import numpy as np
import types
from copy import deepcopy
from typing import Union, Callable

import ComposableDiff.composable_diffusion
import ComposableDiff.composable_diffusion.gaussian_diffusion
import ComposableDiff.composable_diffusion.respace
import ddpm
import torch
from mcmc_yilun_torch import AnnealedMUHASampler
import torch.distributions as dist
import ot
from typing import Tuple, List, Dict
from r_and_r import (
intermediate_distribution,
calculate_interval,
calculate_energy,
need_to_remove_with_thresholds,
calculate_interval_multiple_timesteps,
calculate_interval_to_avoid_multiple_timesteps,
calculate_elbo
)
from utils import plot_two_intervals, plot_energy_histogram
import ComposableDiff


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = torch.randn((eval_batch_size,) + x_shape, device=device)
    indices = list(range(diffusion.num_timesteps))[::-1]

    if progress:
        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    samples = []
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


def ebm_baseline(model_to_test,
                 num_timesteps=50,
                 eval_batch_size=8000,
                 temperature=1,
                 samples_per_step=10,
                 callback=None):

    noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps)
    device = ddpm.device
    model_to_test = model_to_test.to(device)

    num_steps = 50
    dim = 2
    init_std = 1.
    init_mu = 0.
    damping = .5
    mass_diag_sqrt = 1.
    num_leapfrog = 3
    uha_step_size = .03
    uha_step_sizes = torch.ones((num_steps,)) * uha_step_size

    initial_distribution = dist.MultivariateNormal(loc=torch.zeros(dim).to(device) + init_mu, covariance_matrix=torch.eye(dim).to(device) * init_std)

    def energy_function(x, t):
        t = num_steps - 1 - t
        x = x.clone().to(device)
        t_tensor = torch.from_numpy(np.repeat(t, eval_batch_size)).long().to(device)
        return -(model_to_test.energy(x, t_tensor)) * temperature / noise_scheduler.sqrt_one_minus_alphas_cumprod[t]

    def gradient_function(x, t):
        t = num_steps - 1 - t
        x = x.clone().to(device)
        t_tensor = torch.from_numpy(np.repeat(t, eval_batch_size)).long().to(device)
        return -(model_to_test(x, t_tensor)) * temperature / noise_scheduler.sqrt_one_minus_alphas_cumprod[t]

    # Create an instance of the AnnealedMUHASampler
    sampler = AnnealedMUHASampler(num_steps=num_steps,
                                num_samples_per_step=samples_per_step,
                                step_sizes=uha_step_sizes,
                                damping_coeff=damping,
                                mass_diag_sqrt=mass_diag_sqrt,
                                num_leapfrog_steps=num_leapfrog,
                                initial_distribution=initial_distribution,
                                gradient_function=gradient_function,
                                energy_function=energy_function,)

    total_samples, _, _, _ = sampler.sample(n_samples=eval_batch_size, callback=callback)
    return total_samples.cpu().numpy()


def ebm_rejection_baseline(composed_model,
                           models,
                           intervals,
                           algebra,
                           **kwargs):

    assert len(models) == 2
    assert algebra in ['product', 'summation', 'negation']
    device = ddpm.device
    num_timesteps = kwargs.get('num_timesteps', 50)
    filter_ratios = []

    def callback(x, reverse_t):
        t = num_timesteps - 1 - reverse_t
        if isinstance(x, torch.Tensor):
            x_numpy = x.cpu().numpy()
        x = torch.from_numpy(x_numpy).float().to(device)
        t_tensor = torch.from_numpy(np.repeat(t, len(x))).long().to(device)
        energies = [model.energy(x, t_tensor) for model in models]
        interval_mins = [interval[reverse_t][0] for interval in intervals]
        interval_maxs = [interval[reverse_t][1] for interval in intervals]
        out_of_interval = [((energy < interval_min) | (energy > interval_max)) for energy, interval_min, interval_max in zip(energies, interval_mins, interval_maxs)]
        if algebra == 'product':
            need_to_remove = out_of_interval[0] | out_of_interval[1]
        elif algebra == 'summation':
            need_to_remove = out_of_interval[0] & out_of_interval[1]
        elif algebra == 'negation':
            need_to_remove = out_of_interval[0] | (~out_of_interval[1])
        need_to_remove = need_to_remove.cpu().numpy()
        filter_ratios.append(need_to_remove.sum() / len(x))
        # repurpose
        if need_to_remove.sum() != len(x):
            x_numpy[need_to_remove] = x_numpy[~need_to_remove][np.random.choice(np.sum(~need_to_remove), need_to_remove.sum(), replace=True)]

        return torch.from_numpy(x_numpy).float().to(device)

    total_samples = ebm_baseline(composed_model, **kwargs, callback=callback)
    return total_samples, filter_ratios


def diffusion_rejection_baseline(composed_model,
                                 models,
                                 intervals,
                                 algebra,
                                 resample=True,
                                 **kwargs):
    assert len(models) == 2
    assert algebra in ['product', 'summation', 'negation']
    device = ddpm.device
    num_timesteps = kwargs.get('num_timesteps', 50)
    filter_ratios = []

    def callback(x, t):
        reverse_t = num_timesteps - 1 - t
        if isinstance(x, torch.Tensor):
            x_numpy = x.cpu().numpy()
        x = torch.from_numpy(x_numpy).float().to(device)
        t_tensor = torch.full((len(x),), t, dtype=torch.long, device=device)
        energies = [model.energy(x, t_tensor) for model in models]
        interval_mins = [interval[reverse_t][0] for interval in intervals]
        interval_maxs = [interval[reverse_t][1] for interval in intervals]
        out_of_interval = [((energy < interval_min) | (energy > interval_max)) for energy, interval_min, interval_max in zip(energies, interval_mins, interval_maxs)]
        if algebra == 'product':
            need_to_remove = out_of_interval[0] | out_of_interval[1]
        elif algebra == 'summation':
            need_to_remove = out_of_interval[0] & out_of_interval[1]
        elif algebra == 'negation':
            need_to_remove = out_of_interval[0] | (~out_of_interval[1])
        need_to_remove = need_to_remove.cpu().numpy()
        filter_ratios.append(need_to_remove.sum() / len(x))
        # repurpose
        if need_to_remove.sum() != len(x):
            if resample:
                x_numpy[need_to_remove] = x_numpy[~need_to_remove][np.random.choice(np.sum(~need_to_remove), need_to_remove.sum(), replace=True)]
            else:
                x_numpy = x_numpy[~need_to_remove]
        else:
            x_numpy = np.empty((0, x_numpy.shape[1]))

        return torch.from_numpy(x_numpy).float().to(device)


    total_samples = diffusion_baseline(composed_model, **kwargs, callback=callback)
    return total_samples, filter_ratios


def create_rejection_timesteps(method: str,
                               T: int,
                               timesteps_to_select: int):
    """method to create rejection timesteps

    Args:
        method (str): method to create rejection timesteps, should be one of ['uniform', 'latest']
        T (int): number of total timesteps

    Returns:
        List[int]: rejection timesteps
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
                            mini_batch: int,
                            progress: bool=False) -> Callable[[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    """Creates a function to estimate the negative log probability using ELBO.

    Args:
        elbo_cfg (dict[str, Union[bool, str]]): Configuration for ELBO estimation, including whether to use CFG, number of samples, and other parameters.
        noise_scheduler (Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, ComposableDiff.composable_diffusion.respace.SpacedDiffusion]): Noise scheduler for the diffusion process.
        unconditioned_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Denoising function for the unconditioned model.
        mini_batch (int): Mini batch size for ELBO estimation.
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


def rejection_sampling_baseline_with_interval_calculation_elbo(composed_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                                               unconditioned_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                                               conditions_denoise_fn: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                                                               x_shape: Tuple[int, ...],
                                                               algebras: List[str],
                                                               noise_scheduler: Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                                                                                      ComposableDiff.composable_diffusion.respace.SpacedDiffusion],
                                                               reject_using_interval: bool,
                                                               eval_batch_size,
                                                               mini_batch,
                                                               bootstrap_cfg: dict[str, Union[float, str]],
                                                               rejection_scheduler_cfg: dict[str, Union[float, str]],
                                                               elbo_cfg: dict[str, Union[bool, str]],
                                                               resample=True,
        ) -> Tuple[List[torch.Tensor], List[float], List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], List[torch.Tensor], Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]]]:
    """similar to rejection_sampling_baseline_with_interval_calculation, but using elbo to estimate the -log p instead of energy

    Args:
        composed_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the composed model
        unconditioned_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the unconditioned model
        conditions_denoise_fn (List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): denoising function for multiple conditions
        x_shape (Tuple[int, ...]): shape of each sample
        algebras (str): list of the algebra from ["product", "summation", "negation"]
        noise_scheduler (ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, optional): noise scheduler. Defaults to None.
        rejection_using_interval (bool): whether to reject samples using interval or not
        eval_batch_size (int, optional): Number of samples to
                                         (1) calculate the support interval,
                                         (2) generate the samples.
        mini_batch (int, optional): mini batch size for elbo estimation.
        bootstrap_cfg (dict[str, Union[float, str]]): configuration for the bootstrap method, including "method" and "confidence"
        rejection_scheduler_cfg (dict[str, Union[float, str]]): configuration for the rejection scheduler
        elbo_cfg (dict[str, Union[bool, str]]): configuration for the elbo estimation

    Returns:
        Tuple[List[torch.Tensor], List[float], List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], List[torch.Tensor], Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]]:
        total_samples, filter_ratios, intervals_across_timesteps, datasets, original_samples, energies_conditioned_across_timesteps, energies_composed_across_timesteps
    """
    assert all([algebra == "product" for algebra in algebras]), "only support product algebra for now, but got {}".format(algebras)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets_across_timesteps = [diffusion_baseline(lambda x, t: condition_denoise_fn(x, t, use_cfg=True),
                                                    noise_scheduler, x_shape,
                                                    eval_batch_size=eval_batch_size)
                                                    for condition_denoise_fn in conditions_denoise_fn]

    rejection_timesteps = create_rejection_timesteps(**rejection_scheduler_cfg, T=noise_scheduler.num_timesteps)
    if isinstance(noise_scheduler, ComposableDiff.composable_diffusion.respace.SpacedDiffusion):
        rejection_timesteps_unspaced = [noise_scheduler.timestep_map[t] for t in rejection_timesteps]
    else:
        rejection_timesteps_unspaced = rejection_timesteps

    estimate_neg_logp = make_estimate_neg_logp(elbo_cfg, noise_scheduler, unconditioned_denoise_fn, mini_batch)

    # dataset_composed_origin = diffusion_baseline(composed_denoise_fn, noise_scheduler, x_shape, eval_batch_size=eval_batch_size)[-1]
    intervals_across_timesteps = {}
    energies_conditioned_across_timesteps = {}

    for spaced_t, t in zip(rejection_timesteps, rejection_timesteps_unspaced):
        intervals = []
        energies = []
        datasets_at_t = [dataset[::-1][spaced_t] for dataset in datasets_across_timesteps]

        for dataset, condition_denoise_fn in zip(datasets_at_t, conditions_denoise_fn):
            interval, energies_condition = calculate_interval(samples=dataset.to(device),
                                                              denoise_fn=condition_denoise_fn,
                                                              energy_fn=estimate_neg_logp,
                                                              confidence=bootstrap_cfg["confidence"],
                                                              bootstrap_method=bootstrap_cfg["method"],
                                                              t=t,)
            intervals.append(interval)
            energies.append(energies_condition)

        intervals_across_timesteps[spaced_t] = intervals
        energies_conditioned_across_timesteps[spaced_t] = energies

    filter_ratios = []
    energies_composed_across_timesteps = {}

    def callback(x, t):
        if len(t.unique()) == 1 and t[0] in rejection_timesteps:
            intervals_at_t = intervals_across_timesteps[t[0].item()]
            unspaced_t = rejection_timesteps_unspaced[rejection_timesteps.index(t[0])]
            energies = [estimate_neg_logp(denoise_fn, x,
                                          t=torch.full((len(x),), unspaced_t, dtype=torch.long, device=x.device))
                                          for denoise_fn in conditions_denoise_fn]
            energies_composed_across_timesteps[t[0].item()] = [e.cpu().numpy() for e in energies]
            interval_mins = [interval[0] for interval in intervals_at_t]
            interval_maxs = [interval[1] for interval in intervals_at_t]
            if reject_using_interval:
                out_of_interval = [((energy < interval_min) | (energy > interval_max)) for energy, interval_min, interval_max in zip(energies, interval_mins, interval_maxs)]
            else:
                out_of_interval = [(energy > interval_max) for energy, interval_min, interval_max in zip(energies, interval_mins, interval_maxs)]
            need_to_remove = torch.any(torch.stack(out_of_interval), dim=0)
            filter_ratios.append(need_to_remove.sum().item() / (len(x) + 1e-8))
            # resampling
            if need_to_remove.sum() != len(x):
                if resample:
                    x[need_to_remove] = x[~need_to_remove][torch.randint(0, torch.sum(~need_to_remove), (need_to_remove.sum(),))]
                else:
                    x = x[~need_to_remove]
            else:
                x = torch.empty((0, *x.shape[1:]), dtype=x.dtype, device=x.device)

        elif len(t.unique()) == 0:
            filter_ratios.append(1.0)
            x = torch.empty((0, *x.shape[1:]), dtype=x.dtype, device=x.device)

        return x


    final_samples = diffusion_baseline(composed_denoise_fn,
                                       noise_scheduler,
                                       x_shape,
                                       eval_batch_size=eval_batch_size,
                                       callback=callback)

    unfiltered_samples = diffusion_baseline(composed_denoise_fn,
                                            noise_scheduler,
                                            x_shape,
                                            eval_batch_size=eval_batch_size,
                                            callback=None)

    return final_samples, filter_ratios, intervals_across_timesteps, \
           [dataset[-1] for dataset in datasets_across_timesteps], \
            unfiltered_samples, energies_conditioned_across_timesteps, energies_composed_across_timesteps


def best_of_n_sampling_baseline_with_interval_calculation_elbo(composed_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                                               unconditioned_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                                               conditions_denoise_fn: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                                                               x_shape: Tuple[int, ...],
                                                               algebras: List[str],
                                                               noise_scheduler: Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                                                                                      ComposableDiff.composable_diffusion.respace.SpacedDiffusion],
                                                               bootstrap_cfg: dict[str, Union[float, str]],
                                                               rejection_scheduler_cfg: dict[str, Union[float, str]],
                                                               eval_batch_size=8000,
                                                               n_sample_for_elbo=1000,
                                                               mini_batch=20,
                                                               resample=True,
        ) -> Tuple[List[torch.Tensor], List[float], List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], List[torch.Tensor], Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]]]:
    """similar to rejection_sampling_baseline_with_interval_calculation_elbo, but using best of n sampling
    Args:
        composed_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the composed model
        unconditioned_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the unconditioned model
        conditions_denoise_fn (List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): denoising function for multiple conditions
        x_shape (Tuple[int, ...]): shape of each sample
        algebras (str): list of the algebra from ["product", "summation", "negation"]
        noise_scheduler (ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, optional): noise scheduler. Defaults to None.
        bootstrap_cfg (dict[str, Union[float, str]]): configuration for the bootstrap method, including "method" and "confidence"
        rejection_scheduler_cfg (dict[str, Union[float, str]]): configuration for the rejection scheduler
        eval_batch_size (int, optional): Number of samples to
                                         (1) calculate the support interval,
                                         (2) generate the samples.
        n_sample_for_elbo (int, optional): Number of (t, epsilon) pairs to estimate ELBO.
        mini_batch (int, optional): mini batch size for elbo estimation.

    Returns:
        Tuple[List[torch.Tensor], List[float], List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], List[torch.Tensor], Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]]:
        total_samples, filter_ratios, intervals_across_timesteps, datasets, original_samples, energies_conditioned_across_timesteps, energies_composed_across_timesteps
    """
    assert all([algebra == "product" for algebra in algebras]), "only support product algebra for now, but got {}".format(algebras)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def estimate_neg_logp(denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                          x: torch.Tensor,
                          t: torch.Tensor):
        if x.shape[0] == 0:
            return torch.zeros((0,), dtype=x.dtype, device=x.device)
        assert len(t.unique())==1, "t should be the same for all samples, but got {}".format(t.unique())
        log_px_given_c = calculate_elbo(denoise_fn,
                                        noise_scheduler,
                                        x_t=x,
                                        t=t[0],
                                        n_samples=n_sample_for_elbo,
                                        seed=t[0],
                                        mini_batch=mini_batch)
        log_px = calculate_elbo(unconditioned_denoise_fn,
                                noise_scheduler,
                                x_t=x,
                                t=t[0],
                                n_samples=n_sample_for_elbo,
                                seed=t[0],
                                mini_batch=mini_batch)
        log_pc_given_x = log_px_given_c - log_px
        return -log_pc_given_x

    rejection_timesteps = create_rejection_timesteps(**rejection_scheduler_cfg, T=noise_scheduler.num_timesteps)
    if isinstance(noise_scheduler, ComposableDiff.composable_diffusion.respace.SpacedDiffusion):
        rejection_timesteps_unspaced = [noise_scheduler.timestep_map[t] for t in rejection_timesteps]
    else:
        rejection_timesteps_unspaced = rejection_timesteps

    filter_ratios = [0.]
    energies_composed_across_timesteps = {}
    unfiltered_samples = []

    def callback(x, t):
        if len(t.unique()) == 1 and t[0] in rejection_timesteps:
            unfiltered_samples.append(x.clone())
            unspaced_t = rejection_timesteps_unspaced[rejection_timesteps.index(t[0])]
            energies = [estimate_neg_logp(denoise_fn, x,
                                          t=torch.full((len(x),), unspaced_t, dtype=torch.long, device=x.device))
                                          for denoise_fn in conditions_denoise_fn]
            energies_composed_across_timesteps[t[0].item()] = [e.cpu().numpy() for e in energies]
            best_idx = torch.argmin(torch.stack(energies).sum(dim=0), dim=0)
            x = x[None, best_idx].expand(x.shape[0], -1, *x.shape[2:])

        return x


    final_samples = diffusion_baseline(composed_denoise_fn,
                                       noise_scheduler,
                                       x_shape,
                                       eval_batch_size=eval_batch_size,
                                       callback=callback)

    return final_samples, filter_ratios, None, \
           [final_samples[-1]]*len(conditions_denoise_fn), \
            unfiltered_samples, energies_composed_across_timesteps, energies_composed_across_timesteps
