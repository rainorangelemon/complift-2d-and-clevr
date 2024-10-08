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
from typing import Tuple, List
from r_and_r import (
intermediate_distribution,
calculate_interval,
calculate_energy,
need_to_remove_with_thresholds,
calculate_interval_multiple_timesteps,
calculate_interval_to_avoid_multiple_timesteps,
calculate_elbo
)
from utils import plot_two_intervals
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
        t = torch.full((eval_batch_size,), i, dtype=torch.long, device=device)
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


def rejection_sampling_baseline_with_interval_calculation_elbo(composed_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                                               unconditioned_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                                               conditions_denoise_fn: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                                                               x_shape: Tuple[int, ...],
                                                               algebras: List[str],
                                                               noise_scheduler: ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion=None,
                                                               eval_batch_size=8000,
                                                               n_sample_for_elbo=1000,
                                                               mini_batch=20,
                                                               resample=True,
        ) -> Tuple[List[torch.Tensor], List[float], List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]:
    """similar to rejection_sampling_baseline_with_interval_calculation, but using elbo to estimate the -log p instead of energy

    Args:
        composed_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the composed model
        unconditioned_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the unconditioned model
        conditions_denoise_fn (List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): denoising function for multiple conditions
        x_shape (Tuple[int, ...]): shape of each sample
        algebras (str): list of the algebra from ["product", "summation", "negation"]
        noise_scheduler (ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, optional): noise scheduler. Defaults to None.
        eval_batch_size (int, optional): Number of samples to
                                         (1) calculate the support interval,
                                         (2) generate the samples.
        n_sample_for_elbo (int, optional): Number of (t, epsilon) pairs to estimate ELBO.
        mini_batch (int, optional): mini batch size for elbo estimation.

    Returns:
        Tuple[List[torch.Tensor], List[float], List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]: total_samples, filter_ratios, intervals_at_t0, datasets, original_samples
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

    datasets = [diffusion_baseline(lambda x, t: condition_denoise_fn(x, t, use_cfg=True),
                                   noise_scheduler, x_shape,
                                   eval_batch_size=eval_batch_size)[-1]
                                   for condition_denoise_fn in conditions_denoise_fn]

    # from torchvision.utils import make_grid, save_image
    # for condition_idx, dataset in enumerate(datasets):
    #     samples = ((dataset + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu() / 255.
    #     grid = make_grid(samples, nrow=int(np.sqrt(len(samples))), padding=0)
    #     save_image(grid, f'condition_samples_{condition_idx:05d}.png')

    # dataset_composed_origin = diffusion_baseline(composed_denoise_fn, noise_scheduler, x_shape, eval_batch_size=eval_batch_size)[-1]
    intervals_at_t0 = [calculate_interval(samples=dataset.to(device),
                                          denoise_fn=condition_denoise_fn,
                                          energy_fn=estimate_neg_logp)
                                          for dataset, condition_denoise_fn in zip(datasets, conditions_denoise_fn)]

    # intervals_1 = calculate_interval_multiple_timesteps(samples=dataset_composed_origin, model=model_1)[::-1]
    # intervals_2 = calculate_interval_multiple_timesteps(samples=dataset_composed_origin, model=model_2)[::-1]

    filter_ratios = []
    unfiltered_samples = []

    def callback(x, t):
        unfiltered_samples.append(x.clone().cpu())
        if (t == 0).all():
            energies = [estimate_neg_logp(denoise_fn, x, t) for denoise_fn in conditions_denoise_fn]
            interval_mins = [interval[0] for interval in intervals_at_t0]
            interval_maxs = [interval[1] for interval in intervals_at_t0]
            out_of_interval = [((energy < interval_min) | (energy > interval_max)) for energy, interval_min, interval_max in zip(energies, interval_mins, interval_maxs)]
            need_to_remove = torch.any(torch.stack(out_of_interval), dim=0)
            filter_ratios.append(need_to_remove.sum().item() / len(x))
            # resampling
            if need_to_remove.sum() != len(x):
                if resample:
                    x[need_to_remove] = x[~need_to_remove][torch.randint(0, torch.sum(~need_to_remove), (need_to_remove.sum(),))]
                else:
                    x = x[~need_to_remove]
            else:
                x = torch.empty((0, *x.shape[1:]), dtype=x.dtype, device=x.device)

        return x


    total_samples = diffusion_baseline(composed_denoise_fn,
                                       noise_scheduler,
                                       x_shape,
                                       eval_batch_size=eval_batch_size,
                                       callback=callback)

    return total_samples, filter_ratios, intervals_at_t0, datasets, unfiltered_samples


def best_of_n_sampling_baseline_with_interval_calculation_elbo(composed_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                                               unconditioned_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                                               conditions_denoise_fn: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                                                               x_shape: Tuple[int, ...],
                                                               algebras: List[str],
                                                               noise_scheduler: ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion=None,
                                                               eval_batch_size=8000,
                                                               n_sample_for_elbo=1000,
                                                               mini_batch=20,
                                                               resample=True,
        ) -> Tuple[List[torch.Tensor], List[float], List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]:
    """similar to rejection_sampling_baseline_with_interval_calculation_elbo, but using best-of-n sampling

    Args:
        composed_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the composed model
        unconditioned_denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function for the unconditioned model
        conditions_denoise_fn (List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): denoising function for multiple conditions
        x_shape (Tuple[int, ...]): shape of each sample
        algebras (str): list of the algebra from ["product", "summation", "negation"]
        noise_scheduler (ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion, optional): noise scheduler. Defaults to None.
        eval_batch_size (int, optional): Number of samples to
                                         (1) calculate the support interval,
                                         (2) generate the samples.
        n_sample_for_elbo (int, optional): Number of (t, epsilon) pairs to estimate ELBO.
        mini_batch (int, optional): mini batch size for elbo estimation.

    Returns:
        Tuple[List[torch.Tensor], List[float], List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]: total_samples, filter_ratios, intervals_at_t0, datasets, original_samples
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

    # datasets = [diffusion_baseline(lambda x, t: condition_denoise_fn(x, t, use_cfg=True),
    #                                noise_scheduler, x_shape,
    #                                eval_batch_size=eval_batch_size)[-1]
    #                                for condition_denoise_fn in conditions_denoise_fn]

    # from torchvision.utils import make_grid, save_image
    # for condition_idx, dataset in enumerate(datasets):
    #     samples = ((dataset + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu() / 255.
    #     grid = make_grid(samples, nrow=int(np.sqrt(len(samples))), padding=0)
    #     save_image(grid, f'condition_samples_{condition_idx:05d}.png')

    # dataset_composed_origin = diffusion_baseline(composed_denoise_fn, noise_scheduler, x_shape, eval_batch_size=eval_batch_size)[-1]
    # intervals_at_t0 = [calculate_interval(samples=dataset.to(device),
    #                                       denoise_fn=condition_denoise_fn,
    #                                       energy_fn=estimate_neg_logp)
    #                                       for dataset, condition_denoise_fn in zip(datasets, conditions_denoise_fn)]

    # intervals_1 = calculate_interval_multiple_timesteps(samples=dataset_composed_origin, model=model_1)[::-1]
    # intervals_2 = calculate_interval_multiple_timesteps(samples=dataset_composed_origin, model=model_2)[::-1]

    filter_ratios = []
    unfiltered_samples = []

    def callback(x, t):
        unfiltered_samples.append(x.clone().cpu())
        if (t == 0).all():
            energies = [estimate_neg_logp(denoise_fn, x, t) for denoise_fn in conditions_denoise_fn]
            best_idx = torch.argmin(torch.stack(energies).sum(dim=0), dim=0)
            x = x[[best_idx]].expand(x.shape)
            filter_ratios.append((len(x) - 1) / len(x))

        return x


    total_samples = diffusion_baseline(composed_denoise_fn,
                                       noise_scheduler,
                                       x_shape,
                                       eval_batch_size=eval_batch_size,
                                       callback=callback)

    return total_samples, filter_ratios, None, [unfiltered_samples[-1]]*len(conditions_denoise_fn), unfiltered_samples
