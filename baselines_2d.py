from tqdm.auto import tqdm
import numpy as np
import types
from copy import deepcopy
from typing import Union

import ddpm
import torch
from mcmc_yilun_torch import AnnealedMUHASampler
import torch.distributions as dist
import ot
from typing import Tuple, List, Callable, Dict
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


def diffusion_baseline(denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                       diffusion: ddpm.NoiseScheduler,
                       x_shape: Tuple[int, ...] = (2,),
                       eval_batch_size: int=8000,
                       progress: bool = True,
                       callback: Union[None, Callable[[torch.Tensor, int], torch.Tensor]]=None) -> List[np.ndarray]:
    device = ddpm.device
    denoise_fn = denoise_fn.to(device)
    sample = torch.randn((eval_batch_size,) + x_shape).to(device)
    timesteps = list(range(diffusion.num_timesteps))[::-1]

    samples = []
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


def make_estimate_log_lift(elbo_cfg: dict[str, Union[bool, str]],
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
    def estimate_log_lift(denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            x: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.zeros((0,), dtype=x.dtype, device=x.device)
        assert len(t.unique()) == 1, "t should be the same for all samples, but got {}".format(t.unique())
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
        log_px = calculate_elbo(lambda x, t: elbo_cfg["alpha"] * denoise_fn(x, t),
                                noise_scheduler,
                                x_t=x,
                                t=t[0],
                                seed=t[0],
                                mini_batch=elbo_cfg["mini_batch"],
                                n_samples=elbo_cfg["n_samples"],
                                same_noise=elbo_cfg["same_noise"],
                                sample_timesteps=elbo_cfg["sample_timesteps"],
                                progress=progress)
        log_lift = log_px_given_c - log_px
        return log_lift
    return estimate_log_lift


def rejection_baseline(composed_denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                conditions_denoise_fn: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                                algebras: list[str],
                                x_shape: Tuple[int, ...],
                                noise_scheduler: ddpm.NoiseScheduler,
                                num_samples_per_trial: int,
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
        num_samples_per_trial (int, optional): number of samples to generate for each trial.
        elbo_cfg (dict[str, Union[bool, str]]): configuration for the elbo estimation
        progress (bool): whether to show the progress bar

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor], Dict[int, np.ndarray], Dict[int, List[np.ndarray]]]:
        final_samples, unfiltered_samples, need_to_remove_across_timesteps, energies_across_timesteps
    """
    assert all([algebra in ['product', 'summation', 'negation'] for algebra in algebras])

    estimate_log_lift = make_estimate_log_lift(elbo_cfg, noise_scheduler)

    energies_across_timesteps = {}
    unfiltered_samples = []

    def callback(x, t):
        unfiltered_samples.append(x.clone().cpu().numpy())
        if t[0] == 0:
            energies = [estimate_log_lift(denoise_fn, x, t) for denoise_fn in conditions_denoise_fn]
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
                                 eval_batch_size=num_samples_per_trial,
                                 callback=callback,
                                 progress=progress)

    return samples, unfiltered_samples, len(samples[-1]) / len(unfiltered_samples[-1]), energies_across_timesteps


def evaluate_W1(generated_samples, target_samples):
    # Calculate the Wasserstein-1 distance using the optimal transport plan
    w1_distance = ot.emd2(np.ones(len(generated_samples)) / len(generated_samples),
                        np.ones(len(target_samples)) / len(target_samples),
                        ot.dist(generated_samples, target_samples, metric='cityblock'),
                        numItermax=int(1e7))
    return w1_distance


def evaluate_W2(generated_samples, target_samples):
    cost_matrix = ot.dist(generated_samples, target_samples, metric='sqeuclidean')

    # Calculate the Wasserstein-2 distance using the optimal transport plan
    w2_distance = ot.emd2(np.ones(len(generated_samples)) / len(generated_samples),
                        np.ones(len(target_samples)) / len(target_samples),
                        cost_matrix,
                        numItermax=int(1e7))
    return np.sqrt(w2_distance)


def evaluate_chamfer_distance(generated_samples, target_samples):
    """
    Calculate chamfer distance between two point clouds.

    Arguments:
    pc1 -- First point cloud (N1 x D numpy array)
    pc2 -- Second point cloud (N2 x D numpy array)

    Returns:
    chamfer_dist -- Chamfer distance between the point clouds
    """

    pc1, pc2 = generated_samples, target_samples

    # Reshape point clouds if necessary
    pc1 = np.atleast_2d(pc1)
    pc2 = np.atleast_2d(pc2)

    # Calculate pairwise distances
    dist_pc1_to_pc2 = np.sqrt(np.sum((pc1[:, None] - pc2) ** 2, axis=-1))
    dist_pc2_to_pc1 = np.sqrt(np.sum((pc2[:, None] - pc1) ** 2, axis=-1))

    # Minimum distance from each point in pc1 to pc2 and vice versa
    min_dist_pc1_to_pc2 = np.min(dist_pc1_to_pc2, axis=1)
    min_dist_pc2_to_pc1 = np.min(dist_pc2_to_pc1, axis=1)

    # Chamfer distance is the sum of these minimum distances
    chamfer_dist = np.mean(min_dist_pc1_to_pc2) + np.mean(min_dist_pc2_to_pc1)

    return chamfer_dist

# # Example usage:
# pc1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# pc2 = np.array([[2, 3, 4], [5, 6, 7]])
# chamfer_dist = evaluate_chamfer_distance(pc1, pc2)
# print(chamfer_dist)
