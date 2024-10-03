from tqdm.auto import tqdm
import numpy as np
import types
from copy import deepcopy
from typing import Union, Callable

import DiT.diffusion
import DiT.diffusion.gaussian_diffusion
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
import DiT

def diffusion_baseline(denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                       diffusion: DiT.diffusion.gaussian_diffusion.GaussianDiffusion,
                       latent_shape: Tuple[int, ...],
                       eval_batch_size: int=256,
                       clip_denoised: bool=False,
                       progress: bool = True,
                       callback: Union[None, Callable[[torch.Tensor, int], torch.Tensor]] = None,
                       ) -> List[torch.Tensor]:
    """diffusion baseline

    Args:
        denoise_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): denoising function, takes (x, t) as input and returns predicted noise
        diffusion (DiT.diffusion.gaussian_diffusion.GaussianDiffusion): gaussian diffusion procedure, basically a noise scheduler
        latent_shape (Tuple[int, ...]): shape of each sample
        eval_batch_size (int, optional): number of samples to generate. Defaults to 8000.
        clip_denoised (bool, optional): whether to clip the denoised samples. Defaults to False.
        progress (bool, optional): whether to show the progress bar. Defaults to True.
        callback (Union[None, Callable[[torch.Tensor, int], torch.Tensor]], optional): callback function to modify the samples. Defaults to None.

    Returns:
        List[torch.Tensor]: generated samples at each timestep, from the start timestep to the end timestep through reverse diffusion process
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = torch.randn((eval_batch_size,) + latent_shape, device=device)
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
        samples.append(sample)
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


def rejection_sampling_baseline_with_interval_calculation(model_to_test, model_1, model_2, algebra, eval_batch_size=8000
                                                          ) -> Tuple[np.ndarray, List[float], Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
    dataset_1 = diffusion_baseline(model_1, eval_batch_size=eval_batch_size)[-1]
    dataset_2 = diffusion_baseline(model_2, eval_batch_size=eval_batch_size)[-1]
    dataset_3_origin = diffusion_baseline(model_to_test, eval_batch_size=eval_batch_size)[-1]
    interval_1 = calculate_interval(samples=dataset_1, model=model_1)
    interval_2 = calculate_interval(samples=dataset_2, model=model_2)
    energy_1 = calculate_energy(samples=dataset_3_origin, model=model_1)
    energy_2 = calculate_energy(samples=dataset_3_origin, model=model_2)
    need_to_remove = need_to_remove_with_thresholds(algebra=algebra,
                                                    energy_1=energy_1, energy_2=energy_2,
                                                    interval_1=interval_1, interval_2=interval_2)
    dataset_3 = dataset_3_origin[~need_to_remove]

    intervals_1 = calculate_interval_multiple_timesteps(samples=dataset_1, model=model_1)
    if algebra=='negation':
        intervals_2 = calculate_interval_to_avoid_multiple_timesteps(positive_samples=dataset_3, negative_samples=dataset_2, model=model_2)
    else:
        intervals_2 = calculate_interval_multiple_timesteps(samples=dataset_2, model=model_2)

    result = diffusion_rejection_baseline(composed_model=model_to_test,
                                          models=[model_1, model_2],
                                          algebra=algebra,
                                          intervals=[intervals_1, intervals_2],
                                          eval_batch_size=eval_batch_size)
    return *result, (intervals_1, intervals_2)


def rejection_sampling_baseline_with_interval_calculation_elbo(model_to_test: ddpm.CompositionEnergyMLP,
                                                               model_1: ddpm.EnergyMLP,
                                                               model_2: ddpm.EnergyMLP,
                                                               algebra: str,
                                                               eval_batch_size=8000,
                                                               n_sample_for_elbo=1000,
                                                               mini_batch=20,
                                                               num_timesteps=50,
        ) -> Tuple[np.ndarray, List[float], Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
    """similar to rejection_sampling_baseline_with_interval_calculation, but using elbo to estimate the -log p instead of energy

    Args:
        model_to_test (ddpm.CompositionEnergyMLP): the composed model
        model_1 (ddpm.EnergyMLP): model for algebra 1
        model_2 (ddpm.EnergyMLP): model for algebra 2
        algebra (str): type of the algebra, one of ["product", "summation", "negation"]
        eval_batch_size (int, optional): Number of samples to
                                         (1) calculate the support interval,
                                         (2) generate the samples.
        n_sample_for_elbo (int, optional): Number of (t, epsilon) pairs to estimate ELBO.
        mini_batch (int, optional): mini batch size for elbo estimation.
        num_timesteps (int, optional): number of timesteps for the diffusion model.

    Returns:
        Tuple[np.ndarray, List[float], Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]: samples, filter_ratios, intervals
    """
    # overwrite model_to_test.energy to use elbo
    def estimate_neg_logp(model: Union[ddpm.EnergyMLP, ddpm.CompositionEnergyMLP],
                          x: torch.Tensor,
                          t: torch.Tensor):
        if x.shape[0] == 0:
            return torch.zeros((0,), dtype=x.dtype, device=x.device)
        assert len(t.unique())==1, "t should be the same for all samples, but got {}".format(t.unique())
        noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps)
        return -calculate_elbo(model, noise_scheduler, x_t=x, t=t[0], n_samples=n_sample_for_elbo, seed=t[0], mini_batch=mini_batch)

    # deepcopy to avoid modifying the original model
    model_to_test = deepcopy(model_to_test)
    model_1 = deepcopy(model_1)
    model_2 = deepcopy(model_2)

    model_to_test.energy = types.MethodType(estimate_neg_logp, model_to_test)
    model_1.energy = types.MethodType(estimate_neg_logp, model_1)
    model_2.energy = types.MethodType(estimate_neg_logp, model_2)
    return rejection_sampling_baseline_with_interval_calculation(model_to_test, model_1, model_2, algebra, eval_batch_size)


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
