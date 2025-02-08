import torch
from typing import List, Union, Tuple, Callable
import numpy as np
import ComposableDiff.composable_diffusion
import ComposableDiff.composable_diffusion.gaussian_diffusion
import ComposableDiff.composable_diffusion.respace
from ddpm import device, NoiseScheduler
from ddpm import EnergyMLP, CompositionEnergyMLP
import ComposableDiff
from tqdm.auto import tqdm


def to_tensor(data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """convert data to torch.Tensor

    Args:
        data (Union[np.ndarray, torch.Tensor]): data

    Returns:
        torch.Tensor: torch.Tensor
    """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise ValueError("data should be np.ndarray or torch.Tensor")


def intermediate_distribution(data_points: np.ndarray,
                              num_timesteps: int=50) -> List[np.ndarray]:
    """get the intermediate distribution of the data points

    Args:
        data_points (np.ndarray): data points (n_samples, n_features)
        num_timesteps (int, optional): number of diffusion steps. Defaults to 50.

    Returns:
        List[np.ndarray]: list of intermediate data points, len(List) = (num_timesteps + 1),
        the first element is the initial Gaussian distribution,
        the last element is the original data points.
    """
    noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps)
    origin_data = to_tensor(data_points).float().to(device)
    intermediate_data_list = []
    for i in range(num_timesteps):
        noise = torch.randn_like(origin_data)
        intermediate_data = noise_scheduler.add_noise(origin_data, noise, torch.ones(len(origin_data)).long().to(device) * i)
        intermediate_data_list.append(intermediate_data.cpu().numpy())
    return intermediate_data_list[::-1] + [origin_data.cpu().numpy()]


def calculate_energy(samples: np.ndarray,
                     model: Union[EnergyMLP, CompositionEnergyMLP],
                     t: int=0) -> np.ndarray:
    """calculate the energy of the samples.

    Args:
        samples (np.ndarray): samples (n_samples, n_features)
        model (Union[EnergyMLP, CompositionEnergyMLP]): energy model
        t (int, optional): timestep. Defaults to 0.

    Returns:
        np.ndarray: energy values
    """
    with torch.no_grad():
        energy_on_data = model.energy(to_tensor(samples).to(device), t+torch.zeros(len(samples)).long().to(device))
    return energy_on_data.cpu().numpy()


def need_to_remove_with_thresholds(energy_1: np.ndarray,
                                   energy_2: np.ndarray,
                                   interval_1: Tuple[float, float],
                                   interval_2: Tuple[float, float],
                                   algebra: str) -> np.ndarray:
    """determine if the samples need to be removed based on the intervals

    Args:
        energy_1 (np.ndarray): energy values of the first model (n_samples,)
        energy_2 (np.ndarray): energy values of the second model (n_samples,)
        interval_1 (Tuple[float, float]): interval of the first model
        interval_2 (Tuple[float, float]): interval of the second model
        algebra (str): algebra operation, 'product', 'summation', 'negation'

    Returns:
        np.ndarray: boolean array, True if the samples need to be removed (n_samples,)

    Raises:
        ValueError: algebra should be 'product', 'summation', or 'negation'
    """
    if algebra not in ["product", "summation", "negation"]:
        raise ValueError("algebra should be 'product', 'summation', or 'negation'")
    energies = [energy_1, energy_2]
    interval_mins = [interval_1[0], interval_2[0]]
    interval_maxs = [interval_1[1], interval_2[1]]
    out_of_interval = [((energy < interval_min) | (energy > interval_max)) for energy, interval_min, interval_max in zip(energies, interval_mins, interval_maxs)]
    if algebra == "product":
        need_to_remove = out_of_interval[0] | out_of_interval[1]
    elif algebra == "summation":
        need_to_remove = out_of_interval[0] & out_of_interval[1]
    elif algebra == "negation":
        need_to_remove = out_of_interval[0] | (~out_of_interval[1])
    return need_to_remove


def calculate_denoising_matching_term(x_k: torch.Tensor,
                                      true_noise_at_t: torch.Tensor,
                                      pred_noise_at_k: torch.Tensor,
                                      cumprod_alpha_t: torch.Tensor,
                                      cumprod_alpha_k: torch.Tensor) -> torch.Tensor:
    """calculate the denoising matching term

    Args:
        x_k (torch.Tensor): samples at timestep k (batch, n_features)
        true_noise_at_t (torch.Tensor): true noise at timestep t (batch, n_features)
        pred_noise_at_k (torch.Tensor): predicted noise at timestep k (batch, n_features)
        cumprod_alpha_t (torch.Tensor): cumulative product of alpha at timestep t (batch,)
        cumprod_alpha_k (torch.Tensor): cumulative product of alpha at timestep k (batch,)

    Returns:
        torch.Tensor: denoising matching term (batch, n_features)
    """
    x_k_coeff = (cumprod_alpha_t - torch.sqrt(cumprod_alpha_t)) * torch.sqrt(1 - cumprod_alpha_k) / (cumprod_alpha_t - cumprod_alpha_k)
    # -> (batch,)
    noise_t_coeff = torch.sqrt(1 - cumprod_alpha_k) / torch.sqrt(cumprod_alpha_t - cumprod_alpha_k)
    # -> (batch,)

    if len(pred_noise_at_k.shape) > 2:  # an image
        pred_noise_at_k = pred_noise_at_k[:, :3]  # only use the first 3 channels, no learned sigma

    x_k_coeff = x_k_coeff.reshape(-1, *([1] * (len(x_k.shape) - 1)))
    noise_t_coeff = noise_t_coeff.reshape(-1, *([1] * (len(x_k.shape) - 1)))

    return x_k_coeff * x_k + noise_t_coeff * true_noise_at_t - pred_noise_at_k


def add_noise_at_t(noise_scheduler: ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                   x_t: torch.Tensor,
                   x_noise: torch.Tensor,
                   timesteps_t: torch.Tensor,
                   timesteps_k: torch.Tensor) -> torch.Tensor:
    """add noise to the sample at time t

    Args:
        noise_scheduler (ComposableDiff.composable_diffusion.respace.SpacedDiffusion): noise scheduler
        x_t (torch.Tensor): the sample at time t (batch_size, feature_size)
        x_noise (torch.Tensor): add noise to the sample (batch_size, feature_size)
        timesteps_t (torch.Tensor): the timesteps where x_t is at (batch_size,)
        timesteps_k (torch.Tensor): the timesteps to add noise (batch_size,)

    Returns:
        torch.Tensor: the noisy sample (batch_size, feature_size)
    """
    assert (timesteps_t <= timesteps_k).all(), "timesteps_t should be less than or equal to timesteps_k"

    device = x_t.device
    B, T = x_t.size(0), noise_scheduler.num_timesteps

    timesteps_t = timesteps_t.to(device)
    timesteps_k = timesteps_k.to(device)
    alphas = 1 - noise_scheduler.betas

    log_alphas = torch.log(to_tensor(alphas).to(device))

    # -> (T,)
    log_alphas_batched = log_alphas[None, :].repeat(B, 1)
    log_alphas_batched = log_alphas_batched.to(device)
    # -> (batch_size, T)
    is_earlier_timesteps = (torch.arange(T, device=device)[None, :] < timesteps_t[:, None])
    log_alphas_batched[is_earlier_timesteps] = 0
    log_alphas_cumsum = torch.cumsum(log_alphas_batched, dim=-1)
    # -> (batch_size, T)
    alphas_cumprod = torch.exp(log_alphas_cumsum)
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    # -> (batch_size, T)
    sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5
    # -> (batch_size, T)

    s1 = sqrt_alphas_cumprod[torch.arange(B, device=device), timesteps_k]
    s2 = sqrt_one_minus_alphas_cumprod[torch.arange(B, device=device), timesteps_k]

    s1 = s1.reshape(-1, *([1] * (len(x_t.shape) - 1)))
    s2 = s2.reshape(-1, *([1] * (len(x_t.shape) - 1)))

    s1 = s1.to(device).float()
    s2 = s2.to(device).float()

    return s1 * x_t + s2 * x_noise


@torch.no_grad()
def calculate_elbo(model: torch.nn.Module,
                   noise_scheduler: Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                                          ComposableDiff.composable_diffusion.respace.SpacedDiffusion],
                   x_t: torch.Tensor,
                   t: int,
                   n_samples: int,
                   seed: int,
                   mini_batch: int,
                   same_noise: Union[bool, str],
                   sample_timesteps: str,
                   progress: bool=False) -> torch.Tensor:
    """calculate the approximate ELBO

    Args:
        model (torch.nn.Module): a diffusion model
        noise_scheduler (Union[ComposableDiff.composable_diffusion.gaussian_diffusion.GaussianDiffusion,
                               ComposableDiff.composable_diffusion.respace.SpacedDiffusion]): noise scheduler
        x_t (torch.Tensor): samples (batch, n_features)
        t (int): timestep where x_t is at
        n_samples (int): number of samples used for Monte Carlo estimation
        cumprod_alpha (torch.Tensor): cumulative product of alpha (T,)
        seed (int): random seed
        mini_batch (int): mini batch size
        same_noise (bool): whether to use the same noise for all samples
        sample_timesteps (str): how to sample the timesteps, "random" or "interleave" or f"specified{t:d}"
        progress (bool, optional): whether to show the progress bar. Defaults to False.

    Returns:
        torch.Tensor: approximate ELBO (batch,)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if isinstance(noise_scheduler, ComposableDiff.composable_diffusion.respace.SpacedDiffusion):
        noise_scheduler = noise_scheduler.base_diffusion

    B, *D = x_t.shape

    if isinstance(same_noise, bool):
        if same_noise:
            # sample noise (batch, n_sample, n_features)
            # following https://arxiv.org/pdf/2305.15241, use the same noise for all samples
            noise = torch.randn(1, 1, *D, device=x_t.device).expand(B, n_samples, *D)
        else:
            noise = torch.randn(1, n_samples, *D, device=x_t.device).expand(B, n_samples, *D)
    else:
        assert same_noise == "independent"
        noise = torch.randn(B, n_samples, *D, device=x_t.device)

    T = noise_scheduler.num_timesteps
    if sample_timesteps == "interleave":
        # interleave the samples
        ts_k = torch.linspace(t, T-1, n_samples, device=x_t.device).round().long().clamp(t, T-1)
        ts_k = ts_k[None, :].expand(B, n_samples)
    elif sample_timesteps == "random":
        # sample timestep randomly from [t, T): (batch, n_sample)
        ts_k = torch.randint(t, T, (1, n_samples), device=x_t.device).expand(B, n_samples)
    elif sample_timesteps.startswith("specified"):
        # sample timestep from the specified timesteps
        specified_timesteps = [int(s) for s in sample_timesteps.split("specified")[1].split(",")]
        ts_k = torch.tensor(specified_timesteps, device=x_t.device).long().flatten()[None, :].expand(B, n_samples)
    else:
        raise ValueError("sample_timesteps should be 'random' or 'interleave' or 'specified{t:d}'")

    # estimate the ELBO
    cumprod_alpha_prev = to_tensor(noise_scheduler.alphas_cumprod_prev).to(x_t.device).float()
    cumprod_alpha = to_tensor(noise_scheduler.alphas_cumprod).to(x_t.device).float()

    denoising_matching_terms = torch.zeros(B * n_samples, device=x_t.device)

    # vectorized_x_k = x_k.flatten(0, 1)
    vectorized_ts_k = ts_k.flatten()
    vectorized_noise = noise.flatten(0, 1)
    vectorized_ts_t = torch.full((B * n_samples,), t, device=x_t.device)
    vectorized_x_k_idx = torch.arange(B, device=x_t.device)[:, None].expand(B, n_samples).flatten()

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
        batch_term = batch_term.pow(2).reshape(batch_term.shape[0], -1).mean(dim=1)
        # -> (batch * min(mini_batch, n_samples - i),)
        denoising_matching_terms[i:i + mini_batch] = batch_term
        # -> (batch, min(mini_batch, n_samples - i))

    denoising_matching_term = denoising_matching_terms.view(B, n_samples)
    assert denoising_matching_term.shape == (B, n_samples)
    # -> (batch, n_sample)
    denoising_matching_term = denoising_matching_term.mean(dim=1)
    # -> (batch,)
    return -denoising_matching_term
