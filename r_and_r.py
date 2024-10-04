import torch
from typing import List, Union, Tuple, Callable
import numpy as np
from bootstrapping import (
bootstrapping_and_get_max,
bootstrapping_and_get_interval
)
from ddpm import device, NoiseScheduler
from ddpm import EnergyMLP, CompositionEnergyMLP


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
    origin_data = torch.tensor(data_points).float().to(device)
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
        energy_on_data = model.energy(torch.from_numpy(samples).to(device), t+torch.zeros(len(samples)).long().to(device))
    return energy_on_data.cpu().numpy()


def calculate_threshold(samples, model, confidence=0.999):
    # calculate the level-set values
    with torch.no_grad():
        energy_on_data = model.energy(torch.from_numpy(samples).to(device), torch.zeros(len(samples)).long().to(device))
    extreme_value = bootstrapping_and_get_max(energy_on_data.cpu().numpy(), confidence=confidence)
    return extreme_value


def calculate_threshold_multiple_timesteps(samples, model, confidence=0.999):
    intermediate_samples = intermediate_distribution(samples)[1:]  # ignore the initial Gaussian distribution
    extreme_values = []
    for t, samples_t in zip(range(len(intermediate_samples)-1, -1, -1), intermediate_samples):
        # calculate the level-set values
        with torch.no_grad():
            energy_on_data = model.energy(torch.from_numpy(samples_t).to(device), t+torch.zeros(len(samples)).long().to(device))
        extreme_value = bootstrapping_and_get_max(energy_on_data.cpu().numpy(), confidence=confidence)
        extreme_values.append(extreme_value)

    return extreme_values


def calculate_interval(samples: np.ndarray,
                       model: Union[EnergyMLP, CompositionEnergyMLP],
                       confidence: float=0.999) -> Tuple[float, float]:
    """calculate the interval of the samples.

    Args:
        samples (np.ndarray): samples (n_samples, n_features)
        model (Union[EnergyMLP, CompositionEnergyMLP]): energy model
        confidence (float, optional): confidence level. Defaults to 0.999.

    Returns:
        Tuple[float, float]: interval
    """
    # calculate the level-set values
    with torch.no_grad():
        energy_on_data = model.energy(torch.from_numpy(samples).to(device), torch.zeros(len(samples)).long().to(device))
    extreme_value_l, extreme_value_r = bootstrapping_and_get_interval(energy_on_data.cpu().numpy(), confidence=confidence)
    return extreme_value_l, extreme_value_r


def calculate_interval_multiple_timesteps(samples: np.ndarray,
                                          model: Union[EnergyMLP, CompositionEnergyMLP],
                                          confidence: float=0.999,
                                          num_timesteps: int=50) -> List[Tuple[float, float]]:
    """calculate the interval of the samples.

    Args:
        samples (np.ndarray): samples (n_samples, n_features)
        model (Union[EnergyMLP, CompositionEnergyMLP]): energy model
        confidence (float, optional): confidence level. Defaults to 0.999.
        num_timesteps (int, optional): number of diffusion steps. Defaults to 50.

    Returns:
        List[Tuple[float, float]]: list of intervals, len(List) = num_timesteps.
    """
    intermediate_samples = intermediate_distribution(samples, num_timesteps)[1:]  # ignore the initial Gaussian distribution
    intervals = []
    for t, samples_t in zip(range(len(intermediate_samples)-1, -1, -1), intermediate_samples):
        # calculate the level-set values
        with torch.no_grad():
            energy_on_data = model.energy(torch.from_numpy(samples_t).to(device), t+torch.zeros(len(samples)).long().to(device))
        extreme_value_l, extreme_value_r = bootstrapping_and_get_interval(energy_on_data.cpu().numpy(), confidence=confidence)
        intervals.append((extreme_value_l, extreme_value_r))

    return intervals


def calculate_interval_to_avoid_multiple_timesteps(positive_samples: np.ndarray,
                                                   negative_samples: np.ndarray,
                                                   model: Union[EnergyMLP, CompositionEnergyMLP],
                                                   confidence: float=0.999,
                                                   num_timesteps: int=50) -> List[Tuple[float, float]]:
    """calculate the interval to avoid for the samples

    Args:
        positive_samples (np.ndarray): samples to reach (n_samples, n_features)
        negative_samples (np.ndarray): samples to avoid (n_samples, n_features)
        model (Union[EnergyMLP, CompositionEnergyMLP]): energy model
        confidence (float, optional): confidence level. Defaults to 0.999.
        num_timesteps (int, optional): number of diffusion steps. Defaults to 50.

    Returns:
        List[Tuple[float, float]]: list of intervals, len(List) = num_timesteps.
    """
    intermediate_samples = intermediate_distribution(positive_samples, num_timesteps)[1:]  # ignore the initial Gaussian distribution
    intervals_positive = []
    for t, samples_t in zip(range(len(intermediate_samples)-1, -1, -1), intermediate_samples):
        # calculate the level-set values
        with torch.no_grad():
            energy_on_data = model.energy(torch.from_numpy(samples_t).to(device), t+torch.zeros(len(positive_samples)).long().to(device))
        extreme_value_l, extreme_value_r = bootstrapping_and_get_interval(energy_on_data.cpu().numpy(), confidence=confidence)
        intervals_positive.append((extreme_value_l, extreme_value_r))

    intermediate_samples = intermediate_distribution(negative_samples, num_timesteps)[1:]  # ignore the initial Gaussian distribution
    intervals_negative = []
    for t, samples_t in zip(range(len(intermediate_samples)-1, -1, -1), intermediate_samples):
        # calculate the level-set values
        with torch.no_grad():
            energy_on_data = model.energy(torch.from_numpy(samples_t).to(device), t+torch.zeros(len(negative_samples)).long().to(device))
        extreme_value_l, extreme_value_r = bootstrapping_and_get_interval(energy_on_data.cpu().numpy(), confidence=confidence)
        intervals_negative.append((extreme_value_l, extreme_value_r))

    intervals_to_avoid = []
    for interval_positive, interval_negative in zip(intervals_positive, intervals_negative):
        # essentially, we want to the interval_to_avoid = interval_negative - interval_positive
        if interval_negative[0] < interval_positive[0]:
            interval_to_avoid = (interval_negative[0], min(interval_positive[0], interval_negative[1]))
        else:
            interval_to_avoid = (max(interval_positive[1], interval_negative[0]), interval_negative[1])
        # TODO (rainorangelemon): understand the following two cases
        # Currently, it seems that the following two cases are not necessary
        # if interval_negative[1] <= interval_positive[0]:
        #     interval_to_avoid = (float('inf'), float('-inf'))
        # if interval_positive[0] >= interval_positive[1]:
        #     interval_to_avoid = (float('-inf'), float('inf'))
        intervals_to_avoid.append(interval_to_avoid)

    return intervals_to_avoid


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

    return x_k_coeff[:, None] * x_k + noise_t_coeff[:, None] * true_noise_at_t - pred_noise_at_k


@torch.no_grad()
def calculate_elbo(model: torch.nn.Module,
                   noise_scheduler: NoiseScheduler,
                   x_t: torch.Tensor,
                   t: int,
                   n_samples: int,
                   seed: int,
                   mini_batch: int) -> torch.Tensor:
    """calculate the approximate ELBO

    Args:
        model (torch.nn.Module): a diffusion model
        x_t (torch.Tensor): samples (batch, n_features)
        t (int): timestep where x_t is at
        T (int): total number of timesteps
        n_samples (int): number of samples used for Monte Carlo estimation
        cumprod_alpha (torch.Tensor): cumulative product of alpha (T,)
        seed (int): random seed
        mini_batch (int): mini batch size

    Returns:
        torch.Tensor: approximate ELBO (batch,)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    B, D = x_t.shape

    # sample noise (batch, n_sample, n_features)
    noise = torch.randn(B, n_samples, D, device=x_t.device)

    # sample timestep randomly from [t, T): (batch, n_sample)
    T = len(noise_scheduler)
    ts_k = torch.randint(t, T, (B, n_samples), device=x_t.device)
    reshaped_ts_k = ts_k.reshape(B*n_samples)
    reshaped_ts_t = torch.full((B*n_samples,), t, device=x_t.device)

    # sample x_k from p(x_k | x_t, t, k) (batch, n_sample, n_features)
    reshaped_x_t = x_t[:, None, :].expand(B, n_samples, D).reshape(B*n_samples, D)
    reshaped_noise = noise.reshape(B*n_samples, D)
    x_k = noise_scheduler.add_noise_at_t(reshaped_x_t, reshaped_noise, reshaped_ts_t, reshaped_ts_k)
    x_k = x_k.reshape(B, n_samples, D)
    # -> (batch, n_sample, n_features)

    # estimate the ELBO
    cumprod_alpha_prev = noise_scheduler.alphas_cumprod_prev.to(x_t.device)
    cumprod_alpha = noise_scheduler.alphas_cumprod.to(x_t.device)

    denoising_matching_term_list = []
    for i in range(0, n_samples, mini_batch):
        # Prepare mini-batch
        batch_x_k = x_k[:, i:i+mini_batch, :].reshape(-1, D)
        batch_ts_k = ts_k[:, i:i+mini_batch].reshape(-1)
        batch_noise = noise[:, i:i+mini_batch, :].reshape(-1, D)
        batch_ts_t = torch.full((B * min(mini_batch, n_samples - i),), t, device=x_t.device)

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
        batch_term = batch_term.pow(2).mean(dim=-1)
        # -> (batch * min(mini_batch, n_samples - i),)
        denoising_matching_term_list.append(batch_term.view(B, -1))
        # -> (batch, min(mini_batch, n_samples - i))

    denoising_matching_term = torch.cat(denoising_matching_term_list, dim=1)
    assert denoising_matching_term.shape == (B, n_samples)
    # -> (batch, n_sample)
    denoising_matching_term = denoising_matching_term.mean(dim=1)
    # -> (batch,)
    return -denoising_matching_term


# def compose_imagenet_diffusion_models(dit_model: DiT.models.DiT,
#                                       cfg_scale: float,
#                                       algebra: str,
#                                       class_1: int,
#                                       class_2: int,
#                                       null_class: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
#     """compose imagenet diffusion transformer models

#     Args:
#         dit_model (DiT.models.DiT): DiT model
#         cfg_scale (float): scale of the configuration
#         algebra (str): algebra operation, 'product', 'negation'
#         class_1 (int): class 1
#         class_2 (int): class 2
#         null_class (int): null class for unconditional generation

#     Returns:
#         Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: forward function with configuration
#     """

#     assert algebra in ["product", "negation"], "algebra should be 'product' or 'negation'; and 'summation' is not supported yet"

#     def forward_with_cfg(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
#         """
#         half = x
#         combined = torch.cat([half, half, half], dim=0)
#         y = torch.cat([torch.full((len(half),), class_1, dtype=torch.long, device=x.device),
#                        torch.full((len(half),), class_2, dtype=torch.long, device=x.device),
#                        torch.full((len(half),), null_class, dtype=torch.long, device=x.device)], dim=0)
#         t = torch.cat([t, t, t], dim=0)
#         # no need to track gradients for the classifier
#         model_out = dit_model.forward(combined, t, y)
#         # For exact reproducibility reasons, we apply classifier-free guidance on only
#         # three channels by default. The standard approach to cfg applies it to all channels.
#         # This can be done by uncommenting the following line and commenting-out the line following that.
#         # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
#         eps, rest = model_out[:, :3], model_out[:, 3:]
#         cond_1_eps, cond_2_eps, uncond_eps = torch.split(eps, len(eps) // 3, dim=0)
#         _, _, rest = torch.split(rest, len(rest) // 3, dim=0)
#         class_1_eps = cond_1_eps - uncond_eps
#         class_2_eps = cond_2_eps - uncond_eps

#         if algebra == "product":
#             condition_eps = class_1_eps + class_2_eps
#         elif algebra == "negation":
#             # TODO (rainorangelemon): probably finetune the hyperparameters of this setting
#             condition_eps = class_1_eps - class_2_eps
#         else:
#             raise ValueError("algebra should be 'product' or 'negation'")
#         eps = uncond_eps + cfg_scale * condition_eps
#         return torch.cat([eps, rest], dim=1)

#     return forward_with_cfg
