import torch
from typing import List, Union, Tuple
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
        # essentially, we want to interval_negative - interval_positive
        if interval_negative[0] < interval_positive[0]:
            interval_to_avoid = (interval_negative[0], min(interval_positive[0], interval_negative[1]))
        else:
            interval_to_avoid = (max(interval_positive[1], interval_negative[0]), interval_negative[1])
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
