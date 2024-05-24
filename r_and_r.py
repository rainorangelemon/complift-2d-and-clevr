import torch
import numpy as np
from bootstrapping import bootstrapping_and_get_max, bootstrapping_and_get_interval
from baselines import intermediate_distribution
from ddpm import device


def calculate_energy(samples, model, t=0):
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


def calculate_interval(samples, model, confidence=0.999):
    # calculate the level-set values
    with torch.no_grad():
        energy_on_data = model.energy(torch.from_numpy(samples).to(device), torch.zeros(len(samples)).long().to(device))
    extreme_value_l, extreme_value_r = bootstrapping_and_get_interval(energy_on_data.cpu().numpy(), confidence=confidence)
    return extreme_value_l, extreme_value_r


def calculate_interval_multiple_timesteps(samples, model, confidence=0.999):
    intermediate_samples = intermediate_distribution(samples)[1:]  # ignore the initial Gaussian distribution
    intervals = []
    for t, samples_t in zip(range(len(intermediate_samples)-1, -1, -1), intermediate_samples):
        # calculate the level-set values
        with torch.no_grad():
            energy_on_data = model.energy(torch.from_numpy(samples_t).to(device), t+torch.zeros(len(samples)).long().to(device))
        extreme_value_l, extreme_value_r = bootstrapping_and_get_interval(energy_on_data.cpu().numpy(), confidence=confidence)
        intervals.append((extreme_value_l, extreme_value_r))

    return intervals


def calculate_interval_to_avoid_multiple_timesteps(samples_to_reach, samples_to_avoid, model, confidence=0.999):
    intermediate_samples = intermediate_distribution(samples_to_reach)[1:]  # ignore the initial Gaussian distribution
    intervals_reach = []
    for t, samples_t in zip(range(len(intermediate_samples)-1, -1, -1), intermediate_samples):
        # calculate the level-set values
        with torch.no_grad():
            energy_on_data = model.energy(torch.from_numpy(samples_t).to(device), t+torch.zeros(len(samples_to_reach)).long().to(device))
        extreme_value_l, extreme_value_r = bootstrapping_and_get_interval(energy_on_data.cpu().numpy(), confidence=confidence)
        intervals_reach.append((extreme_value_l, extreme_value_r))

    intermediate_samples = intermediate_distribution(samples_to_avoid)[1:]  # ignore the initial Gaussian distribution
    intervals_avoid = []
    for t, samples_t in zip(range(len(intermediate_samples)-1, -1, -1), intermediate_samples):
        # calculate the level-set values
        with torch.no_grad():
            energy_on_data = model.energy(torch.from_numpy(samples_t).to(device), t+torch.zeros(len(samples_to_avoid)).long().to(device))
        extreme_value_l, extreme_value_r = bootstrapping_and_get_interval(energy_on_data.cpu().numpy(), confidence=confidence)
        intervals_avoid.append((extreme_value_l, extreme_value_r))

    intervals = []
    for i, (interval_reach, interval_avoid) in enumerate(zip(intervals_reach, intervals_avoid)):
        if interval_avoid[0] < interval_reach[0]:
            interval_to_avoid = (interval_avoid[0], min(interval_reach[0], interval_avoid[1]))
        else:
            interval_to_avoid = (max(interval_reach[1], interval_avoid[0]), interval_avoid[1])
        if interval_avoid[1] <= interval_reach[0]:
            interval_to_avoid = (float('inf'), float('-inf'))
        if interval_reach[0] >= interval_reach[1]:
            interval_to_avoid = (float('-inf'), float('inf'))
        intervals.append(interval_to_avoid)

    return intervals    


def need_to_remove_with_thresholds(energy_1, energy_2, interval_1, interval_2, algebra):
    energies = [energy_1, energy_2]
    interval_mins = [interval_1[0], interval_2[0]]
    interval_maxs = [interval_1[1], interval_2[1]]
    out_of_interval = [((energy < interval_min) | (energy > interval_max)) for energy, interval_min, interval_max in zip(energies, interval_mins, interval_maxs)]
    if algebra == 'product':
        need_to_remove = out_of_interval[0] | out_of_interval[1]
    elif algebra == 'summation':
        need_to_remove = out_of_interval[0] & out_of_interval[1]
    elif algebra == 'negation':
        need_to_remove = out_of_interval[0] | (~out_of_interval[1])    
    return need_to_remove