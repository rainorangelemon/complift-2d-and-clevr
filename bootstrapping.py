import numpy as np
from tqdm.auto import tqdm
from typing import List


def bootstrapping_and_get_max(data, n=1000, confidence=0.999):
    rho = data.max()

    samples = np.random.choice(data, size=len(data)*n, replace=True)
    samples = samples.reshape(n, len(data))
    sampled_rhos = np.max(samples, axis=1)

    # get the 99.9% confidence interval
    percentile_value = np.percentile(sampled_rhos, 1-confidence)
    return 2*rho-percentile_value


def bootstrapping_and_get_interval(data: np.ndarray,
                                   method: str,
                                   confidence: float,
                                   n: int=None) -> List[float]:
    """bootstrapping, and get the confidence interval

    Args:
        data (np.ndarray): the data
        method (str): the method to use.
        confidence (float): the confidence level.
        n (int, optional): the number of samples. Defaults to None.

    Returns:
        List[float]: the confidence interval, [lower, upper]

    """
    assert method in ["simple", "normal", "pivot"], "method should be simple, normal or pivot"

    if len(data) == 0:
        interval = [float('inf'), float('-inf')]

    elif method == "simple":
        interval = [data.min(), data.max()]

    elif method == "normal":
        required_samples = int(2/(1-confidence))
        if len(data) < required_samples:
            data = np.random.choice(data, size=required_samples, replace=True)
        interval = [np.percentile(data, 100*(1-confidence)/2), np.percentile(data, 100*(1+confidence)/2)]

    else:
        if n is None:
            required_samples = int(2/(1-confidence))
            n = required_samples

        samples = np.random.choice(data, size=n, replace=True)

        # get the pivot confidence interval
        percentile_min = np.percentile(samples, 100*(1-confidence)/2)
        percentile_max = np.percentile(samples, 100*(1+confidence)/2)
        avg = np.mean(samples)
        interval = [2*avg-percentile_max, 2*avg-percentile_min]

    return interval
