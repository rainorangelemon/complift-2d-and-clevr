import numpy as np
from tqdm.auto import tqdm

def bootstrapping_and_get_max(data, n=1000, confidence=0.999):
    rho = data.max()

    samples = np.random.choice(data, size=len(data)*n, replace=True)
    samples = samples.reshape(n, len(data))
    sampled_rhos = np.max(samples, axis=1)

    # get the 99.9% confidence interval
    percentile_value = np.percentile(sampled_rhos, 1-confidence)
    return 2*rho-percentile_value


def bootstrapping_and_get_interval(data, n=1000, confidence=0.999):
    rho = [data.min(), data.max()]

    samples = np.random.choice(data, size=len(data)*n, replace=True)
    samples = samples.reshape(n, len(data))
    
    # get the 99.9% confidence interval
    percentile_min = np.percentile(np.min(samples, axis=1), confidence)
    percentile_max = np.percentile(np.max(samples, axis=1), 1-confidence)
    return [2*rho[0]-percentile_min, 2*rho[1]-percentile_max]