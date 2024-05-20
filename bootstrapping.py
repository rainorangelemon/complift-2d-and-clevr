import numpy as np
from tqdm.auto import tqdm

def bootstrapping_and_get_max(data, n=1000, confidence=0.999):
    rho = data.max()

    sampled_rhos = []
    for _ in tqdm(range(n), desc='calculating bootstrapping'):
        samples = np.random.choice(data, size=len(data), replace=True)
        sampled_rhos.append(samples.max())

    # get the 99.9% confidence interval
    percentile_value = np.percentile(np.array(sampled_rhos), 1-confidence)
    return 2*rho-percentile_value