import numpy as np


def leapfrog(x, p, epsilon, L, grad_logp):
    x_new = x
    p_new = p
    for i in range(L):
        p_new += 0.5 * epsilon * grad_logp(x_new)
        x_new += epsilon * p_new
        p_new += 0.5 * epsilon * grad_logp(x_new)
    return x_new, p_new


# one step of the HMC sampler
def hmc(logp, grad_logp, x0, epsilon, L):
    n_samples, n_params = x0.shape
    x = np.zeros((n_samples, n_params))
    x = x0
    p = np.random.randn(n_samples, n_params)
    x_new, p_new = leapfrog(x, p, epsilon, L, grad_logp)
    new_energy = -logp(x_new) + 0.5 * np.sum(p_new**2, axis=1)
    old_energy = -logp(x) + 0.5 * np.sum(p**2, axis=1)
    # Metropolis-Hastings acceptance criterion
    alpha = np.random.rand(n_samples)
    accept = alpha < np.exp(old_energy - new_energy)
    x[accept] = x_new[accept]
    acceptance_rate = np.mean(accept)
    return x, acceptance_rate


class DonutPDF:
    def __init__(self, radius=3, sigma2=0.05):
        self.radius = radius
        self.sigma2 = sigma2

    def log_density(self, x):
        r = np.linalg.norm(x, axis=1)
        return -(r - self.radius) ** 2 / self.sigma2

    def grad_log_density(self, x):
        r = np.linalg.norm(x, axis=1)
        result = np.zeros_like(x)
        non_zero_result = 2 * x * (self.radius / r - 1).reshape(-1, 1) / self.sigma2
        result[r!=0] = non_zero_result[r!=0]
        return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm
    donut = DonutPDF()
    logp = donut.log_density
    grad_logp = donut.grad_log_density
    x = np.random.randn(1000, 2)
    epsilon = 0.1
    L = 3
    for i in tqdm(range(2000)):
        x, acceptance_rate = hmc(logp, grad_logp, x, epsilon, L)
        plt.scatter(x[:, 0], x[:, 1], s=1)
    plt.savefig('hmc.png')