import torch
import torch.distributions as dist
from typing import Callable, Optional, Tuple, Union, Dict
import numpy as np
from tqdm.auto import tqdm

Array = torch.Tensor
Scalar = Union[float, int]


GradientTarget = Callable[[Array, Array], Array]


def leapfrog_step(x_0: Array,
                  v_0: Array,
                  gradient_target: GradientTarget,
                  step_size: Array,
                  mass_diag_sqrt: Array,
                  num_steps: int):
    """Multiple leapfrog steps with no metropolis correction."""
    x_k = x_0.clone()
    v_k = v_0.clone()
    if mass_diag_sqrt is None:
        mass_diag_sqrt = torch.ones_like(x_k)

    mass_diag = mass_diag_sqrt ** 2.

    for _ in range(num_steps):  # Inefficient version - should combine half steps
        v_k += 0.5 * step_size * gradient_target(x_k)  # half step in v
        x_k += step_size * v_k / mass_diag  # Step in x
        grad = gradient_target(x_k)
        v_k += 0.5 * step_size * grad  # half step in v
    return x_k, v_k


class AnnealedMUHASampler:
    """Implements AIS with ULA"""

    def __init__(self,
                 num_steps: int,
                 num_samples_per_step: int,
                 step_sizes: torch.Tensor,
                 damping_coeff: int,
                 mass_diag_sqrt: float,
                 num_leapfrog_steps: int,
                 initial_distribution: dist.Distribution,
                 gradient_function,
                 energy_function):
        assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
        self._damping_coeff = damping_coeff
        self._mass_diag_sqrt = mass_diag_sqrt
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._num_leapfrog_steps = num_leapfrog_steps
        self._num_samples_per_step = num_samples_per_step
        self._initial_distribution = initial_distribution
        self._gradient_function = gradient_function
        self._energy_function = energy_function

        self._total_steps = self._num_samples_per_step * (self._num_steps - 1)
        self._total_steps_reverse = self._num_samples_per_step * self._num_steps

    def leapfrog_step(self, x, v, i):
        step_size = self._step_sizes[i]
        return leapfrog_step(x, v, lambda _x: self._gradient_function(_x, i), step_size, self._mass_diag_sqrt, self._num_leapfrog_steps)

    @torch.no_grad()
    def sample(self, n_samples: int):
        x_k = self._initial_distribution.sample((n_samples,))

        v_dist = dist.MultivariateNormal(
            loc=torch.zeros(x_k.shape[-1]).to(x_k.device),
            covariance_matrix=torch.eye(x_k.shape[-1]).to(x_k.device) * self._mass_diag_sqrt)

        v_k = v_dist.sample((n_samples,))

        logw = -self._initial_distribution.log_prob(x_k)

        accept_rate = torch.zeros((self._num_steps,)).to(x_k.device)
        total_samples = torch.zeros((self._num_steps, x_k.shape[0], x_k.shape[1])).to(x_k.device)
        for i in tqdm(range(self._total_steps)):
            dist_ind = (i // self._num_samples_per_step) + 1
            eps = torch.randn_like(x_k)

            # resample momentum
            v_k_prime = v_k * self._damping_coeff + torch.sqrt((1. - self._damping_coeff**2) * torch.ones_like(v_k)) * eps * self._mass_diag_sqrt
            # advance samples
            x_k_next, v_k_next = self.leapfrog_step(x_k, v_k_prime, dist_ind)

            # compute change in density
            logp_v_p = v_dist.log_prob(v_k_prime)
            logp_v = v_dist.log_prob(v_k_next)
            # compute target log-probs
            logp_x = self._energy_function(x_k, dist_ind)
            logp_x_hat = self._energy_function(x_k_next, dist_ind)
            # compute joint log-probs
            log_joint_prev = logp_x + logp_v_p
            log_joint_next = logp_x_hat + logp_v
            # acceptance prob
            logp_accept = log_joint_next - log_joint_prev
            u = torch.rand(x_k_next.shape[0]).to(x_k.device)
            accept = (u < torch.exp(logp_accept)).float()
            # update importance weights
            logw += (logp_x - logp_x_hat) * accept
            # update samples
            x_k = accept[:, None] * x_k_next + (1 - accept[:, None]) * x_k
            v_k = accept[:, None] * v_k_next + (1 - accept[:, None]) * v_k_prime
            accept_rate[dist_ind] += accept.mean()
            total_samples[dist_ind] = x_k
        
        accept_rate /= self._num_samples_per_step
        return total_samples, x_k, logw, accept_rate


if __name__ == "__main__":
    import torch
    import torch.distributions as dist

    # Define the ring-like target distribution
    def ring_distribution(x):
        radius = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
        return torch.exp(-(radius - 1)**2 / 0.05)

    def energy_function(x, _):
        radius = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
        log_prob = -(radius - 1)**2 / 0.05
        return log_prob

    # Define the gradient of the energy function
    def gradient_function(x, _):
        x_need_grad = x.clone().detach().requires_grad_(True)
        energy = energy_function(x_need_grad, _)
        grad = torch.autograd.grad(energy.sum(), x_need_grad)[0]
        return grad.clone().detach()
    

    # plot the gradient direction
    import matplotlib.pyplot as plt
    x = torch.linspace(-2, 2, 100)
    y = torch.linspace(-2, 2, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    XY = torch.stack([X, Y], dim=-1).reshape(-1, 2)  # Changed view to reshape
    Z = ring_distribution(XY)
    print(XY.shape, Z.shape)
    X = X.reshape(-1)  # Changed view to reshape
    Y = Y.reshape(-1)  # Changed view to reshape
    Z = Z.reshape(-1, 1)  # Changed view to reshape
    plt.contour(x.numpy(), y.numpy(), Z.reshape(100, 100).numpy())  # Changed view to reshape
    plt.title('Gradient of Ring-Like Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axis('equal')
    plt.savefig('gradient.png')

    # Define the parameters for the AnnealedMUHASampler
    num_steps = 100
    dim = 2
    n_mode = 4
    std = .05
    init_std = 1.
    init_mu = 0.
    damping = .5
    mass_diag_sqrt = 1.
    num_leapfrog = 3
    samples_per_step = 10
    uha_step_size = .03
    ula_step_size = .001
    uha_step_sizes = torch.ones((num_steps,)) * uha_step_size

    batch_size = 1000    

    initial_distribution = dist.MultivariateNormal(loc=torch.zeros(dim) + init_mu, covariance_matrix=torch.eye(dim) * init_std)

    # Create an instance of the AnnealedMUHASampler
    sampler = AnnealedMUHASampler(num_steps=num_steps,
                                num_samples_per_step=samples_per_step,
                                step_sizes=uha_step_sizes,
                                damping_coeff=damping,
                                mass_diag_sqrt=mass_diag_sqrt,
                                num_leapfrog_steps=num_leapfrog,
                                initial_distribution=initial_distribution,
                                gradient_function=gradient_function,
                                energy_function=energy_function)

    # Sample from the distribution using the sampler
    samples, log_weights, accept_rate = sampler.sample(n_samples=8000)

    # Visualize the samples if needed
    import matplotlib.pyplot as plt
    plt.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=5, alpha=0.5)
    plt.title('Samples from Ring-Like Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axis('equal')
    plt.savefig('figures/samples.png')