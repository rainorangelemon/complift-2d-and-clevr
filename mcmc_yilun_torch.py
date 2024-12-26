import torch
import torch.distributions as dist
from typing import Callable, Optional, Tuple, Union, Dict
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist

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


class AnnealedULASampler:
    """Implements AIS with Unadjusted Langevin Algorithm"""

    def __init__(self,
                 num_steps: int,
                 num_samples_per_step: int,
                 step_sizes: torch.Tensor,
                 initial_distribution: dist.Distribution,
                 gradient_function,
                 energy_function):
        assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._num_samples_per_step = num_samples_per_step
        self._initial_distribution = initial_distribution
        self._gradient_function = gradient_function
        self._energy_function = energy_function
        self._total_steps = self._num_samples_per_step * self._num_steps

    def transition_distribution(self, i, x):
        ss = self._step_sizes[i]
        std = torch.sqrt(2 * ss)
        grad = self._gradient_function(x, i)
        mu = x + grad * ss
        return dist.MultivariateNormal(
            loc=mu,
            covariance_matrix=torch.eye(x.shape[-1]).to(x.device) * std)

    def sample(self, n_samples: int, callback=None):
        x = self._initial_distribution.sample((n_samples,))
        logw = -self._initial_distribution.log_prob(x)

        total_samples = torch.zeros((self._num_steps, x.shape[0], x.shape[1])).to(x.device)

        for i in tqdm(range(self._total_steps)):
            dist_ind = (i // self._num_samples_per_step)
            dist_forward = self.transition_distribution(dist_ind, x)
            x_next = dist_forward.sample()

            # reverse transition
            dist_reverse = self.transition_distribution(dist_ind - 1, x_next)

            logw += dist_reverse.log_prob(x) - dist_forward.log_prob(x_next)

            x = x_next
            if callback is not None:
                x = callback(x, dist_ind)

            total_samples[dist_ind] = x

        return total_samples, x, logw, None


class AnnealedUHASampler:
    """Implements AIS with Underdamped Hamiltonian Algorithm"""

    def __init__(self,
                 num_steps: int,
                 num_samples_per_step: int,
                 step_sizes: torch.Tensor,
                 damping_coeff: float,
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
        self._total_steps = self._num_samples_per_step * self._num_steps

    def leapfrog_step(self, x, v, i):
        step_size = self._step_sizes[i]
        return leapfrog_step(x, v, lambda _x: self._gradient_function(_x, i), step_size, self._mass_diag_sqrt, self._num_leapfrog_steps)

    def sample(self, n_samples: int, callback=None):
        x_k = self._initial_distribution.sample((n_samples,))

        v_dist = dist.MultivariateNormal(
            loc=torch.zeros(x_k.shape[-1]).to(x_k.device),
            covariance_matrix=torch.eye(x_k.shape[-1]).to(x_k.device) * self._mass_diag_sqrt)

        v_k = v_dist.sample((n_samples,))
        logw = -self._initial_distribution.log_prob(x_k)

        total_samples = torch.zeros((self._num_steps, x_k.shape[0], x_k.shape[1])).to(x_k.device)

        for i in tqdm(range(self._total_steps)):
            dist_ind = (i // self._num_samples_per_step)
            eps = torch.randn_like(x_k)

            # Resample momentum
            v_k_prime = v_k * self._damping_coeff + torch.sqrt((1. - self._damping_coeff**2) * torch.ones_like(v_k)) * eps * self._mass_diag_sqrt
            # Advance samples
            x_k_next, v_k_next = self.leapfrog_step(x_k, v_k_prime, dist_ind)

            # Update importance weights
            logp_v_p = v_dist.log_prob(v_k_prime)
            logp_v = v_dist.log_prob(v_k_next)
            logw += logp_v - logp_v_p

            x_k = x_k_next
            v_k = v_k_next

            if callback is not None:
                x_k = callback(x_k, dist_ind)

            total_samples[dist_ind] = x_k

        return total_samples, x_k, logw, None


class AnnealedMALASampler:
    """Implements AIS with Metropolis-Adjusted Langevin Algorithm"""

    def __init__(self,
                 num_steps: int,
                 num_samples_per_step: int,
                 step_sizes: torch.Tensor,
                 initial_distribution: dist.Distribution,
                 gradient_function,
                 energy_function):
        assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._num_samples_per_step = num_samples_per_step
        self._initial_distribution = initial_distribution
        self._gradient_function = gradient_function
        self._energy_function = energy_function
        self._total_steps = self._num_samples_per_step * self._num_steps

    def transition_distribution(self, i, x):
        ss = self._step_sizes[i]
        std = torch.sqrt(2 * ss)
        grad = self._gradient_function(x, i)
        mu = x + grad * ss
        return dist.MultivariateNormal(
            loc=mu,
            covariance_matrix=torch.eye(x.shape[-1]).to(x.device) * std)

    def sample(self, n_samples: int, callback=None):
        x = self._initial_distribution.sample((n_samples,))
        logw = -self._initial_distribution.log_prob(x)

        accept_rate = torch.zeros((self._num_steps,)).to(x.device)
        total_samples = torch.zeros((self._num_steps, x.shape[0], x.shape[1])).to(x.device)

        for i in tqdm(range(self._total_steps)):
            dist_ind = (i // self._num_samples_per_step)

            # Propose new sample
            dist_forward = self.transition_distribution(dist_ind, x)
            x_next = dist_forward.sample()
            dist_reverse = self.transition_distribution(dist_ind, x_next)

            # Compute acceptance probability
            logp_x = self._energy_function(x, dist_ind)
            logp_x_next = self._energy_function(x_next, dist_ind)
            logp_forward = dist_forward.log_prob(x_next)
            logp_reverse = dist_reverse.log_prob(x)

            logp_accept = logp_x_next - logp_x + logp_reverse - logp_forward
            u = torch.rand(x_next.shape[0]).to(x.device)
            accept = (u < torch.exp(logp_accept)).float()

            # Update samples and importance weights
            x = accept[:, None] * x_next + (1 - accept[:, None]) * x
            logw += (logp_x - logp_x_next) * accept
            accept_rate[dist_ind] += accept.mean()

            if callback is not None:
                x = callback(x, dist_ind)

            total_samples[dist_ind] = x

        accept_rate /= self._num_samples_per_step
        return total_samples, x, logw, accept_rate


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

        self._total_steps = self._num_samples_per_step * self._num_steps

    def leapfrog_step(self, x, v, i):
        step_size = self._step_sizes[i]
        return leapfrog_step(x, v, lambda _x: self._gradient_function(_x, i), step_size, self._mass_diag_sqrt, self._num_leapfrog_steps)

    def sample(self, n_samples: int, callback=None):
        x_k = self._initial_distribution.sample((n_samples,))

        v_dist = dist.MultivariateNormal(
            loc=torch.zeros(x_k.shape[-1]).to(x_k.device),
            covariance_matrix=torch.eye(x_k.shape[-1]).to(x_k.device) * self._mass_diag_sqrt)

        v_k = v_dist.sample((n_samples,))

        logw = -self._initial_distribution.log_prob(x_k)

        accept_rate = torch.zeros((self._num_steps,)).to(x_k.device)
        total_samples = torch.zeros((self._num_steps, x_k.shape[0], x_k.shape[1])).to(x_k.device)
        for i in tqdm(range(self._total_steps)):
            dist_ind = (i // self._num_samples_per_step)
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

            if callback is not None:
                x_k = callback(x_k, dist_ind)

            total_samples[dist_ind] = x_k

        accept_rate /= self._num_samples_per_step
        return total_samples, x_k, logw, accept_rate


# Define the ring-like target distribution and its derivatives
class RingDistribution:
    def __init__(self, scale=0.05):
        self.scale = scale

    def __call__(self, x):
        radius = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
        return torch.exp(-(radius - 1)**2 / self.scale)

    def log_prob(self, x):
        radius = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
        return -(radius - 1)**2 / self.scale

def energy_function(x, _):
    radius = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
    return -(radius - 1)**2 / 0.05

@torch.enable_grad()
def gradient_function(x, _):
    # Make sure input requires gradient
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)

    assert x.requires_grad, "Input x must have requires_grad=True"
    # Compute energy
    energy = energy_function(x, _)

    # Compute gradient
    grad = torch.autograd.grad(energy.sum(), x)[0]
    return grad.detach()

# Common parameters for all samplers
num_steps = 100
dim = 2
init_std = 1.0
init_mu = 0.0
samples_per_step = 10
batch_size = 1000

# Specific parameters for different samplers
damping = 0.5
mass_diag_sqrt = 1.0
num_leapfrog = 3
uha_step_size = 0.03
ula_step_size = 0.001

# Step sizes for different samplers
step_sizes = torch.ones(num_steps) * uha_step_size
ula_step_sizes = torch.ones(num_steps) * ula_step_size

# Initial distribution
initial_distribution = dist.MultivariateNormal(
    loc=torch.zeros(dim) + init_mu,
    covariance_matrix=torch.eye(dim) * init_std
)

# Ring distribution for plotting
ring_dist = RingDistribution()

# Create instances of all samplers
samplers = {
    "ULA": AnnealedULASampler(
        num_steps=num_steps,
        num_samples_per_step=samples_per_step,
        step_sizes=ula_step_sizes,
        initial_distribution=initial_distribution,
        gradient_function=gradient_function,
        energy_function=energy_function
    ),
    "UHA": AnnealedUHASampler(
        num_steps=num_steps,
        num_samples_per_step=samples_per_step,
        step_sizes=step_sizes,
        damping_coeff=damping,
        mass_diag_sqrt=mass_diag_sqrt,
        num_leapfrog_steps=num_leapfrog,
        initial_distribution=initial_distribution,
        gradient_function=gradient_function,
        energy_function=energy_function
    ),
    "MALA": AnnealedMALASampler(
        num_steps=num_steps,
        num_samples_per_step=samples_per_step,
        step_sizes=step_sizes,
        initial_distribution=initial_distribution,
        gradient_function=gradient_function,
        energy_function=energy_function
    ),
    "MUHA": AnnealedMUHASampler(
        num_steps=num_steps,
        num_samples_per_step=samples_per_step,
        step_sizes=step_sizes,
        damping_coeff=damping,
        mass_diag_sqrt=mass_diag_sqrt,
        num_leapfrog_steps=num_leapfrog,
        initial_distribution=initial_distribution,
        gradient_function=gradient_function,
        energy_function=energy_function
    )
}

if __name__ == "__main__":
    # Plot target distribution
    x = torch.linspace(-2, 2, 100)
    y = torch.linspace(-2, 2, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    XY = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    Z = ring_dist(XY)

    plt.figure(figsize=(10, 8))
    plt.contour(x.numpy(), y.numpy(), Z.reshape(100, 100).numpy())
    plt.title('Target Ring-Like Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.savefig('target_distribution.png')
    plt.close()

    # Sample from all distributions and plot results
    plt.figure(figsize=(15, 12))
    for idx, (name, sampler) in enumerate(samplers.items(), 1):
        plt.subplot(2, 2, idx)

        print(f"\nSampling using {name}...")
        total_samples, final_samples, log_weights, accept_rate = sampler.sample(batch_size)

        # Plot samples
        plt.scatter(final_samples[:, 0].numpy(), final_samples[:, 1].numpy(),
                   s=5, alpha=0.5, label='Samples')

        # Plot target distribution contour
        plt.contour(x.numpy(), y.numpy(), Z.reshape(100, 100).numpy(),
                   colors='r', alpha=0.5, levels=5)

        if accept_rate is not None:
            plt.title(f'{name} (Acceptance Rate: {accept_rate.mean():.2f})')
        else:
            plt.title(name)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

    plt.tight_layout()
    plt.savefig('sampler_comparison.png')
    plt.close()
