import jax
import jax.numpy as jnp
import distrax
from typing import Callable, Optional, Tuple, Union, Dict
Array = jnp.ndarray
Scalar = Union[float, int]
RandomKey = Array

GradientTarget = Callable[[Array, Array], Array]


def leapfrog_step(x_0: Array,
                  v_0: Array,
                  gradient_target: GradientTarget,
                  step_size: Array,
                  mass_diag_sqrt: Array,
                  num_steps: int):
  """Multiple leapfrog steps with no metropolis correction."""
  x_k = x_0
  v_k = v_0
  if mass_diag_sqrt is None:
    mass_diag_sqrt = jnp.ones_like(x_k)

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
               step_sizes: jnp.array,
               damping_coeff: int,
               mass_diag_sqrt: float,
               num_leapfrog_steps: int,
               initial_distribution: distrax.Distribution,
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

  def sample(self, key: RandomKey, n_samples: int):
    key, x_key = jax.random.split(key)
    x_k = self._initial_distribution.sample(seed=x_key, sample_shape=(n_samples,))

    v_dist = distrax.MultivariateNormalDiag(
        loc=jnp.zeros_like(x_k),
        scale_diag=jnp.ones_like(x_k) * self._mass_diag_sqrt)

    key, v_key = jax.random.split(key)
    v_k = v_dist.sample(seed=v_key)

    logw = -self._initial_distribution.log_prob(x_k)

    accept_rate = jnp.zeros((self._num_steps,))
    inputs = (key, logw, x_k, v_k, accept_rate)
    def body_fn(i, inputs):
      # unpack inputs
      key, logw, x_k, v_k, accept_rate = inputs
      dist_ind = (i // self._num_samples_per_step) + 1
      eps_key, accept_key, key = jax.random.split(key, 3)
      eps = jax.random.normal(eps_key, x_k.shape)
      # resample momentum
      v_k_prime = v_k * self._damping_coeff + jnp.sqrt(1. - self._damping_coeff**2) * eps * self._mass_diag_sqrt
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
      u = jax.random.uniform(accept_key, (x_k_next.shape[0],))
      accept = (u < jnp.exp(logp_accept)).astype(jnp.float32)
      # update importance weights
      logw += (logp_x - logp_x_hat) * accept
      # update samples
      x_k = accept[:, None] * x_k_next + (1 - accept[:, None]) * x_k
      v_k = accept[:, None] * v_k_next + (1 - accept[:, None]) * v_k_prime
      accept_rate = accept_rate.at[dist_ind].set(accept_rate[dist_ind] + accept.mean())
      return key, logw, x_k, v_k, accept_rate
    _, logw, x_k, v_k, accept_rate = jax.lax.fori_loop(0, self._total_steps, body_fn, inputs)

    # logw += self._target_distribution(x_k, self._num_steps - 1)
    accept_rate /= self._num_samples_per_step
    return x_k, logw, accept_rate


if __name__ == "__main__":
    import jax.numpy as jnp
    from jax import random, grad
    from distrax import MultivariateNormalDiag

    # Define the ring-like target distribution
    def ring_distribution(x):
        radius = jnp.sqrt(x[:, 0]**2 + x[:, 1]**2)
        return jnp.exp(-0.5 * (radius - 1)**2)

    def energy_function(x, _):
        log_prob = jnp.log(ring_distribution(x))
        return log_prob

    # Define the gradient of the energy function
    gradient_function = lambda x, _: grad(energy_function)(x, _)

    # Define the parameters for the AnnealedMUHASampler
    num_steps = 1000
    num_samples_per_step = 1000
    step_sizes = jnp.ones(num_steps) * 0.03
    damping_coeff = 0.5
    mass_diag_sqrt = 1.0
    num_leapfrog_steps = 3
    initial_distribution = MultivariateNormalDiag(loc=jnp.zeros(2), scale_diag=jnp.ones(2))
    target_distribution = ring_distribution

    # Create an instance of the AnnealedMUHASampler
    sampler = AnnealedMUHASampler(num_steps=num_steps,
                                num_samples_per_step=num_samples_per_step,
                                step_sizes=step_sizes,
                                damping_coeff=damping_coeff,
                                mass_diag_sqrt=mass_diag_sqrt,
                                num_leapfrog_steps=num_leapfrog_steps,
                                initial_distribution=initial_distribution,
                                gradient_function=gradient_function,
                                energy_function=energy_function)

    # Sample from the distribution using the sampler
    key = random.PRNGKey(123)
    samples, log_weights, accept_rate = sampler.sample(key, n_samples=num_samples_per_step)

    # Visualize the samples if needed
    import matplotlib.pyplot as plt
    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
    plt.title('Samples from Ring-Like Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axis('equal')
    plt.savefig('samples.png')