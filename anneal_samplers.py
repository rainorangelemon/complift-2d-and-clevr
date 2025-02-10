# adopted from https://github.com/yilundu/reduce_reuse_recycle
# Line 36-41, 133-138 are modified to handle the case where the input is a batch of samples + Line 36-41 are changed for the correct pdf of the normal distribution
import torch
import numpy as np


class AnnealedMALASampler:
  """Implements AIS with ULA"""

  def __init__(self,
               num_steps,
               num_samples_per_step,
               step_sizes,
               gradient_function,
               ):
    assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
    self._step_sizes = step_sizes
    self._num_steps = num_steps
    self._num_samples_per_step = num_samples_per_step
    self._gradient_function = gradient_function


  def sample_step(self, x, t,ts, model_args):

    e_old, grad = self._gradient_function(x, ts, **model_args)
    for i in range(self._num_samples_per_step):
      ss = self._step_sizes[t]
      std = (2 * ss) ** .5
      noise = torch.randn_like(grad) * std

      x_proposal  = x + grad * ss + noise

      # Compute Energy of the samples
      e_new, grad_new = self._gradient_function(x_proposal, ts, **model_args)

      if x.ndim >= 2:
        log_xhat_given_x = -1.0 * ((x_proposal - x - ss * grad) ** 2).flatten(1).sum(dim=-1) / (4 * ss)
        log_x_given_xhat = -1.0 * ((x - x_proposal - ss * grad_new) ** 2).flatten(1).sum(dim=-1) / (4 * ss)
      else:
        log_xhat_given_x = -1.0 * ((x_proposal - x - ss * grad) ** 2).sum() / (4 * ss)
        log_x_given_xhat = -1.0 * ((x - x_proposal - ss * grad_new) ** 2).sum() / (4 * ss)
      log_alpha = e_new-e_old +log_x_given_xhat - log_xhat_given_x

      # Acceptance Ratio
      accepted = (torch.log(torch.rand(1)) < log_alpha.detach().cpu()).to(x.device)
      x[accepted] = x_proposal[accepted]
      e_old[accepted] = e_new[accepted]
      grad[accepted] = grad_new[accepted]

    return x

def leapfrog_step_c(x_0,
                  v_0,
                  gradient_target,
                  step_size,
                  mass_diag_sqrt,
                  num_steps,
                  grad_i):
  """Multiple leapfrog steps with no metropolis correction."""
  x_k = x_0
  v_k = v_0
  grad_k = grad_i
  if mass_diag_sqrt is None:
    mass_diag_sqrt = torch.ones_like(x_k)

  mass_diag = mass_diag_sqrt ** 2.

  for _ in range(num_steps):
    v_k += 0.5 * step_size *grad_k#  # half step in v
    x_k += step_size * v_k / mass_diag  # Step in x
    grad_k = gradient_target(x_k)[1]
    v_k += 0.5 * step_size * grad_k  # half step in v


  return x_k, v_k



class AnnealedCHASampler:

  def __init__(self,
               num_steps,
               num_samples_per_step,
               step_sizes,
               damping_coeff,
               mass_diag_sqrt,
               num_leapfrog_steps,
               gradient_function,
               ):
    assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
    self._damping_coeff = damping_coeff
    self._mass_diag_sqrt = mass_diag_sqrt
    self._step_sizes = step_sizes
    self._num_steps = num_steps
    self._num_leapfrog_steps = num_leapfrog_steps
    self._num_samples_per_step = num_samples_per_step
    self._gradient_function = gradient_function


  def leapfrog_step_(self, x, v, i, ts, grad_i,model_args):

    step_size = self._step_sizes[i]
    return leapfrog_step_c(x, v, lambda _x: self._gradient_function(_x, ts, **model_args), step_size, self._mass_diag_sqrt[i], self._num_leapfrog_steps,grad_i)


  def sample_step(self, x, t,ts, model_args):



    M=self._mass_diag_sqrt[t] # VAR

    # Sample Momentum
    v = torch.randn_like(x) * M


    for i in range(self._num_samples_per_step):
      # Partial Momentum Refreshment
      eps = torch.randn_like(x)
      v_prime = v * self._damping_coeff + np.sqrt(1. - self._damping_coeff**2) * eps * M

      energy_i, grad_i = self._gradient_function(x,ts,**model_args)
      x_old = x.clone()
      v_old =  v.clone()
      x_new, v_new = self.leapfrog_step_(x, v_prime, t, ts,grad_i, model_args)

      energy_new, grad_new = self._gradient_function(x_new,ts,**model_args)
      energy_diff = energy_new-energy_i


      # log_v = torch.sum(v_dist.log_prob(v_prime))
      # log_v_new = torch.sum(v_dist.log_prob(v_new))

      if x.ndim >= 2:
        log_v_new = (-0.5*(1/M**2)) * (v_new**2).flatten(1).sum(dim=-1) # As mean 0 and Variance M**2
        log_v = (-0.5*(1/M**2)) * (v_prime**2).flatten(1).sum(dim=-1)
      else:
        log_v_new = (-0.5*(1/M**2)) * (v_new**2).sum() # As mean 0 and Variance M**2
        log_v = (-0.5*(1/M**2)) * (v_prime**2).sum()

      logp_accept = energy_diff + (log_v_new - log_v)
      alpha = torch.min(torch.tensor(1.0),torch.exp(logp_accept))

      u = torch.rand(1)
      accepted = (u <= alpha.cpu()).to(x.device)

      x[accepted] = x_new[accepted]
      v[accepted] = v_new[accepted]
      x[~accepted] = x_old[~accepted]
      v[~accepted] = v_old[~accepted]

    return x

def leapfrog_step(x_0,
                  v_0,
                  gradient_target,
                  step_size,
                  mass_diag_sqrt,
                  num_steps,
                  ):
  # """Multiple leapfrog steps with no metropolis correction."""
  x_k = x_0
  v_k = v_0
  if mass_diag_sqrt is None:
    mass_diag_sqrt = torch.ones_like(x_k)

  mass_diag = mass_diag_sqrt ** 2.
  grad = gradient_target(x_k)
  for _ in range(num_steps):  # Inefficient version - should combine half steps
    v_k += 0.5 * step_size * grad#gradient_target(x_k)  # half step in v
    x_k += step_size * v_k / mass_diag  # Step in x
    grad = gradient_target(x_k)
    v_k += 0.5 * step_size * grad  # half step in v
  return x_k, v_k

class AnnealedUHASampler:
  """Implements UHA Sampling"""

  def __init__(self,
               num_steps,
               num_samples_per_step,
               step_sizes,
               damping_coeff,
               mass_diag_sqrt,
               num_leapfrog_steps,
               gradient_function,
               ):
    assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
    self._damping_coeff = damping_coeff
    self._mass_diag_sqrt = mass_diag_sqrt
    self._step_sizes = step_sizes
    self._num_steps = num_steps
    self._num_leapfrog_steps = num_leapfrog_steps
    self._num_samples_per_step = num_samples_per_step
    self._gradient_function = gradient_function


  def leapfrog_step_(self, x, v, i, ts, model_args):
      step_size = self._step_sizes[i]
      return leapfrog_step(x, v, lambda _x: self._gradient_function(_x, ts, **model_args), step_size, self._mass_diag_sqrt[i], self._num_leapfrog_steps)


  def sample_step(self, x, t, ts, model_args):

    # Sample Momentum
    v = torch.randn_like(x) * self._mass_diag_sqrt[t]

    for i in range(self._num_samples_per_step):

      # Partial Momentum Refreshment
      eps = torch.randn_like(x)

      v = v * self._damping_coeff + np.sqrt(1. - self._damping_coeff**2) * eps * self._mass_diag_sqrt[t]

      x, v = self.leapfrog_step_(x, v, t, ts, model_args)

    return x



class AnnealedULASampler:
  """Implements AIS with ULA"""

  def __init__(self,
               num_steps,
               num_samples_per_step,
               step_sizes,
               gradient_function,
               ):
    assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
    self._step_sizes = step_sizes
    self._num_steps = num_steps
    self._num_samples_per_step = num_samples_per_step
    self._gradient_function = gradient_function


  def sample_step(self, x, t,ts, model_args):

    for i in range(self._num_samples_per_step):
      ss = self._step_sizes[t]
      std = (2 * ss) ** .5
      grad = self._gradient_function(x, ts, **model_args)
      noise = torch.randn_like(grad) * std
      x = x + grad * ss + noise
    return x
