from tqdm.auto import tqdm
import numpy as np
import ddpm
import torch
from mcmc_yilun_torch import AnnealedMUHASampler
import torch.distributions as dist
import ot


def diffusion_baseline(model_to_test, num_timesteps=50, eval_batch_size=8000):
    dim = 2
    device = ddpm.device
    model_to_test = model_to_test.to(device)
    noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps)
    sample = torch.randn(eval_batch_size, dim).to(device)
    timesteps = list(range(num_timesteps))[::-1]

    samples = []
    for i, t in enumerate(tqdm(timesteps)):
        t_tensor = torch.from_numpy(np.repeat(t, eval_batch_size)).long().to(device)
        with torch.no_grad():
            residual = model_to_test(sample, t_tensor)
        sample = noise_scheduler.step(residual, t_tensor[0], sample)
        samples.append(sample.cpu().numpy())
    return samples


def ebm_baseline(model_to_test, num_timesteps=50, eval_batch_size=8000, temperature=1):
    
    noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps)
    device = ddpm.device
    model_to_test = model_to_test.to(device)

    num_steps = 50
    dim = 2
    init_std = 1.
    init_mu = 0.
    damping = .5
    mass_diag_sqrt = 1.
    num_leapfrog = 3
    samples_per_step = 10  # aka mcmc_per_step
    uha_step_size = .03
    uha_step_sizes = torch.ones((num_steps,)) * uha_step_size

    initial_distribution = dist.MultivariateNormal(loc=torch.zeros(dim).to(device) + init_mu, covariance_matrix=torch.eye(dim).to(device) * init_std)

    def energy_function(x, t): 
        t = num_steps - t - 1
        x = x.clone().to(device)
        t_tensor = torch.from_numpy(np.repeat(t, eval_batch_size)).long().to(device)
        return -(model_to_test.energy(x, t_tensor)) * temperature / noise_scheduler.sqrt_one_minus_alphas_cumprod[t]

    def gradient_function(x, t):
        t = num_steps - t - 1
        x = x.clone().to(device)
        t_tensor = torch.from_numpy(np.repeat(t, eval_batch_size)).long().to(device)
        return -(model_to_test(x, t_tensor)) * temperature / noise_scheduler.sqrt_one_minus_alphas_cumprod[t]

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

    total_samples, _, _, _ = sampler.sample(n_samples=eval_batch_size)
    return total_samples.cpu().numpy()


def intermediate_distribution(data_points, num_timesteps=50, eval_batch_size=8000):
    noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps)
    device = ddpm.device
    origin_data = torch.tensor(data_points).float().to(device)
    intermediate_data_list = []
    for i in range(num_timesteps):
        noise = torch.randn_like(origin_data)
        intermediate_data = noise_scheduler.add_noise(origin_data, noise, torch.ones(len(origin_data)).long().to(device) * i)
        intermediate_data_list.append(intermediate_data.cpu().numpy())
    return intermediate_data_list[::-1] + [origin_data.cpu().numpy()]


def evaluate_W2(generated_samples, target_samples):
    cost_matrix = ot.dist(generated_samples, target_samples, metric='sqeuclidean')

    # Calculate the Wasserstein-2 distance using the optimal transport plan
    w2_distance = ot.emd2(np.ones(len(generated_samples)) / len(generated_samples),
                        np.ones(len(target_samples)) / len(target_samples),
                        cost_matrix,
                        numItermax=int(1e7))
    return w2_distance