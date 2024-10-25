# Test the ELBO calculation

import io
import torch
import wandb
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import ddpm
from r_and_r import calculate_elbo

wandb.init(project="r_and_r", name="test_elbo")
wandb.run.log_code(".")

num_timesteps = 50
model_1 = ddpm.EnergyMLP()
model_2 = ddpm.EnergyMLP()
noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps)

algebra = "product"
suffix = "a"
device = ddpm.device
model_1.load_state_dict(torch.load(f"exps/{algebra}_{suffix}1/ema_model.pth"))
model_1.to(device)
model_2.load_state_dict(torch.load(f"exps/{algebra}_{suffix}2/ema_model.pth"))
model_2.to(device)

# sample data
model = model_1
eval_batch_size = 1000
probe_samples = 1000
with torch.no_grad():
    sample = torch.randn(eval_batch_size, 2).to(device)
    timesteps = list(range(len(noise_scheduler)))[::-1]
    for i, t in enumerate(tqdm(timesteps)):
        t = torch.full((eval_batch_size,), t).long().to(device)
        with torch.no_grad():
            residual = model(sample, t)
        sample = noise_scheduler.step(residual, t[0], sample)
        elbo = calculate_elbo(model=model_1,
                              noise_scheduler=noise_scheduler,
                              x_t=sample,
                              t=t[0],
                              n_samples=probe_samples,
                              seed=42,
                              mini_batch=100,
                              same_noise=False,
                              sample_timesteps="random")
        energy = model.energy(sample, t)
        # plot the distribution of elbo and energy on 2 subplots
        plt.clf()
        plt.close("all")
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].hist(elbo.cpu().numpy(), bins=50)
        axs[0].set_title("ELBO")
        axs[1].hist(-energy.cpu().numpy(), bins=50)
        axs[1].set_title("Negative Energy")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)

        wandb.log({"elbo vs neg energy": wandb.Image(img)})

        if t[0] == 0:
            # test add_noise_at_t is the same as add_noise when t=0
            noise = torch.randn_like(sample)
            assert torch.allclose(noise_scheduler.add_noise_at_t(sample, noise, t, t*0+0), noise_scheduler.add_noise(sample, noise, t*0+0))
