import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import datasets
from positional_embeddings import PositionalEmbedding
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Block(nn.Module):
    def __init__(self, size: int, widen_factor: int = 2):
        super().__init__()

        self.ff = nn.Linear(size, size * widen_factor)
        self.ff_emb = nn.Linear(size, size * widen_factor)
        self.ff_in = nn.Linear(size * widen_factor, size * widen_factor)
        self.ff_out = nn.Linear(size * widen_factor, size)
        self.ln = nn.LayerNorm(size)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        h = self.ff(self.act(self.ln(x)))
        h += self.ff_emb(emb)
        h = self.act(h)
        h = self.ff_out(self.act(self.ff_in(h)))
        return x + h


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 4, emb_size: int = 128,
                 time_emb: str = "learnable", input_emb: str = "identity"):
        super().__init__()

        if time_emb != "learnable":
            self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        else:
            self.time_mlp = nn.Embedding(1000, emb_size)
        self.input_mlp = nn.Linear(2, emb_size)

        self.layers = torch.nn.ModuleList()
        for _ in range(hidden_layers):
            self.layers.append(Block(hidden_size))

        self.output_mlp = nn.Linear(hidden_size, 2)

    def forward(self, x, t):
        x_emb = self.input_mlp(x)
        t_emb = self.time_mlp(t)
        for layer in self.layers:
            x_emb = layer(x_emb, t_emb)
        x = self.output_mlp(x_emb)
        return x


class EnergyMLP(nn.Module):
    def __init__(self, energy_form='salimans', *args, **kwargs):
        super().__init__()
        self.mlp = MLP(*args, **kwargs)
        self.energy_form = energy_form

    def _energy(self, x, t):
        if self.energy_form == 'salimans':
            return ((x - self.mlp(x, t)) ** 2).sum(dim=-1)
        elif self.energy_form == 'L2':
            return (self.mlp(x, t) ** 2).sum(dim=-1)
        elif self.energy_form == 'inner_product':
            return (x * self.mlp(x, t)).sum(dim=-1)
        else:
            raise NotImplementedError

    def forward(self, x, t):
        # check whether torch.autograd is enabled
        with torch.autograd.enable_grad():
            # take the derivative of the energy
            x = x.clone().detach().requires_grad_(True)
            energy = self._energy(x, t)
            grad = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
        # print(x.max(), energy.shape, energy.max(), grad.shape, grad.max(), x[(grad.abs().sum(-1).argmax())])
        if not torch.is_grad_enabled():
            grad = grad.detach()
        return grad

    def energy(self, x, t):
        return self._energy(x, t)


class CompositionEnergyMLP(nn.Module):
    def __init__(self, *models, algebra='product'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.algebra = algebra

    def _energy(self, x, t):
        energies = [model.energy(x, t) for model in self.models]
        if self.algebra == 'product':
            result = torch.sum(torch.stack(energies), dim=0)
        elif self.algebra == 'summation':
            result = -torch.logsumexp(-torch.stack(energies), dim=0)
        elif self.algebra == 'negation':
            energies = torch.stack(energies)
            energies[-1] = -0.5 * energies[-1]
            energies[:-1] = energies[:-1]
            result = torch.sum(energies, dim=0)
        else:
            raise NotImplementedError
        return result

    def forward(self, x, t):
        if self.algebra == 'product':
            scores = [model(x, t) for model in self.models]
            return torch.sum(torch.stack(scores), dim=0)
        else:
            # check whether torch.autograd is enabled
            with torch.autograd.enable_grad():
                # take the derivative of the energy
                x = x.clone().detach().requires_grad_(True)
                energy = self._energy(x, t)
                grad = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
            # print(x.max(), energy.shape, energy.max(), grad.shape, grad.max(), x[(grad.abs().sum(-1).argmax())])
            if not torch.is_grad_enabled():
                grad = grad.detach()
            # print('composition', grad.shape)
            return grad

    def energy(self, x, t):
        return self._energy(x, t)


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        t = t.to(self.sqrt_inv_alphas_cumprod.device)
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1).to(x_t.device)
        s2 = s2.reshape(-1, 1).to(x_t.device)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        t = t.to(self.posterior_mean_coef1.device)
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1).to(x_t.device)
        s2 = s2.reshape(-1, 1).to(x_t.device)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        # pred_original_sample = pred_original_sample.clamp(-1, 1)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):

        timesteps = timesteps.to(self.sqrt_alphas_cumprod.device)
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        s1 = s1.to(x_start.device)
        s2 = s2.to(x_start.device)

        return s1 * x_start + s2 * x_noise

    def add_noise_at_t(self,
                       x_t: torch.Tensor,
                       x_noise: torch.Tensor,
                       timesteps_t: torch.Tensor,
                       timesteps_k: torch.Tensor) -> torch.Tensor:
        """add noise to the sample at time t

        Args:
            x_t (torch.Tensor): the sample at time t (batch_size, feature_size)
            x_noise (torch.Tensor): add noise to the sample (batch_size, feature_size)
            timesteps_t (torch.Tensor): the timesteps where x_t is at (batch_size,)
            timesteps_k (torch.Tensor): the timesteps to add noise (batch_size,)

        Returns:
            torch.Tensor: the noisy sample (batch_size, feature_size)
        """
        assert (timesteps_t <= timesteps_k).all(), "timesteps_t should be less than or equal to timesteps_k"
        device = x_t.device
        B, T = x_t.size(0), self.num_timesteps

        timesteps_t = timesteps_t.to(device)
        timesteps_k = timesteps_k.to(device)
        log_alphas = torch.log(self.alphas)
        # -> (T,)
        log_alphas_batched = log_alphas[None, :].expand(x_t.size(0), -1)
        log_alphas_batched = log_alphas_batched.to(device)
        # -> (batch_size, T)
        log_alphas_batched[torch.arange(T, device=device)[None, :] < timesteps_t[:, None]] = 0
        log_alphas_cumsum = torch.cumsum(log_alphas_batched, dim=-1)
        # -> (batch_size, T)
        alphas_cumprod = torch.exp(log_alphas_cumsum)
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        # -> (batch_size, T)
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5
        # -> (batch_size, T)

        s1 = sqrt_alphas_cumprod[torch.arange(B, device=device), timesteps_k]
        s2 = sqrt_one_minus_alphas_cumprod[torch.arange(B, device=device), timesteps_k]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        s1 = s1.to(device)
        s2 = s2.to(device)

        return s1 * x_t + s2 * x_noise

    def __len__(self):
        return self.num_timesteps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino")
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--mlp_type", type=str, default="mlp", choices=["energy", "mlp"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument("--time_embedding", type=str, default="learnable", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="identity", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    config = parser.parse_args()

    wandb.init(project="ddpm", name=config.experiment_name, config=config, entity='rainorangelemon')

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    model = {'energy': EnergyMLP, 'mlp': MLP}[config.mlp_type](
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding)

    model.to(device)
    ema_model = deepcopy(model)
    ema_model.eval()

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    global_step = 0
    frames = []
    losses = []
    print("Training model...")
    for epoch in range(config.num_epochs):
        dataset = datasets.get_dataset(config.dataset)
        dataloader = DataLoader(
            dataset, batch_size=config.train_batch_size, shuffle=True)
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            batch = batch[0]
            batch = batch.to(device)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long().to(device)

            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            # update ema model
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(0.999)
                ema_param.data.add_(0.001 * param.data)

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step, "max_predict": noise_pred.max().item()}
            wandb.log(logs)
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            # generate data with the model to later visualize the learning process
            model.eval()
            sample = torch.randn(config.eval_batch_size, 2).to(device)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long().to(device)
                with torch.no_grad():
                    residual = model(sample, t)
                sample = noise_scheduler.step(residual, t[0], sample)
            frames.append(sample.cpu().numpy())

    print("Saving model...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    print("Saving EMA model...")
    torch.save(ema_model.state_dict(), f"{outdir}/ema_model.pth")

    print("Saving images...")
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    frames = np.stack(frames)
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    for i, frame in enumerate(frames):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f"{imgdir}/{i:04}.png")
        plt.close()

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Saving frames...")
    np.save(f"{outdir}/frames.npy", frames)
