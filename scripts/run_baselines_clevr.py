import os
import torch as th
from pathlib import Path
import numpy as np
import baselines_clevr
from tqdm.auto import tqdm
from torchvision.utils import save_image, make_grid
from utils_clevr import conditions_denoise_fn_factory, CLEVRPosDataset
from ComposableDiff.composable_diffusion.model_creation import create_model_and_diffusion
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Callable, Dict, Any, List, Optional, Union
import enum


class Method(str, enum.Enum):
    REJECTION = "rejection"
    CACHE_REJECTION = "cache_rejection"
    EBM = "ebm"


class CacheCallback:
    def __init__(self):
        self.cached_scores = []
        self.cached_xs = []
        self.cached_ts = []

    def __call__(self):
        return {"cached_scores": self.cached_scores, "cached_xs": self.cached_xs, "cached_ts": self.cached_ts}


def create_composed_model_fn(model: th.nn.Module, device: th.device, cfg: DictConfig,
                           cache_callback: Optional[CacheCallback] = None) -> Callable:
    """Creates the composed model function based on the method being used."""

    def composed_model_fn(x_t: th.Tensor, ts: th.Tensor, labels: th.Tensor,
                         batch_size: int = cfg.elbo.mini_batch // cfg.num_constraints) -> th.Tensor:
        num_samples = x_t.shape[0]

        assert len(labels.shape) == 2
        labels = th.cat([labels, th.zeros_like(labels[:1, :])], dim=0).to(device)
        masks = th.ones_like(labels[:, 0], dtype=th.bool).to(device)
        masks[-1] = False

        if cache_callback is not None:
            cache_callback.cached_scores = th.zeros((len(labels), *x_t.shape), device=x_t.device)
            cache_callback.cached_xs = x_t.clone().detach()
            cache_callback.cached_ts = ts.clone().detach()

        labels = labels.unsqueeze(0)
        masks = masks.unsqueeze(0)

        results_eps = []
        results_rest = []

        num_relations_per_sample = labels.shape[1]

        for i in range(0, num_samples, batch_size):
            x_t_batch = x_t[i:i+batch_size]
            ts_batch = ts[i:i+batch_size]

            current_batch_size = x_t_batch.shape[0]
            combined = th.repeat_interleave(x_t_batch, num_relations_per_sample, dim=0)
            ts_batch = th.repeat_interleave(ts_batch, num_relations_per_sample, dim=0)

            current_label = labels.expand(current_batch_size, -1, -1).to(device).flatten(0, 1)
            current_mask = masks.expand(current_batch_size, -1).to(device).flatten(0, 1)

            model_out = model(combined, ts_batch, y=current_label, masks=current_mask)
            eps, rest = model_out[:, :3], model_out[:, 3:]

            if cache_callback is not None:
                cache_callback.cached_scores[:, i:i+batch_size] = eps.view(current_batch_size, -1, *eps.shape[1:]).transpose(0, 1)

            cond_eps, uncond_eps = eps[current_mask], eps[~current_mask]
            uncond_eps = uncond_eps.view(current_batch_size, -1, *uncond_eps.shape[1:])
            cond_eps = cond_eps.view(current_batch_size, -1, *cond_eps.shape[1:])

            eps_batch = uncond_eps + (cfg.cfg_weight * (cond_eps - uncond_eps)).sum(dim=1, keepdim=True)
            eps_batch = eps_batch.flatten(0, 1)
            rest_batch = rest[~current_mask]

            results_eps.append(eps_batch)
            results_rest.append(rest_batch)

        eps = th.cat(results_eps, dim=0)
        rest = th.cat(results_rest, dim=0)

        return th.cat([eps, rest], dim=1)

    return composed_model_fn


def run_rejection(model: th.nn.Module, diffusion: Any, dataset: CLEVRPosDataset,
                 device: th.device, cfg: DictConfig, output_dir: Path) -> None:
    """Run the complift method."""
    packed_samples = th.zeros((len(dataset), cfg.num_samples_to_generate, 3, 128, 128))
    packed_energies = th.zeros((len(dataset), cfg.num_constraints, cfg.num_samples_to_generate))

    for test_idx in tqdm(range(len(dataset))):
        th.manual_seed(0)
        th.cuda.manual_seed(0)

        labels, _ = dataset[test_idx]
        conditions_denoise_fn = conditions_denoise_fn_factory(
            model, th.tensor(labels[np.newaxis], dtype=th.float32),
            batch_size=cfg.elbo.mini_batch, cfg=cfg
        )

        composed_fn = create_composed_model_fn(model, device, cfg)
        filtered_samples, unfiltered_samples_over_time, need_to_remove_across_timesteps, energies_across_timesteps = \
            baselines_clevr.rejection_baseline(
                composed_denoise_fn=lambda x, t: composed_fn(x, t, th.from_numpy(labels).to(device)),
                unconditioned_denoise_fn=conditions_denoise_fn[-1],
                conditions_denoise_fn=conditions_denoise_fn[:-1],
                x_shape=(3, 128, 128),
                noise_scheduler=diffusion,
                num_samples_to_generate=cfg.num_samples_to_generate,
                rejection_scheduler_cfg=cfg.rejection_scheduler,
                elbo_cfg=cfg.elbo,
                progress=False,
            )

        packed_samples[test_idx, ...] = unfiltered_samples_over_time[-1]
        packed_energies[test_idx, ...] = th.from_numpy(np.array(list(energies_across_timesteps.values())))

        if len(filtered_samples):
            sample_at_final_t = filtered_samples[np.random.randint(len(filtered_samples))]
        else:
            score = packed_energies[test_idx].min(dim=0).values
            sample_at_final_t = unfiltered_samples_over_time[-1][score.argmax()]

        sample_at_final_t = (sample_at_final_t + 1) / 2
        grid = make_grid(sample_at_final_t, nrow=1, padding=0)
        save_image(grid, output_dir / f"sample_{test_idx:05d}.png")

        wandb.log({"filtered_ratio": len(filtered_samples) / len(unfiltered_samples_over_time[-1])})

    save_packed_results(packed_samples, packed_energies, output_dir)


def run_cache_rejection(model: th.nn.Module, diffusion: Any, dataset: CLEVRPosDataset,
                       device: th.device, cfg: DictConfig, output_dir: Path) -> None:
    """Run the cache complift sampling method."""
    packed_samples = th.zeros((len(dataset), cfg.num_samples_to_generate, 3, 128, 128))
    packed_energies = th.zeros((len(dataset), cfg.num_constraints, cfg.num_samples_to_generate))

    for test_idx in tqdm(range(len(dataset))):
        th.manual_seed(0)
        th.cuda.manual_seed(0)

        labels, _ = dataset[test_idx]
        conditions_denoise_fn = conditions_denoise_fn_factory(
            model, th.tensor(labels[np.newaxis], dtype=th.float32),
            batch_size=cfg.elbo.mini_batch, cfg=cfg
        )

        cache_callback = CacheCallback()
        composed_fn = create_composed_model_fn(model, device, cfg, cache_callback)
        filtered_samples, final_samples, is_valid, energies = \
            baselines_clevr.cache_rejection_baseline(
                composed_denoise_fn=lambda x, t: composed_fn(x, t, th.from_numpy(labels).to(device)),
                unconditioned_denoise_fn=conditions_denoise_fn[-1],
                conditions_denoise_fn=conditions_denoise_fn[:-1],
                cache_callback=cache_callback,
                x_shape=(3, 128, 128),
                noise_scheduler=diffusion,
                num_samples_to_generate=cfg.num_samples_to_generate,
                elbo_cfg=cfg.elbo,
                progress=False,
            )

        packed_samples[test_idx, ...] = final_samples
        packed_energies[test_idx, ...] = energies

        if len(filtered_samples):
            sample_at_final_t = filtered_samples[np.random.randint(len(filtered_samples))]
        else:
            score = packed_energies[test_idx].min(dim=0).values
            sample_at_final_t = final_samples[score.argmax()]

        sample_at_final_t = (sample_at_final_t + 1) / 2
        grid = make_grid(sample_at_final_t, nrow=1, padding=0)
        save_image(grid, output_dir / f"sample_{test_idx:05d}.png")

        wandb.log({"filtered_ratio": len(filtered_samples) / len(final_samples)})

    save_packed_results(packed_samples, packed_energies, output_dir)


def run_ebm(model: th.nn.Module, diffusion: Any, dataset: CLEVRPosDataset,
            device: th.device, cfg: DictConfig, output_dir: Path) -> None:
    """Run the EBM sampling method."""
    for test_idx in tqdm(range(len(dataset))):
        th.manual_seed(0)
        th.cuda.manual_seed(0)

        labels, _ = dataset[test_idx]
        composed_fn = create_composed_model_fn(model, device, cfg)
        samples = baselines_clevr.ebm_baseline(
            composed_denoise_fn=lambda x, t: composed_fn(x, t, th.from_numpy(labels).to(device)),
            x_shape=(3, 128, 128),
            noise_scheduler=diffusion,
            num_samples_to_generate=cfg.num_samples_to_generate,
            ebm_cfg=cfg.ebm,
            progress=True,
        )

        for i in range(len(samples[-1])):
            sample_at_final_t = samples[-1][i]
            sample_at_final_t = (sample_at_final_t + 1) / 2
            grid = make_grid(sample_at_final_t, nrow=1, padding=0)
            os.makedirs(output_dir / f"trial_{i:02d}", exist_ok=True)
            save_image(grid, output_dir / f"trial_{i:02d}/sample_{test_idx:05d}.png")


def save_packed_results(packed_samples: th.Tensor, packed_energies: th.Tensor, output_dir: Path) -> None:
    """Save the packed samples and energies to disk."""
    packed_samples = (packed_samples + 1) * 127.5
    packed_samples = packed_samples.round().to(dtype=th.uint8)
    th.save(packed_samples, output_dir / "packed_samples.pt")
    th.save(packed_energies, output_dir / "packed_energies.pt")


@hydra.main(config_path="../conf", config_name="clevr_pos")
def main(cfg: DictConfig) -> None:
    # Setup
    th.set_grad_enabled(False)
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')

    # Model setup
    options = OmegaConf.to_container(cfg.model, resolve=True)
    options["use_fp16"] = th.cuda.is_available()
    model, diffusion = create_model_and_diffusion(**options)

    model.eval()
    if options['use_fp16']:
        model.convert_to_fp16()
    model.to(device)

    print(f'Loading checkpoint from {cfg.ckpt_path}')
    checkpoint = th.load(cfg.ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    # optionally, compile the model for faster execution
    # model = th.compile(model, mode="max-autotune")

    print('Total base parameters', sum(x.numel() for x in model.parameters()))

    # Setup output directory
    output_dir = Path(cfg.output_dir)
    experiment_name = cfg.data_path.split('/')[-1].split('.')[0]
    output_dir = output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_save_path = os.path.join(cfg.output_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    dataset = CLEVRPosDataset(data_path=cfg.data_path)

    # Initialize wandb
    project_name = f"{cfg.method}_sampling" if hasattr(cfg, 'method') else "sampling"
    wandb.init(
        project=project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"{cfg.experiment_name}",
    )

    # Run the selected method
    method = Method(cfg.method.lower())
    if method == Method.REJECTION:
        run_rejection(model, diffusion, dataset, device, cfg, output_dir)
    elif method == Method.CACHE_REJECTION:
        run_cache_rejection(model, diffusion, dataset, device, cfg, output_dir)
    elif method == Method.EBM:
        run_ebm(model, diffusion, dataset, device, cfg, output_dir)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")


if __name__ == '__main__':
    main()
