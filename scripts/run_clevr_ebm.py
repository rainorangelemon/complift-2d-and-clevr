import os
import torch as th
from pathlib import Path
import numpy as np
import baselines_clevr
from tqdm.auto import tqdm
from torchvision.utils import save_image, make_grid
from utils_clevr import CLEVRPosDataset
from ComposableDiff.composable_diffusion.model_creation import create_model_and_diffusion
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../conf", config_name="clevr_pos")
def main(cfg: DictConfig):
    # Setup
    th.set_float32_matmul_precision('high')
    th.set_grad_enabled(False)
    has_cuda = th.cuda.is_available()
    if has_cuda:
        # get the local rank
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = th.device(f'cuda:{local_rank}')
        th.cuda.set_device(device)
    else:
        device = th.device('cpu')

    options = OmegaConf.to_container(cfg.model, resolve=True)

    options["use_fp16"] = th.cuda.is_available()

    model, diffusion = create_model_and_diffusion(**options)

    model.eval()
    if options['use_fp16']:
        model.convert_to_fp16()
    model.to(device)

    print(f'Loading checkpoint from {cfg.ckpt_path}')
    checkpoint = th.load(cfg.ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint)

    print('Total base parameters', sum(x.numel() for x in model.parameters()))

    # Create output directory
    output_dir = Path(cfg.output_dir)
    experiment_name = cfg.data_path.split('/')[-1].split('.')[0]
    output_dir = output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the config
    config_save_path = os.path.join(cfg.output_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    dataset = CLEVRPosDataset(data_path=cfg.data_path)

    # optionally, compile the model for faster execution
    model = th.compile(model, mode="max-autotune")

    @th.compile(mode="max-autotune")
    def composed_model_fn(x_t, ts, labels, batch_size=cfg.elbo.mini_batch // cfg.num_constraints):
        num_samples = x_t.shape[0]

        assert len(labels.shape) == 2
        labels = th.cat([labels, th.zeros_like(labels[:1, :])], dim=0).to(device)
        masks = th.ones_like(labels[:, 0], dtype=th.bool).to(device)
        masks[-1] = False

        labels = labels.unsqueeze(0)
        masks = masks.unsqueeze(0)

        results_eps = []
        results_rest = []

        num_relations_per_sample = labels.shape[1]

        for i in range(0, num_samples, batch_size):
            # Create batch slices for current iteration
            x_t_batch = x_t[i:i+batch_size]
            ts_batch = ts[i:i+batch_size]

            current_batch_size = x_t_batch.shape[0]
            combined = th.repeat_interleave(x_t_batch, num_relations_per_sample, dim=0)
            ts_batch = th.repeat_interleave(ts_batch, num_relations_per_sample, dim=0)

            current_label = labels.expand(current_batch_size, -1, -1).to(device).flatten(0, 1)
            current_mask = masks.expand(current_batch_size, -1).to(device).flatten(0, 1)

            model_out = model(combined, ts_batch, y=current_label, masks=current_mask)
            eps, rest = model_out[:, :3], model_out[:, 3:]

            cond_eps, uncond_eps = eps[current_mask], eps[~current_mask]
            uncond_eps = uncond_eps.view(current_batch_size, -1, *uncond_eps.shape[1:])
            cond_eps = cond_eps.view(current_batch_size, -1, *cond_eps.shape[1:])

            eps_batch = uncond_eps + (cfg.cfg_weight * (cond_eps - uncond_eps)).sum(dim=1, keepdim=True)
            eps_batch = eps_batch.flatten(0, 1)
            rest_batch = rest[~current_mask]

            # Collect the results from this batch
            results_eps.append(eps_batch)
            results_rest.append(rest_batch)

        # Concatenate results from all batches
        eps = th.cat(results_eps, dim=0)
        rest = th.cat(results_rest, dim=0)

        return th.cat([eps, rest], dim=1)

    wandb.init(project="ebm_sampling",
               config=OmegaConf.to_container(cfg, resolve=True),
               name=f"{cfg.experiment_name}",)

    NUM_SAMPLES_TO_GENERATE = cfg.num_samples_to_generate

    for test_idx in tqdm(range(100)):
        th.manual_seed(0)
        th.cuda.manual_seed(0)

        labels, _ = dataset[test_idx]

        samples = baselines_clevr.ebm_baseline(
            composed_denoise_fn=lambda x, t: composed_model_fn(x, t, th.from_numpy(labels).to(device)),
            x_shape=(3, 128, 128),
            noise_scheduler=diffusion,
            num_samples_to_generate=NUM_SAMPLES_TO_GENERATE,
            ebm_cfg=cfg.ebm,
            progress=True,
        )

        # save the figure
        for i in range(len(samples[-1])):
            sample_at_final_t = samples[-1][i]
            sample_at_final_t = (sample_at_final_t + 1) / 2
            grid = make_grid(sample_at_final_t, nrow=1, padding=0)
            os.makedirs(output_dir / f"trial_{i:02d}", exist_ok=True)
            save_image(grid, output_dir / f"trial_{i:02d}/sample_{test_idx:05d}.png")


if __name__ == '__main__':
    main()
