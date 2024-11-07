import os
import torch as th
from pathlib import Path
import numpy as np
import baselines_clevr
from tqdm.auto import tqdm
from torchvision.utils import save_image, make_grid
from scripts.best_of_n_on_sam_dataset import create_model_and_diffusion, CLEVRPosDataset, conditions_denoise_fn_factory
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../conf", config_name="clevr_pos")
def main(cfg: DictConfig):
    # Setup
    th.set_float32_matmul_precision('high')
    th.set_grad_enabled(False)
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')

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
    # model = th.compile(model, mode="max-autotune")


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


    wandb.init(project="rejection_sampling",
               config=OmegaConf.to_container(cfg, resolve=True),
               name=f"{cfg.experiment_name}",)

    NUM_SAMPLES_PER_TRIAL = 10

    packed_samples = th.zeros((5000, NUM_SAMPLES_PER_TRIAL, 3, 128, 128))
    packed_energies = th.zeros((5000, cfg.num_constraints, NUM_SAMPLES_PER_TRIAL))
    for test_idx in tqdm(range(2)):

        th.manual_seed(0)
        th.cuda.manual_seed(0)

        labels, _ = dataset[test_idx]
        conditions_denoise_fn = conditions_denoise_fn_factory(model, th.tensor(labels[np.newaxis], dtype=th.float32),
                                                            batch_size=cfg.elbo.mini_batch, cfg=cfg)
        filtered_samples, unfiltered_samples_over_time, need_to_remove_across_timesteps, energies_across_timesteps = \
        baselines_clevr.rejection_sampling_baseline(
                                    composed_denoise_fn=lambda x, t: composed_model_fn(x, t, th.from_numpy(labels).to(device)),
                                    unconditioned_denoise_fn=conditions_denoise_fn[-1],
                                    conditions_denoise_fn=conditions_denoise_fn[:-1],
                                    x_shape=(3, 128, 128),
                                    noise_scheduler=diffusion,
                                    num_samples_per_trial=NUM_SAMPLES_PER_TRIAL,
                                    rejection_scheduler_cfg=cfg.rejection_scheduler,
                                    elbo_cfg=cfg.elbo,
                                    progress=False,
                                    )
        packed_samples[test_idx, ...] = unfiltered_samples_over_time[-1]
        packed_energies[test_idx, ...] = th.from_numpy(np.array(list(energies_across_timesteps.values())))

        # save the figure
        if len(filtered_samples):
            # randomly pick one sample
            sample_at_final_t = filtered_samples[np.random.randint(len(filtered_samples))]
        else:
            # choose the one with minimum energy
            score = packed_energies[test_idx].sum(dim=-1)
            sample_at_final_t = unfiltered_samples_over_time[-1][score.argmin()]

        sample_at_final_t = (sample_at_final_t + 1) / 2
        grid = make_grid(sample_at_final_t, nrow=1, padding=0)
        save_image(grid, output_dir / f"sample_{test_idx:05d}.png")

        wandb.log({f"filtered_ratio": len(filtered_samples) / len(unfiltered_samples_over_time[-1]),
                })

    # Save the packed samples
    packed_samples = (packed_samples + 1) * 127.5
    # convert packed samples to uint8 to save space
    packed_samples = packed_samples.round().to(dtype=th.uint8)
    th.save(packed_samples, output_dir / "packed_samples.pt")
    th.save(packed_energies, output_dir / "packed_energies.pt")


if __name__ == '__main__':
    main()
