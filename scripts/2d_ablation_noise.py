# compare EBM across different timesteps

import ddpm
import torch
from baselines_2d import (
    ebm_baseline,
    diffusion_baseline,
    calculate_elbo
)
from datasets import generate_data_points, get_accuracy
from utils import catchtime, evaluate_chamfer_distance, plot_points
import wandb
import numpy as np
import hydra
from omegaconf import DictConfig
from collections import defaultdict
import pandas as pd
import os
from copy import deepcopy
from typing import Callable, Union


def is_nan_or_none(x):
    return x is None or np.isnan(x)


# Helper function to create multiple rows in a dataframe
def create_metric_row(algebra, metrics_dict, methods):
    # columns: algebra, method, metric1_value, metric2_value, ...
    sub_df = pd.DataFrame()
    sub_detailed_df = pd.DataFrame()
    for n_samples, same_noise in methods:
        sub_df = pd.concat([sub_df, pd.DataFrame([{
            "algebra": algebra,
            "n_samples": n_samples,
            "same_noise": same_noise,
            **{metric_name: np.mean([float(v) for v in metric_values[f"{algebra}", n_samples, same_noise] if not is_nan_or_none(v)]) for metric_name, metric_values in metrics_dict.items()}
        }])], ignore_index=True)
        sub_detailed_df = pd.concat([sub_detailed_df, pd.DataFrame([{
            "algebra": algebra,
            "n_samples": n_samples,
            "same_noise": same_noise,
            **{metric_name: [float(v) for v in metric_values[f"{algebra}", n_samples, same_noise] if not is_nan_or_none(v)] for metric_name, metric_values in metrics_dict.items()}
        }])], ignore_index=True)
    return sub_df, sub_detailed_df


def make_estimate_log_lift(elbo_cfg: dict[str, Union[bool, str]],
                           noise_scheduler: ddpm.NoiseScheduler,
                           seed: int = 0,
                           progress: bool=False) -> Callable[[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    """Creates a function to estimate the log of lift

    Args:
        elbo_cfg (dict[str, Union[bool, str]]): Configuration for ELBO estimation, including number of samples, mini batch size, etc.
        noise_scheduler (ddpm.NoiseScheduler): Noise scheduler.
        progress (bool, optional): Whether to show the progress bar. Defaults to False.

    Returns:
        Callable[[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]: Function to estimate the negative log probability.
    """
    def estimate_log_lift(denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            x: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.zeros((0,), dtype=x.dtype, device=x.device)
        assert len(t.unique()) == 1, "t should be the same for all samples, but got {}".format(t.unique())
        log_px_given_c = calculate_elbo(denoise_fn,
                                        noise_scheduler,
                                        x_t=x,
                                        t=t[0],
                                        seed=seed,
                                        mini_batch=elbo_cfg["mini_batch"],
                                        n_samples=elbo_cfg["n_samples"],
                                        same_noise=elbo_cfg["same_noise"],
                                        sample_timesteps=elbo_cfg["sample_timesteps"],
                                        progress=progress)
        log_px = calculate_elbo(lambda x, t: elbo_cfg["alpha"] * denoise_fn(x, t),
                                noise_scheduler,
                                x_t=x,
                                t=t[0],
                                seed=seed if elbo_cfg["same_noise"] != "independent" else seed+42,
                                mini_batch=elbo_cfg["mini_batch"],
                                n_samples=elbo_cfg["n_samples"],
                                same_noise=elbo_cfg["same_noise"],
                                sample_timesteps=elbo_cfg["sample_timesteps"],
                                progress=progress)
        log_lift = log_px_given_c - log_px
        return log_lift
    return estimate_log_lift


def create_callback(elbo_cfg, noise_scheduler, algebras, conditions_denoise_fn, seed: int = 0):

    if elbo_cfg.same_noise == "share_all_trial":
        cfg = deepcopy(elbo_cfg)
        cfg.same_noise = True
        estimate_log_lift = make_estimate_log_lift(cfg, noise_scheduler, seed)
    elif elbo_cfg.same_noise == "share_each_trial":
        cfg = deepcopy(elbo_cfg)
        cfg.same_noise = False
        estimate_log_lift = make_estimate_log_lift(cfg, noise_scheduler, seed)
    elif elbo_cfg.same_noise == "independent":
        estimate_log_lift = make_estimate_log_lift(elbo_cfg, noise_scheduler, seed)

    def callback(x):
        x = torch.from_numpy(x).to(device='cuda')
        t = torch.zeros(len(x), dtype=torch.long, device=x.device)
        energies = [estimate_log_lift(denoise_fn, x, t) for denoise_fn in conditions_denoise_fn]
        is_valid = torch.ones(len(x), dtype=torch.bool, device=x.device)
        for algebra_idx, algebra in enumerate(algebras):
                if algebra == "negation":
                    is_valid = is_valid & (energies[algebra_idx] <= 0)
                elif algebra == "product":
                    is_valid = is_valid & (energies[algebra_idx] > 0)
                elif algebra == "summation":
                    is_valid = is_valid | (energies[algebra_idx] > 0)

        filtered_x = x[is_valid]
        return filtered_x.cpu().numpy()

    return callback


@torch.no_grad()
@hydra.main(config_path="../conf", config_name="2d")
def main(cfg: DictConfig):
    # Initialize wandb
    wandb.init(project="2d_ablation", name=f"synthetic data 2d {cfg.ebm.temperature.product},{cfg.ebm.temperature.mixture},{cfg.ebm.temperature.negation}")
    wandb.run.log_code(".")

    # Initialize pandas dataframe
    df = pd.DataFrame()
    detailed_df = pd.DataFrame()

    if not os.path.exists("runs/ablation_2d/"):
        os.makedirs("runs/ablation_2d/", exist_ok=True)
    fig_dir = "runs/ablation_2d/figures/"
    os.makedirs(fig_dir, exist_ok=True)

    base_cfg = cfg.copy()
    cfg_list = []
    for t in [50, 100, 200, 500, 1000]:
        for same_noise in ["share_all_trial", "share_each_trial", "independent"]:
            cfg = deepcopy(base_cfg)
            cfg.elbo.n_samples = t
            cfg.elbo.same_noise = same_noise
            cfg_list.append(cfg)

    # Load the models
    model_1 = ddpm.EnergyMLP()
    model_2 = ddpm.EnergyMLP()
    for algebra in ['product', 'summation', 'negation']:
        accs = defaultdict(list)
        chamfer_dists = defaultdict(list)
        NLLs = defaultdict(list)
        acceptance_ratios = defaultdict(list)

        for suffix in ['a', 'b', 'c']:
            generated_samples = dict()

            environment = algebra + '_' + suffix + '3'
            model_1.load_state_dict(torch.load(f"exps/{algebra}_{suffix}1/ema_model.pth", weights_only=True))
            model_2.load_state_dict(torch.load(f"exps/{algebra}_{suffix}2/ema_model.pth", weights_only=True))
            model_to_test = ddpm.CompositionEnergyMLP(model_1, model_2, algebra=algebra, temperature_cfg=cfg.ebm.temperature)
            with catchtime('diffusion'):
                generated_samples_diffusion = diffusion_baseline(model_to_test, diffusion=ddpm.NoiseScheduler(num_timesteps=50))
                generated_samples_diffusion = generated_samples_diffusion[-1]

            for cfg in cfg_list:
                with catchtime('rejection'):
                    noise_scheduler = ddpm.NoiseScheduler(num_timesteps=50)
                    for seed in range(10):
                        callback = create_callback(cfg.elbo, noise_scheduler, ["product", algebra], [model_1, model_2], seed)
                        generated_samples[cfg.elbo.n_samples, cfg.elbo.same_noise, seed] = callback(generated_samples_diffusion)

            dataset_1 = generate_data_points(n=8000, dataset=f"{algebra}_{suffix}1")
            dataset_2 = generate_data_points(n=8000, dataset=f"{algebra}_{suffix}2")
            dataset_composed = generate_data_points(n=8000, dataset=environment)

            for cfg in cfg_list:
                n_samples = cfg.elbo.n_samples
                same_noise = cfg.elbo.same_noise
                for seed in range(10):
                    acc_method = get_accuracy(generated_samples[n_samples, same_noise, seed], environment, 8000)
                    if np.isnan(acc_method):
                        acc_method = None
                    accs[f"{algebra}", n_samples, same_noise].append(acc_method)
                    acceptance_ratio = len(generated_samples[n_samples, same_noise, seed]) / 8000
                    if np.isnan(acceptance_ratio):
                        acceptance_ratio = None
                    acceptance_ratios[f"{algebra}", n_samples, same_noise].append(acceptance_ratio)

            if len(dataset_composed):
                for cfg in cfg_list:
                    n_samples = cfg.elbo.n_samples
                    same_noise = cfg.elbo.same_noise
                    for seed in range(10):
                        if len(generated_samples[n_samples, same_noise, seed]) == 0:
                            chamfer_dist_method = None
                        else:
                            chamfer_dist_method = evaluate_chamfer_distance(generated_samples=generated_samples[n_samples, same_noise, seed], target_samples=dataset_composed)
                        chamfer_dists[f"{algebra}", n_samples, same_noise].append(chamfer_dist_method)

                for cfg in cfg_list:
                    n_samples = cfg.elbo.n_samples
                    same_noise = cfg.elbo.same_noise
                    for seed in range(10):
                        if len(generated_samples[n_samples, same_noise, seed]) == 0:
                            NLL_method = None
                        else:
                            NLL_method = model_to_test.energy(torch.tensor(generated_samples[n_samples, same_noise, seed], dtype=torch.float32, device='cuda'),
                                                      t=torch.zeros((len(generated_samples[n_samples, same_noise, seed]),), dtype=torch.long, device='cuda')).cpu().numpy().mean()
                        NLLs[f"{algebra}", n_samples, same_noise].append(NLL_method)

            else:
                for cfg in cfg_list:
                    n_samples = cfg.elbo.n_samples
                    same_noise = cfg.elbo.same_noise
                    for seed in range(10):
                        chamfer_dists[f"{algebra}", n_samples, same_noise].append(None)
                        NLLs[f"{algebra}", n_samples, same_noise].append(None)

            # wandb.log({**{
            #     f"{environment}/sample_1_gt": wandb.Image(plot_points(dataset_1, filepath=fig_dir + f"{environment}_sample_1_gt.png")),
            #     f"{environment}/sample_2_gt": wandb.Image(plot_points(dataset_2, filepath=fig_dir + f"{environment}_sample_2_gt.png")),
            #     f"{environment}/sample_composed_gt": wandb.Image(plot_points(dataset_composed, filepath=fig_dir + f"{environment}_sample_composed_gt.png"),
            #                                                     caption=f"target samples at {environment}"),},
            #     **{f"{environment}/baseline_{method}": wandb.Image(plot_points(generated_samples[method], filepath=fig_dir + f"{environment}_baseline_{method}.png"),
            #                                                     caption = f"{method} samples at {environment}, " + \
            #                                                               f"#samples: {len(generated_samples[method])}, " + \
            #                                                               (f"chamfer distance: {chamfer_dists[f'{algebra}', method][-1]:0.4f}, " if chamfer_dists[f'{algebra}', method][-1] is not None else 'N/A, ') + \
            #                                                               (f"NLL: {NLLs[f'{algebra}', method][-1]:0.4f}, " if NLLs[f'{algebra}', method][-1] is not None else 'N/A, ') + \
            #                                                               (f"accuracy: {accs[f'{algebra}', method][-1]:0.4f}" if accs[f'{algebra}', method][-1] is not None else 'N/A'))
            #                                             for method in generated_samples},
            #     f"{environment}/acceptance_ratio": acceptance_ratios[f"{algebra}", 'rejection'][-1],
            # })

        # Define methods and metrics
        methods = set([(n_samples, same_noise) for n_samples, same_noise, _ in generated_samples.keys()])
        metrics = {
            "acc": accs,
            "chamfer_dist": chamfer_dists,
            "NLL": NLLs,
            "acceptance_ratio": acceptance_ratios
        }

        # Combine all metrics into a single dictionary
        log_table, log_detailed_table = create_metric_row(algebra, metrics, methods)
        wandb.log({f"{algebra}": wandb.Table(dataframe=log_table)})
        wandb.log({f"{algebra}_detailed": wandb.Table(dataframe=log_detailed_table)})
        df = pd.concat([df, log_table], ignore_index=True)
        detailed_df = pd.concat([detailed_df, log_detailed_table], ignore_index=True)

    df.to_csv("runs/ablation_2d/overview.csv", index=False)
    detailed_df.to_csv("runs/ablation_2d/overview_detailed.csv", index=False)
    wandb.log({"overview": wandb.Table(dataframe=df)})
    wandb.log({"overview_detailed": wandb.Table(dataframe=detailed_df)})

if __name__ == "__main__":
    main()
