# compare EBM across different timesteps

import ddpm
import torch
from baselines_2d import (
    ebm_baseline,
    diffusion_baseline,
    cache_rejection_baseline,
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
def create_metric_row(algebra, metrics_dict):
    # columns: algebra, method, metric1_value, metric2_value, ...
    sub_df = pd.DataFrame()
    sub_detailed_df = pd.DataFrame()
    sub_df = pd.concat([sub_df, pd.DataFrame([{
        "algebra": algebra,
        **{metric_name: np.mean([float(v) for seed in range(10) for v in metric_values[f"{algebra}", seed] if not is_nan_or_none(v)]) for metric_name, metric_values in metrics_dict.items()}
    }])], ignore_index=True)
    sub_detailed_df = pd.concat([sub_detailed_df, pd.DataFrame([{
        "algebra": algebra,
        **{metric_name: [float(v) for seed in range(10) for v in metric_values[f"{algebra}", seed] if not is_nan_or_none(v)] for metric_name, metric_values in metrics_dict.items()}
    }])], ignore_index=True)
    return sub_df, sub_detailed_df


def make_estimate_log_lift(elbo_cfg: dict[str, Union[bool, str]]) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Creates a function to estimate the log of lift

    Args:
        elbo_cfg (dict[str, Union[bool, str]]): Configuration for ELBO estimation, including number of samples, mini batch size, etc.

    Returns:
        Callable[[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]: Function to estimate the negative log probability.
    """
    def estimate_log_lift(cached_scores: torch.Tensor,
                          noise: torch.Tensor) -> torch.Tensor:
        if cached_scores.shape[0] == 0:
            return torch.zeros((0,), dtype=cached_scores.dtype, device=cached_scores.device)
        T, B, D = cached_scores.shape[0], cached_scores.shape[1], cached_scores.shape[2]
        log_px_given_c = torch.zeros(B, device=cached_scores.device)
        log_px = torch.zeros(B, device=cached_scores.device)
        mini_batch = 100
        for i in range(0, B, mini_batch):
            log_px_given_c[i:i+mini_batch] = -(cached_scores[:, i:i+mini_batch] - noise[:, i:i+mini_batch]).pow(2).mean(dim=(0, 2))
            log_px[i:i+mini_batch] = -(elbo_cfg["alpha"] * cached_scores[:, i:i+mini_batch] - noise[:, i:i+mini_batch]).pow(2).mean(dim=(0, 2))
        log_lift = log_px_given_c - log_px
        return log_lift
    return estimate_log_lift


def create_callback(elbo_cfg, algebras):

    estimate_log_lift = make_estimate_log_lift(elbo_cfg)
    def callback(cached_scores, noise):
        cached_scores = cached_scores.to(device='cuda')
        noise = noise.to(device='cuda')
        energies = [estimate_log_lift(cached_scores[i], noise) for i in range(len(cached_scores))]
        is_valid = torch.ones(energies[0].shape[0], dtype=torch.bool, device=cached_scores.device)
        for algebra_idx, algebra in enumerate(algebras):
                if algebra == "negation":
                    is_valid = is_valid & (energies[algebra_idx] <= 0)
                elif algebra == "product":
                    is_valid = is_valid & (energies[algebra_idx] > 0)
                elif algebra == "summation":
                    is_valid = is_valid | (energies[algebra_idx] > 0)

        return is_valid.cpu().numpy()

    return callback


@torch.no_grad()
@hydra.main(config_path="../conf", config_name="2d")
def main(cfg: DictConfig):
    # Initialize wandb
    wandb.init(project="2d_ablation_cache", name=f"synthetic data 2d with cache")
    wandb.run.log_code(".")

    # Initialize pandas dataframe
    df = pd.DataFrame()
    detailed_df = pd.DataFrame()

    if not os.path.exists("runs/ablation_2d_cache/"):
        os.makedirs("runs/ablation_2d_cache/", exist_ok=True)
    fig_dir = "runs/ablation_2d_cache/figures/"
    os.makedirs(fig_dir, exist_ok=True)

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

            # Load models and create cached composition model
            model_1.load_state_dict(torch.load(f"exps/{algebra}_{suffix}1/ema_model.pth", weights_only=True))
            model_2.load_state_dict(torch.load(f"exps/{algebra}_{suffix}2/ema_model.pth", weights_only=True))
            model_to_test = ddpm.CachedCompositionEnergyMLP(model_1, model_2, algebra=algebra, temperature_cfg=cfg.ebm.temperature)

            # Generate samples for each seed using simplified cache_rejection_baseline
            for seed in range(10):
                filtered_samples, acceptance_ratio = cache_rejection_baseline(
                    composed_denoise_fn=model_to_test,
                    algebras=["product", algebra],  # First condition is always product
                    x_shape=(2,),
                    noise_scheduler=ddpm.NoiseScheduler(num_timesteps=50),
                    num_samples_per_trial=8000,
                    elbo_cfg=cfg.elbo
                )
                generated_samples[seed] = filtered_samples
                acceptance_ratios[f"{algebra}", seed].append(acceptance_ratio)

            dataset_1 = generate_data_points(n=8000, dataset=f"{algebra}_{suffix}1")
            dataset_2 = generate_data_points(n=8000, dataset=f"{algebra}_{suffix}2")
            dataset_composed = generate_data_points(n=8000, dataset=environment)

            for seed in range(10):
                acc_method = get_accuracy(generated_samples[seed], environment, 8000)
                if np.isnan(acc_method):
                    acc_method = None
                accs[f"{algebra}", seed].append(acc_method)

            if len(dataset_composed):
                for seed in range(10):
                    if len(generated_samples[seed]) == 0:
                        chamfer_dist_method = None
                    else:
                        chamfer_dist_method = evaluate_chamfer_distance(generated_samples=generated_samples[seed], target_samples=dataset_composed)
                    chamfer_dists[f"{algebra}", seed].append(chamfer_dist_method)

                    if len(generated_samples[seed]) == 0:
                        NLL_method = None
                    else:
                        NLL_method = model_to_test.energy(torch.tensor(generated_samples[seed], dtype=torch.float32, device='cuda'),
                                                      t=torch.zeros((len(generated_samples[seed]),), dtype=torch.long, device='cuda')).cpu().numpy().mean()
                    NLLs[f"{algebra}", seed].append(NLL_method)

            else:
                for seed in range(10):
                    chamfer_dists[f"{algebra}", seed].append(None)
                    NLLs[f"{algebra}", seed].append(None)

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
        metrics = {
            "acc": accs,
            "chamfer_dist": chamfer_dists,
            "NLL": NLLs,
            "acceptance_ratio": acceptance_ratios
        }

        # Combine all metrics into a single dictionary
        log_table, log_detailed_table = create_metric_row(algebra, metrics)
        wandb.log({f"{algebra}": wandb.Table(dataframe=log_table)})
        wandb.log({f"{algebra}_detailed": wandb.Table(dataframe=log_detailed_table)})
        df = pd.concat([df, log_table], ignore_index=True)
        detailed_df = pd.concat([detailed_df, log_detailed_table], ignore_index=True)

    df.to_csv("runs/ablation_2d_cache/overview.csv", index=False)
    detailed_df.to_csv("runs/ablation_2d_cache/overview_detailed.csv", index=False)
    wandb.log({"overview": wandb.Table(dataframe=df)})
    wandb.log({"overview_detailed": wandb.Table(dataframe=detailed_df)})

if __name__ == "__main__":
    main()
