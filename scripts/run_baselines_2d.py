# compare EBM across different timesteps

import ddpm
import torch
from baselines_2d import (
    ebm_baseline,
    diffusion_baseline,
    rejection_baseline,
    evaluate_chamfer_distance,
)
from datasets import generate_data_points, get_accuracy
from utils import catchtime, plot_points, plot_two_intervals, plot_acceptance_ratios
import wandb
import numpy as np
import hydra
from omegaconf import DictConfig
from collections import defaultdict
import pandas as pd

# Helper function to create multiple rows in a dataframe
def create_metric_row(algebra, metrics_dict, methods):
    # columns: algebra, method, metric1_value, metric2_value, ...
    sub_df = pd.DataFrame()
    for method in methods:
        sub_df = pd.concat([sub_df, pd.DataFrame([{
            "algebra": algebra,
            "method": method,
            **{metric_name: np.mean(metric_values[f"{algebra}", method]) for metric_name, metric_values in metrics_dict.items()}
        }])], ignore_index=True)
    return sub_df


@torch.no_grad()
@hydra.main(config_path="../conf", config_name="2d")
def main(cfg: DictConfig):
    # Initialize wandb
    wandb.init(project="r_and_r", name="synthetic data 2d")
    wandb.run.log_code(".")

    # Initialize pandas dataframe
    df = pd.DataFrame()

    # Load the models
    model_1 = ddpm.EnergyMLP()
    model_2 = ddpm.EnergyMLP()
    for algebra in ['product', 'summation', 'negation']:
        accs = defaultdict(list)
        chamfer_dists = defaultdict(list)
        NLLs = defaultdict(list)

        for suffix in ['a', 'b', 'c']:
            generated_samples = dict()

            environment = algebra + '_' + suffix + '3'
            model_1.load_state_dict(torch.load(f"exps/{algebra}_{suffix}1/ema_model.pth", weights_only=True))
            model_2.load_state_dict(torch.load(f"exps/{algebra}_{suffix}2/ema_model.pth", weights_only=True))
            model_to_test = ddpm.CompositionEnergyMLP(model_1, model_2, algebra=algebra)
            with catchtime('diffusion'):
                generated_samples_diffusion = diffusion_baseline(model_to_test, diffusion=ddpm.NoiseScheduler(num_timesteps=50))
                generated_samples_diffusion = generated_samples_diffusion[-1]
                generated_samples['diffusion'] = generated_samples_diffusion

            with catchtime('rejection'):
                generated_samples_rejection, _, acceptance_ratio, _ = rejection_baseline(composed_denoise_fn=model_to_test,
                                                                                         conditions_denoise_fn=[model_1, model_2],
                                                                                         # the first algebra needs to be product, see implementation of rejection_baseline
                                                                                         algebras=["product", algebra],
                                                                                         x_shape=(2,),
                                                                                         noise_scheduler=ddpm.NoiseScheduler(num_timesteps=50),
                                                                                         num_samples_per_trial=8000,
                                                                                         elbo_cfg=cfg.elbo,
                                                                                         progress=True)
                generated_samples_rejection = generated_samples_rejection[-1]
                generated_samples['rejection'] = generated_samples_rejection

            for sampler_type in ["ULA", "UHA", "MALA", "MUHA"]:
                with catchtime(f'ebm_{sampler_type}'):
                    generated_samples_ebm = ebm_baseline(algebra=algebra,
                                                         suffix1=suffix + '1',
                                                         suffix2=suffix + '2',
                                                         eval_batch_size=8000,
                                                         sampler_type=sampler_type)
                    generated_samples[f'ebm_{sampler_type}'] = generated_samples_ebm

            dataset_1 = generate_data_points(n=8000, dataset=f"{algebra}_{suffix}1")
            dataset_2 = generate_data_points(n=8000, dataset=f"{algebra}_{suffix}2")
            dataset_composed = generate_data_points(n=8000, dataset=environment)

            for method in generated_samples:
                acc_method = get_accuracy(generated_samples[method], environment, 8000)
                accs[f"{algebra}", method].append(acc_method)

            if len(dataset_composed):
                for method in generated_samples:
                    chamfer_dist_method = evaluate_chamfer_distance(generated_samples=generated_samples[method], target_samples=dataset_composed)
                    chamfer_dists[f"{algebra}", method].append(chamfer_dist_method)

                for method in generated_samples:
                    NLL_method = model_to_test.energy(torch.tensor(generated_samples[method], dtype=torch.float32, device='cuda'),
                                                    t=torch.zeros((len(generated_samples[method]),), dtype=torch.long, device='cuda')).cpu().numpy().mean()
                    NLLs[f"{algebra}", method].append(NLL_method)

                wandb.log({**{
                    f"{environment}/sample_1_gt": wandb.Image(plot_points(dataset_1)),
                    f"{environment}/sample_2_gt": wandb.Image(plot_points(dataset_2)),
                    f"{environment}/sample_composed_gt": wandb.Image(plot_points(dataset_composed),
                                                                    caption=f"target samples at {environment}"),},
                    **{f"{environment}/baseline_{method}": wandb.Image(plot_points(generated_samples[method]),
                                                                    caption=f"{method} samples at {environment}, chamfer distance: {chamfer_dists[f'{algebra}', method][-1]:0.4f}, NLL: {NLLs[f'{algebra}', method][-1]:0.4f}, accuracy: {accs[f'{algebra}', method][-1]:0.4f}")
                                                           for method in generated_samples},
                    f"{environment}/acceptance_ratio": acceptance_ratio,
                })

            else:

                # Log the target samples
                wandb.log({**{
                    f"{environment}/sample_1_gt": wandb.Image(plot_points(dataset_1)),
                    f"{environment}/sample_2_gt": wandb.Image(plot_points(dataset_2)),
                    f"{environment}/sample_composed_gt": wandb.Image(plot_points(dataset_composed),
                                                                    caption=f"target samples at {environment}"),},
                    **{f"{environment}/baseline_{method}": wandb.Image(plot_points(generated_samples[method]),
                                                                    caption=f"{method} samples at {environment}, #samples: {len(generated_samples[method])}, accuracy: {accs[f'{algebra}', method][-1]:0.4f}")
                                                                    for method in generated_samples},
                    f"{environment}/acceptance_ratio": acceptance_ratio,
                })

        # Define methods and metrics
        methods = generated_samples.keys()
        metrics = {
            "acc": accs,
            "chamfer_dist": chamfer_dists,
            "NLL": NLLs
        }

        # Combine all metrics into a single dictionary
        log_table = create_metric_row(algebra, metrics, methods)
        wandb.log({f"{algebra}": wandb.Table(dataframe=log_table)})
        df = pd.concat([df, log_table], ignore_index=True)

    df.to_csv("exps/baselines_2d.csv", index=False)
    wandb.log({"overview": wandb.Table(dataframe=df)})

if __name__ == "__main__":
    main()
