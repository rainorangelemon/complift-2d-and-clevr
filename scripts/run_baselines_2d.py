# compare EBM across different timesteps

import ddpm
import torch
from baselines_2d import (
    ebm_baseline,
    diffusion_baseline,
    evaluate_W2,
    evaluate_chamfer_distance,
    intermediate_distribution,
    diffusion_baseline,
    rejection_baseline,
)
from datasets import generate_data_points, get_accuracy
from utils import catchtime, plot_points, plot_two_intervals, plot_acceptance_ratios
import wandb
import numpy as np
import hydra
from omegaconf import DictConfig


@torch.no_grad()
@hydra.main(config_path="../conf", config_name="2d")
def main(cfg: DictConfig):
    # Initialize wandb
    wandb.init(project="r_and_r", name="synthetic data 2d")
    wandb.run.log_code(".")

    # Load the models
    model_1 = ddpm.EnergyMLP()
    model_2 = ddpm.EnergyMLP()
    for algebra in ['product', 'summation', 'negation']:
        accs_diffusion = []
        accs_rejection = []
        chamfer_dists_diffusion = []
        chamfer_dists_rejection = []
        NLLs_diffusion = []
        NLLs_rejection = []

        for suffix in ['a', 'b', 'c']:
            environment = algebra + '_' + suffix + '3'
            model_1.load_state_dict(torch.load(f"exps/{algebra}_{suffix}1/ema_model.pth", weights_only=True))
            model_2.load_state_dict(torch.load(f"exps/{algebra}_{suffix}2/ema_model.pth", weights_only=True))
            model_to_test = ddpm.CompositionEnergyMLP(model_1, model_2, algebra=algebra)
            with catchtime('diffusion'):
                generated_samples_diffusion = diffusion_baseline(model_to_test, diffusion=ddpm.NoiseScheduler(num_timesteps=50))
                generated_samples_diffusion = generated_samples_diffusion[-1]
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

            dataset_1 = generate_data_points(n=8000, dataset=f"{algebra}_{suffix}1")
            dataset_2 = generate_data_points(n=8000, dataset=f"{algebra}_{suffix}2")
            dataset_composed = generate_data_points(n=8000, dataset=environment)

            acc_diffusion = get_accuracy(generated_samples_diffusion, environment, 8000)
            acc_rejection = get_accuracy(generated_samples_rejection, environment, 8000)

            accs_diffusion.append(acc_diffusion)
            accs_rejection.append(acc_rejection)

            if len(dataset_composed):
                chamfer_dist_diffusion = evaluate_chamfer_distance(generated_samples=generated_samples_diffusion, target_samples=dataset_composed)
                chamfer_dist_rejection = evaluate_chamfer_distance(generated_samples=generated_samples_rejection, target_samples=dataset_composed)
                NLL_diffusion = model_to_test.energy(torch.tensor(generated_samples_diffusion, dtype=torch.float32, device='cuda'),
                                                    t=torch.zeros((len(generated_samples_diffusion),), dtype=torch.long, device='cuda')).cpu().numpy().mean()
                NLL_rejection = model_to_test.energy(torch.tensor(generated_samples_rejection, dtype=torch.float32, device='cuda'),
                                                    t=torch.zeros((len(generated_samples_rejection),), dtype=torch.long, device='cuda')).cpu().numpy().mean()

                chamfer_dists_diffusion.append(chamfer_dist_diffusion)
                chamfer_dists_rejection.append(chamfer_dist_rejection)
                NLLs_diffusion.append(NLL_diffusion)
                NLLs_rejection.append(NLL_rejection)

                wandb.log({
                    f"{environment}/sample_1_gt": wandb.Image(plot_points(dataset_1)),
                    f"{environment}/sample_2_gt": wandb.Image(plot_points(dataset_2)),
                    f"{environment}/sample_composed_gt": wandb.Image(plot_points(dataset_composed),
                                                                    caption=f"target samples at {environment}"),
                    f"{environment}/baseline_diffusion": wandb.Image(plot_points(generated_samples_diffusion),
                                                                    caption=f"diffusion samples at {environment}, chamfer distance: {chamfer_dist_diffusion:0.4f}, NLL: {NLL_diffusion:0.4f}, accuracy: {acc_diffusion:0.4f}"),
                    f"{environment}/baseline_rejection": wandb.Image(plot_points(generated_samples_rejection),
                                                                    caption=f"rejection samples at {environment}, chamfer distance: {chamfer_dist_rejection:0.4f}, NLL: {NLL_rejection:0.4f}, accuracy: {acc_rejection:0.4f}"),
                    f"{environment}/acceptance_ratio": acceptance_ratio,
                })
            else:
                wandb.log({
                    f"{environment}/sample_1_gt": wandb.Image(plot_points(dataset_1)),
                    f"{environment}/sample_2_gt": wandb.Image(plot_points(dataset_2)),
                    f"{environment}/sample_composed_gt": wandb.Image(plot_points(dataset_composed),
                                                                    caption=f"target samples at {environment}"),
                    f"{environment}/baseline_diffusion": wandb.Image(plot_points(generated_samples_diffusion),
                                                                    caption=f"diffusion samples at {environment}, #samples: {len(generated_samples_diffusion)}, accuracy: {acc_diffusion:0.4f}"),
                    f"{environment}/baseline_rejection": wandb.Image(plot_points(generated_samples_rejection),
                                                                    caption=f"rejection samples at {environment}, #samples: {len(generated_samples_rejection)}, accuracy: {acc_rejection:0.4f}"),
                    f"{environment}/acceptance_ratio": acceptance_ratio,
                })

        wandb.log({f"{algebra}/acc_diffusion": np.mean(accs_diffusion),
                   f"{algebra}/acc_rejection": np.mean(accs_rejection),
                   f"{algebra}/chamfer_dist_diffusion": np.mean(chamfer_dists_diffusion),
                   f"{algebra}/chamfer_dist_rejection": np.mean(chamfer_dists_rejection),
                   f"{algebra}/NLL_diffusion": np.mean(NLLs_diffusion),
                   f"{algebra}/NLL_rejection": np.mean(NLLs_rejection)})

if __name__ == "__main__":
    main()
