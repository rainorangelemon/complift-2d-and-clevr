# compare EBM across different timesteps

import DiT.diffusion
import DiT.download
import DiT.models
import ddpm
import torch
from baselines import (
    ebm_baseline,
    diffusion_baseline,
    evaluate_W2,
    evaluate_chamfer_distance,
    intermediate_distribution,
    diffusion_baseline,
    rejection_sampling_baseline_with_interval_calculation,
    rejection_sampling_baseline_with_interval_calculation_elbo,
)
from datasets import generate_data_points
from utils import catchtime, plot_points, plot_two_intervals, plot_acceptance_ratios
import wandb
import numpy as np
import DiT
import argparse
from typing import Tuple
from diffusers.models import AutoencoderKL


torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(args: argparse.Namespace) -> Tuple[DiT.models.DiT, AutoencoderKL, DiT.diffusion.SpacedDiffusion]:
    """load the Diffusion Transformer model and the VAE model

    Args:
        args (argparse.Namespace): arguments, including the model, image size, number of classes, and the checkpoint

    Returns:
        Tuple[DiT.models.DiT, AutoencoderKL, DiT.diffusion.SpacedDiffusion]: the DiT model, the VAE model, and the diffusion model
    """
    # Load model:
    latent_size = args.image_size // 8
    model = DiT.models.DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = DiT.download.find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = DiT.diffusion.create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    return model, vae, diffusion


# Initialize wandb
wandb.init(project="r_and_r", name="baselines_ImageNet")
wandb.run.log_code(".")

# TODO: Load the models
model_1 = ddpm.EnergyMLP()
model_2 = ddpm.EnergyMLP()
for algebra in ['product', 'summation', 'negation']:
    for suffix in ['a', 'b', 'c']:
        environment = algebra + '_' + suffix + '3'
        model_1.load_state_dict(torch.load(f"exps/{algebra}_{suffix}1/ema_model.pth"))
        model_2.load_state_dict(torch.load(f"exps/{algebra}_{suffix}2/ema_model.pth"))
        model_to_test = ddpm.CompositionEnergyMLP(model_1, model_2, algebra=algebra)
        with catchtime('diffusion'):
            generated_samples_diffusion = diffusion_baseline(model_to_test)[-1]
        with catchtime('rejection'):
            generated_samples_rejection_all_t, rejection_ratios, intervals = rejection_sampling_baseline_with_interval_calculation_elbo(model_to_test, model_1, model_2, algebra=algebra)
            generated_samples_rejection = generated_samples_rejection_all_t[-1]
        dataset_1 = generate_data_points(n=8000, dataset=f"{algebra}_{suffix}1")
        dataset_2 = generate_data_points(n=8000, dataset=f"{algebra}_{suffix}2")
        dataset_composed = generate_data_points(n=8000, dataset=environment)

        cumulative_accept_ratio = np.cumprod(1-np.array(rejection_ratios))

        if len(dataset_composed):
            chamfer_dist_diffusion = evaluate_chamfer_distance(generated_samples=generated_samples_diffusion, target_samples=dataset_composed)
            chamfer_dist_rejection = evaluate_chamfer_distance(generated_samples=generated_samples_rejection, target_samples=dataset_composed)
            wandb.log({
                f"{environment}/sample_1_gt": wandb.Image(plot_points(dataset_1)),
                f"{environment}/sample_2_gt": wandb.Image(plot_points(dataset_2)),
                f"{environment}/sample_composed_gt": wandb.Image(plot_points(dataset_composed),
                                                                 caption=f"target samples at {environment}"),
                f"{environment}/baseline_diffusion": wandb.Image(plot_points(generated_samples_diffusion),
                                                                 caption=f"diffusion samples at {environment}, chamfer distance: {chamfer_dist_diffusion:0.4f}"),
                f"{environment}/baseline_rejection": wandb.Image(plot_points(generated_samples_rejection),
                                                                 caption=f"rejection samples at {environment}, chamfer distance: {chamfer_dist_rejection:0.4f}"),
                f"{environment}/intervals": wandb.Image(plot_two_intervals(intervals[0], intervals[1])),
                f"{environment}/acceptance_ratio": wandb.Image(plot_acceptance_ratios(cumulative_accept_ratio)),
            })
        else:
            wandb.log({
                f"{environment}/sample_1_gt": wandb.Image(plot_points(dataset_1)),
                f"{environment}/sample_2_gt": wandb.Image(plot_points(dataset_2)),
                f"{environment}/sample_composed_gt": wandb.Image(plot_points(dataset_composed),
                                                                 caption=f"target samples at {environment}"),
                f"{environment}/baseline_diffusion": wandb.Image(plot_points(generated_samples_diffusion),
                                                                 caption=f"diffusion samples at {environment}, #samples: {len(generated_samples_diffusion)}"),
                f"{environment}/baseline_rejection": wandb.Image(plot_points(generated_samples_rejection),
                                                                 caption=f"rejection samples at {environment}, #samples: {len(generated_samples_rejection)}"),
                f"{environment}/intervals": wandb.Image(plot_two_intervals(intervals[0], intervals[1])),
                f"{environment}/acceptance_ratio": wandb.Image(plot_acceptance_ratios(cumulative_accept_ratio)),
            })
