# compare EBM across different timesteps

import DiT.diffusion
import DiT.download
import DiT.models
import ddpm
import torch
from baselines_imagenet import (
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
from r_and_r import compose_imagenet_diffusion_models
import wandb
import numpy as np
import DiT
import argparse
from typing import Tuple
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')
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
    model.eval()  # important! to disable dropout of condition labels
    diffusion = DiT.diffusion.create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    return model, vae, diffusion


def main(args: argparse.Namespace):
    # Initialize wandb
    wandb.init(project="r_and_r", name="baselines_ImageNet")
    wandb.run.log_code(".")

    dit_model, vae, diffusion = load_model(args)
    # TODO: Load the models
    model_1 = lambda x, t: dit_model.forward_with_cfg(x, t,
                                                      y=torch.full_like(t, args.class_1),
                                                      cfg_scale=args.cfg_scale)
    model_2 = lambda x, t: dit_model.forward_with_cfg(x, t,
                                                      y=torch.full_like(t, args.class_2),
                                                      cfg_scale=args.cfg_scale)
    model_to_test = compose_imagenet_diffusion_models(dit_model, args.cfg_scale, args.algebra, args.class_1, args.class_2, args.num_classes)
    with catchtime('diffusion'):
        latent_size = args.image_size // 8
        generated_samples_diffusion = diffusion_baseline(model_to_test, diffusion, latent_shape=(4, latent_size, latent_size))[-1]

        batch_size = 32
        num_batches = generated_samples_diffusion.shape[0] // batch_size
        decoded_samples = []

        for i in range(num_batches):
            batch = generated_samples_diffusion[i * batch_size:(i + 1) * batch_size].to(device)
            decoded_batch = vae.decode(batch / 0.18215).sample.cpu()
            decoded_samples.append(decoded_batch)

        # If there are remaining samples that don't fit into a full batch
        if generated_samples_diffusion.shape[0] % batch_size != 0:
            batch = generated_samples_diffusion[num_batches * batch_size:].to(device)
            decoded_batch = vae.decode(batch / 0.18215).sample.cpu()
            decoded_samples.append(decoded_batch)

        generated_samples_diffusion = torch.cat(decoded_samples, dim=0)

    # Save and display images:
    nrow = np.ceil(np.sqrt(generated_samples_diffusion.shape[0])).astype(int)
    save_image(generated_samples_diffusion, "sample_diffusion.png", nrow=nrow, normalize=True, value_range=(-1, 1))

    # with catchtime('rejection'):
    #     generated_samples_rejection_all_t, rejection_ratios, intervals = rejection_sampling_baseline_with_interval_calculation_elbo(model_to_test, model_1, model_2, algebra=algebra)
    #     generated_samples_rejection = generated_samples_rejection_all_t[-1]

    # cumulative_accept_ratio = np.cumprod(1-np.array(rejection_ratios))

    # accuracy_diffusion = calculate_accuracy_imagenet(generated_samples_diffusion, algebra, [args.class_1, args.class_2])
    # accuracy_rejection = calculate_accuracy_imagenet(generated_samples_rejection, algebra, [args.class_1, args.class_2])
    # wandb.log({
    #     f"{environment}/sample_1_gt": wandb.Image(plot_points(dataset_1)),
    #     f"{environment}/sample_2_gt": wandb.Image(plot_points(dataset_2)),
    #     f"{environment}/sample_composed_gt": wandb.Image(plot_points(dataset_composed),
    #                                                     caption=f"target samples at {environment}"),
    #     f"{environment}/baseline_diffusion": wandb.Image(plot_points(generated_samples_diffusion),
    #                                                     caption=f"diffusion samples at {environment}, chamfer distance: {chamfer_dist_diffusion:0.4f}"),
    #     f"{environment}/baseline_rejection": wandb.Image(plot_points(generated_samples_rejection),
    #                                                     caption=f"rejection samples at {environment}, chamfer distance: {chamfer_dist_rejection:0.4f}"),
    #     f"{environment}/intervals": wandb.Image(plot_two_intervals(intervals[0], intervals[1])),
    #     f"{environment}/acceptance_ratio": wandb.Image(plot_acceptance_ratios(cumulative_accept_ratio)),
    # })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT.models.DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--class-1", type=int, default=765)  # 765 is "rocking chair"
    parser.add_argument("--class-2", type=int, default=954)  # 954 is "zebra"
    parser.add_argument("--algebra", type=str, choices=["product", "negation"], default="product")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
