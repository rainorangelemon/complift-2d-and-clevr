import hydra
import numpy as np
import glob
import wandb
import torch as th
import baselines_clevr
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.io import read_image
from utils import plot_energy_histogram
from omegaconf import DictConfig, OmegaConf
from ComposableDiff.composable_diffusion.model_creation import create_model_and_diffusion

device = th.device('cuda' if th.cuda.is_available() else 'cpu')


# utility function copied from segment anything model v2
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_image(img, coord_label, path):
    plt.clf()
    plt.close('all')
    coord_label[..., 0] = coord_label[..., 0] * 128
    coord_label[..., 1] = (1 - coord_label[..., 1]) * 128
    plt.imshow(img)
    coord_label = coord_label[None, ...]
    show_points(coord_label, np.ones(len(coord_label)), plt.gca())
    plt.savefig(path)


def load_img(path):
    # Read the image
    image = read_image(path)
    # Convert to float and scale to [0, 1]
    image = image.float() / 255.0
    # Reverse the normalization
    image = (image * 2 - 1)
    return image


# Sampling function
def energy_on_single_image(cfg, model, diffusion, samples, labels):
    samples = samples[None, ...]
    labels = labels[None, ...]
    masks = th.ones_like(labels[:, 0], dtype=th.bool).to(device)

    def conditions_denoise_fn_factory(current_label, current_mask, batch_size=cfg.mini_batch):
        def condition_denoise_fn(x_t, ts, batch_size=batch_size, use_cfg=False):

            num_samples = x_t.shape[0]
            results = []

            for i in range(0, num_samples, batch_size):
                # Create batch slices for current iteration
                x_t_batch = x_t[i:i+batch_size]
                ts_batch = ts[i:i+batch_size]
                current_batch_size = x_t_batch.shape[0]

                # Expand the current label and mask for the current batch size
                expanded_label = current_label.expand(current_batch_size, -1)
                expanded_mask = current_mask.expand(current_batch_size)

                if use_cfg:
                    # Add the unconditioned label
                    x_t_batch = th.cat([x_t_batch, x_t_batch], dim=0)
                    ts_batch = th.cat([ts_batch, ts_batch], dim=0)
                    expanded_label = th.cat([expanded_label, th.zeros_like(expanded_label)], dim=0)
                    expanded_mask = th.cat([expanded_mask, th.zeros_like(expanded_mask)], dim=0)
                    result = model(x_t_batch, ts_batch, y=expanded_label, masks=expanded_mask)
                    eps, rest = result[:, :3], result[:, 3:]
                    cond_eps, uncond_eps = eps[expanded_mask], eps[~expanded_mask]
                    eps = uncond_eps + (cfg.cfg_weight * (cond_eps - uncond_eps))
                    result = th.cat([eps, rest[~expanded_mask]], dim=1)
                else:
                    result = model(x_t_batch, ts_batch, y=expanded_label, masks=expanded_mask)

                results.append(result)

            # Concatenate the results from all batches
            return th.cat(results, dim=0)
        return condition_denoise_fn


    conditions_denoise_fn = conditions_denoise_fn_factory(labels, masks)
    unconditioned_denoise_fn = conditions_denoise_fn_factory(labels * 0, masks.clone().fill_(False),
                                                             batch_size=cfg.mini_batch)

    estimate_neg_logp = baselines_clevr.make_estimate_neg_logp(elbo_cfg=cfg.elbo,
                                                               noise_scheduler=diffusion,
                                                               unconditioned_denoise_fn=unconditioned_denoise_fn,
                                                               mini_batch=cfg.mini_batch,
                                                               progress=False if cfg.elbo.n_samples <=10 else True)

    condition_samples_all_t = baselines_clevr.diffusion_baseline(
        denoise_fn=lambda x, t: conditions_denoise_fn(x, t, use_cfg=True),
        diffusion=diffusion,
        x_shape=(3, cfg.model.image_size, cfg.model.image_size),
        eval_batch_size=100,
    )
    condition_samples_last_t = condition_samples_all_t[-1].to(samples.device)
    samples = th.cat([samples, condition_samples_last_t], dim=0)

    energies = estimate_neg_logp(conditions_denoise_fn, samples,
                                 t=th.full((len(samples),), 0, dtype=th.long, device=samples.device))

    # check the percentile of the 0th element among the energies
    percentile = th.sum(energies[0] > energies) / len(energies)
    return energies, percentile


@hydra.main(config_path="../conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    wandb.init(project="rejection_sampling_on_sam_dataset",
               name=f"{cfg.experiment_name}",
               # to dict
               config=OmegaConf.to_container(cfg, resolve=True))

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

    positive_files = np.load('../sam2/notebooks/positive_samples_5.npz')
    negative_files = np.load('../sam2/notebooks/negative_samples_5.npz')
    im_paths_positive = positive_files['im_paths']
    im_paths_negative = negative_files['im_paths']

    coords_labels_positive = positive_files['coords_labels']
    coords_labels_negative = negative_files['coords_labels']

    np.random.seed(1)
    positive_idxs = np.arange(len(im_paths_positive))[np.random.choice(len(im_paths_positive), 10, replace=False)]
    np.random.seed(2)
    negative_idxs = np.arange(len(im_paths_negative))[np.random.choice(len(im_paths_negative), 10, replace=False)]

    im_paths_positive = im_paths_positive[positive_idxs]
    im_paths_negative = im_paths_negative[negative_idxs]
    coords_labels_positive = coords_labels_positive[positive_idxs]
    coords_labels_negative = coords_labels_negative[negative_idxs]

    ims_positive = [load_img("ComposableDiff/output/test_clevr_pos_5000_5/"+im_path) for im_path in im_paths_positive]
    ims_negative = [load_img("ComposableDiff/output/test_clevr_pos_5000_5/"+im_path) for im_path in im_paths_negative]

    ims_positive = th.stack(ims_positive).to(device)
    ims_negative = th.stack(ims_negative).to(device)
    coords_labels_positive = th.tensor(coords_labels_positive).to(device).float()
    coords_labels_negative = th.tensor(coords_labels_negative).to(device).float()

    # save images
    for i, (im_positive, coords_label_positive) in enumerate(zip(ims_positive, coords_labels_positive)):
        show_image(im_positive.cpu().numpy().transpose(1, 2, 0) / 2 + 0.5, coords_label_positive.cpu().numpy(), f"positive_{i}.png")

    for i, (im_negative, coords_label_negative) in enumerate(zip(ims_negative, coords_labels_negative)):
        show_image(im_negative.cpu().numpy().transpose(1, 2, 0) / 2 + 0.5, coords_label_negative.cpu().numpy(), f"negative_{i}.png")

    energies_positive = [energy_on_single_image(cfg, model, diffusion, im_positive, coords_label_positive) for im_positive, coords_label_positive in tqdm(list(zip(ims_positive, coords_labels_positive)))]
    energies_negative = [energy_on_single_image(cfg, model, diffusion, im_negative, coords_label_negative) for im_negative, coords_label_negative in tqdm(list(zip(ims_negative, coords_labels_negative)))]

    energies_positive = th.stack(energies_positive)
    energies_negative = th.stack(energies_negative)
    img_positive = plot_energy_histogram(energies_positive.flatten().cpu().numpy())
    img_negative = plot_energy_histogram(energies_negative.flatten().cpu().numpy())
    wandb.log({f"energy_correct": [wandb.Image(img_positive, caption=f"Energy histogram for correct samples")],
               f"energy_incorrect": [wandb.Image(img_negative, caption=f"Energy histogram for incorrect samples")]})



if __name__ == "__main__":
    main()
