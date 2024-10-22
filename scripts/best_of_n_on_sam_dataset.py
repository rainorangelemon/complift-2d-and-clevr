import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from typing import Union
import os
import json

import argparse
import torch as th

from ComposableDiff.composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict
)
from ComposableDiff.composable_diffusion.image_datasets import load_data
from ComposableDiff.classifier.eval import load_classifier
import baselines_clevr

from PIL import Image
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import wandb
from utils import plot_energy_histogram
from copy import deepcopy
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


device = th.device('cuda' if th.cuda.is_available() else 'cpu')
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    th.autocast("cuda", dtype=th.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if th.cuda.get_device_properties(0).major >= 8:
        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True
sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()


# utility function copied from segment anything model v2
def show_points(coords, ax, marker_size=375):
    ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_image(img, coord_label, path, mask=None):
    plt.clf()
    plt.close('all')
    coord_label = coord_label.copy()
    coord_label[..., 0] = coord_label[..., 0] * 128
    coord_label[..., 1] = (1 - coord_label[..., 1]) * 128
    plt.imshow(img)
    if len(coord_label.shape) == 1:
        coord_label = coord_label[None, ...]
    show_points(coord_label, plt.gca())
    if mask is not None:
        show_mask(mask, plt.gca())
    plt.savefig(path)


class CLEVRPosDataset(Dataset):
    def __init__(
        self,
        data_path,
    ):
        self.data_path = data_path

        data = np.load(self.data_path)
        self.labels = data['coords_labels']

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        label = self.labels[index]
        return label.astype(np.float32), self.convert_caption(label)

    def convert_caption(self, label):
        paragraphs = []
        for j in range(label.shape[0]):
            x, y = label[j, :2]
            paragraphs.append(f'object at position {x}, {y}')
        return ' and '.join(paragraphs)


@th.no_grad()
def score_single_image(image, labels):
    n_constraints = len(labels)
    img_batch = [image] * n_constraints
    predictor.set_image_batch(img_batch)

    input_points = labels
    input_points[:, 0] = input_points[:, 0] * len(image)
    input_points[:, 1] = (1 - input_points[:, 1]) * len(image[0])
    input_labels = np.array([1] * len(input_points))

    masks_batch, _, _ = predictor.predict_batch(
        point_coords_batch=[p[np.newaxis] for p in input_points],
        point_labels_batch=[l[np.newaxis] for l in input_labels],
        multimask_output=True,
    )

    success = True
    successes_point = []
    for i, masks in zip(range(len(input_points)), masks_batch):

        success_per_point = ((masks.sum(axis=(1, 2)) > 100) & (masks.sum(axis=(1, 2)) < 1000)).any()
        success = success and success_per_point

        successes_point.append(success_per_point)

    return success, successes_point, masks_batch


@th.no_grad()
@hydra.main(config_path="../conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    wandb.init(project="rejection_sampling",
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

    # Create output directory
    output_dir = Path(cfg.output_dir)
    experiment_name = cfg.data_path.split('/')[-1].split('.')[0]
    output_dir = output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the config
    config_save_path = os.path.join(cfg.output_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Create dataset and dataloader
    if "clevr_pos" in cfg.data_path:
        dataset = CLEVRPosDataset(data_path=cfg.data_path)
    else:
        raise ValueError("Unknown dataset")

    def conditions_denoise_fn_factory(model, labels, batch_size=cfg.mini_batch):
        # add zeros to the labels for unconditioned sampling
        labels = th.cat([labels, th.zeros_like(labels[:, :1, :])], dim=1).to(device)
        masks = th.ones_like(labels[:, :, 0], dtype=th.bool).to(device)
        masks[:, -1] = False
        num_relations_per_sample = labels.shape[1]
        def create_condition_denoise_fn(rel_idx):
            def condition_denoise_fn(x_t, ts, use_cfg=False, batch_size=batch_size):
                current_label = labels[:, rel_idx, :].to(device)
                current_mask = masks[:, rel_idx].to(device)

                num_samples = x_t.shape[0]
                if use_cfg:
                    batch_size = batch_size // 2
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

        return [create_condition_denoise_fn(rel_idx) for rel_idx in range(num_relations_per_sample)]


    # Sampling loop
    for test_idx in [1]:
        ims = [np.array(Image.open(f"runs/10-21_17-39-55/test_clevr_pos_5000_5/original_sample_{test_idx:05d}_{i:05d}.png").convert("RGB")) for i in range(100)]
        labels, _ = dataset[test_idx]
        n_success = 0
        n_success_per_point = [0] * len(labels)

        conditions_denoise_fn = conditions_denoise_fn_factory(model, th.tensor(labels[np.newaxis], dtype=th.float32), batch_size=cfg.mini_batch)
        estimate_neg_logp = baselines_clevr.make_estimate_neg_logp(elbo_cfg=cfg.elbo,
                                                                noise_scheduler=diffusion,
                                                                unconditioned_denoise_fn=conditions_denoise_fn[-1],
                                                                mini_batch=cfg.mini_batch,
                                                                progress=True)
        samples = th.from_numpy(np.stack(ims)).to(device).permute(0, 3, 1, 2).float()
        # normalize the images
        samples = (samples / 255.0) * 2.0 - 1.0
        energies = [estimate_neg_logp(condition_denoise_fn, samples, t=th.full((len(samples),), 0, dtype=th.long, device=samples.device)) for condition_denoise_fn in conditions_denoise_fn[:-1]]
        is_correct = {k: th.tensor([False] * len(ims), device=energies[k].device, dtype=th.bool) for k in range(len(labels))}

        # for im_idx, im in tqdm(list(enumerate(ims))):
        #     success, successes_point, masks = score_single_image(im, deepcopy(labels))
        #     n_success += success
        #     for i, s in enumerate(successes_point):
        #         n_success_per_point[i] += bool(s)
        #         is_correct[i][im_idx] = bool(s)
        #         # if s:
        #         #     for mask_idx, mask in enumerate(masks[i]):
        #         #         show_image(im, labels[i], output_dir / f"correct_{im_idx}_{i}_{mask_idx}.png", mask=mask)
        #         # else:
        #         #     for mask_idx, mask in enumerate(masks[i]):
        #         #         show_image(im, labels[i], output_dir / f"incorrect_{im_idx}_{i}_{mask_idx}.png", mask=mask)
        # print(n_success, n_success_per_point)
        # all_correct = th.stack([is_correct[k] for k in range(len(labels))], dim=0).all(dim=0)

        # correct_energy = energies[is_correct]
        # incorrect_energy = energies[~is_correct]
        # img_correct = plot_energy_histogram(correct_energy.flatten().cpu().numpy())
        # img_incorrect = plot_energy_histogram(incorrect_energy.flatten().cpu().numpy())

        energies_vector = th.stack(energies, dim=1)
        best_of_n = 10
        # top_correct_rates = []
        # for _ in range(1):
        #     perm = th.randperm(len(energies_vector))
        #     permuted_energies = energies_vector[perm]
        #     permuted_all_correct = all_correct[perm]

        #     permuted_energies = permuted_energies.reshape(-1, best_of_n, len(labels))
        #     # calculate the rank for each individual label, lower rank is better -> (B // N, L)
        #     rank_idx = permuted_energies.argsort(dim=1)
        #     top_idx = rank_idx.sum(dim=-1).argmin(dim=1)
        #     # sum the ranks for each label, lower sum is better -> (B // N)
        #     top_correct = permuted_all_correct.reshape(-1, best_of_n)[th.arange(len(top_idx)), top_idx]
        #     top_correct_rates.append(top_correct.sum().item() / len(top_correct))

        # rank the energies
        rank_global = energies_vector[:, 1]
        top_idx = rank_global.topk(best_of_n, largest=False).indices
        # save the images
        for idx in top_idx:
            show_image(ims[idx], labels[1], output_dir / f"best_of_n_{idx}.png")

        # plot the worst images
        worst_idx = rank_global.topk(best_of_n, largest=True).indices
        for idx in worst_idx:
            show_image(ims[idx], labels[1], output_dir / f"worst_of_n_{idx}.png")

        # avg_top_correct_rate = sum(top_correct_rates) / best_of_n

        # wandb.log({f"success_rate": n_success / len(ims),
        #            **{f"success_rate_per_point/{i}": n_success_per_point[i] / len(ims) for i in range(len(labels))},
        #            f"top_correct_rate": avg_top_correct_rate})



if __name__ == '__main__':
    main()
