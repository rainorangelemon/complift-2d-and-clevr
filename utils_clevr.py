import numpy as np
import torch as th
from torch.utils.data import Dataset
import os
import torch


def get_device():
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return device


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


def conditions_denoise_fn_factory(model, labels, batch_size, cfg):
    device = get_device()
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
                results.append(result.clone())

            # Concatenate the results from all batches
            return th.cat(results, dim=0)
        return condition_denoise_fn

    return [create_condition_denoise_fn(rel_idx) for rel_idx in range(num_relations_per_sample)]
