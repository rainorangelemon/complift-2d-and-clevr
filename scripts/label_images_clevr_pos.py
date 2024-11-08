from scripts.best_of_n_on_sam_dataset import CLEVRPosDataset
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm.auto import tqdm


def classification_with_sam(mask_generator, image, input_points):
    anns = mask_generator.generate(image)

    input_points = np.array(input_points)
    input_points[:, 0] = input_points[:, 0] * len(image)
    input_points[:, 1] = (1 - input_points[:, 1]) * len(image[0])
    input_labels = np.array([1] * len(input_points))
    draw_labels =  np.array([1] * len(input_points))

    # get the background mask which has the largest area
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    background_mask = sorted_anns[0]['segmentation']

    success = True
    for i in range(len(input_points)):

        # check if the point is inside the background mask
        point = input_points[i]
        success_per_point = (background_mask[int(point[1]), int(point[0])] == 0)
        success = success and success_per_point

        if not success_per_point:
            draw_labels[i] = 0

    return success, background_mask, draw_labels


@hydra.main(config_path="../conf", config_name="clevr_pos", version_base="1.1")
def main(cfg: DictConfig):

    dataset = CLEVRPosDataset(data_path=cfg.data_path)
    image_dir = cfg.image_dir

    # clear the global hydra config to avoid conflicts with the SAM2 config
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )


    np.random.seed(3)

    os.chdir("../sam2")
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    successes = []
    pbar = tqdm(range(len(dataset)))
    for i in pbar:
        coords, _ = dataset[i]
        img_path = os.path.join("../tiny-diffusion/", image_dir, f"sample_{i:05d}.png")
        img = Image.open(img_path)
        img = np.array(img)
        success, _, draw_labels = classification_with_sam(mask_generator, img, coords)
        successes.append(success)
        pbar.set_description(f"Success rate: {np.mean(successes):.3f}")

    # save the results under the same directory as the images
    results_path = os.path.join("../tiny-diffusion/", image_dir, "results.npy")
    np.save(results_path, successes)

    print(f"Success rate: {np.mean(successes):.3f}")


if __name__ == "__main__":
    main()
