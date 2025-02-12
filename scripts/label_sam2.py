from utils_clevr import CLEVRPosDataset
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm.auto import tqdm
from typing import Tuple

class SAMClassifier:
    def __init__(self, device):
        self.device = device
        self._setup_device_settings()
        self._initialize_sam()

    def _setup_device_settings(self):
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

    def _initialize_sam(self):
        # clear the global hydra config to avoid conflicts with the SAM2 config
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()

        current_dir = os.getcwd()
        os.chdir("../sam2")

        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2)

        os.chdir(current_dir)

    def classify_image(self,
                       image: np.ndarray,
                       input_points: np.ndarray,
                       return_anns: bool = False) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Classifies an image with SAM2.

        Args:
            image: The image to classify, as a numpy array, range 0-255.
            input_points: The points to classify the image with, as a numpy array, range 0-1.
            return_anns: Whether to return the annotations.
        """
        anns = self.mask_generator.generate(image)

        input_points = np.array(input_points)
        input_points[:, 0] = input_points[:, 0] * len(image)
        input_points[:, 1] = (1 - input_points[:, 1]) * len(image[0])
        draw_labels = np.array([1] * len(input_points))

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

        if return_anns:
            return success, background_mask, draw_labels, anns
        else:
            return success, background_mask, draw_labels

    def classify_image_batch(self,
                             images: np.ndarray,
                             input_points: np.ndarray,
                             return_anns: bool = False) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Classifies a batch of images with SAM2.

        Args:
            images: The images to classify, as a numpy array, range 0-255.
            input_points: The points to classify the image with, as a numpy array, range 0-1.
            return_anns: Whether to return the annotations.
        """
        successes, background_masks, draw_labels, anns = [], [], [], []
        for image, input_points in zip(images, input_points):
            if return_anns:
                success, background_mask, draw_label, ann = self.classify_image(image, input_points, return_anns)
            else:
                success, background_mask, draw_label = self.classify_image(image, input_points, return_anns)
            successes.append(success)
            background_masks.append(background_mask)
            draw_labels.append(draw_label)
            if return_anns:
                anns.append(ann)
        if return_anns:
            return successes, background_masks, draw_labels, anns
        else:
            return successes, background_masks, draw_labels


@hydra.main(config_path="../conf", config_name="clevr_pos", version_base="1.1")
def main(cfg: DictConfig):
    dataset = CLEVRPosDataset(data_path=cfg.data_path)
    image_dir = cfg.image_dir

    # select the device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else
                         "cpu")
    print(f"using device: {device}")

    np.random.seed(3)

    classifier = SAMClassifier(device)

    successes = []
    pbar = tqdm(range(len(dataset)))
    for i in pbar:
        coords, _ = dataset[i]
        img_path = os.path.join("../tiny-diffusion/", image_dir, f"sample_{i:05d}.png")
        img = Image.open(img_path)
        img = np.array(img)
        success, _, draw_labels = classifier.classify_image(img, coords)
        successes.append(success)
        pbar.set_description(f"Success rate: {np.mean(successes):.3f}")

    # save the results under the same directory as the images
    results_path = os.path.join("../tiny-diffusion/", image_dir, "results.npy")
    np.save(results_path, successes)

    print(f"Success rate: {np.mean(successes):.3f}")


if __name__ == "__main__":
    main()
