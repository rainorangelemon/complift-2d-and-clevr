from PIL import Image
from time import perf_counter
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import io


def plot_points(points: np.ndarray, filepath: str = None) -> np.ndarray:
    """plot the points

    Args:
        points (np.ndarray): (n, 2) array of points

    Returns:
        io.BytesIO: buffer of the plot
    """
    plt.clf()
    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(points[:, 0], points[:, 1], 'o', alpha=0.3)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)

    if filepath is not None:
        img.save(filepath)
    return img


def plot_two_intervals(interval_1: List[Tuple[float, float]], interval_2: List[Tuple[float, float]]) -> np.ndarray:
    # Sample data: timestep and (min, max) for each interval
    timesteps = list(range(len(interval_1)))
    # Creating the plot
    fig, ax = plt.subplots()

    # Extracting min and max values for each dataset
    min1, max1 = zip(*interval_1)
    min2, max2 = zip(*interval_2)

    # Plotting intervals as bands
    ax.fill_between(timesteps, min1, max1, color='blue', alpha=0.3, label='Data 1 Band')  # Band for the first data set
    ax.fill_between(timesteps, min2, max2, color='red', alpha=0.3, label='Data 2 Band')  # Band for the second data set

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Values')
    ax.set_title('Interval Band Plot Over Timesteps')
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_energy_histogram(energies: np.ndarray) -> np.ndarray:
    # plot the histograms of the energy
    plt.clf()
    plt.close("all")
    # Create a figure and a set of subplots
    fig, _ = plt.subplots()

    # Plot histogram for support_energy
    plt.hist(energies, bins=50, color='blue', alpha=0.7)

    if len(energies) > 0:
        # plot the percentile using red dashed line
        percentile_95 = np.percentile(energies, 95)
        percentile_05 = np.percentile(energies, 5)
        # label in scientific notation
        plt.axvline(x=percentile_95, color='red', linestyle='--', label=f'95th percentile: {percentile_95:.2e}')
        plt.axvline(x=percentile_05, color='red', linestyle='--', label=f'5th percentile: {percentile_05:.2e}')

    plt.legend()

    # in case the x-axis overlaps
    plt.xticks(rotation=45)
    # set the xticks to scientific notation
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_acceptance_ratios(rejection_ratios: np.ndarray) -> np.ndarray:
    timesteps = list(range(len(rejection_ratios)-1, -1, -1))
    fig, ax = plt.subplots()
    ax.plot(timesteps, rejection_ratios, 'o-')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cumulative Acceptance Ratio')
    ax.set_title('Cumulative Acceptance Ratio Over Timesteps')
    plt.gca().invert_xaxis()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img


def merge_pic(image_paths, column, row, save_path):
    images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = column * max(widths)
    total_height = row * max(heights)

    new_im = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    index = 0
    for i in range(row):
        for j in range(column):
            new_im.paste(images[index], (j * max(widths), i * max(heights)))
            index += 1

    new_im.save(save_path)


def merge_pic_in_a_row(image_paths, save_path):
    images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    total_height = max(heights)

    new_im = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    offset = 0
    for im in images:
        new_im.paste(im, (offset, 0))
        offset += im.size[0]

    new_im.save(save_path)


class catchtime:
    def __init__(self, module_str):
        self.module_str = module_str

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f'Time for {self.module_str}: {self.time:.3f} seconds'
        print(self.readout)


def evaluate_chamfer_distance(generated_samples, target_samples):
    """
    Calculate chamfer distance between two point clouds.

    Arguments:
    generated_samples -- First point cloud (N1 x D numpy array)
    target_samples -- Second point cloud (N2 x D numpy array)

    Returns:
    chamfer_dist -- Chamfer distance between the point clouds

    Example:
    >>> generated_samples = np.array([[1, 2], [3, 4], [5, 6]])
    >>> target_samples = np.array([[2, 3], [4, 5], [6, 7]])
    >>> evaluate_chamfer_distance(generated_samples, target_samples)
    """

    pc1, pc2 = generated_samples, target_samples

    # Reshape point clouds if necessary
    pc1 = np.atleast_2d(pc1)
    pc2 = np.atleast_2d(pc2)

    # Calculate pairwise distances
    dist_pc1_to_pc2 = np.sqrt(np.sum((pc1[:, None] - pc2) ** 2, axis=-1))
    dist_pc2_to_pc1 = np.sqrt(np.sum((pc2[:, None] - pc1) ** 2, axis=-1))

    # Minimum distance from each point in pc1 to pc2 and vice versa
    min_dist_pc1_to_pc2 = np.min(dist_pc1_to_pc2, axis=1)
    min_dist_pc2_to_pc1 = np.min(dist_pc2_to_pc1, axis=1)

    # Chamfer distance is the sum of these minimum distances
    chamfer_dist = np.mean(min_dist_pc1_to_pc2) + np.mean(min_dist_pc2_to_pc1)

    return chamfer_dist
