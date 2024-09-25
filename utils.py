from PIL import Image
from time import perf_counter
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import io


def plot_points(points: np.ndarray) -> np.ndarray:
    """plot the points

    Args:
        points (np.ndarray): (n, 2) array of points

    Returns:
        io.BytesIO: buffer of the plot
    """
    plt.clf()
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot(points[:, 0], points[:, 1], 'o')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
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