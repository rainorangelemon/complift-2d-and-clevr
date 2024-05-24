import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def circle_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def dino_dataset(n=8000):
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def gaussian_from_centers(centers, n_samples_per_center):
    X = []
    for i, center in enumerate(centers):
        x = np.random.normal(center[0], 0.03, n_samples_per_center)
        y = np.random.normal(center[1], 0.03, n_samples_per_center)
        X.append(np.stack((x, y), axis=1))
    X = np.concatenate(X)
    X = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    return X


def composition_product_a1_dataset(n=8000):
    # mixture of gaussians, a GMM of 8 Gaussians in a ring of radius 0.5 around the origin, with each Gaussian having a standard deviation of 0.3
    n_gaussians = 8
    n_samples_per_gaussian = n // n_gaussians
    centers = [[0.5 * np.cos(2 * np.pi * i / n_gaussians), 0.5 * np.sin(2 * np.pi * i / n_gaussians)] for i in range(n_gaussians)]
    return gaussian_from_centers(centers, n_samples_per_gaussian)


def composition_product_a2_dataset(n=8000):
    # a uniform distribution of points with x between -0.1 and 0.1 and y between -1 and 1
    x = np.random.uniform(-0.1, 0.1, n)
    y = np.random.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def composition_product_a3_dataset(n=8000):
    # the target distribution
    centers = [[0, 0.5], [0, -0.5]]
    n_samples_per_center = n // len(centers)
    X = gaussian_from_centers(centers, n_samples_per_center)
    return X


def composition_product_b1_dataset(n=8000):
    theta = np.random.uniform(0, 2 * np.pi, size=n)
    x = np.cos(theta) - 0.5
    y = np.sin(theta)
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def composition_product_b2_dataset(n=8000):
    theta = np.random.uniform(0, 2 * np.pi, size=n)
    x = np.cos(theta) + 0.5
    y = np.sin(theta)
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def composition_product_b3_dataset(n=8000):
    intersection_1 = [0, 0.5 * np.sqrt(3)]
    intersection_2 = [0, -0.5 * np.sqrt(3)]
    centers = np.array([intersection_1, intersection_2])
    center_id = np.random.randint(0, 2, n)
    X = centers[center_id]
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def composition_product_c1_dataset(n=8000):
    region = [[-1, -1], [-0.5, 1]]
    x = np.random.uniform(region[0][0], region[1][0], n)
    y = np.random.uniform(region[0][1], region[1][1], n)
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def composition_product_c2_dataset(n=8000):
    region = [[0.5, -1], [1, 1]]
    x = np.random.uniform(region[0][0], region[1][0], n)
    y = np.random.uniform(region[0][1], region[1][1], n)
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def composition_product_c3_dataset(n=8000):
    return TensorDataset(torch.empty((0, 2), dtype=torch.float32))

def composition_summation_a1_dataset(n=8000):
    left_centers = [[-0.25, 0.5], [-0.25, 0.], [-0.25, -0.5]]
    n_samples_per_center = n // len(left_centers)
    X = gaussian_from_centers(left_centers, n_samples_per_center)
    return X

def composition_summation_a2_dataset(n=8000):
    right_centers = [[0.25, 0.5], [0.25, 0.], [0.25, -0.5]]
    n_samples_per_center = n // len(right_centers)
    X = gaussian_from_centers(right_centers, n_samples_per_center)
    return X

def composition_summation_a3_dataset(n=8000):
    centers = [[-0.25, 0.5], [-0.25, 0.], [-0.25, -0.5], [0.25, 0.5], [0.25, 0.], [0.25, -0.5]]
    n_samples_per_center = n // len(centers)
    X = gaussian_from_centers(centers, n_samples_per_center)
    return X

def composition_summation_b1_dataset(n=8000):
    region = [[-1, -1], [-0.5, 1]]
    x = np.random.uniform(region[0][0], region[1][0], n)
    y = np.random.uniform(region[0][1], region[1][1], n)
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def composition_summation_b2_dataset(n=8000):
    region = [[0.5, -1], [1, 1]]
    x = np.random.uniform(region[0][0], region[1][0], n)
    y = np.random.uniform(region[0][1], region[1][1], n)
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def composition_summation_b3_dataset(n=8000):
    n_samples_per_region = n // 2
    samples_1 = composition_summation_b1_dataset(n_samples_per_region).tensors[0]
    samples_2 = composition_summation_b2_dataset(n_samples_per_region).tensors[0]
    X = torch.cat((samples_1, samples_2), dim=0)
    return TensorDataset(X)

def composition_summation_c1_dataset(n=8000):
    return composition_product_b1_dataset(n)

def composition_summation_c2_dataset(n=8000):
    return composition_product_b2_dataset(n)

def composition_summation_c3_dataset(n=8000):
    samples_per_region = n // 2
    samples_1 = composition_summation_c1_dataset(samples_per_region).tensors[0]
    samples_2 = composition_summation_c2_dataset(samples_per_region).tensors[0]
    X = torch.cat((samples_1, samples_2), dim=0)
    return TensorDataset(X)

def composition_negation_a1_dataset(n=8000):
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def composition_negation_a2_dataset(n=8000):
    x = np.random.uniform(-0.5, 0.5, n)
    y = np.random.uniform(-0.5, 0.5, n)
    X = np.stack((x, y), axis=1)
    X = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    return X

def composition_negation_a3_dataset(n=8000):
    region1 = [[-1, -1], [-0.5, 0.5]]
    region2 = [[-0.5, -1], [1, -0.5]]
    region3 = [[-1, 0.5], [0.5, 1]]
    region4 = [[0.5, -0.5], [1, 1]]
    n_samples_per_region = n // 4
    X = []
    for region in [region1, region2, region3, region4]:
        x = np.random.uniform(region[0][0], region[1][0], n_samples_per_region)
        y = np.random.uniform(region[0][1], region[1][1], n_samples_per_region)
        X.append(np.stack((x, y), axis=1))
    X = np.concatenate(X)
    X = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    return X

def composition_negation_b1_dataset(n=8000):
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def composition_negation_b2_dataset(n=8000):
    x = np.random.uniform(-0.5, 0.5, n)
    y = np.random.uniform(-2, 2, n)
    X = np.stack((x, y), axis=1)
    X = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    return X

def composition_negation_b3_dataset(n=8000):
    region1 = [[-1, -1], [-0.5, 1]]
    region2 = [[0.5, -1], [1, 1]]
    n_samples_per_region = n // 2
    X = []
    for region in [region1, region2]:
        x = np.random.uniform(region[0][0], region[1][0], n_samples_per_region)
        y = np.random.uniform(region[0][1], region[1][1], n_samples_per_region)
        X.append(np.stack((x, y), axis=1))
    X = np.concatenate(X)
    X = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    return X

def composition_negation_c1_dataset(n=8000):
    region = [[-0.5, -0.5], [0.5, 0.5]]
    x = np.random.uniform(region[0][0], region[1][0], n)
    y = np.random.uniform(region[0][1], region[1][1], n)
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def composition_negation_c2_dataset(n=8000):
    region = [[-1, -1], [1, 1]]
    x = np.random.uniform(region[0][0], region[1][0], n)
    y = np.random.uniform(region[0][1], region[1][1], n)
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def composition_negation_c3_dataset(n=8000):
    return TensorDataset(torch.empty((0, 2), dtype=torch.float32))


def get_dataset(name, n=8000):
    dataset_mapping = {
        "moons": moons_dataset,
        "dino": dino_dataset,
        "line": line_dataset,
        "circle": circle_dataset,
        
        "product_a1": composition_product_a1_dataset,
        "product_a2": composition_product_a2_dataset,
        "product_a3": composition_product_a3_dataset,
        "summation_a1": composition_summation_a1_dataset,
        "summation_a2": composition_summation_a2_dataset,
        "summation_a3": composition_summation_a3_dataset,
        "negation_a1": composition_negation_a1_dataset,
        "negation_a2": composition_negation_a2_dataset,
        "negation_a3": composition_negation_a3_dataset,
        
        "product_b1": composition_product_b1_dataset,
        "product_b2": composition_product_b2_dataset,
        "product_b3": composition_product_b3_dataset,
        "summation_b1": composition_summation_b1_dataset,
        "summation_b2": composition_summation_b2_dataset,
        "summation_b3": composition_summation_b3_dataset,
        "negation_b1": composition_negation_b1_dataset,
        "negation_b2": composition_negation_b2_dataset,
        "negation_b3": composition_negation_b3_dataset,

        "product_c1": composition_product_c1_dataset,
        "product_c2": composition_product_c2_dataset,
        "product_c3": composition_product_c3_dataset,
        "summation_c1": composition_summation_c1_dataset,
        "summation_c2": composition_summation_c2_dataset,
        "summation_c3": composition_summation_c3_dataset,
        "negation_c1": composition_negation_c1_dataset,
        "negation_c2": composition_negation_c2_dataset,
        "negation_c3": composition_negation_c3_dataset,
    }
    if name in dataset_mapping:
        dataset = dataset_mapping[name](n)
        dataset.tensors = (dataset.tensors[0].to(device),)
        return dataset
    else:
        raise ValueError(f"Unknown dataset: {name}")
    

def generate_data_points(n=8000, dataset="moons"):
    dataset = get_dataset(dataset, n)
    return dataset.tensors[0].cpu().numpy()    