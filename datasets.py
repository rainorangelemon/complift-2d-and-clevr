import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset


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


def composition_product_1_dataset(n=8000):
    # mixture of gaussians, a GMM of 8 Gaussians in a ring of radius 0.5 around the origin, with each Gaussian having a standard deviation of 0.3
    rng = np.random.default_rng(42)
    n_gaussians = 8
    n_samples_per_gaussian = n // n_gaussians
    X = []
    for i in range(n_gaussians):
        theta = 2 * np.pi * i / n_gaussians
        x = rng.normal(0.5 * np.cos(theta), 0.03, n_samples_per_gaussian)
        y = rng.normal(0.5 * np.sin(theta), 0.03, n_samples_per_gaussian)
        X.append(np.stack((x, y), axis=1))
    
    X = np.concatenate(X)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def composition_product_2_dataset(n=8000):
    # a uniform distribution of points with x between -0.1 and 0.1 and y between -1 and 1
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.1, 0.1, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def get_dataset(name, n=8000):
    dataset_mapping = {
        "moons": moons_dataset,
        "dino": dino_dataset,
        "line": line_dataset,
        "circle": circle_dataset,
        "composition_product_1": composition_product_1_dataset,
        "composition_product_2": composition_product_2_dataset,
    }
    if name in dataset_mapping:
        return dataset_mapping[name](n)
    else:
        raise ValueError(f"Unknown dataset: {name}")
