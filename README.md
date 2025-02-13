# CompLift for 2D and CLEVR Tasks

The official PyTorch implementation of *Improving Compositional Generation with Diffusion Models Using Lift Scores*.

## Overview

CompLift is a novel approach that improves compositional generation in diffusion models by using lift scores. The project focuses on two main tasks:

1. **2D Point Generation**: Generating 2D points with specific spatial relationships.
2. **CLEVR Object Generation**: Generating CLEVR-style images with objects having specific spatial relationships and attributes.

## Features

- Implementation of CompLift scoring mechanism for improved compositional generation
- Support for both standard and cached CompLift variants
- Integration with Composable Diffusion models
- Evaluation tools for measuring generation quality
- Support for various spatial relationships and object attributes in CLEVR
- Configurable sampling and generation parameters

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd complift

# Install dependencies (recommend using a virtual environment)
pip install -r requirements.txt
```

## Usage

### CLEVR Tasks

1. **Position-based Generation**:
```python
python 2d_and_clevr/scripts/run_baselines_clevr.py --data_path [path_to_clevr_pos_data]
```

2. **Relation-based Generation**:
```python
python 2d_and_clevr/scripts/run_baselines_clevr.py --data_path [path_to_clevr_rel_data]
```

### 2D Point Generation

```python
python 2d_and_clevr/scripts/run_baselines_2d.py
```

## Model Architecture

The implementation uses a UNet-based architecture with the following key components:

- Conditional diffusion model for compositional generation
- Lift score computation for improved sampling
- Support for both position-based and relation-based conditions
- Cached scoring mechanism for efficient generation

## Data Format

### CLEVR Position Dataset
- Coordinates labels for object positions
- Format: `(x, y)` coordinates for each object

### CLEVR Relation Dataset
- Object attributes: color, shape, material, size
- Supported relations: left, right, front, behind, above, below

## References

* tanelp's [tiny diffusion](https://github.com/tanelp/tiny-diffusion)
* HuggingFace's [diffusers](https://github.com/huggingface/diffusers) library
* lucidrains' [DDPM implementation in PyTorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
* Jonathan Ho's [implementation of DDPM](https://github.com/hojonathanho/diffusion)
* InFoCusp's [DDPM implementation in tf](https://github.com/InFoCusp/diffusion_models)

