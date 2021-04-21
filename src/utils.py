"""Small helpers: yaml load, sample grid save, seed, base64 grid."""
import base64
import io
import os
import random

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.utils import make_grid, save_image


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_sample_grid(tensor, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # un-normalize from [-1, 1] -> [0, 1]
    tensor = (tensor.clamp(-1, 1) + 1) / 2.0
    save_image(tensor, path, nrow=nrow)


def make_grid_uint8(tensor, nrow=8):
    """Return HxWxC uint8 numpy array for GIF building."""
    tensor = (tensor.clamp(-1, 1) + 1) / 2.0
    grid = make_grid(tensor, nrow=nrow)
    grid = grid.mul(255).byte().cpu().numpy()
    # CHW -> HWC
    return grid.transpose(1, 2, 0)


def tensor_grid_to_b64(tensor, nrow=8):
    """Encode a batch tensor as a base64 PNG grid."""
    arr = make_grid_uint8(tensor, nrow=nrow)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')
