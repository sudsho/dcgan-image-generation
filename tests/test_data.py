"""Tests for the data transforms / loader (uses tmp ImageFolder root)."""
import os

import numpy as np
import torch
from PIL import Image

from src.data import build_transform, get_celeba_loader


def _write_dummy_jpeg(path, size=(178, 218)):
    arr = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, 'JPEG')


def test_build_transform_outputs_64x64(tmp_path):
    p = tmp_path / 'a.jpg'
    _write_dummy_jpeg(str(p))
    t = build_transform(image_size=64)
    img = Image.open(str(p)).convert('RGB')
    out = t(img)
    assert out.shape == (3, 64, 64)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_celeba_loader_yields_batches(tmp_path):
    root = tmp_path / 'celeba'
    cls = root / 'img'
    cls.mkdir(parents=True)
    for i in range(8):
        _write_dummy_jpeg(str(cls / f'{i:04d}.jpg'))
    loader = get_celeba_loader(
        root=str(root), image_size=64, batch_size=4, num_workers=0,
    )
    x, _ = next(iter(loader))
    assert x.shape == (4, 3, 64, 64)
