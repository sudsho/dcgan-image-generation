"""Tests for utils helpers."""
import base64
import os
import tempfile

import torch

from src.utils import (
    load_config, set_seed, save_sample_grid, make_grid_uint8,
    tensor_grid_to_b64,
)


def test_set_seed_is_deterministic():
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.allclose(a, b)


def test_save_sample_grid_creates_file():
    x = torch.randn(4, 3, 64, 64)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'sub', 'g.png')
        save_sample_grid(x, path, nrow=2)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_make_grid_uint8_shape():
    x = torch.randn(4, 3, 16, 16)
    arr = make_grid_uint8(x, nrow=2)
    # 2x2 grid w/ default 2px padding -> ~36x36, 3 channels
    assert arr.dtype.kind == 'u'
    assert arr.shape[2] == 3


def test_tensor_grid_to_b64_decodable():
    x = torch.randn(2, 3, 16, 16)
    s = tensor_grid_to_b64(x, nrow=2)
    raw = base64.b64decode(s)
    # PNG magic bytes
    assert raw[:8] == b'\x89PNG\r\n\x1a\n'


def test_load_config_roundtrip():
    cfg_text = 'a:\n  b: 1\n'
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        f.write(cfg_text)
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg['a']['b'] == 1
    finally:
        os.unlink(path)
