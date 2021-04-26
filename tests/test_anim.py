"""Tests for the GIF builder."""
import os

import imageio
import numpy as np
import pytest

from src.anim import build_gif


def test_build_gif_writes_file(tmp_path):
    sdir = tmp_path / 'samples'
    sdir.mkdir()
    for i in range(3):
        # use a deterministic gradient instead of random so imageio doesn't
        # get tripped by all-uniform frames
        arr = np.zeros((32, 32, 3), dtype='uint8')
        arr[:, :, 0] = (i * 80) % 256
        arr[:, :, 1] = (i * 50) % 256
        imageio.imwrite(str(sdir / f'epoch_{i:03d}.png'), arr)
    out = tmp_path / 'training.gif'
    build_gif(str(sdir), str(out), fps=2)
    assert out.exists()
    assert out.stat().st_size > 0


def test_build_gif_raises_on_empty_dir(tmp_path):
    sdir = tmp_path / 'empty'
    sdir.mkdir()
    out = tmp_path / 'x.gif'
    with pytest.raises(FileNotFoundError):
        build_gif(str(sdir), str(out))
