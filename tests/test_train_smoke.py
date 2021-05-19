"""Quick training smoke test on a tiny dataset (1 epoch, 8 images, no FID)."""
import os

import numpy as np
import pytest
from PIL import Image

from src.train import train


def _make_tiny_dataset(root, n=8):
    cls = os.path.join(root, 'img')
    os.makedirs(cls, exist_ok=True)
    for i in range(n):
        arr = np.random.randint(0, 255, (218, 178, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(cls, f'{i:04d}.jpg'), 'JPEG')


@pytest.mark.slow
def test_train_runs_one_epoch(tmp_path):
    _make_tiny_dataset(str(tmp_path / 'celeba'), n=4)
    cfg = {
        'data': {
            'root': str(tmp_path / 'celeba'),
            'image_size': 64,
            'batch_size': 2,
            'num_workers': 0,
            'subset_size': 4,
        },
        'model': {
            'latent_dim': 16, 'ngf': 8, 'ndf': 8, 'channels': 3,
        },
        'train': {
            'epochs': 1, 'lr_g': 2e-4, 'lr_d': 2e-4,
            'beta1': 0.5, 'beta2': 0.999,
            'label_smoothing': 0.1,
            'device': 'cpu', 'seed': 0,
            'log_interval': 1,
            'output_dir': str(tmp_path / 'out'),
        },
        'fid': {'enabled': False, 'num_real': 4, 'num_fake': 4,
                'every_epochs': 99},
    }
    train(cfg)
    assert os.path.exists(str(tmp_path / 'out' / 'samples' / 'epoch_000.png'))
    assert os.path.exists(
        str(tmp_path / 'out' / 'checkpoints' / 'epoch_000.pt'))
