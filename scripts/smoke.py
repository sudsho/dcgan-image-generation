"""Tiny-CPU offline smoke for the DCGAN.

Runs the whole thing small, on CPU, with SYNTHETIC images so there is no
CelebA download and no GPU needed. It:

  1. forces CPU (ignores any available GPU),
  2. writes a handful of tiny synthetic JPEGs to a temp dir (guards the
     dataset download - the real CelebA path is never touched),
  3. runs a few adversarial steps and checks that both the discriminator
     and generator losses compute, stay finite, and actually update the
     weights (the adversarial loop really runs),
  4. exercises the real training entrypoint (src.train.train) for one epoch
     on the synthetic data and confirms it writes a checkpoint + sample grid,
  5. loads that checkpoint and generates a small batch from noise, asserting
     the ConvTranspose output dimensions (B, 3, 64, 64).

The headline result - realistic 64x64 faces - needs a GPU and the real
CelebA dataset. This smoke only proves the code runs end to end on CPU.
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# make `import src...` work when run as `python scripts/smoke.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import Generator, Discriminator, weights_init
from src.train import train
from src.sample import load_generator, random_samples

# tiny everything: this is a smoke, not a training run
LATENT_DIM = 16
NGF = 8
NDF = 8
CHANNELS = 3
IMAGE_SIZE = 64  # DCGAN G is fixed to a 64x64 output
BATCH = 8
STEPS = 5


def _force_cpu():
    device = torch.device('cpu')
    torch.manual_seed(0)
    np.random.seed(0)
    return device


def _make_tiny_dataset(root, n=8):
    """Write n tiny synthetic CelebA-shaped JPEGs under root/img/."""
    cls = os.path.join(root, 'img')
    os.makedirs(cls, exist_ok=True)
    for i in range(n):
        arr = np.random.randint(0, 255, (218, 178, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(cls, f'{i:04d}.jpg'), 'JPEG')
    return root


def check_adversarial_loop(device):
    """Run a few D/G steps on synthetic tensors; assert losses update."""
    print('[1/3] adversarial loop (synthetic tensors, CPU)')
    torch.manual_seed(0)
    G = Generator(LATENT_DIM, NGF, CHANNELS).to(device)
    D = Discriminator(NDF, CHANNELS).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # snapshot a generator weight to prove it moves after a G update
    w_before = G.net[0].weight.detach().clone()

    first_ld = first_lg = None
    for step in range(STEPS):
        real = torch.randn(BATCH, CHANNELS, IMAGE_SIZE, IMAGE_SIZE,
                           device=device)
        b = real.size(0)

        # D step
        opt_d.zero_grad()
        real_labels = torch.full((b,), 0.9, device=device)
        fake_labels = torch.full((b,), 0.0, device=device)
        loss_d_real = criterion(D(real), real_labels)
        z = torch.randn(b, LATENT_DIM, device=device)
        fake = G(z)
        loss_d_fake = criterion(D(fake.detach()), fake_labels)
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        opt_d.step()

        # G step
        opt_g.zero_grad()
        loss_g = criterion(D(fake), real_labels)
        loss_g.backward()
        opt_g.step()

        if first_ld is None:
            first_ld, first_lg = loss_d.item(), loss_g.item()
        print(f'      step {step} loss_d {loss_d.item():.4f} '
              f'loss_g {loss_g.item():.4f}')
        assert np.isfinite(loss_d.item()), 'loss_d is not finite'
        assert np.isfinite(loss_g.item()), 'loss_g is not finite'

    w_after = G.net[0].weight.detach()
    delta = (w_after - w_before).abs().max().item()
    print(f'      generator weight max-delta after {STEPS} steps: {delta:.3e}')
    assert delta > 0, 'generator weights did not update (loop did not run)'
    print(f'      losses moved: loss_d {first_ld:.4f} -> {loss_d.item():.4f}, '
          f'loss_g {first_lg:.4f} -> {loss_g.item():.4f}')


def check_real_training_entrypoint(tmp_root, device):
    """Run src.train.train for one epoch on synthetic images."""
    print('[2/3] real training entrypoint (src.train.train, 1 epoch, CPU)')
    data_root = _make_tiny_dataset(os.path.join(tmp_root, 'celeba'), n=8)
    out_dir = os.path.join(tmp_root, 'out')
    cfg = {
        'data': {
            'root': data_root,
            'image_size': IMAGE_SIZE,
            'batch_size': BATCH // 2,
            'num_workers': 0,
            'subset_size': 8,
        },
        'model': {
            'latent_dim': LATENT_DIM, 'ngf': NGF, 'ndf': NDF,
            'channels': CHANNELS,
        },
        'train': {
            'epochs': 1, 'lr_g': 2e-4, 'lr_d': 2e-4,
            'beta1': 0.5, 'beta2': 0.999,
            'label_smoothing': 0.1,
            # force CPU regardless of any GPU present
            'device': 'cpu', 'seed': 0,
            'log_interval': 1,
            'output_dir': out_dir,
        },
        # guard: FID (needs a pretrained InceptionV3 download) stays off
        'fid': {'enabled': False, 'num_real': 8, 'num_fake': 8,
                'every_epochs': 99},
    }
    train(cfg)
    ckpt = os.path.join(out_dir, 'checkpoints', 'epoch_000.pt')
    grid = os.path.join(out_dir, 'samples', 'epoch_000.png')
    assert os.path.exists(ckpt), f'no checkpoint written at {ckpt}'
    assert os.path.exists(grid), f'no sample grid written at {grid}'
    print(f'      wrote checkpoint {os.path.relpath(ckpt, tmp_root)} '
          f'and sample grid {os.path.relpath(grid, tmp_root)}')
    return ckpt


def check_generate_and_dims(ckpt, device):
    """Load the checkpoint, sample from noise, assert ConvTranspose dims."""
    print('[3/3] generate from noise + assert ConvTranspose output dims')
    G = load_generator(ckpt, LATENT_DIM, NGF, CHANNELS, device)
    n = 4
    imgs = random_samples(G, n, LATENT_DIM, device)
    print(f'      generated batch shape: {tuple(imgs.shape)}')
    assert imgs.shape == (n, CHANNELS, IMAGE_SIZE, IMAGE_SIZE), (
        f'expected {(n, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)}, got '
        f'{tuple(imgs.shape)}')
    # explicit ConvTranspose spatial-dim assertion: 1x1 -> 4 -> 8 -> 16 -> 32 -> 64
    assert imgs.shape[-1] == 64 and imgs.shape[-2] == 64, (
        'ConvTranspose stack did not upsample to 64x64')
    print('      ConvTranspose output dims OK (1x1 noise -> 64x64 image)')


def main():
    device = _force_cpu()
    print(f'device: {device} (cuda available: {torch.cuda.is_available()}, '
          f'forced off for smoke)')
    with tempfile.TemporaryDirectory(prefix='dcgan_smoke_') as tmp_root:
        check_adversarial_loop(device)
        ckpt = check_real_training_entrypoint(tmp_root, device)
        check_generate_and_dims(ckpt, device)
    print('SMOKE OK')


if __name__ == '__main__':
    main()
