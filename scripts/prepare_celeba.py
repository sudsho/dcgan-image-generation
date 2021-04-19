"""Prepare a CelebA subset for training.

Expects the CelebA aligned `img_align_celeba/` directory (download separately
from the CelebA site or kaggle). Copies the first N images into
data/celeba/img/ so torchvision.ImageFolder picks them up.
"""
import argparse
import os
import shutil


def prepare(src, dst, n):
    os.makedirs(dst, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if f.endswith('.jpg'))[:n]
    for f in files:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print(f'copied {len(files)} images to {dst}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True,
                   help='path to img_align_celeba/')
    p.add_argument('--dst', default='data/celeba/img')
    p.add_argument('--n', type=int, default=30000)
    args = p.parse_args()
    prepare(args.src, args.dst, args.n)
