"""Generate samples from a trained Generator. Random or interpolated z."""
import argparse
import os
import torch

from src.model import Generator
from src.utils import load_config, save_sample_grid


def load_generator(ckpt_path, latent_dim, ngf, channels, device):
    G = Generator(latent_dim, ngf, channels).to(device)
    state = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(state['G'])
    G.eval()
    return G


def random_samples(G, n, latent_dim, device):
    z = torch.randn(n, latent_dim, device=device)
    with torch.no_grad():
        return G(z).cpu()


def interpolate(G, n, latent_dim, device, z_a=None, z_b=None):
    if z_a is None:
        z_a = torch.randn(1, latent_dim, device=device)
    if z_b is None:
        z_b = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, n, device=device).view(-1, 1)
    z = (1 - alphas) * z_a + alphas * z_b
    with torch.no_grad():
        return G(z).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--mode', choices=['random', 'interp'], default='random')
    p.add_argument('--out', default='artifacts/samples/out.png')
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = load_generator(args.ckpt, cfg['model']['latent_dim'],
                       cfg['model']['ngf'], cfg['model']['channels'], device)

    if args.mode == 'random':
        imgs = random_samples(G, args.n, cfg['model']['latent_dim'], device)
    else:
        imgs = interpolate(G, args.n, cfg['model']['latent_dim'], device)

    save_sample_grid(imgs, args.out, nrow=8)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
