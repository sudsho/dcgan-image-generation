"""DCGAN training loop (alternating G/D updates)."""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_celeba_loader
from src.model import Generator, Discriminator, weights_init
from src.utils import load_config, set_seed, save_sample_grid


def train(cfg):
    device = torch.device(cfg['train']['device']
                          if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['train']['seed'])

    loader = get_celeba_loader(
        root=cfg['data']['root'],
        image_size=cfg['data']['image_size'],
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        subset_size=cfg['data'].get('subset_size'),
    )

    G = Generator(cfg['model']['latent_dim'], cfg['model']['ngf'],
                  cfg['model']['channels']).to(device)
    D = Discriminator(cfg['model']['ndf'],
                      cfg['model']['channels']).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = optim.Adam(G.parameters(), lr=cfg['train']['lr_g'],
                       betas=(cfg['train']['beta1'], cfg['train']['beta2']))
    opt_d = optim.Adam(D.parameters(), lr=cfg['train']['lr_d'],
                       betas=(cfg['train']['beta1'], cfg['train']['beta2']))

    out_dir = cfg['train']['output_dir']
    samples_dir = os.path.join(out_dir, 'samples')
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    fixed_z = torch.randn(64, cfg['model']['latent_dim'], device=device)

    for epoch in range(cfg['train']['epochs']):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            b = real.size(0)

            # ===== D step =====
            opt_d.zero_grad()
            real_labels = torch.full((b,), 1.0, device=device)
            fake_labels = torch.full((b,), 0.0, device=device)
            out_real = D(real)
            loss_d_real = criterion(out_real, real_labels)
            z = torch.randn(b, cfg['model']['latent_dim'], device=device)
            fake = G(z)
            out_fake = D(fake.detach())
            loss_d_fake = criterion(out_fake, fake_labels)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # ===== G step =====
            opt_g.zero_grad()
            out_fake_for_g = D(fake)
            loss_g = criterion(out_fake_for_g, real_labels)
            loss_g.backward()
            opt_g.step()

            if i % cfg['train']['log_interval'] == 0:
                print(f'epoch {epoch} step {i} '
                      f'loss_d {loss_d.item():.4f} '
                      f'loss_g {loss_g.item():.4f}')

        # epoch end: save sample grid + checkpoint
        with torch.no_grad():
            G.eval()
            samples = G(fixed_z).detach().cpu()
            G.train()
        save_sample_grid(samples,
                         os.path.join(samples_dir, f'epoch_{epoch:03d}.png'))
        torch.save({'G': G.state_dict(), 'D': D.state_dict()},
                   os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    args = p.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == '__main__':
    main()
