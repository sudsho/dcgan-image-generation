"""DCGAN training loop (alternating G/D updates).

Tricks used:
- One-sided label smoothing: real labels are (1 - eps).
- Adam beta1 = 0.5 (DCGAN paper).
- Track losses with MLflow if available.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.data import get_celeba_loader
from src.model import Generator, Discriminator, weights_init
from src.utils import load_config, set_seed, save_sample_grid
from src.fid import compute_fid


def train(cfg):
    device = torch.device(cfg['train']['device']
                          if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['train']['seed'])
    smoothing = float(cfg['train'].get('label_smoothing', 0.0))
    flip_p = float(cfg['train'].get('label_flip_prob', 0.0))

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

    if HAS_MLFLOW:
        mlflow.start_run()
        for k in ('lr_g', 'lr_d', 'beta1', 'epochs', 'label_smoothing'):
            if k in cfg['train']:
                mlflow.log_param(k, cfg['train'][k])
        mlflow.log_param('batch_size', cfg['data']['batch_size'])
        mlflow.log_param('latent_dim', cfg['model']['latent_dim'])

    for epoch in range(cfg['train']['epochs']):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            b = real.size(0)

            # ===== D step =====
            opt_d.zero_grad()
            # one-sided label smoothing: real -> 1 - smoothing
            real_labels = torch.full((b,), 1.0 - smoothing, device=device)
            fake_labels = torch.full((b,), 0.0, device=device)
            # occasionally flip a small fraction of D's labels - helps when
            # D dominates and G stops getting useful gradients (mode collapse)
            if flip_p > 0:
                flip_mask = (torch.rand(b, device=device) < flip_p)
                real_labels[flip_mask] = 0.0
                fake_labels[flip_mask] = 1.0 - smoothing
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
                if HAS_MLFLOW:
                    step = epoch * len(loader) + i
                    mlflow.log_metric('loss_d', loss_d.item(), step=step)
                    mlflow.log_metric('loss_g', loss_g.item(), step=step)

        # epoch end: save sample grid + checkpoint
        with torch.no_grad():
            G.eval()
            samples = G(fixed_z).detach().cpu()
            G.train()
        save_sample_grid(samples,
                         os.path.join(samples_dir, f'epoch_{epoch:03d}.png'))
        ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt')
        torch.save({'G': G.state_dict(), 'D': D.state_dict()}, ckpt_path)
        # also keep a stable "latest" pointer for the API to pick up
        latest = os.path.join(ckpt_dir, 'latest.pt')
        try:
            if os.path.exists(latest):
                os.remove(latest)
            torch.save({'G': G.state_dict(), 'D': D.state_dict()}, latest)
        except OSError:
            pass

        # FID every few epochs - it's slow
        fid_every = int(cfg.get('fid', {}).get('every_epochs', 5))
        if cfg.get('fid', {}).get('enabled') and (epoch + 1) % fid_every == 0:
            try:
                fid = compute_fid(
                    G, loader, cfg['model']['latent_dim'],
                    cfg['fid']['num_real'], cfg['fid']['num_fake'], device)
                print(f'  FID @ epoch {epoch}: {fid:.4f}')
                if HAS_MLFLOW:
                    mlflow.log_metric('fid', fid, step=epoch)
            except Exception as e:
                print(f'  FID failed: {e}')

    if HAS_MLFLOW:
        mlflow.end_run()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    args = p.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == '__main__':
    main()
