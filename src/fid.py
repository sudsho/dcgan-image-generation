"""Frechet Inception Distance using torchvision InceptionV3 features.

FID = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2*sqrt(C_r * C_g))

We pull the 2048-d pool3 features from a pretrained InceptionV3, compute
mean and covariance for real vs generated, then plug into the formula.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torchvision.models import inception_v3
from torchvision import transforms

from src.data import get_celeba_loader
from src.model import Generator
from src.utils import load_config


class InceptionFeatureExtractor(nn.Module):
    """Returns the 2048-d pool3 vector."""
    def __init__(self):
        super().__init__()
        net = inception_v3(pretrained=True, aux_logits=False)
        net.fc = nn.Identity()
        net.eval()
        self.net = net
        self.up = nn.Upsample(size=(299, 299), mode='bilinear',
                              align_corners=False)
        # FID uses ImageNet mean/std after rescale to [0, 1]
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def forward(self, x):
        # x in [-1, 1]; rescale to [0, 1] then normalize
        x = (x.clamp(-1, 1) + 1) / 2.0
        x = self.up(x)
        x = torch.stack([self.norm(xi) for xi in x])
        return self.net(x)


def gather_features(extractor, batches, device):
    feats = []
    for x in batches:
        x = x.to(device)
        f = extractor(x)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def calc_stats(feats):
    mu = np.mean(feats, axis=0)
    cov = np.cov(feats, rowvar=False)
    return mu, cov


def fid_from_stats(mu_r, cov_r, mu_g, cov_g, eps=1e-6):
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(cov_r.dot(cov_g), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_r.shape[0]) * eps
        covmean = linalg.sqrtm((cov_r + offset).dot(cov_g + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    return float(diff.dot(diff) + np.trace(cov_r) + np.trace(cov_g)
                 - 2 * np.trace(covmean))


def compute_fid(G, loader, latent_dim, num_real, num_fake, device):
    extractor = InceptionFeatureExtractor().to(device).eval()

    real_batches, n_real = [], 0
    for x, _ in loader:
        real_batches.append(x)
        n_real += x.size(0)
        if n_real >= num_real:
            break
    real_feats = gather_features(extractor, real_batches, device)[:num_real]

    fake_batches, n_fake = [], 0
    was_training = G.training
    G.eval()
    try:
        with torch.no_grad():
            while n_fake < num_fake:
                b = min(64, num_fake - n_fake)
                z = torch.randn(b, latent_dim, device=device)
                fake_batches.append(G(z).cpu())
                n_fake += b
    finally:
        if was_training:
            G.train()
    fake_feats = gather_features(extractor, fake_batches, device)[:num_fake]

    mu_r, cov_r = calc_stats(real_feats)
    mu_g, cov_g = calc_stats(fake_feats)
    return fid_from_stats(mu_r, cov_r, mu_g, cov_g)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--ckpt', required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(cfg['model']['latent_dim'], cfg['model']['ngf'],
                  cfg['model']['channels']).to(device)
    state = torch.load(args.ckpt, map_location=device)
    G.load_state_dict(state['G'])
    loader = get_celeba_loader(
        cfg['data']['root'], cfg['data']['image_size'],
        batch_size=64, num_workers=cfg['data']['num_workers'],
        subset_size=cfg['fid']['num_real'],
    )
    fid = compute_fid(G, loader, cfg['model']['latent_dim'],
                      cfg['fid']['num_real'], cfg['fid']['num_fake'], device)
    print(f'FID: {fid:.4f}')


if __name__ == '__main__':
    main()
