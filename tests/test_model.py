"""Smoke tests for Generator + Discriminator output shapes."""
import torch

from src.model import Generator, Discriminator


def test_generator_shape():
    G = Generator(latent_dim=100, ngf=64, channels=3)
    z = torch.randn(2, 100)
    out = G(z)
    assert out.shape == (2, 3, 64, 64)


def test_discriminator_shape():
    D = Discriminator(ndf=64, channels=3)
    x = torch.randn(2, 3, 64, 64)
    out = D(x)
    assert out.shape == (2,)
