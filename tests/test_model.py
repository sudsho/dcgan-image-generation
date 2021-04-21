"""Smoke tests for Generator + Discriminator output shapes."""
import pytest
import torch

from src.model import Generator, Discriminator, weights_init


@pytest.fixture(autouse=True)
def fix_seed():
    # tests were occasionally flaky on a few platforms; deterministic now
    torch.manual_seed(0)


def test_generator_shape():
    G = Generator(latent_dim=100, ngf=64, channels=3)
    z = torch.randn(2, 100)
    out = G(z)
    assert out.shape == (2, 3, 64, 64)


def test_generator_accepts_4d_z():
    G = Generator(latent_dim=100, ngf=64, channels=3)
    z = torch.randn(2, 100, 1, 1)
    out = G(z)
    assert out.shape == (2, 3, 64, 64)


def test_discriminator_shape():
    D = Discriminator(ndf=64, channels=3)
    x = torch.randn(2, 3, 64, 64)
    out = D(x)
    assert out.shape == (2,)


def test_weights_init_runs():
    G = Generator()
    G.apply(weights_init)
    # just confirms apply doesn't blow up across submodules
