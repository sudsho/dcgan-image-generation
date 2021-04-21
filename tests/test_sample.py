"""Tests for sample.random_samples and sample.interpolate."""
import torch

from src.model import Generator
from src.sample import random_samples, interpolate


def test_random_samples_shape():
    G = Generator(latent_dim=100).eval()
    out = random_samples(G, 8, 100, torch.device('cpu'))
    assert out.shape == (8, 3, 64, 64)


def test_interpolate_endpoints_match():
    G = Generator(latent_dim=100).eval()
    z_a = torch.zeros(1, 100)
    z_b = torch.ones(1, 100)
    out = interpolate(G, 5, 100, torch.device('cpu'), z_a=z_a, z_b=z_b)
    assert out.shape == (5, 3, 64, 64)
    # first frame should be G(z_a), last G(z_b) -- check via re-eval
    with torch.no_grad():
        first = G(z_a)
        last = G(z_b)
    assert torch.allclose(out[0], first[0], atol=1e-5)
    assert torch.allclose(out[-1], last[0], atol=1e-5)
