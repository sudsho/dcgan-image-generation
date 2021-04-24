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


def test_interpolate_n2_yields_endpoints_only():
    """Edge case: n=2 should just be [G(z_a), G(z_b)]."""
    G = Generator(latent_dim=100).eval()
    z_a = torch.randn(1, 100)
    z_b = torch.randn(1, 100)
    out = interpolate(G, 2, 100, torch.device('cpu'), z_a=z_a, z_b=z_b)
    assert out.shape == (2, 3, 64, 64)
    with torch.no_grad():
        a = G(z_a)
        b = G(z_b)
    assert torch.allclose(out[0], a[0], atol=1e-5)
    assert torch.allclose(out[1], b[0], atol=1e-5)


def test_random_samples_distinct():
    """Different draws should produce different images."""
    torch.manual_seed(0)
    G = Generator(latent_dim=100).eval()
    # need to advance the rng between calls so the two batches don't collide
    a = random_samples(G, 4, 100, torch.device('cpu'))
    torch.manual_seed(1)
    b = random_samples(G, 4, 100, torch.device('cpu'))
    # extremely unlikely to be the same
    diff = (a - b).abs().mean().item()
    assert diff > 1e-2, f'samples too similar (mean abs diff {diff})'
