"""API smoke tests using a tiny on-the-fly Generator checkpoint."""
import base64
import os
import tempfile

import pytest
import torch
from fastapi.testclient import TestClient


def _write_ckpt(path):
    from src.model import Generator, Discriminator
    G = Generator(latent_dim=100, ngf=64, channels=3)
    D = Discriminator(ndf=64, channels=3)
    torch.save({'G': G.state_dict(), 'D': D.state_dict()}, path)


@pytest.fixture
def client(tmp_path, monkeypatch):
    ckpt = tmp_path / 'tiny.pt'
    _write_ckpt(str(ckpt))
    monkeypatch.setenv('DCGAN_CKPT', str(ckpt))
    # import here so env is read
    import importlib
    import src.api.main as m
    importlib.reload(m)
    return TestClient(m.app)


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_generate(client):
    r = client.get('/generate', params={'n': 4})
    assert r.status_code == 200
    body = r.json()
    assert body['n'] == 4
    raw = base64.b64decode(body['png_b64'])
    assert raw[:8] == b'\x89PNG\r\n\x1a\n'


def test_generate_bad_n(client):
    r = client.get('/generate', params={'n': 0})
    assert r.status_code == 400
    r = client.get('/generate', params={'n': 999})
    assert r.status_code == 400


def test_interpolate(client):
    r = client.get('/interpolate', params={'n': 6})
    assert r.status_code == 200
    body = r.json()
    assert body['n'] == 6
    raw = base64.b64decode(body['png_b64'])
    assert raw[:8] == b'\x89PNG\r\n\x1a\n'
