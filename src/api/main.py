"""FastAPI server: /generate and /interpolate endpoints.

The Generator checkpoint path is read from env DCGAN_CKPT (defaults to
artifacts/checkpoints/latest.pt).
"""
import os

import torch
from fastapi import FastAPI, HTTPException

from src.api.schemas import GenerateResponse, HealthResponse
from src.model import Generator
from src.sample import interpolate, random_samples
from src.utils import tensor_grid_to_b64


app = FastAPI(title='DCGAN demo')

CKPT_PATH = os.environ.get('DCGAN_CKPT', 'artifacts/checkpoints/latest.pt')
LATENT_DIM = int(os.environ.get('DCGAN_LATENT_DIM', 100))
NGF = int(os.environ.get('DCGAN_NGF', 64))
CHANNELS = 3

_state = {'G': None, 'device': None}


def get_generator():
    # cache hit
    if _state['G'] is not None:
        return _state['G'], _state['device']
    # re-read CKPT_PATH on every call so tests can override via env
    ckpt_path = os.environ.get('DCGAN_CKPT', CKPT_PATH)
    if not os.path.exists(ckpt_path):
        raise HTTPException(status_code=503,
                            detail=f'checkpoint not found at {ckpt_path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(LATENT_DIM, NGF, CHANNELS).to(device)
    state = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(state['G'])
    G.eval()
    _state['G'] = G
    _state['device'] = device
    return G, device


@app.get('/health', response_model=HealthResponse)
def health():
    return HealthResponse(status='ok', ckpt_present=os.path.exists(CKPT_PATH))


@app.get('/generate', response_model=GenerateResponse)
def generate(n: int = 16):
    if n < 1 or n > 64:
        raise HTTPException(400, 'n must be in [1, 64]')
    G, device = get_generator()
    imgs = random_samples(G, n, LATENT_DIM, device)
    return GenerateResponse(n=n, png_b64=tensor_grid_to_b64(imgs, nrow=4))


@app.get('/interpolate', response_model=GenerateResponse)
def interpolate_endpoint(n: int = 10):
    if n < 2 or n > 32:
        raise HTTPException(400, 'n must be in [2, 32]')
    G, device = get_generator()
    imgs = interpolate(G, n, LATENT_DIM, device)
    return GenerateResponse(n=n, png_b64=tensor_grid_to_b64(imgs, nrow=n))
