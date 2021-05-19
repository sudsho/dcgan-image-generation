# dcgan-image-generation

Deep Convolutional GAN (Radford et al. 2016) implemented in PyTorch, with a
small FastAPI demo that serves generated samples. Trains on a CelebA face
subset at 64x64.

## What it does

Given random noise z ~ N(0, I) of size 100, generate 64x64 face images from
CelebA. The repo covers the training loop, sample saving per epoch, a GIF
builder, an FID script, and a tiny inference API. No trained checkpoints are
committed; you train it yourself on your own CelebA copy.

## Architecture

```
G:  z(100, 1x1) -> ConvT(4x4, s1) -> BN, ReLU          (1 -> 4)
             -> ConvT(4x4, s2) -> BN, ReLU  x3         (4 -> 8 -> 16 -> 32)
             -> ConvT(4x4, s2) -> Tanh                 (32 -> 64, 3 channels)

D:  3x64x64 -> Conv(s2), LeakyReLU(0.2)                (no BN at input)
            -> Conv(s2), BN, LeakyReLU(0.2)  x3
            -> Conv -> 1                               (no BN at output)
```

Tricks used:
- BN in G but not in D's input layer, and not in D's output (logit) layer
- LeakyReLU(0.2) in D, ReLU in G, Tanh on G output
- Adam(lr=2e-4, beta1=0.5)
- One-sided label smoothing: real labels are 0.9 instead of 1.0
- Weight init N(0, 0.02) on Conv/ConvTranspose

## Setup

```
pip install -r requirements.txt
```

Place CelebA aligned faces under `data/celeba/img/*.jpg` (the loader uses
ImageFolder so the inner `img/` directory is required).

## Train

```
make train
# or
python -m src.train --config configs/default.yaml
```

Per-epoch sample grids land under `artifacts/samples/epoch_XXX.png` and
checkpoints under `artifacts/checkpoints/epoch_XXX.pt`. Neither is checked
into git.

## Build the training GIF

```
python -m src.anim --samples-dir artifacts/samples --out artifacts/training.gif
```

## FID

Requires a trained checkpoint. Note that FID is computed against a subset of
the same training data, not a held-out split, so the number is only useful
for comparing runs against each other in this repo. It is not directly
comparable to published FID numbers (this implementation uses torchvision's
ImageNet-trained InceptionV3, not the TF-ported pt_inception used by the
standard FID reference).

```
python -m src.fid --config configs/default.yaml --ckpt artifacts/checkpoints/epoch_024.pt
```

## Sample from a checkpoint

```
python -m src.sample --ckpt artifacts/checkpoints/epoch_024.pt --n 64
```

## API demo

```
make api
# then GET http://localhost:8000/generate?n=16
#      GET http://localhost:8000/interpolate?n=10
```

Response is `{"png_b64": "..."}` so the client can decode and display.

## Layout

```
.
+-- configs/default.yaml
+-- src/
|   +-- data.py     CelebA dataloader
|   +-- model.py    G + D
|   +-- train.py    training loop
|   +-- sample.py   random + interp
|   +-- anim.py     GIF builder
|   +-- fid.py      FID metric
|   +-- utils.py
|   +-- api/main.py FastAPI app
+-- tests/          shape + API tests
+-- notebooks/exploration.ipynb
+-- Dockerfile, docker-compose.yml
+-- ci/test.yml.example
```

## Deploy

```
docker build -t dcgan-api .
docker run --rm -p 8000:8000 \
    -v $(pwd)/artifacts:/app/artifacts:ro \
    dcgan-api
```

Or with compose:

```
docker-compose up --build
```

The image runs uvicorn on port 8000. Mount `artifacts/` read-only so the
trained checkpoint at `artifacts/checkpoints/latest.pt` is visible inside
the container. The CKPT path can be overridden via the `DCGAN_CKPT` env
var.

## License

MIT.
