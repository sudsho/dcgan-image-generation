# dcgan-image-generation

Deep Convolutional GAN (Radford et al. 2016) implemented in PyTorch, with a
small FastAPI demo that serves generated samples. Trains on a CelebA face
subset at 64x64.

## Quick start (tiny-CPU smoke, no GPU/download)

You can prove the whole thing runs end to end on a CPU with no CelebA download
and no GPU. It builds a tiny generator + discriminator, trains a few adversarial
steps on tiny synthetic images, then generates a batch from noise and checks the
ConvTranspose output dimensions.

```
python scripts/smoke.py
# or: make smoke
```

Real output:

```
device: cpu (cuda available: True, forced off for smoke)
[1/3] adversarial loop (synthetic tensors, CPU)
      step 0 loss_d 1.3482 loss_g 1.0785
      step 1 loss_d 1.4580 loss_g 0.9699
      step 2 loss_d 1.3633 loss_g 1.0438
      step 3 loss_d 1.3585 loss_g 1.0533
      step 4 loss_d 1.3120 loss_g 0.9716
      generator weight max-delta after 5 steps: 1.138e-03
      losses moved: loss_d 1.3482 -> 1.3120, loss_g 1.0785 -> 0.9716
[2/3] real training entrypoint (src.train.train, 1 epoch, CPU)
epoch 0 step 0 loss_d 1.5718 loss_g 0.8867
epoch 0 step 1 loss_d 1.3758 loss_g 1.0750
      wrote checkpoint out\checkpoints\epoch_000.pt and sample grid out\samples\epoch_000.png
[3/3] generate from noise + assert ConvTranspose output dims
      generated batch shape: (4, 3, 64, 64)
      ConvTranspose output dims OK (1x1 noise -> 64x64 image)
SMOKE OK
```

Tests (the shape, data, sample, FID-math, and API tests) run on CPU too:

```
python -m pytest -q
# 25 passed, 1 deselected
```

The deselected test is the slower one-epoch training smoke; run it with
`python -m pytest -q -m slow` (also CPU, uses synthetic images).

This smoke only proves the code runs. The headline result, realistic 64x64
faces, needs a GPU and the real CelebA dataset (see Train below). Expect many
epochs on a GPU before samples look like faces.

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
