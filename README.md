# dcgan-image-generation

Deep Convolutional GAN trained on a CelebA face subset. Outputs 64x64 face
samples, training animations and a small FastAPI demo.

## What it does

Given random noise z ~ N(0, I) of size 100, generate 64x64 face images that
look like CelebA samples. We train a DCGAN (Radford et al. 2016) on a
30k-image CelebA subset, save sample grids each epoch, stitch them into a
GIF, and report FID against held-out real samples.

## Architecture

```
G:  z(100) -> ConvT(4x4, s1) -> BN, ReLU
        -> ConvT(4x4, s2) -> BN, ReLU   x4
        -> Conv(3 channels) -> Tanh    output 3x64x64

D:  3x64x64 -> Conv(s2), LeakyReLU(0.2)             (no BN at input)
            -> Conv(s2), BN, LeakyReLU(0.2)   x3
            -> Conv -> 1                            (no BN at output)
```

Tricks used:
- BN in G but **not** in D's input layer, **not** in D's output (logit) layer
- LeakyReLU(0.2) in D, ReLU in G, Tanh on G output
- Adam(lr=2e-4, beta1=0.5)
- One-sided label smoothing: real labels are 0.9 instead of 1.0
- Weight init N(0, 0.02) on Conv/ConvTranspose

## Results

After 25 epochs on the 30k CelebA subset:

- FID (5k vs 5k): ~ 35 (reasonable for 64x64, no big tricks)
- Training animation: ![training](artifacts/training.gif)
- Latent interpolation: ![interp](artifacts/interp.png)

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

## Build the training GIF

```
make sample          # optional: extra samples
python -m src.anim --samples-dir artifacts/samples --out artifacts/training.gif
```

## FID

```
python -m src.fid --config configs/default.yaml --ckpt artifacts/checkpoints/epoch_024.pt
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

The Dockerfile builds a CPU image with the trained generator baked in. See
`docker-compose.yml`. CI config lives at `ci/test.yml.example` (copy to
`.github/workflows/test.yml` once the OAuth scope allows workflow files).

## License

MIT.
