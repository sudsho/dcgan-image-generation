# dcgan-image-generation

Deep Convolutional GAN trained on a CelebA face subset. Generates 64x64 face
samples and short training animations.

## Plan

1. Get CelebA aligned subset, center-crop, normalize to [-1, 1].
2. Build Generator (transposed convs, BN, ReLU) and Discriminator (strided
   convs, LeakyReLU, no BN on input/output).
3. Adam with beta1=0.5, lr=2e-4, batch 128, ~25 epochs.
4. Save sample grid every epoch, stitch into a GIF.
5. FID against held-out set.
6. Wrap a small FastAPI demo that returns generated faces as base64.

## Notes

- DCGAN paper: Radford, Metz, Chintala 2016.
- Watch for mode collapse. Maybe try label smoothing / one-sided.
