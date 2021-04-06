"""Build a GIF from per-epoch sample grids."""
import argparse
import glob
import os
import imageio


def build_gif(samples_dir, out_path, fps=4):
    paths = sorted(glob.glob(os.path.join(samples_dir, 'epoch_*.png')))
    if not paths:
        raise FileNotFoundError(f'no epoch_*.png under {samples_dir}')
    frames = [imageio.imread(p) for p in paths]
    imageio.mimsave(out_path, frames, fps=fps)
    print(f'wrote {out_path} ({len(frames)} frames)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples-dir', default='artifacts/samples')
    p.add_argument('--out', default='artifacts/training.gif')
    p.add_argument('--fps', type=int, default=4)
    args = p.parse_args()
    build_gif(args.samples_dir, args.out, args.fps)


if __name__ == '__main__':
    main()
