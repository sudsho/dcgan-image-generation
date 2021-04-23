"""Math-only tests for FID computation. Doesn't load InceptionV3."""
import numpy as np

from src.fid import calc_stats, fid_from_stats


def test_fid_zero_for_identical_distributions():
    rng = np.random.RandomState(0)
    x = rng.randn(200, 16)
    mu, cov = calc_stats(x)
    fid = fid_from_stats(mu, cov, mu, cov)
    assert abs(fid) < 1e-4


def test_fid_positive_for_shifted_means():
    rng = np.random.RandomState(0)
    x = rng.randn(500, 8)
    y = x + 3.0  # shift mean
    mu_x, cov_x = calc_stats(x)
    mu_y, cov_y = calc_stats(y)
    fid = fid_from_stats(mu_x, cov_x, mu_y, cov_y)
    # mean shift = 3 in 8 dims -> fid >= 8 * 9 = 72
    assert fid > 60
