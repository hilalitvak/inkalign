"""Unit tests on synthetic volumes (development only — the submission's evidence
comes from real scroll data, per villa's CONTRIBUTING.md)."""

import numpy as np
import pytest

from inkalign.profile import depth_profile, locate_ridge, profile_volume


def make_volume(z=28, y=256, x=256, ridge_at=13.5, width=3.0, noise=2.0, seed=0):
    """Synthetic surface volume: gaussian bright band at `ridge_at` plus noise."""
    rng = np.random.default_rng(seed)
    zz = np.arange(z, dtype=np.float64)
    band = 60 + 40 * np.exp(-0.5 * ((zz - ridge_at) / width) ** 2)
    vol = np.broadcast_to(band[:, None, None], (z, y, x)) + rng.normal(0, noise, (z, y, x))
    return np.clip(vol, 1, 255).astype(np.uint8)  # min 1: zero means "outside mask"


def test_locate_ridge_centered():
    vol = make_volume(ridge_at=13.5)
    prof, cov = depth_profile(vol)
    assert cov == 1.0
    pos, prom = locate_ridge(prof)
    assert abs(pos - 13.5) < 0.3
    assert prom > MIN_PROM_SANITY


MIN_PROM_SANITY = 1.0


@pytest.mark.parametrize("ridge_at", [9.0, 11.5, 13.5, 16.0, 18.0])
def test_locate_ridge_offsets(ridge_at):
    vol = make_volume(ridge_at=ridge_at)
    prof, _ = depth_profile(vol)
    pos, _ = locate_ridge(prof)
    assert abs(pos - ridge_at) < 0.4


def test_edge_ridge_rejected():
    """A profile only rising toward the stack edge has no localizable ridge."""
    vol = make_volume(ridge_at=27.0)
    prof, _ = depth_profile(vol)
    assert locate_ridge(prof) is None


def test_flat_profile_rejected():
    rng = np.random.default_rng(1)
    vol = np.clip(rng.normal(80, 2.0, (28, 128, 128)), 1, 255).astype(np.uint8)
    prof, _ = depth_profile(vol)
    assert locate_ridge(prof) is None


def test_masked_tile_skipped():
    """Tiles mostly outside the segment mask (zeros) are not scored."""
    vol = make_volume(y=128, x=128)
    vol[:, :, :] = 0
    prof, cov = depth_profile(vol)
    assert cov == 0.0


def test_profile_volume_recovers_global_offset():
    vol = make_volume(z=28, y=512, x=512, ridge_at=16.0)
    result = profile_volume(vol, tile=128, max_tiles=50)
    m = result.median_offset
    assert m is not None
    assert abs(m - (16.0 - 13.5)) < 0.4
    start, end = result.recommended_window()
    # window [start, end) must be centered on the ridge, within a layer
    assert abs((start + end - 1) / 2 - 16.0) < 1.0


def test_adjacent_wrap_does_not_hijack_tiles():
    """A tile whose adjacent-wrap band is brighter than its own sheet must still
    report the sheet near the global ridge, not the wrap at the stack edge."""
    vol = make_volume(z=28, y=512, x=512, ridge_at=13.5)
    # one tile gets a brighter second band near the far edge (the next wrap)
    zz = np.arange(28, dtype=np.float64)
    wrap = 80 * np.exp(-0.5 * ((zz - 25.0) / 1.5) ** 2)
    tile = vol[:, :128, :128].astype(np.float64) + wrap[:, None, None]
    vol[:, :128, :128] = np.clip(tile, 1, 255).astype(np.uint8)

    result = profile_volume(vol, tile=128, max_tiles=50)
    for t in result.tiles:
        if t.offset is not None:
            assert abs(t.offset) < 5.0, f"tile ({t.y0},{t.x0}) locked onto the wrap: {t.offset}"


def test_zero_mean_handling_in_partial_tiles():
    """Half-masked tile: profile must come from the valid half only."""
    vol = make_volume(y=128, x=128, ridge_at=13.5)
    vol[:, :, 64:] = 0
    prof, cov = depth_profile(vol)
    assert 0.4 < cov < 0.6
    pos, _ = locate_ridge(prof)
    assert abs(pos - 13.5) < 0.4
