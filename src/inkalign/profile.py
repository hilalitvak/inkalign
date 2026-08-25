"""Depth-profile analysis: locate the papyrus density ridge in a surface volume.

The core idea: in a correctly rendered surface volume the papyrus sheet is a
bright band centered in the depth (Z) axis. For each spatial tile we average CT
intensity per Z-layer, find the peak of that profile with sub-voxel precision,
and report its offset from the stack center. A consistent nonzero offset means
the whole surface sits too shallow or too deep for the ink models, which expect
the sheet centered — exactly the failure mode the official tutorial tells users
to fix by hand-shifting --layer-start/--layer-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .io import read_block

#: Tiles with less than this fraction of nonzero voxels are outside the segment
#: mask (zarr fill value is 0) and are not scored.
MIN_COVERAGE = 0.2

#: Peaks with prominence (peak minus profile median, in profile-std units) below
#: this are too flat to trust.
MIN_PROMINENCE = 0.8

#: The peak must also beat the profile median by this fraction of the median —
#: std-normalized prominence alone lets pure noise through (a flat profile has a
#: tiny std). Real papyrus ridges show ~25-30% contrast.
MIN_CONTRAST = 0.02

#: Per-tile peak search is confined to this many layers around the segment-wide
#: ridge, so a bright adjacent wrap near the stack edge cannot win.
WRAP_SEARCH_RADIUS = 5.0


@dataclass
class TileResult:
    y0: int
    x0: int
    offset: float | None  # ridge center minus stack center, in layers; None if invalid
    prominence: float | None
    coverage: float
    profile: np.ndarray | None


@dataclass
class ProfileResult:
    shape: tuple[int, int, int]  # (Z, Y, X) of the analyzed level
    tile: int
    stride: int
    tiles: list[TileResult] = field(default_factory=list)

    @property
    def offsets(self) -> np.ndarray:
        return np.array([t.offset for t in self.tiles if t.offset is not None])

    @property
    def median_offset(self) -> float | None:
        offs = self.offsets
        return float(np.median(offs)) if offs.size else None

    def offset_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dense (rows, cols) grids of offsets (NaN where invalid) plus extents."""
        ys = sorted({t.y0 for t in self.tiles})
        xs = sorted({t.x0 for t in self.tiles})
        grid = np.full((len(ys), len(xs)), np.nan)
        yi = {v: i for i, v in enumerate(ys)}
        xi = {v: i for i, v in enumerate(xs)}
        for t in self.tiles:
            if t.offset is not None:
                grid[yi[t.y0], xi[t.x0]] = t.offset
        return grid, np.array(ys), np.array(xs)

    def mean_profile(self) -> np.ndarray | None:
        profs = [t.profile for t in self.tiles if t.profile is not None]
        return np.mean(profs, axis=0) if profs else None

    def recommended_window(self) -> tuple[int, int] | None:
        """(--layer-start, --layer-end) that re-centers the ridge.

        For a stack of Z layers with the ridge at center + m, the window
        [2m, Z) (m > 0) or [0, Z + 2m) (m < 0) has its center on the ridge
        while keeping as many layers as possible.
        """
        m = self.median_offset
        if m is None:
            return None
        z = self.shape[0]
        shift = int(round(2 * m))
        if shift >= 0:
            return min(shift, z - 1), z
        return 0, max(z + shift, 1)


def depth_profile(block: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-layer mean intensity over nonzero voxels, and the nonzero fraction.

    Zero voxels are outside the segment mask (zarr fill value), so a plain mean
    would drag edge tiles toward zero and bias the ridge estimate.
    """
    nonzero = block != 0
    counts = nonzero.reshape(block.shape[0], -1).sum(axis=1)
    sums = np.where(nonzero, block, 0).reshape(block.shape[0], -1).sum(axis=1, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    coverage = float(nonzero.mean())
    return profile, coverage


def locate_ridge(
    profile: np.ndarray,
    smooth_sigma: float = 0.8,
    search_center: float | None = None,
    search_radius: float | None = None,
) -> tuple[float, float] | None:
    """Sub-voxel ridge position and prominence, or None if no credible peak.

    The peak must be an interior maximum: a profile that only rises toward an
    edge (the sheet cut off by the stack boundary) has no localizable ridge.

    When `search_center`/`search_radius` are given, the peak is only sought in
    that layer window. Surface volumes often show the *adjacent* papyrus wrap
    near the stack edges; without the restriction a tile can lock onto the
    neighboring sheet and report a wildly wrong offset.
    """
    if np.isnan(profile).any():
        return None
    sm = gaussian_filter1d(profile.astype(np.float64), smooth_sigma)
    lo, hi = 1, len(sm) - 1
    if search_center is not None and search_radius is not None:
        lo = max(lo, int(np.floor(search_center - search_radius)))
        hi = min(hi, int(np.ceil(search_center + search_radius)) + 1)
        if hi - lo < 3:
            return None
    interior = sm[lo:hi]
    peak = int(np.argmax(interior)) + lo
    # Must be a strict local maximum: a profile that only rises toward an edge
    # (sheet cut off by the stack boundary) has no localizable ridge there.
    if not (sm[peak] > sm[peak - 1] and sm[peak] > sm[peak + 1]):
        return None
    std = float(sm.std())
    if std == 0:
        return None
    med = float(np.median(sm))
    prominence = (sm[peak] - med) / std
    if prominence < MIN_PROMINENCE:
        return None
    if med <= 0 or (sm[peak] - med) / med < MIN_CONTRAST:
        return None
    p1, p2, p3 = sm[peak - 1 : peak + 2]
    denom = p1 - 2 * p2 + p3
    subvoxel = peak + (0.5 * (p1 - p3) / denom if denom != 0 else 0.0)
    return float(subvoxel), prominence


def profile_volume(
    arr,
    tile: int = 128,
    max_tiles: int = 200,
    keep_profiles: bool = True,
    progress=None,
) -> ProfileResult:
    """Sample tiles across the segment and estimate the ridge offset in each.

    Tiles are chunk-aligned and sampled on a regular grid with a stride chosen
    so at most `max_tiles` tiles are read — full coverage of a large segment is
    rarely needed to see whether the surface is systematically off-center.
    """
    z, y, x = arr.shape
    center = (z - 1) / 2

    ny, nx = max(y // tile, 1), max(x // tile, 1)
    stride = 1
    while (ny // stride + 1) * (nx // stride + 1) > max_tiles:
        stride += 1

    result = ProfileResult(shape=(z, y, x), tile=tile, stride=stride)
    positions = [
        (ty * tile, tx * tile)
        for ty in range(0, ny, stride)
        for tx in range(0, nx, stride)
    ]

    # Pass 1: stream every tile once and keep its depth profile.
    profiles: list[tuple[int, int, np.ndarray | None, float]] = []
    for i, (y0, x0) in enumerate(positions):
        block = read_block(arr, y0, x0, tile, tile)
        prof, coverage = depth_profile(block)
        profiles.append((y0, x0, prof if coverage >= MIN_COVERAGE else None, coverage))
        if progress:
            progress(i + 1, len(positions))

    # Global ridge from the segment-wide mean profile anchors the per-tile
    # search, so tiles cannot lock onto the adjacent papyrus wrap.
    valid = [p for _, _, p, _ in profiles if p is not None and not np.isnan(p).any()]
    anchor = locate_ridge(np.mean(valid, axis=0)) if valid else None

    # Pass 2: per-tile ridge, restricted around the anchor when we have one.
    for y0, x0, prof, coverage in profiles:
        offset = prominence = None
        if prof is not None:
            ridge = locate_ridge(
                prof,
                search_center=anchor[0] if anchor else None,
                search_radius=WRAP_SEARCH_RADIUS if anchor else None,
            )
            if ridge is not None:
                offset = ridge[0] - center
                prominence = ridge[1]
        result.tiles.append(
            TileResult(y0, x0, offset, prominence, coverage, prof if keep_profiles else None)
        )
    return result
