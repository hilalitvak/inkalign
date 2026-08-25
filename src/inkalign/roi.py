"""Extract a small ROI from a surface volume into a local zarr.

Sweeping inference over many depth windows multiplies GPU cost, so the sweep
should run on a small region, not the whole segment. The extracted ROI keeps
the (Z, Y, X) layout and OME-style multiscale group structure that
`vesuvius.ink_detection.inference.infer` expects, with the full depth stack
preserved (depth is what we're sweeping).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from .io import read_block


def extract_roi(
    arr: zarr.Array,
    y0: int,
    x0: int,
    height: int,
    width: int,
    outpath: str | Path,
) -> Path:
    """Copy arr[:, y0:y0+height, x0:x0+width] into a local multiscale zarr."""
    outpath = Path(outpath)
    z, y, x = arr.shape
    y0, x0 = max(0, y0), max(0, x0)
    height = min(height, y - y0)
    width = min(width, x - x0)
    if height <= 0 or width <= 0:
        raise ValueError("ROI is empty — check y0/x0 against the volume shape")

    block = read_block(arr, y0, x0, height, width)

    root = zarr.open_group(str(outpath), mode="w")
    root.attrs["multiscales"] = [
        {
            "axes": [{"name": n, "type": "space"} for n in ("z", "y", "x")],
            "datasets": [{"path": "0"}],
            "version": "0.4",
        }
    ]
    root.attrs["inkalign_roi"] = {"source_offset_yx": [int(y0), int(x0)]}
    root.create_dataset(
        "0",
        data=np.ascontiguousarray(block),
        chunks=(z, min(128, height), min(128, width)),
        dtype=block.dtype,
    )
    return outpath


def pick_inked_roi(arr: zarr.Array, size: int = 768) -> tuple[int, int]:
    """Heuristic ROI placement: the segment's central region, clamped to bounds.

    The center of a rendered segment is the most likely place to contain
    papyrus (and therefore possibly ink); callers with better knowledge —
    e.g. a region where letters are already suspected — should pass explicit
    coordinates instead.
    """
    _, y, x = arr.shape
    return max(0, (y - size) // 2), max(0, (x - size) // 2)
