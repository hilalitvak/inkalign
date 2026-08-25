"""Open Vesuvius surface volumes from local paths or remote zarr stores.

Surface volumes are OME-zarr groups whose level-0 array is (Z, Y, X) with Z the
depth axis perpendicular to the papyrus surface (typically 28 layers at ~9 um).
Remote stores are streamed chunk-by-chunk; nothing is downloaded in bulk.

dl.ash2txt.org rate-limits concurrent chunk fetches (HTTP 429), so all remote
reads go through `read_block`, which retries with exponential backoff.
"""

from __future__ import annotations

import time

import numpy as np
import zarr


#: s3://bucket/key -> anonymous HTTPS endpoint (works for the open-data bucket
#: without any AWS credentials or s3fs).
def _s3_to_https(url: str) -> str:
    rest = url[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def open_surface_volume(path: str, level: int | str = 0) -> zarr.Array:
    """Open a surface volume and return its (Z, Y, X) array at `level`.

    `path` may be a local directory, an https:// URL, or an s3:// URL, and may
    point either at the OME-zarr group or directly at one resolution level.
    """
    if path.startswith("s3://"):
        path = _s3_to_https(path)

    store = zarr.storage.FSStore(path)
    try:
        node = zarr.open(store, mode="r")
    except zarr.errors.PathNotFoundError:
        # HTTP stores cannot list directories, so zarr.open on a group URL can
        # fail even though `<url>/<level>` is a valid array. Fall through.
        node = None

    if isinstance(node, zarr.Array):
        return node

    # Group (or unlistable remote group): address the level explicitly.
    arr = zarr.open_array(zarr.storage.FSStore(f"{path.rstrip('/')}/{level}"), mode="r")
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D (Z, Y, X) surface volume, got shape {arr.shape}")
    return arr


def read_block(
    arr: zarr.Array,
    y0: int,
    x0: int,
    height: int,
    width: int,
    retries: int = 5,
    backoff: float = 2.0,
) -> np.ndarray:
    """Read arr[:, y0:y0+height, x0:x0+width], retrying on transient HTTP errors.

    Callers should keep blocks chunk-aligned where possible so each read maps to
    whole chunks and no chunk is fetched twice.
    """
    for attempt in range(retries + 1):
        try:
            return arr[:, y0 : y0 + height, x0 : x0 + width]
        except Exception as exc:  # aiohttp raises several exception types for 429/5xx
            transient = "429" in str(exc) or "503" in str(exc) or "timeout" in str(exc).lower()
            if not transient or attempt == retries:
                raise
            time.sleep(backoff * (2**attempt))
    raise RuntimeError("unreachable")
