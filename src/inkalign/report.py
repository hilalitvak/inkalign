"""Plots and summaries for depth-profile results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .profile import ProfileResult


def save_offset_heatmap(result: ProfileResult, path: Path) -> None:
    grid, ys, xs = result.offset_grid()
    z = result.shape[0]
    span = max(2.0, float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 2.0)
    fig, ax = plt.subplots(figsize=(8, 8 * grid.shape[0] / max(grid.shape[1], 1)))
    im = ax.imshow(
        grid,
        cmap="RdBu_r",
        vmin=-span,
        vmax=span,
        extent=(xs[0], xs[-1] + result.tile, ys[-1] + result.tile, ys[0]),
        interpolation="nearest",
    )
    ax.set_title(
        f"Papyrus ridge offset from stack center (layers)\n"
        f"stack depth {z}, median offset "
        f"{result.median_offset:+.2f}" if result.median_offset is not None else "no valid tiles"
    )
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="offset (layers); blue = shallow, red = deep")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_profile_curve(result: ProfileResult, path: Path) -> None:
    prof = result.mean_profile()
    if prof is None:
        return
    z = result.shape[0]
    center = (z - 1) / 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(z), prof, marker="o", ms=3, label="mean intensity")
    ax.axvline(center, ls="--", c="gray", label=f"stack center ({center:.1f})")
    m = result.median_offset
    if m is not None:
        ax.axvline(center + m, ls="-", c="crimson", label=f"ridge (median offset {m:+.2f})")
    ax.set_xlabel("depth layer (z)")
    ax.set_ylabel("CT intensity")
    ax.set_title("Mean depth profile across sampled tiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_dict(result: ProfileResult, source: str) -> dict:
    offs = result.offsets
    window = result.recommended_window()
    valid = int(offs.size)
    return {
        "source": source,
        "shape_zyx": list(result.shape),
        "tile": result.tile,
        "stride": result.stride,
        "tiles_sampled": len(result.tiles),
        "tiles_valid": valid,
        "median_offset_layers": result.median_offset,
        "offset_iqr_layers": (
            [float(np.percentile(offs, 25)), float(np.percentile(offs, 75))] if valid else None
        ),
        "recommended_layer_start": window[0] if window else None,
        "recommended_layer_end": window[1] if window else None,
    }


def write_outputs(result: ProfileResult, source: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    save_offset_heatmap(result, outdir / "offset_heatmap.png")
    save_profile_curve(result, outdir / "depth_profile.png")
    summary = summary_dict(result, source)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
