"""inkalign command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .io import open_surface_volume
from .profile import profile_volume
from .report import write_outputs


def _progress(done: int, total: int) -> None:
    sys.stderr.write(f"\r  tiles {done}/{total}")
    sys.stderr.flush()
    if done == total:
        sys.stderr.write("\n")


def cmd_profile(args: argparse.Namespace) -> int:
    arr = open_surface_volume(args.volume)
    print(f"surface volume: shape {arr.shape}, dtype {arr.dtype}, chunks {arr.chunks}")
    result = profile_volume(
        arr, tile=args.tile, max_tiles=args.max_tiles, progress=_progress
    )
    summary = write_outputs(result, args.volume, Path(args.outdir))

    m = summary["median_offset_layers"]
    if m is None:
        print("No tile produced a credible papyrus ridge — the volume may be "
              "mostly outside the segment mask, or the surface badly misplaced.")
        return 1
    print(f"median ridge offset: {m:+.2f} layers "
          f"(IQR {summary['offset_iqr_layers'][0]:+.2f} .. {summary['offset_iqr_layers'][1]:+.2f}, "
          f"{summary['tiles_valid']}/{summary['tiles_sampled']} tiles valid)")
    if abs(m) < 0.5:
        print("surface is well centered — no window shift needed")
    else:
        print(f"suggested re-centering: --layer-start {summary['recommended_layer_start']} "
              f"--layer-end {summary['recommended_layer_end']}")
    print(f"outputs written to {args.outdir}/ (offset_heatmap.png, depth_profile.png, summary.json)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="inkalign",
        description="Depth-offset and orientation calibration for Vesuvius surface volumes.",
    )
    parser.add_argument("--version", action="version", version=f"inkalign {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "profile",
        help="locate the papyrus density ridge and report its offset from stack center",
    )
    p.add_argument("volume", help="surface volume: local zarr path, https:// or s3:// URL")
    p.add_argument("--tile", type=int, default=128, help="tile size in px (default 128)")
    p.add_argument("--max-tiles", type=int, default=200,
                   help="max tiles to sample (default 200; stride chosen automatically)")
    p.add_argument("--outdir", default="inkalign_out", help="output directory")
    p.set_defaults(func=cmd_profile)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
