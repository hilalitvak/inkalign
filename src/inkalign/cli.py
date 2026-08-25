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


def cmd_extract_roi(args: argparse.Namespace) -> int:
    from .roi import extract_roi, pick_inked_roi

    arr = open_surface_volume(args.volume)
    if args.y0 is None or args.x0 is None:
        y0, x0 = pick_inked_roi(arr, args.size)
        print(f"no --y0/--x0 given; using central ROI at y={y0}, x={x0}")
    else:
        y0, x0 = args.y0, args.x0
    out = extract_roi(arr, y0, x0, args.size, args.size, args.out)
    print(f"ROI (depth {arr.shape[0]}, {args.size}x{args.size} at y={y0}, x={x0}) -> {out}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from .report import save_sweep_curve
    from .sweep import DEFAULT_TEMPLATE, execute_sweep, plan_runs, write_sweep_summary

    arr = open_surface_volume(args.volume)
    z = arr.shape[0]
    window = args.window_depth or z // 2
    offsets = list(range(args.offset_min, args.offset_max + 1, args.offset_step))
    directions = ["forward", "reverse"] if args.direction == "both" else [args.direction]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = plan_runs(z, window, offsets, directions, outdir)
    print(f"stack depth {z}, window {window}: {len(runs)} inference runs planned")
    if args.dry_run:
        for r in runs:
            print(f"  offset {r.offset:+.1f} [{r.layer_start},{r.layer_end}) {r.direction}")
        return 0

    execute_sweep(runs, args.volume, args.checkpoint, args.um_per_px,
                  template=args.infer_template or DEFAULT_TEMPLATE)
    summary = write_sweep_summary(runs, outdir)
    save_sweep_curve(runs, outdir / "sweep_curve.png")

    if summary["best"] is None:
        print("every inference run failed — check the infer command template")
        return 1
    print(f"\nbest window: offset {summary['best']['offset']:+.1f}, "
          f"direction {summary['best']['direction']}, "
          f"line_score {summary['best']['line_score']:.3f}")
    if summary["direction"]:
        d = summary["direction"]
        conf = "confident" if d["confident"] else "weak signal — inspect the curve"
        print(f"orientation: {d['direction']} ({conf}, margin x{d['margin']:.1f})")
    print(f"recommended flags: {summary['recommended_flags']}")
    print(f"outputs written to {outdir}/ (sweep_curve.png, sweep_summary.json)")
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

    p = sub.add_parser(
        "extract-roi",
        help="copy a small ROI (full depth) to a local zarr for cheap GPU sweeps",
    )
    p.add_argument("volume", help="surface volume: local zarr path, https:// or s3:// URL")
    p.add_argument("--y0", type=int, help="ROI top edge (default: segment center)")
    p.add_argument("--x0", type=int, help="ROI left edge (default: segment center)")
    p.add_argument("--size", type=int, default=768, help="ROI side length in px (default 768)")
    p.add_argument("--out", default="inkalign_roi.zarr", help="output zarr path")
    p.set_defaults(func=cmd_extract_roi)

    p = sub.add_parser(
        "sweep",
        help="run ink inference over depth windows and orientations, score each output",
    )
    p.add_argument("volume", help="surface volume (use extract-roi first to keep GPU cost small)")
    p.add_argument("checkpoint", help="ink-detection checkpoint path passed to the infer command")
    p.add_argument("--window-depth", type=int,
                   help="depth window size in layers (default: half the stack)")
    p.add_argument("--offset-min", type=int, default=-6, help="smallest window offset (default -6)")
    p.add_argument("--offset-max", type=int, default=6, help="largest window offset (default +6)")
    p.add_argument("--offset-step", type=int, default=2, help="offset step (default 2)")
    p.add_argument("--direction", choices=("forward", "reverse", "both"), default="both")
    p.add_argument("--um-per-px", type=float, default=9.362,
                   help="prediction scale for text-likeness scoring (default 9.362)")
    p.add_argument("--infer-template",
                   help="override the inference command; placeholders: {python} {volume} "
                        "{checkpoint} {output} {layer_start} {layer_end} {direction}")
    p.add_argument("--outdir", default="inkalign_sweep", help="output directory")
    p.add_argument("--dry-run", action="store_true", help="print planned runs and exit")
    p.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
