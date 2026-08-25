"""Sweep ink-detection inference over depth windows and orientations.

Wraps villa's `vesuvius.ink_detection.inference.infer` (or any command with the
same flag contract): for each (layer_start, direction) combination it runs
inference, scores the resulting prediction for text-likeness, and reports which
window and orientation the ink model actually responds to.

infer.py selects source layers [layer_start, layer_end), center-crops them to
the model's input depth, and reverses the order for direction=reverse — so
passing exact-size windows [k, k+depth) walks the model's receptive window
through the stack one offset at a time.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import tifffile

from .score import text_likeness

#: Default inference command; {python} resolves to the current interpreter.
#: Any template using the same placeholders can be substituted (e.g. to add
#: --batch-size, a uv prefix, or a different entry point).
DEFAULT_TEMPLATE = (
    "{python} -m vesuvius.ink_detection.inference.infer "
    "{volume} {checkpoint} {output} "
    "--layer-start {layer_start} --layer-end {layer_end} --direction {direction}"
)


@dataclass
class SweepRun:
    layer_start: int
    layer_end: int
    direction: str
    offset: float  #: window center minus stack center, in layers
    output: str
    line_score: float | None = None
    ink_fraction: float | None = None
    error: str | None = None


def plan_runs(
    stack_depth: int,
    window_depth: int,
    offsets: list[int],
    directions: list[str],
    outdir: Path,
) -> list[SweepRun]:
    """One run per (offset, direction); offsets that slide the window off the
    stack are clamped out."""
    center = (stack_depth - 1) / 2
    runs = []
    for off in offsets:
        start = round(center + off - (window_depth - 1) / 2)
        if start < 0 or start + window_depth > stack_depth:
            continue
        for direction in directions:
            runs.append(
                SweepRun(
                    layer_start=start,
                    layer_end=start + window_depth,
                    direction=direction,
                    offset=start + (window_depth - 1) / 2 - center,
                    output=str(outdir / f"pred_off{off:+03d}_{direction}.tif"),
                )
            )
    return runs


def execute_sweep(
    runs: list[SweepRun],
    volume: str,
    checkpoint: str,
    um_per_px: float,
    template: str = DEFAULT_TEMPLATE,
    log=print,
) -> list[SweepRun]:
    """Run each planned inference and score its output in place."""
    # The template is tokenized *before* substitution, then each token is
    # formatted and the list run without a shell — so Windows paths (backslashes,
    # spaces) in volume/checkpoint/output never meet shell quoting rules.
    tokens = shlex.split(template)
    for i, run in enumerate(runs):
        mapping = dict(
            python=sys.executable,
            volume=volume,
            checkpoint=checkpoint,
            output=run.output,
            layer_start=run.layer_start,
            layer_end=run.layer_end,
            direction=run.direction,
        )
        cmd = [t.format(**mapping) for t in tokens]
        log(f"[{i + 1}/{len(runs)}] offset {run.offset:+.1f} {run.direction}: "
            + subprocess.list2cmdline(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            run.error = proc.stderr.strip().splitlines()[-1] if proc.stderr else "inference failed"
            log(f"  FAILED: {run.error}")
            continue
        try:
            pred = tifffile.imread(run.output)
        except Exception as exc:
            run.error = f"could not read output: {exc}"
            continue
        score = text_likeness(np.asarray(pred), um_per_px)
        run.line_score = score.line_score
        run.ink_fraction = score.ink_fraction
        log(f"  line_score={run.line_score:.3f} ink={run.ink_fraction:.2f}")
    return runs


def best_run(runs: list[SweepRun]) -> SweepRun | None:
    scored = [r for r in runs if r.line_score is not None]
    return max(scored, key=lambda r: r.line_score) if scored else None


def direction_verdict(runs: list[SweepRun]) -> dict | None:
    """Compare the best score per direction; a clear margin decides recto/verso."""
    by_dir = {}
    for r in runs:
        if r.line_score is not None:
            if r.direction not in by_dir or r.line_score > by_dir[r.direction]:
                by_dir[r.direction] = r.line_score
    if len(by_dir) < 2:
        return None
    winner = max(by_dir, key=by_dir.get)
    loser = min(by_dir, key=by_dir.get)
    margin = by_dir[winner] / by_dir[loser] if by_dir[loser] > 0 else float("inf")
    return {"direction": winner, "scores": by_dir, "margin": margin, "confident": margin >= 1.5}


def write_sweep_summary(runs: list[SweepRun], outdir: Path) -> dict:
    best = best_run(runs)
    verdict = direction_verdict(runs)
    summary = {
        "runs": [asdict(r) for r in runs],
        "best": asdict(best) if best else None,
        "direction": verdict,
        "recommended_flags": (
            f"--layer-start {best.layer_start} --layer-end {best.layer_end} "
            f"--direction {best.direction}"
        )
        if best
        else None,
    }
    (outdir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
