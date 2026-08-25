"""Sweep orchestration tests with a stub inference command (no GPU, no model)."""

import json
import sys
from pathlib import Path

import numpy as np
import zarr

from inkalign.roi import extract_roi
from inkalign.sweep import (
    best_run,
    direction_verdict,
    execute_sweep,
    plan_runs,
    write_sweep_summary,
)

#: Stub "inference": writes a prediction whose text-line structure degrades
#: with distance from the true window (center offset +2, forward). Mirrors how
#: a real checkpoint responds when the sheet is centered and upright.
FAKE_INFER = """
import argparse, math
import numpy as np, tifffile

p = argparse.ArgumentParser()
p.add_argument("volume"); p.add_argument("checkpoint"); p.add_argument("output")
p.add_argument("--layer-start", type=int); p.add_argument("--layer-end", type=int)
p.add_argument("--direction")
a = p.parse_args()

center = (a.layer_start + a.layer_end - 1) / 2
quality = math.exp(-0.5 * ((center - 15.5) / 2.0) ** 2)  # true offset: +2 of 13.5
if a.direction == "reverse":
    quality *= 0.15

rng = np.random.default_rng(a.layer_start * 2 + (a.direction == "reverse"))
img = rng.random((1024, 1024)) * 0.2
pitch = 0.8 * 1000 / 19.2
y = 0.0
while y < 1024:
    for x0 in range(0, 1024, 60):
        if rng.random() < 0.7 * quality:
            img[int(y):int(y) + int(pitch * 0.45), x0:x0 + 45] += 0.8
    y += pitch
tifffile.imwrite(a.output, (np.clip(img, 0, 1) * 255).astype("uint8"))
"""

TEMPLATE = ("{python} " + "{stub} " +
            "{volume} {checkpoint} {output} "
            "--layer-start {layer_start} --layer-end {layer_end} --direction {direction}")


def test_plan_runs_clamps_offsets():
    runs = plan_runs(28, 14, offsets=list(range(-20, 21, 2)), directions=["forward"],
                     outdir=Path("."))
    assert runs, "some offsets must survive"
    for r in runs:
        assert 0 <= r.layer_start and r.layer_end <= 28
        assert r.layer_end - r.layer_start == 14


def test_sweep_finds_true_window_and_direction(tmp_path):
    stub = tmp_path / "fake_infer.py"
    stub.write_text(FAKE_INFER)
    # as_posix: the template is shlex-tokenized, where backslashes are escapes
    template = TEMPLATE.replace("{stub}", stub.as_posix())

    runs = plan_runs(28, 14, offsets=list(range(-6, 7, 2)),
                     directions=["forward", "reverse"], outdir=tmp_path)
    execute_sweep(runs, "vol.zarr", "ckpt.pth", um_per_px=19.2,
                  template=template, log=lambda *_: None)

    best = best_run(runs)
    assert best is not None
    assert best.direction == "forward"
    assert abs(best.offset - 2.0) <= 1.0, f"best offset {best.offset}, expected ~+2"

    verdict = direction_verdict(runs)
    assert verdict["direction"] == "forward" and verdict["confident"]

    summary = write_sweep_summary(runs, tmp_path)
    assert "--direction forward" in summary["recommended_flags"]
    assert json.loads((tmp_path / "sweep_summary.json").read_text())["best"]


def test_sweep_reports_failures(tmp_path):
    runs = plan_runs(28, 14, offsets=[0], directions=["forward"], outdir=tmp_path)
    execute_sweep(runs, "vol.zarr", "ckpt.pth", um_per_px=19.2,
                  template="{python} -c \"import sys; sys.exit(1)\"",
                  log=lambda *_: None)
    assert runs[0].error is not None
    assert best_run(runs) is None


def test_extract_roi_roundtrip(tmp_path):
    src = tmp_path / "src.zarr"
    vol = np.arange(28 * 256 * 256, dtype=np.uint8).reshape(28, 256, 256)
    zarr.save_array(str(src), vol, chunks=(28, 128, 128))
    arr = zarr.open_array(str(src), mode="r")

    out = extract_roi(arr, 64, 64, 128, 128, tmp_path / "roi.zarr")
    g = zarr.open_group(str(out), mode="r")
    roi = g["0"][:]
    assert roi.shape == (28, 128, 128)
    np.testing.assert_array_equal(roi, vol[:, 64:192, 64:192])
    assert g.attrs["multiscales"][0]["datasets"] == [{"path": "0"}]
