# inkalign

**Is your surface at the right depth — and is it facing the right way?**

`inkalign` is a depth-offset and orientation calibrator for [Vesuvius
Challenge](https://scrollprize.org) surface volumes. Ink-detection models are
sensitive to depth offsets: if a rendered segment's papyrus sheet is not
centered in the depth stack, checkpoints respond poorly and the official advice
is to shift `--layer-start` / `--layer-end` by hand and retry. `inkalign`
measures the misplacement instead of guessing it.

## What it does

Streams a rendered surface volume (local zarr, `https://`, or `s3://` — no bulk
download), samples tiles across the segment, and for each tile:

1. computes the CT intensity profile along the depth (Z) axis,
2. locates the papyrus density ridge with sub-voxel precision (anchored to the
   segment-wide ridge, so bright *adjacent wraps* near the stack edges can't
   hijack the estimate),
3. reports the ridge's offset from the stack center.

Outputs: a per-tile **offset heatmap**, the segment's **mean depth profile**, a
machine-readable `summary.json`, and a concrete recommendation for
`--layer-start` / `--layer-end` that re-centers the sheet.

## Install

```bash
pip install git+https://github.com/hilalitvak/inkalign
```

## Usage

Point it at any surface volume — for example, straight at the open S3 bucket
(no credentials, no registration):

```bash
inkalign profile "s3://vesuvius-challenge-open-data/PHerc0139/segments/20250108000000-w025_2025010863/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr"
```

```
surface volume: shape (28, 7280, 6800), dtype uint8, chunks (28, 128, 128)
median ridge offset: -1.36 layers (IQR -3.29 .. -0.44, 80/132 tiles valid)
suggested re-centering: --layer-start 0 --layer-end 25
outputs written to inkalign_out/ (offset_heatmap.png, depth_profile.png, summary.json)
```

The whole run streams ~60 MB and takes about a minute on a laptop — no GPU.

Options:

| flag | meaning |
|---|---|
| `--tile N` | tile size in pixels (default 128, chunk-aligned) |
| `--max-tiles N` | sampling budget; stride is chosen automatically (default 200) |
| `--outdir DIR` | where to write the heatmap, curve, and summary |

### Model-in-the-loop sweep

The depth profile answers "is my surface centered?" from the CT signal alone.
The sweep answers the sharper question — *which window does the ink model
actually respond to, and which way is the surface facing?* — by running real
inference over a grid of depth windows and both orientations and scoring each
prediction for text-likeness (FFT line-pitch structure, no extra model needed).

Extract a small ROI first so the GPU only ever sees a few hundred pixels:

```bash
inkalign extract-roi "s3://…/surface-volumes/9.362um-….zarr" --out roi.zarr
inkalign sweep roi.zarr checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth
```

```
stack depth 28, window 14: 14 inference runs planned
...
best window: offset +2.0, direction forward, line_score 0.121
orientation: forward (confident, margin x6.3)
recommended flags: --layer-start 9 --layer-end 23 --direction forward
```

The inference command is a template (`--infer-template`), so any runner with
`--layer-start/--layer-end/--direction` flags can be swept. `--dry-run` prints
the planned runs without touching a GPU.

## Status

Under active development (August 2026). Roadmap:

- [x] depth-profile core: per-tile ridge offset, heatmap, recommended window
- [x] model-in-the-loop sweep over `(layer_start, direction)` with model-free
      text-likeness scoring, resolving the recto/verso ambiguity that
      `--direction both` brute-forces (validated against a stub runner; real
      checkpoint run pending)
- [ ] self-contained HTML report (score-vs-offset curve, filmstrip)
- [ ] label snapping: per-vertex offsets along mesh normals for tifxyz meshes

## License

MIT
