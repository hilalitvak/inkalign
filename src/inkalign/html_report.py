"""Render inkalign outputs into a single self-contained report.html.

Follows the conventions of villa's `make_ink_report.py`: one shareable file,
every image embedded as a data URI, headline numbers first, then the evidence.
It needs only what the `profile` and `sweep` commands persist (summary.json /
sweep_summary.json plus their PNGs and prediction TIFFs), so the report can be
rebuilt at any time:

    inkalign report --profile-dir inkalign_out --sweep-dir inkalign_sweep
"""

from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CHIP_COLORS = {"good": "#1a9641", "mid": "#ff8c00", "bad": "#d7191c"}

#: line_score grading for table chips; sweep scores are relative, so grade
#: against the best score in the sweep rather than absolute thresholds.
STYLE = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 980px;
       color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 1.6rem; } h2 { font-size: 1.2rem; margin-top: 2rem; }
img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
table { border-collapse: collapse; margin: 1rem 0; }
td, th { padding: .35rem .8rem; border-bottom: 1px solid #eee; text-align: right; }
th { border-bottom: 2px solid #999; }
.chip { display: inline-block; min-width: 3.2em; padding: .1em .45em;
        border-radius: 1em; color: #fff; text-align: center; font-size: .9em; }
.headline { background: #f4f6f8; border-radius: 8px; padding: 1rem 1.4rem; }
.headline code { font-size: 1.05em; background: #fff; padding: .15em .4em;
                 border-radius: 4px; border: 1px solid #ddd; }
.filmstrip { display: flex; flex-wrap: wrap; gap: .6rem; }
.filmstrip figure { margin: 0; text-align: center; font-size: .85em; }
.filmstrip img { width: 150px; }
.muted { color: #777; font-size: .9em; }
"""


def _data_uri_png(path: Path) -> str | None:
    if not path.is_file():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _tif_thumbnail_uri(path: Path, width: int = 300) -> str | None:
    """Downsampled JPEG data URI of a prediction TIFF (PIL only when needed)."""
    try:
        import tifffile
        from PIL import Image

        arr = np.asarray(tifffile.imread(path))
        if arr.dtype != np.uint8:
            lo, hi = np.percentile(arr, [1, 99])
            arr = np.clip((arr - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.thumbnail((width, width * 4))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _chip(value: float, best: float) -> str:
    grade = "good" if value >= 0.7 * best else "mid" if value >= 0.3 * best else "bad"
    return f'<span class="chip" style="background:{CHIP_COLORS[grade]}">{value:.3f}</span>'


def _img_section(title: str, uri: str | None) -> str:
    if uri is None:
        return ""
    return f"<h2>{title}</h2><img src='{uri}'>"


def build_report(
    profile_dir: Path | None = None,
    sweep_dir: Path | None = None,
    out: Path = Path("report.html"),
) -> Path:
    profile = sweep = None
    if profile_dir and (profile_dir / "summary.json").is_file():
        profile = json.loads((profile_dir / "summary.json").read_text())
    if sweep_dir and (sweep_dir / "sweep_summary.json").is_file():
        sweep = json.loads((sweep_dir / "sweep_summary.json").read_text())
    if profile is None and sweep is None:
        raise FileNotFoundError("no summary.json or sweep_summary.json found in the given dirs")

    parts = [f"<style>{STYLE}</style>"]
    parts.append("<h1>inkalign calibration report</h1>")
    parts.append(
        f"<p class='muted'>generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
        + (f" &middot; source: <code>{profile['source']}</code>" if profile else "")
        + "</p>"
    )

    # Headline: the answer first.
    headline = []
    if profile and profile.get("median_offset_layers") is not None:
        m = profile["median_offset_layers"]
        headline.append(
            f"CT depth profile: papyrus ridge sits <b>{m:+.2f} layers</b> from stack center "
            f"({profile['tiles_valid']}/{profile['tiles_sampled']} tiles)."
        )
    if sweep and sweep.get("best"):
        b = sweep["best"]
        headline.append(
            f"Ink model responds best at window offset <b>{b['offset']:+.1f}</b>, "
            f"direction <b>{b['direction']}</b> (line score {b['line_score']:.3f})."
        )
        if sweep.get("direction"):
            d = sweep["direction"]
            conf = "confident" if d["confident"] else "weak — inspect the curve"
            headline.append(f"Orientation verdict: <b>{d['direction']}</b> ({conf}, margin &times;{d['margin']:.1f}).")
    flags = (sweep or {}).get("recommended_flags") or (
        profile
        and profile.get("recommended_layer_start") is not None
        and f"--layer-start {profile['recommended_layer_start']} "
            f"--layer-end {profile['recommended_layer_end']}"
    )
    if flags:
        headline.append(f"Recommended flags: <code>{flags}</code>")
    parts.append("<div class='headline'><p>" + "</p><p>".join(headline) + "</p></div>")

    if profile_dir:
        parts.append(_img_section("Depth profile", _data_uri_png(profile_dir / "depth_profile.png")))
        parts.append(_img_section("Per-tile ridge offset", _data_uri_png(profile_dir / "offset_heatmap.png")))

    if sweep:
        parts.append(_img_section("Model response vs depth window", _data_uri_png(sweep_dir / "sweep_curve.png")))
        runs = [r for r in sweep["runs"] if r.get("line_score") is not None]
        if runs:
            best = max(r["line_score"] for r in runs)
            rows = "".join(
                f"<tr><td>{r['offset']:+.1f}</td><td>[{r['layer_start']}, {r['layer_end']})</td>"
                f"<td>{r['direction']}</td><td>{_chip(r['line_score'], best)}</td>"
                f"<td>{r['ink_fraction']:.2f}</td></tr>"
                for r in sorted(runs, key=lambda r: (r["direction"], r["offset"]))
            )
            parts.append(
                "<h2>Sweep runs</h2><table><tr><th>offset</th><th>window</th>"
                "<th>direction</th><th>line score</th><th>ink frac</th></tr>" + rows + "</table>"
            )
            failed = [r for r in sweep["runs"] if r.get("error")]
            if failed:
                parts.append(f"<p class='muted'>{len(failed)} run(s) failed and are not shown.</p>")

            thumbs = []
            for r in sorted(runs, key=lambda r: (r["direction"], r["offset"])):
                uri = _tif_thumbnail_uri(Path(r["output"]))
                if uri:
                    thumbs.append(
                        f"<figure><img src='{uri}'><figcaption>{r['offset']:+.1f} "
                        f"{r['direction']}</figcaption></figure>"
                    )
            if thumbs:
                parts.append("<h2>Prediction filmstrip</h2><div class='filmstrip'>" + "".join(thumbs) + "</div>")

    out.write_text("\n".join(p for p in parts if p), encoding="utf-8")
    return out
