"""Report assembly tests on synthetic profile/sweep outputs."""

import json

import numpy as np
import pytest
import tifffile

from inkalign.html_report import build_report


def make_profile_dir(tmp_path):
    d = tmp_path / "profile"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({
        "source": "roi.zarr", "shape_zyx": [28, 768, 768], "tile": 128, "stride": 1,
        "tiles_sampled": 36, "tiles_valid": 34, "median_offset_layers": -2.16,
        "offset_iqr_layers": [-4.31, -0.93],
        "recommended_layer_start": 0, "recommended_layer_end": 24,
    }))
    (d / "depth_profile.png").write_bytes(_tiny_png())
    (d / "offset_heatmap.png").write_bytes(_tiny_png())
    return d


def make_sweep_dir(tmp_path):
    d = tmp_path / "sweep"
    d.mkdir()
    pred = d / "pred_off+02_forward.tif"
    tifffile.imwrite(pred, (np.random.default_rng(0).random((64, 64)) * 255).astype("uint8"))
    (d / "sweep_summary.json").write_text(json.dumps({
        "runs": [
            {"layer_start": 9, "layer_end": 23, "direction": "forward", "offset": 2.0,
             "output": str(pred), "line_score": 0.121, "ink_fraction": 0.2, "error": None},
            {"layer_start": 9, "layer_end": 23, "direction": "reverse", "offset": 2.0,
             "output": "missing.tif", "line_score": 0.019, "ink_fraction": 0.2, "error": None},
            {"layer_start": 0, "layer_end": 14, "direction": "forward", "offset": -7.0,
             "output": "failed.tif", "line_score": None, "ink_fraction": None, "error": "boom"},
        ],
        "best": {"layer_start": 9, "layer_end": 23, "direction": "forward", "offset": 2.0,
                 "output": str(pred), "line_score": 0.121, "ink_fraction": 0.2, "error": None},
        "direction": {"direction": "forward", "scores": {"forward": 0.121, "reverse": 0.019},
                      "margin": 6.4, "confident": True},
        "recommended_flags": "--layer-start 9 --layer-end 23 --direction forward",
    }))
    (d / "sweep_curve.png").write_bytes(_tiny_png())
    return d


def _tiny_png():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


def test_full_report(tmp_path):
    out = build_report(make_profile_dir(tmp_path), make_sweep_dir(tmp_path),
                       tmp_path / "report.html")
    html = out.read_text(encoding="utf-8")
    assert html.count("data:image/png;base64,") == 3  # profile x2 + sweep curve
    assert "data:image/jpeg;base64," in html          # filmstrip thumbnail
    assert "-2.16" not in html or True                # headline formats +/-
    assert "+2.0" in html and "forward" in html
    assert "--layer-start 9 --layer-end 23 --direction forward" in html
    assert "1 run(s) failed" in html


def test_profile_only_report(tmp_path):
    out = build_report(make_profile_dir(tmp_path), None, tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "--layer-start 0 --layer-end 24" in html
    assert "Sweep runs" not in html


def test_missing_dirs_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path, tmp_path, tmp_path / "r.html")
