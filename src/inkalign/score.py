"""Model-free text-likeness scoring for ink predictions.

Used to rank sweep outputs: an ink prediction produced with the *right* depth
window and orientation shows writing-like structure — horizontal text lines at
a regular pitch — while a wrong window yields diffuse noise. The line-pitch
prior comes from villa's `get_ink_metrics.py` (80-120 px at its ~7.91 um strip
scale) converted to physical units, so predictions at any resolution can be
scored by passing their um-per-pixel scale.

Unlike `get_ink_metrics.py` (whole-scroll strips, nnU-Net ensemble, multi-GPU)
this is a cheap 1D-projection heuristic meant for small ROIs on a CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

#: Expected text-line pitch band, in mm (80-120 px at 7.91 um/px).
LINE_PITCH_MM = (0.63, 0.95)

#: Width of the sliding x-window used for row projections, in mm (~half a
#: column, following villa's LINE_WIN_PX = 512 at 7.91 um/px).
LINE_WIN_MM = 4.0

#: Windows whose mean ink fraction is below this carry no signal to score.
MIN_INK_FRACTION = 0.01


@dataclass
class TextScore:
    line_score: float  #: spectral power in the text-pitch band vs background, 0..~1
    ink_fraction: float  #: fraction of image above the ink threshold
    windows_scored: int

    @property
    def total(self) -> float:
        """Single number for ranking sweep outputs. Line structure is the
        discriminative part; ink fraction alone is easily fooled by noise."""
        return self.line_score


def _normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def _band_power_ratio(proj: np.ndarray, pitch_band_px: tuple[float, float]) -> float | None:
    """Fraction of (detrended) spectral power inside the expected pitch band.

    Regular text lines put a peak at frequency 1/pitch in the row projection;
    diffuse noise spreads power across all frequencies.
    """
    n = proj.size
    lo_px, hi_px = pitch_band_px
    if n < 3 * hi_px:  # need at least ~3 pitch periods to see a line pattern
        return None
    detrended = proj - gaussian_filter1d(proj, hi_px)  # remove sub-pitch trends
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(detrended * window)) ** 2
    freqs = np.fft.rfftfreq(n)
    band = (freqs >= 1.0 / hi_px) & (freqs <= 1.0 / lo_px)
    total = spectrum[1:].sum()  # skip DC
    if total <= 0 or not band.any():
        return None
    return float(spectrum[band].sum() / total)


def text_likeness(img: np.ndarray, um_per_px: float) -> TextScore:
    """Score how much a 2D ink prediction looks like written text lines.

    `img` is any 2D array of ink probability/intensity; `um_per_px` its scale.
    Rows of text are assumed roughly horizontal (true for standard renders).
    """
    norm = _normalize(img)
    ink_fraction = float((norm > 0.5).mean())

    px_per_mm = 1000.0 / um_per_px
    pitch_band = (LINE_PITCH_MM[0] * px_per_mm, LINE_PITCH_MM[1] * px_per_mm)
    win = max(int(LINE_WIN_MM * px_per_mm), 32)

    ratios = []
    for x0 in range(0, max(norm.shape[1] - win, 1), win // 2):
        strip = norm[:, x0 : x0 + win]
        if strip.mean() < MIN_INK_FRACTION:
            continue
        ratio = _band_power_ratio(strip.sum(axis=1), pitch_band)
        if ratio is not None:
            ratios.append(ratio)

    # Baseline: white noise spreads power uniformly, so its in-band fraction is
    # the band's width share of the spectrum. Score relative to that.
    line_score = 0.0
    if ratios:
        n_rows = norm.shape[0]
        freqs = np.fft.rfftfreq(n_rows)
        band = (freqs >= 1.0 / pitch_band[1]) & (freqs <= 1.0 / pitch_band[0])
        chance = band.sum() / max(len(freqs) - 1, 1)
        best = float(np.percentile(ratios, 90))  # text often occupies part of a segment
        line_score = float(max(0.0, (best - chance) / (1 - chance)))

    return TextScore(line_score=line_score, ink_fraction=ink_fraction, windows_scored=len(ratios))
