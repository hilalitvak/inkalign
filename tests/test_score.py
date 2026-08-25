"""Unit tests for the text-likeness scorer (synthetic; real-data validation
lives in the repo's evidence images, per villa's CONTRIBUTING.md)."""

import numpy as np

from inkalign.score import text_likeness

UM = 19.2  # scale of a ds8 9um-family render, um/px
PX_PER_MM = 1000.0 / UM


def make_text_image(h=1024, w=1024, pitch_mm=0.8, seed=0):
    """Horizontal bright 'text lines' at a regular pitch plus noise."""
    rng = np.random.default_rng(seed)
    img = rng.random((h, w)) * 0.2
    pitch_px = pitch_mm * PX_PER_MM
    line_h = max(int(pitch_px * 0.45), 1)
    y = 0.0
    while y < h:
        y0 = int(y)
        # lines are broken into word-like runs, not solid bars
        for x0 in range(0, w, 60):
            if rng.random() < 0.7:
                img[y0 : y0 + line_h, x0 : x0 + 45] += 0.8
        y += pitch_px
    return np.clip(img, 0, 1)


def test_text_scores_higher_than_shuffled():
    img = make_text_image()
    rng = np.random.default_rng(1)
    shuffled = img[rng.permutation(img.shape[0])]
    s_text = text_likeness(img, UM)
    s_shuf = text_likeness(shuffled, UM)
    assert s_text.line_score > 3 * s_shuf.line_score
    assert s_text.line_score > 0.05


def test_noise_scores_near_zero():
    rng = np.random.default_rng(2)
    noise = rng.random((1024, 1024))
    assert text_likeness(noise, UM).line_score < 0.05


def test_wrong_pitch_scores_lower():
    """Periodic structure at 3x the expected pitch is not text."""
    right = text_likeness(make_text_image(pitch_mm=0.8), UM)
    wrong = text_likeness(make_text_image(pitch_mm=2.4), UM)
    assert right.line_score > 2 * wrong.line_score


def test_empty_image():
    s = text_likeness(np.zeros((512, 512)), UM)
    assert s.line_score == 0.0
    assert s.windows_scored == 0
