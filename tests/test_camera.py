"""
The camera looming extractor (brain/sensory/camera.py) on synthetic video with
known geometry: it must report approach as positive expansion in the right
direction, and report sideways motion, recession and camera shake as no
looming. No camera hardware needed.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from brain.sensory.camera import HEIGHT, WIDTH, LoomingExtractor  # noqa: E402

S = 4                                  # render at 4x, then downsample like the camera
_rng = np.random.default_rng(0)
_BG = cv2.GaussianBlur((_rng.random((HEIGHT * S, WIDTH * S)) * 255).astype(np.uint8), (0, 0), 3)
_TEX = cv2.normalize(cv2.GaussianBlur((_rng.random((800, 800)) * 255).astype(np.uint8), (0, 0), 2),
                     None, 0, 255, cv2.NORM_MINMAX)
DPP = 0.5 * (66 / WIDTH + 41 / HEIGHT)


def _frame(cx, cy, r_px, shift=(0, 0)):
    img = np.roll(_BG, shift, axis=(0, 1)).copy()
    if r_px > 0:
        R, x0, y0 = int(r_px * S), int(cx * S), int(cy * S)
        patch = cv2.resize(_TEX, (2 * R + 1, 2 * R + 1))     # texture scales with the object
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        m = xx ** 2 + yy ** 2 <= R * R
        ys, xs = yy[m] + y0, xx[m] + x0
        ok = (ys >= 0) & (ys < HEIGHT * S) & (xs >= 0) & (xs < WIDTH * S)
        img[ys[ok], xs[ok]] = patch[m][ok] // 2 + 20
    return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def _run(traj, fps=30.0):
    ex = LoomingExtractor()
    return [ex.update(_frame(*t), i / fps) for i, t in enumerate(traj)]


def test_approach_is_positive_expansion_in_the_right_direction():
    traj = [(120, 25, 60 / (30 - 0.9 * i), (0, 0)) for i in range(30)]
    est = _run(traj)[-1]
    true_rate = (traj[-1][2] - traj[-2][2]) * DPP * 30
    assert est["active"]
    assert est["expansion_rate_deg_s"] == pytest.approx(true_rate, rel=0.35)
    assert est["azimuth_deg"] == pytest.approx((120 / WIDTH - 0.5) * 66, abs=3)
    assert est["elevation_deg"] == pytest.approx((0.5 - 25 / HEIGHT) * 41, abs=3)


@pytest.mark.parametrize("name,traj", [
    ("sideways", [(30 + 3 * i, 45, 12, (0, 0)) for i in range(30)]),
    ("shake", [(0, 0, 0, (int(4 * np.sin(i)), int(6 * np.cos(i)))) for i in range(30)]),
])
def test_non_looming_motion_gives_no_expansion(name, traj):
    for est in _run(traj)[3:]:
        assert est["expansion_rate_deg_s"] <= 2.0, name


def test_receding_is_negative_expansion():
    est = _run([(100, 30, 30 - 0.8 * i, (0, 0)) for i in range(30)])[-1]
    assert est["active"] and est["expansion_rate_deg_s"] < -5.0


def test_static_scene_is_inactive():
    assert not any(e["active"] for e in _run([(80, 45, 10, (0, 0))] * 20))
