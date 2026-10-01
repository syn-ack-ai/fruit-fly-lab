"""robot/eye.py: the panorama the robot's eyes see (camera + lidar), the
steadied lidar, and the encoder that hands the eye's rates to the brain.
(flyvis itself: tests/test_flyvis_eye.py, where it is installed.)"""
import time

import numpy as np

import robot.eye as E
from robot.retina import PANO_H, PANO_W


def test_area_resize_averages():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.testing.assert_allclose(E.area_resize(a, 2, 2), [[2.5, 4.5], [10.5, 12.5]])
    assert E.area_resize(np.ones((96, 128), np.float32), 26, 20).shape == (20, 26)


def test_compose_lidar_and_camera():
    r = np.full(360, 8.0)
    r[175:185] = 1.0                                         # an obstacle straight ahead (beam 180 = 0 deg)
    img = E.compose(None, ranges=r)
    assert img.shape == (PANO_H, PANO_W)
    col = PANO_W // 2                                        # azimuth ~0
    assert img[PANO_H // 2 - 2, col] < 0.3                   # dark band ahead
    assert img[PANO_H // 2 - 2, 10] == E.SKY                 # nothing behind
    near = r.copy()
    near[175:185] = 0.5
    rows = lambda im: int((im[:, col] < 0.3).sum())          # noqa: E731
    assert rows(E.compose(None, ranges=near)) > rows(img)    # nearer = taller
    cam = np.zeros((96, 128), np.uint8)                      # a black camera image where the head points
    img2 = E.compose(cam, pan_deg=40.0, tilt_deg=0.0)
    x = int((40.0 + 180.0) / 2)
    assert img2[PANO_H // 2, x] == 0.0 and img2[PANO_H // 2, PANO_W // 2] != 0.0


def test_lidar_steadier_ignores_noise_and_edges_but_sees_obstacles():
    rng = np.random.default_rng(0)
    L = E.LidarSteadier(360)
    base = np.full(360, 2.0)
    outs = []
    for k in range(40):
        r = base + rng.normal(0, 0.015, 360)
        r[rng.random(360) < 0.1] = 0.0                       # dropouts
        r[50] = 1.0 if k % 2 else 3.0                        # an edge beam alternating near / far
        outs.append(L.update(r).copy())
    o = np.array(outs[10:])
    assert (np.abs(np.diff(o, axis=0)) > 0).sum() == 0       # still: nothing redrawn
    base[100:110] = 1.0                                      # a real obstacle arrives
    seen = [float(L.update(base.copy())[105]) for _ in range(8)]
    assert seen[0] > 1.5 and seen[-1] == 1.0                 # after it lasts a few scans


class FakeFeed:
    def __init__(self, n):
        self.indices = np.arange(n, dtype=np.int64) * 3
        self._h = np.zeros(len(E._F))
        self._seq = 0
        self.calls = 0

    def publish(self, r):
        self._r = np.asarray(r, np.float64)
        self._seq += 2
        self._h[E._I["seq"]] = self._seq
        self._h[E._I["ready"]] = 1.0
        self._h[E._I["t_wall"]] = time.time()

    def rates(self, since=None):
        self.calls += 1
        if self._seq == since:
            return None
        return self._seq, self._r.copy()


def test_eye_rates_pickup():
    f = FakeFeed(4)
    er = E.EyeRates(f)
    r0 = er.rates_hz(0.0)
    assert not r0.any() and not r0.flags.writeable           # not ready: silent
    f.publish([1.0, 2.0, 3.0, 4.0])
    P = E.PICKUP_MS
    assert er.rates_hz(P / 2) is r0                          # within a pickup period: not looked at
    r1 = er.rates_hz(P)
    np.testing.assert_array_equal(r1, [1, 2, 3, 4])
    assert not r1.flags.writeable
    assert er.rates_hz(2 * P) is r1                          # nothing new: the same array
    f.publish([5.0, 5.0, 5.0, 5.0])
    assert er.rates_hz(3 * P)[0] == 5.0
    f._h[E._I["t_wall"]] = time.time() - 10                  # the eye stopped: silent again
    assert not er.rates_hz(4 * P).any()
    assert er.rates_hz(0.0) is not None                      # a new day (the clock restarts)
