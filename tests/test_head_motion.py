"""robot.head.ObjectEncoder's person-motion estimate (review 2026-09-25): it must
work per new detection and survive a simulation clock that restarts at 0."""
import numpy as np

import robot.head as rh


class _Feed:
    def __init__(self):
        self.s = {}

    def set(self, t, az):
        self.s = {"ready": True, "stale": False, "moving": 0.0, "person_active": 1.0,
                  "person_stale": False, "person_t": t, "person_az": az, "person_el": 0.0,
                  "person_half_deg": 6.0}

    def state(self):
        return self.s


def _encoder():
    enc = rh.ObjectEncoder.__new__(rh.ObjectEncoder)     # skip receptive-field loading
    enc.feed = _Feed()
    enc.indices = np.arange(3)
    enc._az = np.array([-30.0, 0.0, 30.0]); enc._el = np.zeros(3); enc._sigma = np.full(3, 20.0)
    enc.last = {}
    return enc


def _run(enc, t0, moving, steps=30):
    gains = []
    for k in range(steps):
        t = t0 + 0.1 * k
        enc.feed.set(t, (20.0 * k * 0.1) if moving else 5.0)
        for _ in range(5):                                # several calls per detection
            enc.rates_hz(0.0)
        gains.append(enc._pmove)
    return gains[-1]


def test_moving_person_reads_as_moving_and_still_as_still():
    enc = _encoder()
    assert _run(enc, 0.0, moving=True) > 0.8
    assert _run(enc, 10.0, moving=False) < 0.2


def test_clock_restart_does_not_fake_motion():
    enc = _encoder()
    _run(enc, 0.0, moving=True)
    assert _run(enc, 0.0, moving=False) < 0.2              # new episode, person standing still
