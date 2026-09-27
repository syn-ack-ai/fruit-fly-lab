"""
The central-complex compass (brain/navigation/compass.py): a heading put into
the E-PG ring as a landmark-like input is read back from the E-PG bump, and a
rotating heading moves the bump the same way. Whole brain, calibrated dynamics.
"""
from __future__ import annotations

import numpy as np
import pytest

from brain.neurons.registry import load_connectome
from native import lif_native

pytestmark = pytest.mark.skipif(not lif_native.available(),
                                reason="native engine not built (make -C native)")


def _circ(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


@pytest.fixture(scope="module")
def setup():
    from brain.navigation.compass import Compass, CompassDrive
    from simulation.engine.session import apply_dynamics
    c = load_connectome()
    e = lif_native.NativeLIFEngine.from_connectome(c, seed=3)
    from simulation.engine.session import apply_calibrated_gain
    apply_calibrated_gain(e)             # the dataset's calibrated gain (male-based: 0.62)
    apply_dynamics(e, c, "calibrated")
    return e, CompassDrive(Compass(c))


def _window(e, drive, ms, omega=0.0, step=20):
    s0 = e.spike_counts.copy()
    for _ in range(int(ms // step)):
        e.set_poisson(drive.indices, drive.rates_hz())
        e.run(float(step))
        drive.turn(omega, step)
    return e.spike_counts - s0


@pytest.mark.parametrize("heading", [45.0, 200.0])
def test_bump_reads_back_the_heading(setup, heading):
    e, drive = setup
    e.reset(seed=3)
    drive.set_heading(heading)
    _window(e, drive, 200)
    est, strength = drive.heading_estimate(_window(e, drive, 300))
    assert strength > 0.5
    assert _circ(est, heading) < 30.0


@pytest.mark.parametrize("omega", [180.0, -180.0])
def test_rotating_heading_moves_the_bump(setup, omega):
    e, drive = setup
    e.reset(seed=3)
    drive.set_heading(0.0)
    _window(e, drive, 200)
    _window(e, drive, 500, omega)                  # heading ends at +-90 deg
    est, _ = drive.heading_estimate(_window(e, drive, 100, omega))
    assert _circ(est, drive.heading) < 35.0
    assert _circ(est, 90.0 if omega > 0 else 270.0) < 45.0
