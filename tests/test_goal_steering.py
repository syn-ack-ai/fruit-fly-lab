"""
Goal steering (brain/navigation/goal.py): a goal set in the fan-shaped body
(FC2) is compared with the heading (E-PG, via CompassDrive) by PFL3 neurons,
and the fly's steering output turns it toward the goal. Whole brain,
calibrated dynamics, resting olfactory input. Sign convention: angles
counter-clockwise positive; a goal clockwise of the heading (goal - heading
< 0) must give more right-side PFL3 and DNa02 output than a goal
counter-clockwise of it. The comparison of the two cancels the fly's resting
left-turn bias ("handedness").
"""
from __future__ import annotations

import numpy as np
import pytest

from brain.neurons.registry import load_connectome
from native import lif_native

pytestmark = pytest.mark.skipif(not lif_native.available(),
                                reason="native engine not built (make -C native)")


@pytest.fixture(scope="module")
def setup():
    from brain.navigation.compass import Compass, CompassDrive
    from brain.navigation.goal import GoalCircuit, GoalDrive
    from robot.head import RestingOlfaction
    from simulation.engine.session import apply_dynamics
    c = load_connectome()
    cx = Compass(c)
    e = lif_native.NativeLIFEngine.from_connectome(c, seed=1)
    apply_dynamics(e, c, "calibrated")
    return e, CompassDrive(cx, 270.0), GoalDrive(GoalCircuit(c, cx), peak_hz=120.0), RestingOlfaction(c)


def _steer(setup, offset_deg):
    e, comp, goal, rest = setup
    comp.set_heading(270.0)
    goal.set_goal(270.0 + offset_deg)
    out = []
    for seed in (1, 2):
        e.reset(seed=seed)
        parts = [(rest.indices, rest._rates), (comp.indices, comp.rates_hz()), (goal.indices, goal.rates_hz())]
        idx = np.concatenate([p[0] for p in parts])
        r = np.concatenate([p[1] for p in parts])
        o = np.argsort(idx)
        u, first = np.unique(idx[o], return_index=True)
        e.set_poisson(u, r[o][first])
        e.run(300.0)
        s0 = e.spike_counts.copy()
        e.run(600.0)
        out.append(goal.readout(e.spike_counts - s0, 600.0))
    return {k: float(np.mean([x[k] for x in out])) for k in out[0]}


def test_pfl3_and_dna02_steer_toward_the_goal(setup):
    right = _steer(setup, -90.0)        # goal 90 deg clockwise: turn right
    left = _steer(setup, +90.0)         # goal 90 deg counter-clockwise: turn left
    assert right["PFL3_R"] - right["PFL3_L"] > left["PFL3_R"] - left["PFL3_L"] + 2.0
    assert right["DNa02_R"] - right["DNa02_L"] > left["DNa02_R"] - left["DNa02_L"] + 5.0
