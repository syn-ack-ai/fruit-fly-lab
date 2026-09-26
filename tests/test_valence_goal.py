"""
Learned valence -> central-complex goal (brain/navigation/valence_goal.py).

Unit checks on a small stand-in mushroom body (no connectome needed): a naive
fly has no goal; depressing the Kenyon cell -> MBON synapses of a reward-taught
(PAM, avoidance) MBON gives positive valence and a goal up the experienced
odour gradient (or upwind); depressing those of a punishment-taught (PPL1,
approach) MBON gives negative valence and the reversed goal.
"""
from __future__ import annotations

import types

import numpy as np

from brain.navigation.valence_goal import VALENCE_FULL, ValenceGoal


class _Goal:
    def __init__(self):
        self.enabled, self.goal, self.gain = True, 0.0, 1.0

    def set_goal(self, g):
        self.goal = g % 360.0


def _mb():
    # neurons 0-3 KCs, 4-5 MBONs (4: PAM-taught, 5: PPL1-taught), 6-7 DANs
    mb = types.SimpleNamespace()
    mb.kc = np.arange(4)
    mb.mbon = np.array([4, 5])
    mb.dan = np.array([6, 7])
    mb.types = np.array(["KCg", "KCg", "KCg", "KCg", "MBON01", "MBON11", "PAM01", "PPL101"])
    mb.teacher = np.array([[1.0, 0.0], [0.0, 1.0]])              # DAN x MBON
    mb.edge_kc = np.repeat(np.arange(4), 2)
    mb.edge_mbon = np.tile([0, 1], 4)
    mb.weights = np.ones(8)
    mb._kc_pos = np.full(8, -1)
    mb._kc_pos[:4] = np.arange(4)
    return mb


def _run(mb, heading_path, odour_path, **kw):
    vg = ValenceGoal(mb, _Goal(), None)
    for h, od in zip(heading_path, odour_path):
        vg.step(np.array([0, 1, 2, 3]), 10.0, h, odour=od, **kw)
    return vg


# the fly heads along +x (0 deg) and the odour rises: the gradient points to 0 deg
H = [0.0] * 50
C = list(np.linspace(0.0, 1.0, 50))


def test_naive_fly_has_no_goal():
    vg = _run(_mb(), H, C)
    assert vg.valence == 0.0 and not vg.goal.enabled


def test_reward_learning_sets_goal_up_gradient():
    mb = _mb()
    mb.weights[mb.edge_mbon == 0] = 0.5                          # avoidance MBON depressed
    vg = _run(mb, H, C)
    assert vg.valence > 0 and vg.goal.enabled
    assert min(vg.goal.goal, 360 - vg.goal.goal) < 1.0           # toward 0 deg
    assert vg.goal.gain == min(1.0, vg.valence / VALENCE_FULL)


def test_punishment_learning_reverses_goal():
    mb = _mb()
    mb.weights[mb.edge_mbon == 1] = 0.5                          # approach MBON depressed
    vg = _run(mb, H, C)
    assert vg.valence < 0 and vg.goal.enabled
    assert abs(vg.goal.goal - 180.0) < 1.0


def test_upwind_goal_when_air_moves():
    mb = _mb()
    mb.weights[mb.edge_mbon == 0] = 0.5
    # heading 90 deg, air from 30 deg to the right -> upwind is 60 deg
    vg = _run(mb, [90.0] * 5, [0.5] * 5, air_from_deg=30.0, airspeed=200.0)
    assert abs(vg.goal.goal - 60.0) < 1e-6
