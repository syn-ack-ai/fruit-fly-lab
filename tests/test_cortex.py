"""The prototype neocortex (cortex/): top-down sign conventions, the learned
cognitive map, and (if PyTorch is installed) that v0 learns where food is."""
import math

import numpy as np
import pytest

from cortex.topdown import bearing_deg, goal_azimuth
from cortex.v0 import CELL_M, CognitiveMap, cell_of, centre


def test_bearing_matches_habitat_yaw_frame():
    # habitat_server: yaw CCW from +x in the (x, -z) plane
    assert bearing_deg((0, 0), (1, 0)) == pytest.approx(0.0)
    assert bearing_deg((0, 0), (0, -1)) == pytest.approx(90.0)     # -z is "left" of +x
    assert bearing_deg((0, 0), (-1, 0)) == pytest.approx(180.0)


def test_goal_azimuth_left_is_negative():
    assert goal_azimuth(90.0, 0.0) == pytest.approx(-90.0)         # CCW goal -> left
    assert goal_azimuth(350.0, 0.0) == pytest.approx(10.0)         # CW goal -> right
    assert goal_azimuth(10.0, 10.0) == pytest.approx(0.0)


def test_map_plans_only_over_moves_made():
    m = CognitiveMap()
    prev = None
    # an L-shaped route: along +x, then along +z
    for c in [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]:
        m.visit(c, prev)
        prev = c
    m.visit((0, 2), None)                                          # seen, never walked to
    dist = m.distances((0, 0))
    assert CognitiveMap.path_to(dist, (2, 2)) == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert dist[(2, 2)][0] == pytest.approx(4 * CELL_M)
    assert (0, 2) not in dist
    assert (1, 1) in m.frontier((2, 1))


def test_cells():
    assert cell_of(0.1, -0.1) == (0, -1)
    assert centre((0, -1)) == (0.5 * CELL_M, -0.5 * CELL_M)


def test_v0_learns_where_food_is():
    pytest.importorskip("torch")
    from cortex.v0 import CortexV0

    class Home:
        sources = [("bowl", (3.0, 0.0), "f", 1, "sweet")]

        def __init__(self):
            self.conc = {"L": np.zeros(1), "R": np.zeros(1)}
            self.taste, self.petting = None, False
            self.stats = {"pets": 0, "treats": 0}

    cx, h = CortexV0(seed=1), Home()
    firsts = []
    for day in range(2):
        cx.reset(day, h, None)
        x = z = yaw = 0.0
        rng = np.random.default_rng(day)
        first = None
        for k in range(1200):
            d = math.hypot(x - 3, z)
            eat = d < 0.6
            h.taste = "sweet" if eat else None
            if eat and first is None:
                first = k
            obs = {"robot": [x, z, yaw], "visible": False, "half": 0, "az": 0, "dist": 9}
            cmd = cx.act(obs, h, k * 0.1, {"body": {"proboscis_extension": float(eat)}})
            if eat:
                continue
            if cmd["goal_deg"] is not None and cmd["goal_gain"] > 0:
                err = (cmd["goal_deg"] - yaw + 180) % 360 - 180
                yaw += np.clip(err, -12, 12) + rng.normal(0, 5)
            else:
                yaw += rng.normal(0, 15)
            x = float(np.clip(x + 0.03 * math.cos(math.radians(yaw)), -4, 4))
            z = float(np.clip(z - 0.03 * math.sin(math.radians(yaw)), -4, 4))
        firsts.append(first)
        cx.save()
    assert firsts[0] is not None and firsts[1] is not None
    assert firsts[1] < firsts[0]                                    # it remembers the way
    assert cx.summary()["food_places"] >= 1


def test_orienting_reflex_attends_to_a_person_who_appears():
    """A person coming into view snaps the pursuit channel (attend) to their
    bearing for ORIENT_S; not while they stay in view, not when resting."""
    from cortex.v0 import CortexV0, ORIENT_S
    cx = CortexV0(seed=0, manners=True, naps=True, orient=True)
    cx.reset(0, None, None)
    obs = lambda t, vis, az=40.0: {"robot": [0.0, 0.0, 0.0], "visible": vis, "az": az, "half": 5.0, "human": [1.0, 1.0], "dist": 2.0}
    cmds = [cx.act(obs(0.1 * k, k >= 20), None, 0.1 * k, None) for k in range(80)]
    on = [("attend_az" in c) for c in cmds]
    assert not any(on[:20]) and on[20] and on[20 + int(ORIENT_S / 0.1) - 2]
    assert not any(on[20 + int(ORIENT_S / 0.1) + 1:])          # stays in view: no re-trigger
    assert cmds[20]["attend_az"] == 40.0 and cmds[20]["attend_gain"] == 1.0



def test_orienting_reflex_stays_quiet_when_it_should():
    """Not while resting, not mid-meal, not while giving the person space."""
    import types
    from cortex.v0 import CortexV0
    obs = lambda vis: {"robot": [0.0, 0.0, 0.0], "visible": vis, "az": 40.0, "half": 5.0, "human": [1.0, 1.0], "dist": 2.0}
    def run(setup, home=None):
        cx = CortexV0(seed=0, manners=True, naps=True, orient=True)
        cx.reset(0, None, None)
        setup(cx)
        cmds = [cx.act(obs(k >= 5), home, 0.1 * k, None) for k in range(12)]
        return any("attend_az" in c for c in cmds[5:])
    assert run(lambda cx: None)                                    # the control: it orients
    assert not run(lambda cx: setattr(cx, "resting", True) or setattr(cx, "want_rest", True) or setattr(cx, "sleepy", 1.0))
    meal = types.SimpleNamespace(battery=types.SimpleNamespace(meal=True, sense_hunger=1.0, hunger=0.5, soc=0.5),
                                 taste=None, stats={"pets": 0, "treats": 0}, conc={"L": np.zeros(1), "R": np.zeros(1)},
                                 petting=False, hunger=0.5)
    assert not run(lambda cx: None, home=meal)
