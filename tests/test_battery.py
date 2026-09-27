"""Battery = stomach (robot/battery.py) and the hunger-dependent dock senses
(sim/habitat_bridge/home.py battery mode)."""
import pytest

from robot.battery import FULL, HUNGRY, ONSET, Battery
from sim.habitat_bridge.home import BOWL_XZ, HomeWorld


def test_hunger_follows_charge():
    assert Battery(0.95).hunger == 0.0
    assert Battery(ONSET).hunger == 0.0 and Battery(0.7).hunger == 0.0
    assert Battery(HUNGRY).hunger == pytest.approx(1.0)
    assert Battery(0.4).hunger == pytest.approx(0.5)


def test_a_meal_lasts_until_full():
    b = Battery(0.5)
    b.step(1.0, 0.0, False, charging=True)
    assert b.meal and b.sense_hunger == 1.0
    while b.soc < FULL:
        b.step(1.0, 0.0, False, charging=True)
    assert not b.meal and b.stats["meals"] == 1 and b.sense_hunger == 0.0
    c = Battery(0.5)
    c.step(1.0, 0.0, False, charging=True)
    for _ in range(10):
        c.step(1.0, 0.3, False, charging=False)    # a short wander between bursts
    assert c.meal
    for _ in range(6):
        c.step(1.0, 0.3, False, charging=False)    # walked off the dock
    assert not c.meal


def test_moving_drains_faster_than_resting_and_the_dock_charges():
    a, b, c = Battery(0.5), Battery(0.5), Battery(0.5)
    for _ in range(100):
        a.step(0.1, 0.5, False, False)
        b.step(0.1, 0.0, True, False)
        c.step(0.1, 0.0, False, True)
    assert a.soc < b.soc < 0.5 < c.soc
    d = Battery(0.001)
    d.step(1.0, 0.5, False, False)
    assert d.flat and d.summary()["flat_s"] == 1.0


def test_dock_charges_only_while_eating_there():
    home = HomeWorld(seed=0, battery=Battery(0.3))
    at_dock = {"robot": [BOWL_XZ[0], BOWL_XZ[1], 0.0], "dist": 5.0, "visible": False}
    home.step(at_dock, 1.0, proboscis=0.0)
    s0 = home.battery.soc
    home.step(at_dock, 1.0, proboscis=1.0)
    assert home.battery.soc > s0


def test_a_charged_pet_senses_the_dock_weakly():
    full, low = HomeWorld(seed=0, battery=Battery(0.7)), HomeWorld(seed=0, battery=Battery(0.1))
    assert full.sense_gain()[0] == pytest.approx(0.15) and low.sense_gain()[0] == pytest.approx(1.0)
    assert full.sense_gain()[1] < low.sense_gain()[1]
    assert HomeWorld(seed=0).sense_gain() == (1.0, 1.0)          # no battery: as before


def test_a_flat_pet_is_rescued_next_day_and_it_is_recorded():
    b = Battery(0.0)
    assert b.flat
    b.day_start()
    assert b.soc == pytest.approx(0.3) and b.stats["rescued"] is True
    b.day_start()
    assert b.stats["rescued"] is False


def test_a_pet_born_on_its_dock_plans_to_it_when_hungry():
    pytest.importorskip("torch")
    from cortex.v0 import CortexV0, cell_of
    cx = CortexV0(seed=1, manners=True, naps=True)
    cx.reset(0, None, None)
    cx.know_place((-5.29, -4.49))                 # the dock, never walked to
    here = cell_of(0.0, 0.0)
    cx.map.visit(here, None)
    cx.hunger, cx.social, cx.intent = 0.9, 0.1, "none"
    U, _ = cx._utilities(here, 10.0)
    dock = cell_of(-5.29, -4.49)
    assert dock in U and U[dock][1] == "food"
    assert max(U.items(), key=lambda kv: kv[1][0])[0] == dock


def test_bored_and_alone_it_naps_and_wakes_when_its_person_is_back():
    pytest.importorskip("torch")
    from cortex.v0 import CortexV0
    cx = CortexV0(seed=1, manners=True, naps=True)
    cx.reset(0, None, None)
    cx.hunger, cx.sleepy = 0.1, 0.25
    obs = {"robot": [0.2, 0.2, 0.0], "visible": False, "half": 0.0, "az": 0.0, "el": 0.0}
    cx._u_best = {"food": 0.0, "explore": 0.05, "owner": 0.0}   # nothing worth doing
    for k in range(260):                                         # 26 s alone, standing still
        cx.next_plan = 1e9                                       # keep the utilities fixed
        cx.act(obs, None, 0.1 * k, None)
        if cx.want_rest:
            break
    assert cx.want_rest and cx._want_kind == "bored"
    cx.resting, cx.nap_kind = True, "bored"
    seen = dict(obs, visible=True, half=6.0)                    # its person walks in
    cx.act(seen, None, 30.0, None)
    assert not cx.resting


def test_no_meal_when_not_hungry_and_none_restarts_at_full():
    b = Battery(0.7)                                # not hungry (above ONSET)
    b.step(1.0, 0.0, False, charging=True)
    assert not b.meal and b.sense_hunger == 0.0
    c = Battery(0.5)
    while c.soc < FULL:
        c.step(1.0, 0.0, False, charging=True)
    for _ in range(5):                              # still on the contacts
        c.step(1.0, 0.0, False, charging=True)
    assert not c.meal and c.stats["meals"] == 1


def test_the_cortex_stays_hungry_until_the_meal_is_done():
    home = HomeWorld(seed=0, battery=Battery(0.5))
    at_dock = {"robot": [BOWL_XZ[0], BOWL_XZ[1], 0.0], "dist": 5.0, "visible": False}
    home.step(at_dock, 1.0, proboscis=1.0)
    for _ in range(8):                               # charge past ONSET
        home.step(at_dock, 1.0, proboscis=1.0)
    assert home.battery.soc > 0.6 and home.hunger == 1.0


def test_no_meal_carries_over_night_and_nav_docking_is_not_a_meal():
    b = Battery(0.5)
    b.step(1.0, 0.0, False, charging=True)
    assert b.meal
    b.day_start()
    assert not b.meal and b.sense_hunger == b.hunger
    c = Battery(0.3)
    c.step(1.0, 0.0, False, charging=True, feeding=False)     # emergency dock, not eating
    assert not c.meal and c.soc > 0.3


def test_a_bored_pet_really_naps():
    """Regression (review 2026-09-26): a small explore option beat the nap."""
    pytest.importorskip("torch")
    from cortex.v0 import CortexV0, cell_of
    cx = CortexV0(seed=1, manners=True, naps=True)
    cx.reset(0, None, None)
    here = cell_of(0.2, 0.2)
    # a well-known neighbourhood (every cell visited often): little left to explore
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            c = (here[0] + dx, here[1] + dz)
            for _ in range(100):
                cx.map.visit(c, here if c != here else None)
    cx.hunger, cx.social, cx.sleepy, cx.intent = 0.0, 0.1, 0.25, "none"
    U, _ = cx._utilities(here, 30.0)
    assert max(u for u, k in U.values() if k == "explore") < 0.15      # genuinely bored
    cx.want_rest, cx._want_kind = True, "bored"
    cx._plan(here, 30.0)
    if not cx.resting:                                                # goes to a nap spot nearby first
        assert cx.goal is not None and cx.goal[1] == "rest"
        cx._plan(cx.goal[0], 31.0)                                    # arrived
    assert cx.resting and cx.nap_kind == "bored"


def test_older_cortex_kinds_do_not_use_the_new_planning():
    pytest.importorskip("torch")
    from cortex.v0 import CortexV0, cell_of
    cx = CortexV0(seed=1, manners=True, naps=False)          # v0_manners
    cx.reset(0, None, None)
    cx.know_place((-5.29, -4.49))
    here = cell_of(0.0, 0.0)
    cx.map.visit(here, None)
    cx.hunger, cx.social, cx.intent = 0.9, 0.1, "none"
    U, _ = cx._utilities(here, 10.0)
    assert cell_of(-5.29, -4.49) not in U
