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
