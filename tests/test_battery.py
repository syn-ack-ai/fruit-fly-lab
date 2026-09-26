"""Battery = stomach (robot/battery.py) and the hunger-dependent dock senses
(sim/habitat_bridge/home.py battery mode)."""
import pytest

from robot.battery import FULL, HUNGRY, Battery
from sim.habitat_bridge.home import BOWL_XZ, HomeWorld


def test_hunger_follows_charge():
    assert Battery(0.95).hunger == 0.0
    assert Battery(FULL).hunger == 0.0
    assert Battery(HUNGRY).hunger == pytest.approx(1.0)
    assert Battery(0.55).hunger == pytest.approx(0.5)


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
    full, low = HomeWorld(seed=0, battery=Battery(0.95)), HomeWorld(seed=0, battery=Battery(0.1))
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
