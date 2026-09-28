"""The escape readouts (brain/motor/descending.py): the long-mode takeoff is a
population behaviour of its looming DNs, the Giant Fibre a single command cell
(review 2026-09-27: two spikes of one DNp11, driven by the pursuit pathway,
launched the robot 2.7 times a day with no threat)."""
import numpy as np


def _readout():
    from brain.neurons.registry import load_connectome
    from brain.motor.descending import DescendingReadout
    return DescendingReadout(load_connectome())


def test_one_long_mode_cell_does_not_launch_a_takeoff():
    from fly.body.fly_body import LONG_MODE_THRESHOLD
    r = _readout()
    L, R = r._chan_groups["escape_long_mode"]
    sums = np.zeros(len(r._gscale))
    sums[L[-1]] = 2                                     # two spikes of one cell in the 50 ms window
    assert r.channels(None, 50.0, sums=sums)["escape_long_mode"] < LONG_MODE_THRESHOLD
    sums[np.array(L + R)] = 3                           # the whole looming population
    assert r.channels(None, 50.0, sums=sums)["escape_long_mode"] >= LONG_MODE_THRESHOLD


def test_giant_fibre_keeps_the_single_cell_reading():
    from fly.body.fly_body import ESCAPE_THRESHOLD
    r = _readout()
    L, R = r._chan_groups["escape_takeoff"]
    sums = np.zeros(len(r._gscale))
    sums[L[0]] = 3                                      # one Giant Fibre, 60 Hz over 50 ms
    assert r.channels(None, 50.0, sums=sums)["escape_takeoff"] >= ESCAPE_THRESHOLD
