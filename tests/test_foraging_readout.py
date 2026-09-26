"""Readout used by the foraging body: DNg100 (BDN2) forward walking and
per-type steering signals (Rayshubskiy et al. 2025)."""
import numpy as np
import pytest

from brain.motor.descending import DescendingReadout
from brain.neurons.registry import load_connectome


@pytest.fixture(scope="module")
def connectome():
    return load_connectome()


def test_dng100_forward_and_per_type_steering(connectome):
    ro = DescendingReadout(connectome)
    n = connectome.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    side = n["side"].astype(str).to_numpy()
    sc = np.zeros(connectome.n)
    sc[t == "DNg100"] = 2                       # 40 Hz over a 50 ms window
    sc[(t == "DNa02") & (side == "left")] = 1   # 20 Hz on the left only
    ch = ro.channels(sc, 50.0)
    assert ch["forward_walk"] > 0.3
    assert abs(ch["hz_DNg100"] - 40.0) < 1e-9
    assert ch["lr_DNa02"] < 0 and "lr_DNa01" in ch


def test_body_steers_transiently_on_dna02():
    from fly.body.foraging_body import ForagingBody
    b = ForagingBody(neural=True, seed=0)
    b._walking = True
    ch = {"lr_DNa02": -0.3, "hz_DNg100": 14.6}   # sustained left-dominant DNa02
    turns = []
    for k in range(3000):                        # 3 s at 1 ms
        b._turn_noise = 0.0
        b.update(1.0, ch, float(k))
        b._walking = True
        turns.append(b.state.turn_rate_deg_s)
    assert max(turns[:400]) > 100                # a brisk left turn at onset ...
    assert abs(turns[-1]) < 20                   # ... that fades (biphasic filter)
