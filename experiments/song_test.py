"""
What makes the male fly brain sing? P1 (the male courtship-arousal neurons:
MaleCNS pC1 types named pMP4 / pMP-e) against the song command pIP10, the
nerve cord's song pattern generator (dPR1, vPR9, TN1a), the proboscis motor
neuron MN9 (courtship "licking") and walking (DNg100 forward, MDN backward).
Every condition runs on top of the resting receptor input the robot always has
(cognition.exam.core.resting); calibrated dynamics; 3 seeds x 1 s.

    FLY_DATASET=merged python -m experiments.song_test

2026-09-28 (merged): rest -> pIP10 0 Hz; P1 12 / 24 / 30 / 40 Hz -> pIP10
21 / 41 / ~55 / 66 Hz, MN9 1.5 / 4 / - / 20 Hz, DNg100 1.7 / 1.0 / - / 0.3 Hz;
a visual target alone (LC10a) -> pIP10 ~2 Hz; Or47b ORNs -> ~0.3 Hz; P1 24 Hz
with no resting input -> about the same (44 Hz, last row). All with the
calibrated dynamics the robot runs; other dynamics (e.g. the published model)
differ. Hence cortex/topdown.P1_MAX_HZ = 24.
"""
from __future__ import annotations

import types

import numpy as np

from cognition.exam import core
from cortex.topdown import AttendEncoder, p1_indices


def main():
    c = core.connectome()
    t = c.neurons["primary_type"].fillna("").astype(str).to_numpy()
    p1 = p1_indices(c)
    if not len(p1):
        raise SystemExit("no P1 neurons: run with a male brain (FLY_DATASET=merged or malecns)")
    g = {"pIP10": np.flatnonzero(t == "pIP10"), "dPR1": np.flatnonzero(t == "dPR1"),
         "vPR9": np.flatnonzero(np.char.startswith(t.astype(str), "vPR9")),
         "TN1a": np.flatnonzero(np.char.startswith(t.astype(str), "TN1a")),
         "MN9": core.groups()["mn9"], "DNg100": np.flatnonzero(t == "DNg100"), "MDN": np.flatnonzero(t == "MDN")}
    from brain.sensory.retinotopy import load_retinotopy, receptive_fields_2hop
    rf = receptive_fields_2hop(load_retinotopy(c), "LC10a").dropna(subset=["azimuth_deg"])
    o = np.argsort(rf["idx"].to_numpy())
    att = AttendEncoder(types.SimpleNamespace(
        indices=rf["idx"].to_numpy(np.int64)[o], _az=rf["azimuth_deg"].to_numpy(float)[o],
        _el=rf["elevation_deg"].to_numpy(float)[o],
        _sigma=np.clip(rf["rf_radius_deg"].to_numpy(float)[o], 8.0, 40.0), MAX_HZ=150.0))
    att.set(0.0, 1.0)
    ctx = core.Ctx({"dynamics": "calibrated"})
    ri, rr = core.resting()
    print(f"P1 cells: {len(p1)}")
    conds = [("rest", [])] + [(f"P1 {hz} Hz", [(p1, float(hz))]) for hz in (12, 24, 30, 40)]
    conds += [("visual target (LC10a)", [(att.indices, att.rates_hz())]),
              ("Or47b ORNs (VA1v) 80 Hz", [(np.flatnonzero(t == "ORN_VA1v"), 80.0)])]
    for name, inputs in conds:
        r = ctx.rates([(ri, rr)] + inputs, 1000.0, [1, 2, 3])
        print(f"{name:24s} " + "  ".join(f"{k} {r[v].mean():5.1f}" for k, v in g.items()))
    r = ctx.rates([(p1, 24.0)], 1000.0, [1, 2, 3])                    # no background input at all
    print(f"{'P1 24 Hz, no resting in':24s} " + "  ".join(f"{k} {r[v].mean():5.1f}" for k, v in g.items()))


if __name__ == "__main__":
    main()
