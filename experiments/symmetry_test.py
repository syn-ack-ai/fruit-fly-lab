"""
Does the brain have a left/right bias? With SYMMETRIC input a balanced brain's
steering output should average zero; any bias would make the robot drift or
favour a side. Drive symmetric stimuli and read, per seed, left minus right:

  DNa01, DNa02   steering descending neurons (Hz)
  turn index     leg coxa movers (R - L) / (R + L) (nerve-cord datasets only;
                 experiments/vnc_turn_test.py)

Stimuli (each on top of resting ORNs at 8 Hz, calibrated dynamics):
  rest       nothing else
  odour      the vinegar ORNs of both antennae (modality odor_vinegar), 60 Hz
  looming    every LC4 and LPLC2 cell (both eyes), 60 Hz
  walk       DNg100 both sides, 80 Hz (forward walking command)

    FLY_DATASET=merged python experiments/symmetry_test.py [--seeds 8]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                                          # noqa: E402
from brain.neurons.registry import load_connectome                     # noqa: E402
from native.lif_native import NativeLIFEngine                          # noqa: E402
from simulation.engine.session import apply_dynamics                   # noqa: E402


def turn_index(d, coxa, t, side, seg):
    """Leg turn index (R - L) / (R + L) from the coxa movers, as the mean rate
    per cell of each (motor neuron type, segment) present on both sides, summed
    over those types (a plain sum would count the extra cells of whichever side
    has more: 29 left / 27 right in the MaleCNS)."""
    L = R = 0.0
    for key in set(zip(t[coxa], seg[coxa])):
        m = (t[coxa] == key[0]) & (seg[coxa] == key[1])
        l, r = coxa[m & (side[coxa] == "left")], coxa[m & (side[coxa] == "right")]
        if len(l) and len(r):
            L += d[l].mean(); R += d[r].mean()
    return (R - L) / max(R + L, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dynamics", default=os.environ.get("FLY_DYNAMICS", "calibrated"))
    a = ap.parse_args()
    c = load_connectome()
    n = c.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    side = n["side"].fillna("").astype(str).to_numpy()
    from brain.sensory.modalities import ALL_MODALITIES, resolve_neurons
    vinegar = resolve_neurons(next(m for m in ALL_MODALITIES if m.key == "odor_vinegar"), c)
    orn = np.flatnonzero(np.char.startswith(t.astype(str), "ORN_"))
    loom = np.flatnonzero(np.isin(t, ["LC4", "LPLC2"]))
    walk = np.flatnonzero(t == "DNg100")
    ids = np.unique(np.concatenate([orn, vinegar, loom, walk]))
    stim = {"rest": {}, "odour": {"v": 60.0}, "looming": {"l": 60.0}, "walk": {"w": 80.0}}
    coxa = seg = None
    if config.MALE_CNS:
        sc = n["super_class"].fillna("").astype(str).to_numpy()
        seg = n["soma_neuromere"].fillna("").astype(str).to_numpy()
        low = np.char.lower(t.astype(str))
        coxa = np.flatnonzero((sc == "vnc_motor") & np.isin(seg, ["T1", "T2", "T3"])
                              & ((np.char.find(low, "rotator") >= 0) | (np.char.find(low, "promotor") >= 0)
                                 | (np.char.find(low, "remotor") >= 0)))
    e = NativeLIFEngine.from_connectome(c, seed=1, threads=a.threads)
    from simulation.engine.session import apply_calibrated_gain
    apply_calibrated_gain(e)             # the dataset's calibrated gain (male-based: 0.62)
    apply_dynamics(e, c, a.dynamics)
    cell = lambda ty, sd: np.flatnonzero((t == ty) & (side == sd))
    print(f"dataset {config.DATASET_KEY}, dynamics {a.dynamics}, {a.seeds} seeds x {a.seconds} s")
    from scipy.stats import ttest_1samp
    for name, s in stim.items():
        hz = np.zeros(len(ids))
        pos = {int(i): k for k, i in enumerate(ids)}
        hz[[pos[i] for i in orn]] = 8.0
        if "v" in s:
            hz[[pos[i] for i in vinegar]] = s["v"]
        if "l" in s:
            hz[[pos[i] for i in loom]] = s["l"]
        if "w" in s:
            hz[[pos[i] for i in walk]] = s["w"]
        rows = []
        for k in range(a.seeds):
            e.reset(seed=700 + k)
            e.set_poisson(ids, hz)
            e.run(200.0)
            c0 = e.spike_counts.copy()
            e.run(a.seconds * 1000.0)
            d = (e.spike_counts - c0) / a.seconds
            r = {ty: d[cell(ty, "left")].mean() - d[cell(ty, "right")].mean() for ty in ("DNa01", "DNa02")}
            r["DN mean"] = np.mean([d[cell(ty, sd)].mean() for ty in ("DNa01", "DNa02") for sd in ("left", "right")])
            if coxa is not None:
                r["turn idx"] = turn_index(d, coxa, t, side, seg)
            rows.append(r)
        out = []
        for key in rows[0]:
            v = np.array([r_[key] for r_ in rows])
            if key == "DN mean":
                out.append(f"{key} {v.mean():5.1f} Hz")
                continue
            # identical values every seed: p = 1 if they are all 0, else a
            # perfectly consistent bias (review 2026-09-27)
            p = ttest_1samp(v, 0.0).pvalue if v.std() > 0 else (1.0 if v.mean() == 0 else 0.0)
            out.append(f"{key} L-R {v.mean():+6.2f} (sd {v.std():.2f}, p {p:.2f})")
        print(f"  {name:8s} " + " | ".join(out))


if __name__ == "__main__":
    main()
