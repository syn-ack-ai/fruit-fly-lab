"""
Experiment 04 - aversive olfactory learning in the mushroom body.

Reproduces the design of Hige et al. 2015 (Neuron 88:985), who paired an odour
with optogenetic activation of the punishment dopamine neuron PPL1-gamma1pedc
and recorded MBON-gamma1pedc>a/b: its response to the paired odour dropped,
its response to an unpaired odour did not. In FlyWire hemibrain naming these
are PPL101 and MBON11, and the connectome's strongest DAN->MBON link is
PPL101 -> MBON11 (1,365 synapses).

Uses the calibrated dynamics (data/metadata/dynamics_calibrated.json), which
make Kenyon cell odour codes sparse, and the dopamine-gated KC->MBON rule of
brain/plasticity/mushroom_body.py. An odour is either one receptor type driven
at 60 Hz ("ORN_DM1") or an odour-like mixture of 6 receptor types at 30-80 Hz
("mix:<seed>:<i>", as in cognition/calibrate_dynamics.py); it lasts 500 ms.
Brain activity is reset between trials (a long inter-trial interval), learned
weights are kept.

    python -m experiments.04_mb_learning [--a mix:7:0 --b mix:7:2]
"""
from __future__ import annotations

import json
import sys

import numpy as np

import config
from brain.neurons.registry import load_connectome
from brain.plasticity.mushroom_body import MushroomBody
from native.lif_native import NativeLIFEngine
from simulation.engine.session import apply_dynamics

ODOUR_A, ODOUR_B = "mix:7:0", "mix:7:2"      # two 6-glomerulus odours with no glomerulus in common
ODOUR_HZ, ODOUR_MS = 60.0, 500
DAN_TYPE, DAN_HZ, DAN_FROM_MS = "PPL101", 100.0, 100      # DAN on for the last 400 ms
TARGET = "MBON11"
KC_MBON_GAIN = 8.0            # calibrated: MBON11 odour responses ~20 Hz (see mushroom_body.py)
TRIALS = 3
TEST_SEEDS = (11, 12, 13)


def _odour(spec, n, t):
    """'ORN_<glomerulus>' at ODOUR_HZ, or 'mix:<seed>:<i>' -> (indices, rates)."""
    if spec.startswith("mix:"):
        from cognition.calibrate_dynamics import _mixtures
        _, seed, i = spec.split(":")
        idx, rate, types = _mixtures(n, t.to_numpy(), n_mix=int(i) + 1, seed=int(seed))[int(i)]
        print(f"{spec} = {', '.join(ty[4:] for ty in types)}")
        order = np.argsort(idx)
        return idx[order], rate[order]
    idx = np.sort(n[t == spec]["idx"].to_numpy())
    return idx, np.full(len(idx), ODOUR_HZ)


def main():
    import argparse
    global ODOUR_A, ODOUR_B, TRIALS
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=ODOUR_A); ap.add_argument("--b", default=ODOUR_B)
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args(); ODOUR_A, ODOUR_B, TRIALS = args.a, args.b, args.trials
    c = load_connectome(); n = c.neurons
    t = n["primary_type"].fillna("").astype(str)
    e = NativeLIFEngine.from_connectome(c, seed=1, threads=3)
    apply_dynamics(e, c, "calibrated")
    mb = MushroomBody(c, e, plastic=True, kc_mbon_gain=KC_MBON_GAIN)
    odour = {o: _odour(o, n, t) for o in (ODOUR_A, ODOUR_B)}
    dan = np.sort(n[t == DAN_TYPE]["idx"].to_numpy())
    mbon_types = t.to_numpy()[mb.mbon]

    def present(o, seed, with_dan=False):
        e.reset(seed=seed); mb.reset_activity()
        idx, rate = odour[o]
        e.set_poisson(idx, rate)
        for ms in range(ODOUR_MS):
            if with_dan and ms == DAN_FROM_MS:
                e.set_poisson(np.concatenate([idx, dan]),
                              np.concatenate([rate, np.full(len(dan), DAN_HZ)]))
            mb.step(e.run_collect(10))
        return e.spike_counts[mb.mbon].astype(float)

    def test():
        mb.learning = False
        r = {o: np.mean([present(o, s) for s in TEST_SEEDS], axis=0) for o in odour}
        mb.learning = True
        return r

    kc_sets = {}
    for o in odour:
        e.reset(seed=1); e.set_poisson(*odour[o]); e.run(ODOUR_MS)
        kc_sets[o] = set(np.flatnonzero(e.spike_counts[mb.kc] > 0))
    a, b = kc_sets[ODOUR_A], kc_sets[ODOUR_B]
    print(f"Kenyon cells: {ODOUR_A} {len(a)}, {ODOUR_B} {len(b)}, shared {len(a & b)} "
          f"(Jaccard {len(a & b) / max(1, len(a | b)):.2f})")

    before = test()
    for trial in range(TRIALS):                       # A paired with the DAN, B alone
        present(ODOUR_A, seed=100 + trial, with_dan=True)
        present(ODOUR_B, seed=200 + trial, with_dan=False)
    after = test()

    tgt = mbon_types == TARGET
    print(f"\n{TARGET} spikes per trial ({ODOUR_MS} ms odour; mean of {len(TEST_SEEDS)} test trials):")
    res = {}
    for o, label in ((ODOUR_A, "paired with " + DAN_TYPE), (ODOUR_B, "unpaired")):
        pre, post = before[o][tgt].sum(), after[o][tgt].sum()
        res[o] = {"before": pre, "after": post, "change": (post - pre) / max(pre, 1e-9)}
        print(f"  {o:8s} ({label:17s}): {pre:6.1f} -> {post:6.1f}  ({100 * res[o]['change']:+.0f}%)")

    w = mb.weights
    to_t = mbon_types[mb.edge_mbon] == TARGET
    print(f"\nKC->{TARGET} synaptic weights: mean {w[to_t].mean():.3f} "
          f"(min {w[to_t].min():.3f}); other KC->MBON: mean {w[~to_t].mean():.3f}")
    kc_w = np.full(len(mb.kc), np.nan)
    for grp, name in ((a - b, "A-only KCs"), (b - a, "B-only KCs"), (a & b, "shared KCs")):
        sel = to_t & np.isin(mb.edge_kc, list(grp))
        if sel.any():
            print(f"  from {name:10s}: mean weight {w[sel].mean():.3f} over {int(sel.sum())} synapses")
    # which other MBONs changed
    ch = {}
    for mt in np.unique(mbon_types):
        m = mbon_types == mt
        pa, qa = before[ODOUR_A][m].sum(), after[ODOUR_A][m].sum()
        if pa >= 5:
            ch[mt] = (qa - pa) / pa
    top = sorted(ch.items(), key=lambda kv: kv[1])[:6]
    print("\nMBONs whose response to the paired odour changed most:",
          ", ".join(f"{k} {100 * v:+.0f}%" for k, v in top))
    out = config.OUTPUT_DIR / "experiment_04_mb_learning.json"
    out.write_text(json.dumps({"target": TARGET, "dan": DAN_TYPE, "results": res,
                               "mbon_changes_paired": ch}, indent=1, default=float))
    print("\nSaved:", out)


if __name__ == "__main__":
    sys.exit(main())
