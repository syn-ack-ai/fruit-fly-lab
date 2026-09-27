"""
Can "full" live inside the connectome? Hunger/satiety handles with synaptic
outputs in FlyWire, tested on the feeding reflex (MN9 proboscis motor neurons)
to a weak and a strong sugar taste (calibrated dynamics):

  ISN   interoceptive SEZ neurons: sense hunger, promote sugar ingestion
        (Jourjine et al. 2016 Cell 166:855)
  Hugin SEZ hugin neurons: suppress feeding initiation (Melcher & Pankratz
        2005 PLoS Biol 3:e305; Schoofs et al. 2014)

Every condition drives the SAME Poisson set (sugar GRNs + ISN + Hugin), the
handles at 0 Hz when off (a changing Poisson set shifts the RNG streams).

    python experiments/satiety_drive_test.py [--seeds 6] [--hz 80]

Result (Pi, 6 seeds x 1 s, 2026-09-26): proboscis motor neurons to sugar at
20 / 60 / 120 Hz: 8.0 / 18.4 / 25.6 Hz. Hugin at 80 Hz: no change at all (its
synaptic outputs do not reach the feeding circuit; its effects are hormonal).
ISN at 80 Hz: -10% / -15% (p = 0.03, one of 6 uncorrected comparisons) / -6%, i.e. if anything the WRONG direction (real
ISNs promote sugar intake, largely through neuropeptides). The fast-synapse
model cannot carry hunger through these cells; the pet's satiety therefore
acts on sugar-taste and food-odour sensitivity (sim/habitat_bridge/home.py),
which is itself the documented effect of hunger state in flies.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.neurons.registry import load_connectome                     # noqa: E402
from brain.sensory.modalities import BY_KEY, resolve_neurons           # noqa: E402
from native.lif_native import NativeLIFEngine                          # noqa: E402
from simulation.engine.session import apply_dynamics                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=80.0)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    c = load_connectome()
    t = c.neurons["primary_type"].fillna("").astype(str).to_numpy()
    sugar = np.sort(np.asarray(resolve_neurons(BY_KEY["taste_sugar"], c), np.int64))
    isn = np.flatnonzero(t == "ISN")
    hug = np.flatnonzero(t == "SEZ_NSC_Hugin")
    from brain.motor.descending import DescendingReadout
    mn9 = np.asarray(DescendingReadout(c).proboscis_idx, np.int64)   # the model's proboscis motor neurons
    ids = np.concatenate([sugar, isn, hug])
    order = np.argsort(ids)
    e = NativeLIFEngine.from_connectome(c, seed=1, threads=4)
    from simulation.engine.session import apply_calibrated_gain
    apply_calibrated_gain(e)             # the dataset's calibrated gain (male-based: 0.62)
    apply_dynamics(e, c, "calibrated")
    print(f"sugar GRNs {len(sugar)}, ISN {len(isn)}, Hugin {len(hug)}, proboscis motor neurons {len(mn9)}; {a.seeds} seeds x {a.seconds} s")
    rows = []
    for sugar_hz in (20.0, 60.0, 120.0):
        for name, isn_hz, hug_hz in (("none", 0, 0), ("+ ISN (hungry)", a.hz, 0), ("+ Hugin (full)", 0, a.hz)):
            hz = np.concatenate([np.full(len(sugar), sugar_hz), np.full(len(isn), float(isn_hz)),
                                 np.full(len(hug), float(hug_hz))])
            r = []
            for s in range(a.seeds):
                e.reset(seed=200 + s)
                e.set_poisson(ids[order], hz[order])
                e.run(200.0)
                c0 = e.spike_counts.copy()
                e.run(a.seconds * 1000.0)
                r.append(float((e.spike_counts - c0)[mn9].mean() / a.seconds))
            rows.append((sugar_hz, name, np.array(r)))
    from scipy.stats import ttest_rel
    base = {s: r for s, n, r in rows if n == "none"}
    print("%-10s %-16s %s" % ("sugar Hz", "handle", "MN9 Hz mean +- sd   vs none"))
    for s, n, r in rows:
        extra = "" if n == "none" else "  %+.0f%%  p = %.3f" % (100 * (r.mean() / max(base[s].mean(), 1e-9) - 1), ttest_rel(r, base[s]).pvalue)
        print("%-10g %-16s %6.1f +- %-5.1f%s" % (s, n, r.mean(), r.std(ddof=1), extra))


if __name__ == "__main__":
    main()
