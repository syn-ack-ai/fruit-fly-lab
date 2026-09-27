"""
Does top-down drive to the fly's sleep-promoting neurons quiet its walking
command? (calibrated dynamics; odour pushes DNg100 = BDN2 up)

Sleep handles (research 2026-09-26):
  dFB (R23E10 dorsal fan-shaped body; Hulse et al. 2021 matched FB6A, FB6C,
  FB6E, FB6G, FB6I, FB6Z, FB7A, FB7K; Donlea et al. 2011; Jones et al. 2025)
  ER5 ring neurons (sleep drive; Liu et al. 2016)

    python experiments/sleep_drive_test.py [--hz 80] [--seconds 2] [--seeds 6]

Result (Pi, 6 seeds x 2 s, 2026-09-26): DNg100 odour 24.3 +- 2.8 Hz; + ER5
19.9 (-18%, paired p = 0.075; ExR1 0.4 -> 0.0); + dFB 22.3 (-8%, p = 0.42);
+ both 21.7 (-11%, p = 0.026).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.neurons.registry import load_connectome                     # noqa: E402
from native.lif_native import NativeLIFEngine                          # noqa: E402
from simulation.engine.session import apply_dynamics                   # noqa: E402

DFB = ("FB6A", "FB6C", "FB6E", "FB6G", "FB6I", "FB6Z", "FB7A", "FB7K")
READ = ("DNg100", "DNa02", "DNa01", "DNp09", "MDN", "hDeltaF", "ExR1", "ER5", "PFL3", "FB6A", "FB6H", "FB5H")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=80.0)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    c = load_connectome()
    t = c.neurons["primary_type"].fillna("").astype(str).to_numpy()
    idx = lambda types: np.flatnonzero(np.isin(t, list(types)))
    orn = np.flatnonzero(np.char.startswith(t.astype(str), "ORN_"))
    from brain.sensory.olfaction import OlfactorySpace
    rest = OlfactorySpace([x[4:] for x in sorted({x for x in t if x.startswith("ORN_")})])
    base = np.where(np.isin(t[orn], ["ORN_DM1", "ORN_DM2", "ORN_VA2", "ORN_DM4"]), 120.0, 8.0)
    er5 = idx(["ER5"])
    dfb = idx(DFB)
    # Every condition drives the SAME Poisson set (ORNs + ER5 + dFB), with the
    # sleep neurons at 0 Hz when "off" (review 2026-09-26: adding them to the
    # set only when on re-indexed the ORNs' Poisson draws and made them
    # non-refractory, a confound as large as the effect).
    ids = np.concatenate([orn, er5, dfb])
    order = np.argsort(ids)
    rest_orn = np.full(len(orn), 8.0)
    conds = {"rest (spontaneous ORNs)": (rest_orn, 0, 0),
             "odour": (base, 0, 0),
             "odour + dFB": (base, 0, a.hz),
             "odour + ER5": (base, a.hz, 0),
             "odour + dFB + ER5": (base, a.hz, a.hz)}
    e = NativeLIFEngine.from_connectome(c, seed=1, threads=4)
    from simulation.engine.session import apply_calibrated_gain
    apply_calibrated_gain(e)             # the dataset's calibrated gain (male-based: 0.62)
    apply_dynamics(e, c, "calibrated")
    print("driven: dFB", len(dfb), "cells; ER5", len(er5), "cells; at", a.hz, "Hz;", a.seeds, "seeds x", a.seconds, "s")
    res = {}
    for name, (orn_hz, er5_hz, dfb_hz) in conds.items():
        hz = np.concatenate([orn_hz, np.full(len(er5), float(er5_hz)), np.full(len(dfb), float(dfb_hz))])
        per_seed = []
        for s_ in range(a.seeds):
            e.reset(seed=100 + s_)
            e.set_poisson(ids[order], hz[order])
            e.run(300.0)                                   # settle
            c0 = e.spike_counts.copy()
            e.run(a.seconds * 1000.0)
            per_seed.append((e.spike_counts - c0) / a.seconds)
        P = np.array(per_seed)
        res[name] = {k: (P[:, idx([k])].mean(axis=1) if len(idx([k])) else np.full(a.seeds, np.nan)) for k in READ}
    print("%-24s" % "Hz, mean +- sd over seeds" + "".join("%13s" % k for k in READ))
    for name, v in res.items():
        print("%-24s" % name + "".join("%7.1f+-%-5.1f" % (v[k].mean(), v[k].std(ddof=1)) for k in READ))
    from scipy.stats import ttest_rel
    b = res["odour"]["DNg100"]
    for name in ("odour + dFB", "odour + ER5", "odour + dFB + ER5"):
        x = res[name]["DNg100"]
        print(f"DNg100 {name} vs odour: {100 * (x.mean() / b.mean() - 1):+.0f}%  paired t p = {ttest_rel(x, b).pvalue:.3f}")


if __name__ == "__main__":
    main()
