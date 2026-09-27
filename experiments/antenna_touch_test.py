"""
What does the connectome do when its antennae (and vibrissae) on one side are
touched? The lidar's "touch" goes there (robot/lidar.py). Read the steering
(DNa01/DNa02 left-right), walking (DNg100), backing up (MDN), antennal
grooming (DNge078 = aDN) and escape (DNp01, DNp11) neurons. Every condition
drives the same Poisson set (resting ORNs + BM_Ant/BM_Vib both sides), the
bristles at 0 Hz when off.

    python experiments/antenna_touch_test.py [--seeds 6] [--hz 80]
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

READ = ("DNa01", "DNa02", "DNg100", "MDN", "DNge078", "DNp01", "DNp11", "DNg13")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=80.0)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    c = load_connectome()
    t = c.neurons["primary_type"].fillna("").astype(str).to_numpy()
    side = c.neurons["side"].fillna("").astype(str).to_numpy()
    orn = np.flatnonzero(np.char.startswith(t.astype(str), "ORN_"))
    bm = np.flatnonzero(np.isin(t, ["BM_Ant", "BM_Vib"]) & np.isin(side, ["left", "right"]))
    bm_right = side[bm] == "right"
    ids = np.concatenate([orn, bm])
    order = np.argsort(ids)
    e = NativeLIFEngine.from_connectome(c, seed=1, threads=4)
    from simulation.engine.session import apply_calibrated_gain
    apply_calibrated_gain(e)             # the dataset's calibrated gain (male-based: 0.62)
    apply_dynamics(e, c, "calibrated")
    cell = lambda ty, sd: np.flatnonzero((t == ty) & (side == sd))
    print(f"BM_Ant+BM_Vib: {len(bm)} ({bm_right.sum()} right); {a.seeds} seeds x {a.seconds} s at {a.hz} Hz")
    res = {}
    for name, l, r in (("none", 0, 0), ("left touched", a.hz, 0), ("right touched", 0, a.hz), ("front (both)", a.hz, a.hz)):
        hz = np.concatenate([np.full(len(orn), 8.0), np.where(bm_right, float(r), float(l))])
        rows = []
        for s in range(a.seeds):
            e.reset(seed=300 + s)
            e.set_poisson(ids[order], hz[order])
            e.run(200.0)
            c0 = e.spike_counts.copy()
            e.run(a.seconds * 1000.0)
            d = (e.spike_counts - c0) / a.seconds
            rows.append({f"{ty}_{sd[0]}": float(d[cell(ty, sd)].mean()) if len(cell(ty, sd)) else np.nan
                         for ty in READ for sd in ("left", "right")})
        res[name] = {k: np.array([r_[k] for r_ in rows]) for k in rows[0]}
    keys = [f"{ty}_{s}" for ty in READ for s in "lr"]
    print("%-15s" % "Hz" + "".join("%9s" % k for k in keys))
    for name, v in res.items():
        print("%-15s" % name + "".join("%9.1f" % np.nanmean(v[k]) for k in keys))
    from scipy.stats import ttest_rel
    for ty in ("DNa01", "DNa02"):
        for name in ("left touched", "right touched"):
            lr = res[name][f"{ty}_r"] - res[name][f"{ty}_l"]
            base = res["none"][f"{ty}_r"] - res["none"][f"{ty}_l"]
            print(f"{ty} R-L, {name}: {lr.mean():+.1f} Hz vs none {base.mean():+.1f}  paired p = {ttest_rel(lr, base).pvalue:.3f}")


if __name__ == "__main__":
    main()
