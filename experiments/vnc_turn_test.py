"""
Does the male CNS nerve cord turn the fly? Drive the brain's walking and
steering command neurons in the MaleCNS connectome (brain + ventral nerve
cord, FLY_DATASET=malecns) and read the LEG motor neurons: is there a
consistent left/right difference we could read as a turn (for the rover's
wheels), and a forward/backward signature?

  walk      DNg100 both sides (forward walking command; Bidaye et al. 2020)
  walk+L/R  plus DNa02 on one side (steering; Rayshubskiy et al. 2020 bioRxiv)
  walk+a01  plus DNa01 on one side
  back      MDN both sides (backward walking; Bidaye et al. 2014)

Every condition drives the same Poisson set (resting ORNs at 8 Hz + the
command neurons, at 0 Hz when off), paired seeds. Readout: mean rate of leg
motor neurons (super_class vnc_motor, soma in T1/T2/T3) by side and segment,
and the motor neuron types whose left-right difference flips with the side
of DNa02 (exploratory: picked from this same data). Then a readout fixed in
advance from anatomy, not from these results: per side, the summed rate of
the coxa movers (MN types named rotator / promotor / remotor: the muscles
that swing a leg forward and back, i.e. stride), turn index = (R - L) /
(R + L), reported per seed (--seed0 for held-out seeds).

    FLY_DATASET=malecns python experiments/vnc_turn_test.py [--seeds 6] [--hz 80] [--dynamics published]
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

SEGMENTS = ("T1", "T2", "T3")


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
    ap.add_argument("--hz", type=float, default=80.0)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--dynamics", default=os.environ.get("FLY_DYNAMICS", "published"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=500, help="first seed (use another range for held-out checks)")
    a = ap.parse_args()
    if not config.MALE_CNS:
        raise SystemExit("needs FLY_DATASET=malecns (the FAFB brain has no nerve cord)")
    c = load_connectome()
    n = c.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    side = n["side"].fillna("").astype(str).to_numpy()
    sc = n["super_class"].fillna("").astype(str).to_numpy()
    seg = n["soma_neuromere"].fillna("").astype(str).to_numpy()
    leg = np.flatnonzero((sc == "vnc_motor") & np.isin(seg, SEGMENTS) & np.isin(side, ["left", "right"]))
    print(f"leg motor neurons: {len(leg)} "
          + str({(s, g): int(((side[leg] == s) & (seg[leg] == g)).sum()) for s in ("left", "right") for g in SEGMENTS}))
    orn = np.flatnonzero(np.char.startswith(t.astype(str), "ORN_"))
    cmd = {k: np.flatnonzero((t == ty) & (side == sd)) for k, ty, sd in
           (("DNg100_l", "DNg100", "left"), ("DNg100_r", "DNg100", "right"),
            ("DNa02_l", "DNa02", "left"), ("DNa02_r", "DNa02", "right"),
            ("DNa01_l", "DNa01", "left"), ("DNa01_r", "DNa01", "right"))}
    cmd["MDN"] = np.flatnonzero(t == "MDN")
    ids = np.concatenate([orn] + list(cmd.values()))
    order = np.argsort(ids)
    conds = {"none": [], "walk": ["DNg100_l", "DNg100_r"],
             "walk+DNa02 L": ["DNg100_l", "DNg100_r", "DNa02_l"], "walk+DNa02 R": ["DNg100_l", "DNg100_r", "DNa02_r"],
             "walk+DNa01 L": ["DNg100_l", "DNg100_r", "DNa01_l"], "walk+DNa01 R": ["DNg100_l", "DNg100_r", "DNa01_r"],
             "back (MDN)": ["MDN"]}
    e = NativeLIFEngine.from_connectome(c, seed=1, threads=a.threads)
    from simulation.engine.session import apply_calibrated_gain
    apply_calibrated_gain(e)             # the dataset's calibrated gain (male-based: 0.62)
    apply_dynamics(e, c, a.dynamics)
    print(f"dataset {config.DATASET_KEY}, dynamics {a.dynamics}, {a.seeds} seeds x {a.seconds} s, commands at {a.hz} Hz")
    rates = {}
    for name, on in conds.items():
        hz = np.full(len(ids), 0.0)
        hz[:len(orn)] = 8.0
        pos = len(orn)
        for k, v in cmd.items():
            if k in on:
                hz[pos:pos + len(v)] = a.hz
            pos += len(v)
        rows = []
        for s in range(a.seeds):
            e.reset(seed=a.seed0 + s)
            e.set_poisson(ids[order], hz[order])
            e.run(200.0)
            c0 = e.spike_counts.copy()
            e.run(a.seconds * 1000.0)
            rows.append((e.spike_counts - c0) / a.seconds)
        rates[name] = np.array(rows)                       # seeds x neurons
    # --- per side and segment
    print("\nmean leg MN rate (Hz), L / R per segment; active = MNs > 1 Hz")
    for name, R in rates.items():
        m = R.mean(0)
        cells = []
        for g in SEGMENTS:
            l = m[leg[(side[leg] == "left") & (seg[leg] == g)]].mean()
            r = m[leg[(side[leg] == "right") & (seg[leg] == g)]].mean()
            cells.append(f"{g} {l:5.2f}/{r:5.2f}")
        print(f"{name:14s} " + "  ".join(cells) + f"  active {int((m[leg] > 1).sum())}"
              + "  DNs " + " ".join(f"{k}={m[v].mean():.0f}" for k, v in cmd.items()))
    # --- which MN types flip with the side of DNa02 / DNa01
    from scipy.stats import ttest_rel
    types = sorted(set((t[i], seg[i]) for i in leg))
    for steer in ("DNa02", "DNa01"):
        L, Rr = rates[f"walk+{steer} L"], rates[f"walk+{steer} R"]
        out = []
        for ty, g in types:
            il = leg[(t[leg] == ty) & (seg[leg] == g) & (side[leg] == "left")]
            ir = leg[(t[leg] == ty) & (seg[leg] == g) & (side[leg] == "right")]
            if not len(il) or not len(ir):
                continue
            dL = L[:, il].mean(1) - L[:, ir].mean(1)         # left-minus-right, steering left
            dR = Rr[:, il].mean(1) - Rr[:, ir].mean(1)       # ... steering right
            eff = (dL - dR) / 2
            if np.abs(eff).mean() < 0.5:
                continue
            p = ttest_rel(dL, dR).pvalue if np.std(dL - dR) > 0 else 0.0
            out.append((abs(eff.mean()), ty, g, eff.mean(), p))
        out.sort(reverse=True)
        print(f"\n{steer}: leg MN types whose L-R difference follows the steering side "
              f"(effect = half of [L-R | {steer} left] - [L-R | {steer} right], Hz)")
        for _, ty, g, eff, p in out[:15]:
            print(f"   {g} {ty:32s} {eff:+7.2f} Hz  paired p = {p:.3f}")
        if not out:
            print("   none above 0.5 Hz")
    # --- the fixed readout: stride drive per side from the coxa movers
    coxa = leg[np.char.find(np.char.lower(t[leg].astype(str)), "rotator") >= 0]
    coxa = np.union1d(coxa, leg[(np.char.find(np.char.lower(t[leg].astype(str)), "promotor") >= 0)
                                | (np.char.find(np.char.lower(t[leg].astype(str)), "remotor") >= 0)])
    cl, cr = coxa[side[coxa] == "left"], coxa[side[coxa] == "right"]
    print(f"\nfixed readout: coxa movers {len(cl)} left / {len(cr)} right; turn index (R-L)/(R+L) per seed"
          " (+ = right legs drive harder = turning LEFT)")
    for name, R in rates.items():
        L_, R_ = R[:, cl].sum(1), R[:, cr].sum(1)
        ti = np.array([turn_index(row, coxa, t, side, seg) for row in R])     # per-type means
        print(f"   {name:14s} " + " ".join(f"{x:+.3f}" for x in ti) + f"   mean {ti.mean():+.3f}  stride drive {(L_ + R_).mean() / 2:.0f}")


if __name__ == "__main__":
    main()

