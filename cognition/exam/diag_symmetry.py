"""
Where does the left/right asymmetry of odour steering arise?

Odour (fermenting fruit, as in the exam's odour_lateralization test) at the
left antenna only, the right antenna only, or neither; every neuron's rate.
For each cell type present on both sides, its response to IPSILATERAL odour
on the left (left cells, left odour) is compared with the mirror case (right
cells, right odour), and likewise contralateral. Types are listed by their
synaptic distance from the ORNs, so the first stage that breaks the mirror
symmetry shows up first.

    python -m cognition.exam.diag_symmetry --dynamics calibrated --out sym.npz
"""
from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from cognition.exam import core

MS = 1500.0


def hops_from_orns(c, max_hops=6):
    t = c.neurons["primary_type"].fillna("").astype(str).to_numpy()
    src = np.flatnonzero(np.char.startswith(t.astype(str), "ORN_"))
    w = c.w.tocsr()
    hop = np.full(c.n, -1)
    hop[src] = 0
    q = deque(src.tolist())
    while q:
        i = q.popleft()
        if hop[i] >= max_hops:
            continue
        for j in w.indices[w.indptr[i]:w.indptr[i + 1]]:
            if hop[j] < 0:
                hop[j] = hop[i] + 1
                q.append(j)
    return hop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics", default="calibrated")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--load", default=None, help="analyse saved rates instead of simulating")
    a = ap.parse_args()
    if a.load:
        R = dict(np.load(a.load))
        c = core.connectome()
    else:
        ctx = core.Ctx({"dynamics": a.dynamics})
        _, _, _, osp = core.orn_space()
        od = osp.sources["fermenting_fruit"] * 0.5
        z = np.zeros_like(od)
        R = {k: ctx.rates(core.odour_inputs(cl, cr), MS, a.seeds)
             for k, (cl, cr) in {"L": (od, z), "R": (z, od), "0": (z, z)}.items()}
        c = ctx.c
        if a.out:
            np.savez_compressed(a.out, **R)
    n = c.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    side = n["side"].fillna("").astype(str).to_numpy()
    hop = hops_from_orns(c)
    ro = core.readout()
    for k in ("L", "R", "0"):
        ch = ro.channels(R[k] * MS / 1000.0, MS)
        print(f"odour {k}: turn_bias {ch['turn_bias']:+.3f}")
    rows = []
    for ty in sorted(set(t[(hop > 0) & (hop <= 5)])):
        if not ty:
            continue
        Lc = np.flatnonzero((t == ty) & (side == "left"))
        Rc = np.flatnonzero((t == ty) & (side == "right"))
        if not len(Lc) or not len(Rc):
            continue
        ipsiL = R["L"][Lc].mean() - R["0"][Lc].mean()
        ipsiR = R["R"][Rc].mean() - R["0"][Rc].mean()
        conL = R["R"][Lc].mean() - R["0"][Lc].mean()
        conR = R["L"][Rc].mean() - R["0"][Rc].mean()
        restL, restR = R["0"][Lc].mean(), R["0"][Rc].mean()
        big = max(abs(ipsiL), abs(ipsiR), abs(conL), abs(conR), abs(restL - restR))
        if big < 2.0:
            continue
        asym = abs(ipsiL - ipsiR) + abs(conL - conR) + abs(restL - restR)
        rows.append((int(hop[Lc].min()), ty, len(Lc), len(Rc), restL, restR, ipsiL, ipsiR, conL, conR,
                     asym / (big + 1e-9)))
    rows.sort(key=lambda r: (r[0], -r[-1]))
    print("hop type nL nR | rest L/R | ipsi L/R | contra L/R | asym/size")
    shown = 0
    for r in rows:
        if r[-1] < 0.5:
            continue
        print("%d %-18s %3d %3d | %6.1f %6.1f | %+7.1f %+7.1f | %+7.1f %+7.1f | %.2f" % r)
        shown += 1
        if shown >= a.top:
            break


if __name__ == "__main__":
    main()
