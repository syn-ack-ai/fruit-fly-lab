"""
Are two brains' battery-pet lifetimes different? Per condition and per-day
metric (score_pets.py), a Mann-Whitney U test over days and whether every
seed's mean of one brain lies beyond every seed's mean of the other. Days within
one seed share a learned map and battery, so they are not independent, and with
3 vs 3 seeds complete separation happens by chance 10% of the time (review
2026-09-27): a difference is marked REAL only if the day test survives a
Bonferroni correction over all tests AND all seeds are apart.

    .venv/bin/python -m sim.habitat_bridge.compare_brains_stats \\
        simulation/outputs/habitat/real_v1_fafb simulation/outputs/habitat/real_v3_merged
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.stats import mannwhitneyu

from sim.habitat_bridge.score_pets import load

KEYS = ("furniture", "person", "brake", "pinned", "walked", "company",
        "speed_cv", "reversals", "bouts", "orient", "eating_s")


def main(a_root: str, b_root: str) -> None:
    A, B = load(a_root), load(b_root)
    conds = sorted(set(A) & set(B))
    alpha = 0.05 / (len(conds) * len(KEYS))
    print(f"A = {a_root}\nB = {b_root}\nREAL: p(days) < {alpha:.2g} (Bonferroni) and all seeds apart")
    for cond in conds:
        print("==", cond)
        for k in KEYS:
            a = [d[k] for s in A[cond].values() for d in s if k in d]
            b = [d[k] for s in B[cond].values() for d in s if k in d]
            if not a or not b:
                continue
            sa = [np.mean([d[k] for d in s if k in d]) for s in A[cond].values()]
            sb = [np.mean([d[k] for d in s if k in d]) for s in B[cond].values()]
            p = 1.0 if len(set(a) | set(b)) == 1 else float(mannwhitneyu(a, b).pvalue)
            apart = max(sa) < min(sb) or max(sb) < min(sa)
            tag = "REAL" if (apart and p < alpha) else ("seeds apart only" if apart else "")
            print(f"  {k:10s} A {np.mean(a):7.2f}  B {np.mean(b):7.2f}  p(days)={p:.3g}  {tag}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
