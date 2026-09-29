"""
Head-bristle subtypes of the male CNS's untyped "BM" cells.

The MaleCNS types 69 head bristle mechanosensory neurons only as "BM", with
the FlyWire type "BM_Ant,BM_Fr,BM_FrOr,BM_InOc,BM_Oc,BM_Or,BM_Vib"; all enter
through the antennal nerve (BM_Vib, typed separately, through the
maxillary-labial nerve). FlyWire FAFB labels these subtypes. Each male cell is
assigned the FAFB subtype whose mean OUTPUT profile (fraction of its synapses
onto each postsynaptic cell type, in FAFB's names: a male partner's
unambiguous flywire_type, else its type if FAFB has it) is closest (cosine).

Validation: on FAFB itself, leave-one-out, 72/81 cells are assigned their own
subtype (BM_Ant 38/41). Why it matters: driven alone (40 Hz, resting input),
the cells assigned fronto-orbital (BM_FrOr, 22) back the male fly up (MDN 23
Hz) and extend its proboscis (MN9 37 Hz), orbital (BM_Or, 8) backs it (MDN
13); antennal, frontal and inter-ocular do neither (MDN 4-5, MN9 <= 2). The
lidar's "antennal touch" (robot/lidar.LidarTouch) had driven all 69 and kept
the robot backing and freezing at walls; it now drives TOUCH_SUBTYPES
(2026-09-28, results/touch_subtypes_2026-09-28/).

    python -m brain.sensory.bm_subtypes      # -> data/metadata/bm_subtypes_malecns.csv

(The same neurons in merged and malecns; merged's connectivity, as the robot
uses it. Merge note: rule 2 mirrored "BM" as one group by rank, ignoring these
subtypes, so the sides' subtype mixes differ: BM_Ant 16 left / 3 right.)
"""
from __future__ import annotations

import importlib
import os

import numpy as np
import pandas as pd

SUBTYPES = ["BM_Ant", "BM_Fr", "BM_FrOr", "BM_InOc", "BM_Oc", "BM_Or"]
OUT_NAME = "bm_subtypes_malecns.csv"      # in data/metadata (root ids: malecns and merged)
# the subtypes the lidar's touch drives on male brains (robot/lidar.LidarTouch)
TOUCH_SUBTYPES = ("BM_Ant", "BM_Fr", "BM_InOc")


def _load(ds: str):
    os.environ["FLY_DATASET"] = ds
    import config
    importlib.reload(config)
    import brain.neurons.registry as reg
    importlib.reload(reg)
    return reg.load_connectome.__wrapped__()


def _profiles(W, rows, names) -> list:
    W = abs(W.tocsr())
    out = []
    for i in rows:
        r = W.getrow(i)
        s = pd.Series(r.data, index=names[r.indices])
        s = s[s.index != ""].groupby(level=0).sum()
        out.append(s / s.sum() if s.sum() else s)
    return out


def _cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def assign(male_ds: str = "merged") -> tuple:
    """(male assignment DataFrame, FAFB leave-one-out accuracy)."""
    F = _load("fafb")
    tf = F.neurons["primary_type"].fillna("").astype(str).to_numpy()
    M = _load(male_ds)
    nm = M.neurons
    tm = nm["primary_type"].fillna("").astype(str).to_numpy()
    fw = nm["flywire_type"].fillna("").astype(str).to_numpy()
    ftypes = set(tf) - {""}
    mname = np.where((fw != "") & (np.char.find(fw.astype(str), ",") < 0), fw,
                     np.where(np.isin(tm, list(ftypes)), tm, ""))
    frows = np.flatnonzero(np.isin(tf, SUBTYPES))
    fprof = _profiles(F.w, frows, tf)
    flab = tf[frows]
    keys = sorted(set().union(*[set(p.index) for p in fprof]))
    X = np.array([p.reindex(keys).fillna(0).to_numpy() for p in fprof])
    cent = lambda Xs, lab: {s: Xs[lab == s].mean(0) for s in SUBTYPES if (lab == s).any()}
    hit = 0
    for i in range(len(X)):
        m = np.ones(len(X), bool)
        m[i] = False
        C = cent(X[m], flab[m])
        hit += max(C, key=lambda s: _cos(X[i], C[s])) == flab[i]
    C = cent(X, flab)
    mrows = np.flatnonzero(tm == "BM")
    Y = np.array([p.reindex(keys).fillna(0).to_numpy() for p in _profiles(M.w, mrows, mname)])
    sims = np.array([[_cos(y, C[s]) for s in SUBTYPES] for y in Y])
    order = np.sort(sims, 1)
    df = pd.DataFrame({"root_id": nm["root_id"].to_numpy()[mrows], "side": nm["side"].astype(str).to_numpy()[mrows],
                       "subtype": np.array(SUBTYPES)[sims.argmax(1)], "cos": order[:, -1].round(4),
                       "margin": (order[:, -1] - order[:, -2]).round(4)})
    for j, s in enumerate(SUBTYPES):
        df["cos_" + s] = sims[:, j].round(4)
    return df, hit / len(X)


def load():
    """The saved assignment (root_id, side, subtype, ...), or None."""
    import config
    p = config.METADATA_DIR / OUT_NAME
    return pd.read_csv(p) if p.exists() else None


def main():
    df, acc = assign("merged")
    import config
    out = config.METADATA_DIR / OUT_NAME
    df.to_csv(out, index=False)
    print(f"FAFB leave-one-out {acc:.0%}; wrote {out}")
    print(df.groupby(["subtype", "side"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
