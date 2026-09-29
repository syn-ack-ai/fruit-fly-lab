"""
Which head bristles should the lidar's touch drive on a male brain?
(results/touch_subtypes_2026-09-28/)

Drives each candidate set on top of the resting input (calibrated dynamics,
3 seeds x 1 s): both sides at the adapted tonic level (12 Hz) and at a fresh
contact's burst (80 Hz), and one side at 40 Hz. Reports backward walking
(MDN), forward walking (DNg100), proboscis extension (MN9) and steering
(DNa02 right - left). The male "BM" subtypes come from
brain/sensory/bm_subtypes.py.

    FLY_DATASET=merged python -m experiments.touch_subtypes_test
    FLY_DATASET=fafb   python -m experiments.touch_subtypes_test
"""
import numpy as np
import pandas as pd

import config
from cognition.exam import core
c = core.connectome(); n = c.neurons
t = n.primary_type.fillna("").astype(str).to_numpy(); sd = n.side.fillna("").astype(str).to_numpy()
g = {k: np.flatnonzero(t == k) for k in ("MDN", "DNg100", "MN9")}
g = {k: v for k, v in g.items() if len(v)}          # FAFB has no cell typed MN9
aL, aR = np.flatnonzero((t == "DNa02") & (sd == "left")), np.flatnonzero((t == "DNa02") & (sd == "right"))
ctx = core.Ctx({"dynamics": "calibrated"}); ri, rr = core.resting()
sets = {}
if config.MALE_CNS:
    from brain.sensory.bm_subtypes import load
    a = load()
    pos = pd.Series(np.arange(len(n)), index=n.root_id.to_numpy())
    ix = pos.reindex(a.root_id)
    assert ix.notna().all(), "bm_subtypes_malecns.csv has root ids this connectome lacks"
    a["idx"] = ix.to_numpy().astype(np.int64)
    vib = np.flatnonzero(t == "BM_Vib")
    sub = lambda ss: np.concatenate([a.idx[a.subtype.isin(ss)].to_numpy(), vib])
    sets["all BM + Vib (now)"] = np.concatenate([np.flatnonzero(t == "BM"), vib])
    sets["Ant+Fr+InOc + Vib"] = sub(["BM_Ant", "BM_Fr", "BM_InOc"])
    sets["Ant + Vib"] = sub(["BM_Ant"])
    sets["Vib only"] = vib
else:
    sets["FAFB Ant + Vib"] = np.flatnonzero(np.isin(t, ["BM_Ant", "BM_Vib"]))
for name, idx in sets.items():
    idx = idx[np.isin(sd[idx], ["left", "right"])]
    L, R = idx[sd[idx] == "left"], idx[sd[idx] == "right"]
    out = []
    for lab, drive in (("both 12Hz", [(idx, 12.0)]), ("both 80Hz", [(idx, 80.0)]), ("L 40", [(L, 40.0)]), ("R 40", [(R, 40.0)])):
        r = ctx.rates([(ri, rr)] + drive, 1000.0, [1, 2, 3])
        out.append(f"{lab}: " + " ".join(f"{k} {r[v].mean():4.1f}" for k, v in g.items())
                   + f" a02 R-L {r[aR].mean() - r[aL].mean():+5.1f}")
    print(f"{name} (L {len(L)} / R {len(R)})\n   " + "\n   ".join(out), flush=True)
r = ctx.rates([(ri, rr)], 1000.0, [1, 2, 3])
print("rest: " + " ".join(f"{k} {r[v].mean():4.1f}" for k, v in g.items()) + f" a02 R-L {r[aR].mean() - r[aL].mean():+5.1f}")
