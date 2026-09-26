"""
Map the sensory modalities (defined on FlyWire FAFB) onto the Janelia MaleCNS.

The stimuli in brain/sensory/modalities.py name FlyWire cell types, or for
sugar and bitter taste a list of individually labelled FlyWire neurons. The
MaleCNS annotates cell types independently, often more finely (FlyWire JO-B is
JO-B1_a, JO-B3, ... there), and carries a `flywireType` cross-reference.

For each modality:
  1. type match: MaleCNS neurons whose flywireType or own type equals a FlyWire
     type of the modality, or is a subtype of it (JO-B -> JO-B1_a, JO-B-unclear;
     LB3 -> LB3a; but JO-E does not absorb JO-EV1, which has its own entry, and
     LC4 does not absorb LC40, a different numbered type).
  2. if the FlyWire definition is a labelled subset of a type (sugar and bitter
     GRNs are 23 of 122 LB3 and 38 LB1 neurons), candidate MaleCNS subtypes are
     ranked by connectivity fingerprint: cosine similarity between the synapses
     they send to each downstream cell type (by FlyWire type) and those the
     labelled FlyWire neurons send. The best subtype is kept, plus any other
     within FINGERPRINT_KEEP of it.
The result, with every choice and score, is written to
data/derived/malecns/modality_map.json and read by resolve_neurons() when
FLY_DATASET=malecns.

Run:  python -m brain.sensory.crossmap
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

FINGERPRINT_KEEP = 0.9


def _subtype_of(child: str, parent: str) -> bool:
    if child == parent:
        return True
    if not child.startswith(parent):
        return False
    rest = child[len(parent):]
    if parent[-1].isdigit() and rest[0].isdigit():
        return False                  # LC4 -> LC40 is a different type, not a subtype
    return bool(re.match(r"^([0-9_\-]|[a-z])", rest))


def _output_profile(c, idx, types_of) -> pd.Series:
    """Synapses from `idx` to each downstream cell type (FlyWire naming)."""
    sub = abs(c.w[np.asarray(idx), :]).tocoo()
    t = types_of[sub.col]
    s = pd.Series(sub.data, index=t)
    s = s[s.index != ""]
    return s.groupby(level=0).sum()


def _cos(a: pd.Series, b: pd.Series) -> float:
    k = a.index.union(b.index)
    x, y = a.reindex(k, fill_value=0).to_numpy(float), b.reindex(k, fill_value=0).to_numpy(float)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def build(fafb, mcns) -> dict:
    from brain.sensory.modalities import ALL_MODALITIES, resolve_neurons
    mn = mcns.neurons
    # (pandas 3 keeps missing values as NaN through astype(str): fill first)
    ft = mn["flywire_type"].fillna("").astype(str).to_numpy()
    ty = mn["primary_type"].fillna("").astype(str).to_numpy()
    # FlyWire-convention type for every MaleCNS neuron (for fingerprints)
    mc_fw = np.where(ft != "", ft, ty)
    fafb_types = fafb.neurons["primary_type"].fillna("").astype(str).to_numpy()

    out = {}
    for m in ALL_MODALITIES:
        if not m.supported:
            continue
        fi = resolve_neurons(m, fafb)
        ftypes = sorted(set(fafb_types[fi]) - {""}) if len(fi) else []
        types = sorted(set(m.cell_types) | set(ftypes))
        match = np.array([bool(a or b) and any(_subtype_of(a, t) or _subtype_of(b, t) for t in types)
                          for a, b in zip(ft, ty)])
        entry = {"fafb_n": int(len(fi)), "fafb_types": types, "method": "type",
                 "candidates": {}}
        idx = np.flatnonzero(match)
        # labelled subset of a type: choose MaleCNS subtypes by fingerprint
        n_fafb_type = int(np.isin(fafb_types, ftypes).sum()) if ftypes else 0
        if m.label_group and len(fi) and n_fafb_type > 1.3 * len(fi):
            ref = _output_profile(fafb, fi, fafb_types)
            scores = {}
            for sub in sorted(set(ty[idx])):
                sidx = idx[ty[idx] == sub]
                scores[sub] = (_cos(ref, _output_profile(mcns, sidx, mc_fw)), int(len(sidx)))
            best = max(s for s, _ in scores.values())
            keep = [k for k, (s, _) in scores.items() if s >= FINGERPRINT_KEEP * best]
            idx = idx[np.isin(ty[idx], keep)]
            entry.update(method="type + connectivity fingerprint",
                         candidates={k: {"cosine": round(s, 3), "n": n} for k, (s, n) in scores.items()},
                         chosen=keep)
        entry["malecns_n"] = int(len(idx))
        entry["malecns_types"] = dict(pd.Series(ty[idx]).value_counts().head(20))
        entry["idx"] = [int(x) for x in idx]
        out[m.key] = entry
    return out


def main():
    import subprocess, sys
    os.environ["FLY_DATASET"] = "fafb"
    from brain.neurons.registry import load_connectome
    fafb = load_connectome()
    # a second interpreter loads the MaleCNS (config is fixed per process)
    code = ("import os, pickle, sys; os.environ['FLY_DATASET']='malecns';"
            "from brain.neurons.registry import load_connectome; c=load_connectome();"
            "pickle.dump((c.neurons, c.w, c.manifest), sys.stdout.buffer)")
    import pickle
    blob = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True).stdout
    from brain.neurons.registry import Connectome
    mcns = Connectome(*pickle.loads(blob))
    res = build(fafb, mcns)
    import config
    path = config.DERIVED_DIR / "malecns" / "modality_map.json"
    path.write_text(json.dumps(res, indent=1, default=int))
    for k, v in res.items():
        extra = ""
        if "chosen" in v:
            extra = "  chosen " + ", ".join(f"{c} (cos {v['candidates'][c]['cosine']})" for c in v["chosen"])
            extra += " | rejected " + ", ".join(f"{c} ({d['cosine']})" for c, d in v["candidates"].items()
                                                  if c not in v["chosen"])
        print(f"{k:16s} FlyWire {v['fafb_n']:5d} -> MaleCNS {v['malecns_n']:5d}  [{v['method']}]{extra}")
    print("wrote", path)


if __name__ == "__main__":
    main()
