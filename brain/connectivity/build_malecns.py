"""
Build a simulation-ready connectome from the Janelia MaleCNS v1.0 release
(brain AND ventral nerve cord of one male fly), in the same format as the
FlyWire FAFB build, so the same engine, session and tools can run it.

Source (CC-BY 4.0): https://male-cns.janelia.org/download/
    body-annotations-male-cns-v1.0-minconf-0.5.feather
    body-neurotransmitters-male-cns-v1.0.feather
    connectome-weights-male-cns-v1.0-minconf-0.5.feather

Choices, all recorded in the manifest:
  - neurons: bodies with status "Traced" (fragments, glia, orphans excluded)
  - connections: >= MIN_SYNAPSES synapses between two traced neurons (the
    same threshold as FlyWire's connections table used for FAFB)
  - sign of a neuron's output from its consensus neurotransmitter, falling
    back to the cell-type prediction: ACh/DA/5-HT/OA excitatory; GABA,
    glutamate and HISTAMINE inhibitory (FlyEM predicts histamine, which the
    FlyWire release did not). Unresolved neurons make no output.

Run:  python -m brain.connectivity.build_malecns /path/to/malecns_v1.0
Then: FLY_DATASET=malecns python -m ...   (see config.py)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf
import scipy.sparse as sp

MIN_SYNAPSES = 5
ORDER_SEED = 783   # neurons are stored in a fixed pseudo-random order (see below)
VOXEL_NM = 8.0
NT_ABBR = {"acetylcholine": "ACH", "gaba": "GABA", "glutamate": "GLUT", "dopamine": "DA",
           "serotonin": "SER", "octopamine": "OCT", "histamine": "HIST"}
EXCITATORY = {"ACH", "DA", "SER", "OCT"}
INHIBITORY = {"GABA", "GLUT", "HIST"}


def _log(m):
    print("[malecns] " + m, flush=True)


def _flow(superclass):
    s = str(superclass)
    if "sensory" in s:
        return "afferent"
    if "motor" in s or "efferent" in s or "endocrine" in s:
        return "efferent"
    return "intrinsic"


def build(src: Path, out_dir: Path, meta_dir: Path) -> dict:
    t0 = time.time()
    ann = pd.read_feather(src / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    nts = pd.read_feather(src / "body-neurotransmitters-male-cns-v1.0.feather")
    neu = ann[ann["status"] == "Traced"].copy()
    # Fixed pseudo-random order rather than by body ID: low IDs are the large,
    # busy, early-annotated neurons, and the engine gives each thread a
    # contiguous index range, so ID order puts most of the work on one thread.
    # The order changes nothing else (each neuron keeps its bodyId).
    neu = neu.sort_values("bodyId").reset_index(drop=True)
    neu = neu.iloc[np.random.default_rng(ORDER_SEED).permutation(len(neu))].reset_index(drop=True)
    neu["idx"] = np.arange(len(neu), dtype=np.int64)
    _log("traced neurons: %d" % len(neu))

    # --- neurotransmitter -> sign -----------------------------------------
    nts = nts.set_index("body")
    cons = neu["bodyId"].map(nts["consensus_nt"])
    ctype = neu["bodyId"].map(nts["celltype_predicted_nt"])
    nt = cons.where(cons.notna() & (cons != "unclear"), ctype)
    nt = nt.where(nt.notna() & (nt != "unclear"), None)
    neu["nt_resolved"] = nt.map(NT_ABBR).fillna("")
    neu["nt_type_score"] = neu["bodyId"].map(nts["predicted_nt_confidence"])
    neu["sign"] = np.where(neu["nt_resolved"].isin(EXCITATORY), 1,
                           np.where(neu["nt_resolved"].isin(INHIBITORY), -1, 0)).astype(np.int8)
    _log("signs: %d excitatory, %d inhibitory, %d unresolved" % (
        (neu.sign == 1).sum(), (neu.sign == -1).sum(), (neu.sign == 0).sum()))

    # --- connections among traced neurons ------------------------------------
    _log("reading connection weights ...")
    w = pf.read_table(src / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    ids = pa.array(neu["bodyId"].to_numpy())
    keep = pc.and_(pc.and_(pc.is_in(w["body_pre"], value_set=ids),
                           pc.is_in(w["body_post"], value_set=ids)),
                   pc.greater_equal(w["weight"], MIN_SYNAPSES))
    w = w.filter(keep)
    pre = w["body_pre"].to_numpy(); post = w["body_post"].to_numpy()
    cnt = w["weight"].to_numpy().astype(np.int64)
    del w
    pos = pd.Index(neu["bodyId"].to_numpy())
    i, j = pos.get_indexer(pre), pos.get_indexer(post)
    sign = neu["sign"].to_numpy()[i]
    m = sign != 0
    n = len(neu)
    W = sp.csr_matrix((cnt[m] * sign[m], (i[m], j[m])), shape=(n, n), dtype=np.int64)
    W.sum_duplicates(); W.sort_indices()
    if np.abs(W.data).max() > 32767:
        raise ValueError("synapse counts exceed int16")
    _log("connections: %d pairs (>= %d synapses), %d synapses; %d pairs dropped (unsigned pre)" % (
        W.nnz, MIN_SYNAPSES, int(np.abs(W.data).sum()), int((~m).sum())))

    # --- neuron index in the FAFB layout -------------------------------------
    loc = neu["somaLocation"].apply(lambda v: v if isinstance(v, np.ndarray) and len(v) == 3 else None)
    xyz = np.array([v if v is not None else [np.nan] * 3 for v in loc], dtype=np.float64) * VOXEL_NM
    side = neu["somaSide"].fillna(neu["rootSide"]).map({"L": "left", "R": "right", "M": "center"})
    index = pd.DataFrame({
        "idx": neu["idx"], "root_id": neu["bodyId"].astype(np.int64),
        "nt_type": neu["nt_resolved"], "nt_type_score": neu["nt_type_score"],
        "flow": neu["superclass"].map(_flow), "super_class": neu["superclass"],
        "class": neu["class"], "sub_class": neu["subclass"], "side": side,
        "nerve": neu["entryNerve"].fillna(neu["exitNerve"]),
        "primary_type": neu["type"].fillna(neu["flywireType"]),
        "pos_x_nm": xyz[:, 0], "pos_y_nm": xyz[:, 1], "pos_z_nm": xyz[:, 2],
        "nt_resolved": neu["nt_resolved"], "sign": neu["sign"],
        "primary_neuropil": neu["somaNeuromere"].fillna(""),
        "flywire_type": neu["flywireType"], "manc_type": neu["mancType"],
        "soma_neuromere": neu["somaNeuromere"],
    })

    out_dir.mkdir(parents=True, exist_ok=True); meta_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "connectome_malecns_v1.0.npz"
    np.savez_compressed(npz, root_ids=index["root_id"].to_numpy(), indptr=W.indptr,
                        indices=W.indices.astype(np.int32), data=W.data.astype(np.int32),
                        shape=np.array(W.shape), dataset=np.array(["Janelia MaleCNS v1.0"]))
    index.to_csv(out_dir / "neuron_index_malecns_v1.0.csv.gz", index=False)
    manifest = {
        "dataset": "Janelia MaleCNS", "version": "1.0",
        "source_url": "https://male-cns.janelia.org/download/", "license": "CC-BY 4.0",
        "source_dir": str(src), "built_utc": datetime.now(timezone.utc).isoformat(),
        "n_neurons": int(n), "n_neuron_pairs": int(W.nnz),
        "n_synapses": int(np.abs(W.data).sum()), "min_synapses": MIN_SYNAPSES,
        "neuron_filter": "status == Traced",
        "neuron_order": "pseudo-random permutation, seed %d (for thread load balance)" % ORDER_SEED,
        "excitatory_neurons": int((index.sign == 1).sum()),
        "inhibitory_neurons": int((index.sign == -1).sum()),
        "unknown_sign_neurons": int((index.sign == 0).sum()),
        "nt_sign_convention": {"excitatory": sorted(EXCITATORY), "inhibitory": sorted(INHIBITORY)},
        "super_class_counts": {str(k): int(v) for k, v in index.super_class.value_counts().items()},
        "model_reference": "Shiu et al. 2024, Nature 634:210-219 (LIF parameters reused unchanged)",
    }
    src_manifest = src / "manifest.json"
    if src_manifest.exists():
        manifest["source_md5"] = {Path(it["name"]).name: it["md5Hash"]
                                  for it in json.loads(src_manifest.read_text())["items"]}
    (meta_dir / "build_manifest_malecns.json").write_text(json.dumps(manifest, indent=2))
    _log("done in %.0f s -> %s" % (time.time() - t0, out_dir))
    return manifest


def main():
    import config
    src = Path(sys.argv[1] if len(sys.argv) > 1 else config.PROJECT_ROOT.parent / "malecns_v1.0")
    m = build(src, config.DERIVED_DIR / "malecns", config.METADATA_DIR)
    print(json.dumps({k: m[k] for k in ("n_neurons", "n_neuron_pairs", "n_synapses",
                                         "excitatory_neurons", "inhibitory_neurons",
                                         "unknown_sign_neurons")}, indent=1))


if __name__ == "__main__":
    main()
