"""
Loader and query interface for the real FlyWire FAFB v783 neuron set.

Everything returned by this module traces back to a FlyWire root ID. There is no
synthetic, generated, or placeholder neuron anywhere in this file.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config


class Connectome:
    """The real FlyWire v783 connectome, as a sparse signed-weight graph."""

    def __init__(self, neurons: pd.DataFrame, weights: sp.csr_matrix,
                 manifest: dict):
        self.neurons = neurons
        self.w = weights                    # CSR, signed synapse counts, pre -> post
        self.manifest = manifest
        self.root_ids = neurons["root_id"].to_numpy()
        self._id2idx = {int(r): int(i) for i, r in enumerate(self.root_ids)}
        self.n = len(neurons)

    # ---- identity -------------------------------------------------------
    @property
    def dataset(self) -> str:
        return "%s v%s" % (self.manifest["dataset"], self.manifest["version"])

    def __repr__(self) -> str:
        return "<Connectome %s: %d neurons, %d pairs, %d synapses>" % (
            self.dataset, self.n, self.w.nnz, int(np.abs(self.w.data).sum()))

    # ---- lookup ---------------------------------------------------------
    def idx(self, root_id) -> int:
        """Simulation index for a FlyWire root ID."""
        try:
            return self._id2idx[int(root_id)]
        except KeyError:
            raise KeyError("root_id %s is not in %s" % (root_id, self.dataset))

    def indices(self, root_ids) -> np.ndarray:
        return np.array([self.idx(r) for r in root_ids], dtype=np.int64)

    def root_id(self, idx) -> int:
        return int(self.root_ids[int(idx)])

    def by_cell_type(self, cell_type, side=None) -> pd.DataFrame:
        """All neurons whose consolidated primary_type equals `cell_type`."""
        m = self.neurons["primary_type"].astype(str) == str(cell_type)
        if side is not None:
            m &= self.neurons["side"].astype(str) == str(side)
        return self.neurons.loc[m]

    def by_cell_types(self, cell_types, side=None) -> pd.DataFrame:
        wanted = set(str(c) for c in cell_types)
        m = self.neurons["primary_type"].astype(str).isin(wanted)
        if side is not None:
            m &= self.neurons["side"].astype(str) == str(side)
        return self.neurons.loc[m]

    def by_super_class(self, super_class) -> pd.DataFrame:
        return self.neurons.loc[self.neurons["super_class"].astype(str) == str(super_class)]

    def search_type(self, pattern) -> pd.DataFrame:
        """Regex search over consolidated cell types."""
        s = self.neurons["primary_type"].astype(str)
        return self.neurons.loc[s.str.contains(pattern, regex=True, na=False)]

    # ---- connectivity queries -------------------------------------------
    def outputs(self, root_id, top=None) -> pd.DataFrame:
        """Downstream partners of a neuron, with real signed synapse counts."""
        i = self.idx(root_id)
        row = self.w.getrow(i)
        return self._partner_frame(row.indices, row.data, top)

    def inputs(self, root_id, top=None) -> pd.DataFrame:
        """Upstream partners of a neuron, with real signed synapse counts."""
        i = self.idx(root_id)
        col = self.w.getcol(i).tocoo()
        return self._partner_frame(col.row, col.data, top)

    def _partner_frame(self, idxs, data, top):
        df = self.neurons.iloc[idxs][
            ["root_id", "primary_type", "super_class", "class", "side", "nt_resolved"]
        ].copy()
        df["syn_count"] = np.abs(data)
        df["signed_weight"] = data
        df = df.sort_values("syn_count", ascending=False)
        return df.head(top) if top else df

    def connection_strength(self, pre_root_id, post_root_id) -> int:
        """Signed synapse count from one real neuron to another (0 if unconnected)."""
        i, j = self.idx(pre_root_id), self.idx(post_root_id)
        row = self.w.getrow(i)
        hit = np.where(row.indices == j)[0]
        return int(row.data[hit[0]]) if len(hit) else 0

    def subgraph(self, indices) -> "Connectome":
        """
        A Connectome restricted to the given real neurons, keeping real root IDs
        and real synapse counts. Used for unit tests and focused experiments;
        never used to invent connectivity.
        """
        idx = np.sort(np.asarray(indices, dtype=np.int64))
        neurons = self.neurons.iloc[idx].copy().reset_index(drop=True)
        neurons["idx"] = np.arange(len(idx), dtype=np.int32)
        sub = self.w[idx, :][:, idx].tocsr()
        sub.sort_indices()
        m = dict(self.manifest)
        m["subgraph_of"] = "%s v%s" % (m["dataset"], m["version"])
        m["n_neurons"] = int(len(idx))
        return Connectome(neurons, sub, m)

    def type_to_type(self, pre_type, post_type) -> dict:
        """Aggregate real connectivity between two annotated cell types."""
        pre = self.by_cell_type(pre_type)
        post = self.by_cell_type(post_type)
        if pre.empty or post.empty:
            return {"pre_type": pre_type, "post_type": post_type,
                    "n_pre": len(pre), "n_post": len(post),
                    "total_synapses": 0, "connected_pairs": 0}
        sub = self.w[pre["idx"].to_numpy(), :][:, post["idx"].to_numpy()]
        return {
            "pre_type": pre_type, "post_type": post_type,
            "n_pre": int(len(pre)), "n_post": int(len(post)),
            "total_synapses": int(np.abs(sub.data).sum()),
            "connected_pairs": int(sub.nnz),
        }


@lru_cache(maxsize=1)
def load_connectome() -> Connectome:
    """Load the built connectome. Raises if the build has not been run."""
    if not all(p.exists() for p in (config.CONNECTOME_NPZ, config.NEURON_INDEX, config.BUILD_MANIFEST)):
        build = {
            "fafb": "python -m brain.connectivity.build_connectome",
            "malecns": "python -m brain.connectivity.build_malecns /path/to/malecns_v1.0",
            "merged": ("python -m brain.connectivity.merge "
                       "(requires the FAFB and MaleCNS builds first)"),
        }[config.DATASET_KEY]
        raise FileNotFoundError(
            f"Connectome for FLY_DATASET={config.DATASET_KEY} not built. Run:  {build}"
        )
    z = np.load(config.CONNECTOME_NPZ, allow_pickle=False)
    shape = tuple(int(x) for x in z["shape"])
    w = sp.csr_matrix((z["data"].astype(np.int32), z["indices"], z["indptr"]), shape=shape)
    neurons = pd.read_csv(config.NEURON_INDEX, dtype={"root_id": np.int64})
    manifest = json.loads(config.BUILD_MANIFEST.read_text())
    if config.MALE_CNS:
        neurons = harmonize_malecns(neurons)
    return Connectome(neurons, w, manifest)


def harmonize_malecns(neurons: pd.DataFrame) -> pd.DataFrame:
    """Small, explicit label fixes so code written against FlyWire labels runs
    on the MaleCNS: primary_neuropil, mostly empty there, becomes "" rather
    than NaN; an empty class is filled from the FlyWire type. Types and classes stay the MaleCNS's own annotation; senses are
    mapped separately (brain/sensory/crossmap.py). (Head bristles: the MaleCNS
    types 69 of them only as "BM", cross-referenced to a mix of seven FlyWire
    bristle types including BM_Ant; robot/lidar.LidarTouch uses them.)"""
    n = neurons.copy()
    n["primary_neuropil"] = n["primary_neuropil"].fillna("").astype(str)
    # an empty class takes the FlyWire class of the neuron's FlyWire type
    # (e.g. 445 lateral-horn local neurons, FlyWire class LHLN, have none in
    # the MaleCNS); the MaleCNS's own class labels are never overridden
    fafb = pd.read_csv(config.DERIVED_DIR / "neuron_index_v783.csv.gz", usecols=["primary_type", "class"])
    fcls = fafb.dropna().groupby("primary_type")["class"].agg(lambda s: s.value_counts().index[0])
    fill = n["class"].isna() & n["flywire_type"].isin(fcls.index)
    n.loc[fill, "class"] = n.loc[fill, "flywire_type"].map(fcls)
    return n


def canonical_super_class(neurons) -> np.ndarray:
    """Super classes under common names for both datasets: FlyWire's
    ("sensory", "descending", "motor", "ascending", "sensory_ascending",
    "visual_projection", ...) and the MaleCNS's (cb_/vnc_/ol_sensory,
    descending_neuron, cb_/vnc_motor, ascending_neuron, *_tbc ...). Kinds:
    sensory, descending, motor, ascending, sensory_ascending, visual_projection,
    or the dataset's own name. (Review 2026-09-27: code matching FlyWire names
    silently found nothing on the male brains.)"""
    sc = neurons["super_class"].fillna("").astype(str).str.replace("_tbc", "", regex=False)
    out = sc.copy()
    out[sc.isin(["descending_neuron", "efferent_descending", "sensory_descending"])] = "descending"
    out[sc.isin(["ascending_neuron", "efferent_ascending"])] = "ascending"
    out[sc.str.endswith("_motor")] = "motor"
    out[sc.str.endswith("_sensory")] = "sensory"
    out[sc == "sensory_ascending"] = "sensory_ascending"
    return out.to_numpy()
