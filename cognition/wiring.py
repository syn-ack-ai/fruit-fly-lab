"""
Wiring for the brain substrate: the real FlyWire connectome and controls.

Every variant keeps the same 139,255 neurons, the same synapse weights and the
same sign per presynaptic neuron (the neurotransmitter), so sensory input still
enters through the same real neurons. Only *who connects to whom* changes:

  real      FlyWire FAFB v783 as built by brain/connectivity/build_connectome.
  shuffled  degree-preserving rewiring: each neuron keeps its exact out-going
            synapse list (count, strengths, sign) and its exact in-degree; the
            targets of all edges are permuted together (configuration model).
  random    each neuron keeps its out-going synapse list, but targets are
            drawn uniformly at random (in-degree not preserved).

Duplicate edges created by rewiring are merged (weights summed); self-loops
are kept, as they occur at the rate the model produces them. The result is
cached next to the connectome so every run of a variant uses the same graph.
"""
from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp

import config

SEED = 20260924


def _csr_arrays(w: sp.csr_matrix):
    w = w.tocsr()
    w.sum_duplicates()
    w.sort_indices()
    data = np.clip(w.data, -32767, 32767).astype(np.int16)
    return w.indptr.astype(np.int32), w.indices.astype(np.int32), data


def _rewire(w: sp.csr_matrix, kind: str, seed: int) -> sp.csr_matrix:
    w = w.tocsr()
    rows = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
    rng = np.random.default_rng(seed)
    if kind == "shuffled":
        cols = rng.permutation(w.indices)
    elif kind == "random":
        cols = rng.integers(0, w.shape[1], size=w.nnz)
    else:
        raise ValueError(kind)
    return sp.csr_matrix((w.data.astype(np.int32), (rows, cols)), shape=w.shape)


def wiring_for(connectome, kind: str = "real", seed: int = SEED):
    """(indptr, indices, int16 weights) for the native engine."""
    if kind == "real":
        return _csr_arrays(connectome.w)
    tag = hashlib.sha1(f"{kind}-{seed}-{connectome.w.nnz}".encode()).hexdigest()[:10]
    cache = config.DERIVED_DIR / f"wiring_{kind}_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["indptr"], z["indices"], z["weights"]
    indptr, indices, weights = _csr_arrays(_rewire(connectome.w, kind, seed))
    np.savez(cache, indptr=indptr, indices=indices, weights=weights)
    return indptr, indices, weights


def describe(connectome, kind: str) -> dict:
    ip, ix, wt = wiring_for(connectome, kind)
    n = len(ip) - 1
    outdeg = np.diff(ip)
    indeg = np.bincount(ix, minlength=n)
    return {"kind": kind, "edges": int(len(ix)), "synapses": int(np.abs(wt.astype(np.int64)).sum()),
            "self_loops": int(np.sum(ix == np.repeat(np.arange(n), outdeg))),
            "outdeg_max": int(outdeg.max()), "indeg_max": int(indeg.max()),
            "excitatory_weight_frac": float((wt > 0).mean())}
