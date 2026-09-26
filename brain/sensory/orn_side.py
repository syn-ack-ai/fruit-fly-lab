"""
Antenna side of every olfactory receptor neuron (ORN).

FlyWire labels most ORNs "left"/"right"; a few (30 in v783) have no side. Each
ORN projects to both antennal lobes, more strongly to the lobe on its own side
(Gaudry et al. 2013), so an unlabelled ORN is given the side of the projection
neurons that receive most of its synapses. Used wherever ORNs are split by
side (exam odour inputs, the world's and home's senses, ipsilateral release).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


def orn_sides(connectome) -> np.ndarray:
    """Array of "left" / "right" / "" for every neuron ("" = not an ORN or unknown)."""
    return _orn_sides(id(connectome), connectome)


@lru_cache(maxsize=4)
def _orn_sides(_key, connectome) -> np.ndarray:
    n = connectome.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    side = n["side"].fillna("").astype(str).to_numpy()
    is_orn = np.char.startswith(t.astype(str), "ORN_")
    is_pn = (n["class"].fillna("").astype(str) == "ALPN").to_numpy()
    out = np.where(is_orn & np.isin(side, ["left", "right"]), side, "").astype(object)
    w = connectome.w.tocsr()
    for i in np.flatnonzero(is_orn & (out == "")):
        a, b = w.indptr[i], w.indptr[i + 1]
        j = w.indices[a:b]
        m = is_pn[j]
        syn = np.abs(w.data[a:b][m])
        left = syn[side[j[m]] == "left"].sum()
        right = syn[side[j[m]] == "right"].sum()
        if left != right:
            out[i] = "left" if left > right else "right"
    return out.astype(str)
