"""
Fly exam infrastructure: engines for a (dynamics, wiring, perturbation)
configuration, neuron groups, and stimulus / read-out helpers.

A configuration is a plain dict:
    dynamics      "published" | "calibrated"            (simulation.engine.session.apply_dynamics)
    wiring        "real" | "shuffled" | "random"        (cognition/wiring.py; degree-preserving = shuffled)
    wsyn          global synaptic gain (Shiu et al. robustness test: +-30%)
    edge_drop     fraction of connections removed at random
    nt_flip       fraction of neurons whose output sign is flipped (transmitter misprediction)
    lesion        fraction of central neurons silenced (sensory, descending and motor spared)
    sensor_drop   fraction of stimulated sensory neurons that deliver no input
    sensor_gain   multiplier on every sensory input rate
    seed          perturbation seed
"""
from __future__ import annotations

import copy
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd

import config

DEFAULT_CFG = dict(dynamics="calibrated", wiring="real", wsyn=1.0, edge_drop=0.0,
                   nt_flip=0.0, lesion=0.0, sensor_drop=0.0, sensor_gain=1.0, seed=0)


def cfg_key(cfg: dict) -> tuple:
    c = {**DEFAULT_CFG, **cfg}
    return tuple(sorted(c.items()))


@lru_cache(maxsize=1)
def connectome():
    from brain.neurons.registry import load_connectome
    return load_connectome()


@lru_cache(maxsize=1)
def labels() -> pd.DataFrame:
    return pd.read_csv(config.FLYWIRE_DIR / "labels.csv.gz", usecols=["root_id", "label"])


def label_cells(pattern: str, exclude: str | None = None) -> np.ndarray:
    """Simulation indices of neurons whose FlyWire community label matches."""
    c, lab = connectome(), labels()
    m = lab.label.str.contains(pattern, na=False, regex=True)
    if exclude:
        m &= ~lab.label.str.contains(exclude, na=False, regex=True)
    ids = lab[m].root_id.unique()
    return np.sort(np.array([c.idx(r) for r in ids if int(r) in c._id2idx], np.int64))


def types(pattern: str, side: str | None = None) -> np.ndarray:
    n = connectome().neurons
    t = n["primary_type"].fillna("").astype(str)
    m = t.str.fullmatch(pattern)
    if side:
        m &= n["side"].astype(str) == side
    return np.sort(n[m]["idx"].to_numpy(np.int64))


class Ctx:
    """One engine for one configuration, with helpers the tests use."""

    def __init__(self, cfg: dict):
        from native.lif_native import NativeLIFEngine
        from simulation.engine.session import apply_dynamics
        import scipy.sparse as sp
        self.cfg = {**DEFAULT_CFG, **cfg}
        c0 = connectome()
        c = c0
        if self.cfg["wiring"] != "real":
            from cognition.wiring import wiring_for
            ip, ix, wt = wiring_for(c0, self.cfg["wiring"])
            c = copy.copy(c0)
            c.w = sp.csr_matrix((wt.astype(np.int32), ix, ip), shape=c0.w.shape)
        self.c = c
        threads = int(os.environ.get("FLY_THREADS", "1"))
        self.e = NativeLIFEngine.from_connectome(c, seed=1, threads=threads)
        apply_dynamics(self.e, c, self.cfg["dynamics"])
        rng = np.random.default_rng(1000 + int(self.cfg["seed"]))
        if self.cfg["wsyn"] != 1.0:
            self.e.set_gain(float(self.cfg["wsyn"]))
        if self.cfg["edge_drop"] > 0 or self.cfg["nt_flip"] > 0:
            mult = self.e.plastic_multipliers()
            if self.cfg["edge_drop"] > 0:
                drop = rng.random(len(mult)) < self.cfg["edge_drop"]
                mult[drop] = 0.0
            if self.cfg["nt_flip"] > 0:
                w = c.w.tocsr()
                flip = np.flatnonzero(rng.random(c.n) < self.cfg["nt_flip"])
                for i in flip:
                    mult[w.indptr[i]:w.indptr[i + 1]] *= -1.0
            if hasattr(self.e, "commit_plastic"):
                self.e.commit_plastic()
        self.lesioned = np.empty(0, np.int64)
        if self.cfg["lesion"] > 0:
            sc = c.neurons["super_class"].astype(str).to_numpy()
            spare = np.isin(sc, ["sensory", "descending", "motor", "visual_projection",
                                 "ascending", "sensory_ascending"])
            cand = np.flatnonzero(~spare)
            k = int(round(self.cfg["lesion"] * len(cand)))
            self.lesioned = np.sort(rng.choice(cand, k, replace=False))
            self.e.silence(self.lesioned)
        self._sensor_rng = np.random.default_rng(2000 + int(self.cfg["seed"]))
        self._drop_cache = {}

    # --------------------------------------------------------------- inputs
    def _sensor(self, idx: np.ndarray, rate) -> tuple:
        rate = np.broadcast_to(np.asarray(rate, float), idx.shape).astype(float) * self.cfg["sensor_gain"]
        if self.cfg["sensor_drop"] > 0 and len(idx):
            key = (idx.tobytes()[:64], len(idx))
            if key not in self._drop_cache:
                self._drop_cache[key] = self._sensor_rng.random(len(idx)) >= self.cfg["sensor_drop"]
            keep = self._drop_cache[key]
            idx, rate = idx[keep], rate[keep]
        return idx, rate

    def set_inputs(self, inputs: list) -> None:
        """inputs: [(indices, rate or rates), ...]; overlapping neurons take the max."""
        if not inputs:
            self.e.clear_poisson()
            return
        parts = [self._sensor(np.asarray(i, np.int64), r) for i, r in inputs]
        idx = np.concatenate([p[0] for p in parts])
        r = np.concatenate([p[1] for p in parts])
        if not len(idx):
            self.e.clear_poisson()
            return
        o = np.lexsort((-r, idx))
        idx, r = idx[o], r[o]
        u, first = np.unique(idx, return_index=True)
        self.e.set_poisson(u, r[first])

    def window(self, ms: float) -> np.ndarray:
        s0 = self.e.spike_counts.copy()
        self.e.run(float(ms))
        return self.e.spike_counts - s0

    def trial(self, inputs: list, ms: float, seed: int, warm_inputs: list | None = None,
              warm_ms: float = 0.0) -> np.ndarray:
        """Spike counts over `ms` after an optional warm-up; engine reset per trial."""
        self.e.reset(seed=seed)
        if self.lesioned.size:
            self.e.silence(self.lesioned)
        if warm_ms > 0:
            self.set_inputs(warm_inputs or [])
            self.e.run(float(warm_ms))
        self.set_inputs(inputs)
        return self.window(ms)

    def rates(self, inputs: list, ms: float, seeds, **kw) -> np.ndarray:
        """Mean firing rate (Hz) per neuron over trials."""
        acc = np.zeros(self.c.n)
        for s in seeds:
            acc += self.trial(inputs, ms, s, **kw)
        return acc / len(seeds) / (ms / 1000.0)


# ------------------------------------------------------------------ groups
@lru_cache(maxsize=1)
def groups() -> dict:
    """Neuron groups the tests use (indices in the real connectome's order,
    which every wiring variant shares)."""
    from brain.sensory.modalities import BY_KEY, resolve_neurons
    c = connectome()
    n = c.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    cl = n["class"].fillna("").astype(str).to_numpy()
    side = n["side"].fillna("").astype(str).to_numpy()
    g = {}
    g["sugar"] = np.sort(resolve_neurons(BY_KEY["taste_sugar"], c))
    g["bitter"] = np.sort(resolve_neurons(BY_KEY["taste_bitter"], c))
    g["ir94e"] = label_cells(r"Ir94e")
    g["mn9"] = label_cells(r"\bMN9\b")
    g["mn9_L"] = g["mn9"][side[g["mn9"]] == "left"]
    g["mn9_R"] = g["mn9"][side[g["mn9"]] == "right"]
    fdg = label_cells(r"\bFdg\b")
    g["named"] = {
        "Roundup": label_cells(r"Roundup"),
        "Fdg": fdg if fdg.size else label_cells(r"FDG"),
        "FMIn": label_cells(r"FMIN|FMIn"),
        "Clavicle": label_cells(r"Clavicle"),
        "Fudog": label_cells(r"Fudog"),
        "Phantom": label_cells(r"Phantom"),
        "Rattle": label_cells(r"Rattle"),
        "Zorro": label_cells(r"Zorro"),
        "Bract": label_cells(r"Bract"),
        "Usnea": label_cells(r"Usnea"),
    }
    g["loom"] = np.sort(c.by_cell_types(["LC4", "LPLC2"])["idx"].to_numpy(np.int64))
    g["gf"] = types("DNp01")
    pn = cl == "ALPN"
    g["upn"] = np.flatnonzero(pn & np.array(["PN" in x and not x.startswith(("M_", "MZ_", "Z_")) for x in t]))
    g["ln"] = np.flatnonzero(cl == "ALLN")
    g["kc"] = np.flatnonzero(np.char.startswith(t.astype(str), "KC"))
    g["lhn"] = np.flatnonzero(cl == "LHLN")
    g["mbon"] = np.flatnonzero(np.char.startswith(t.astype(str), "MBON"))
    g["dan"] = np.flatnonzero(np.char.startswith(t.astype(str), "PAM") | np.char.startswith(t.astype(str), "PPL1"))
    g["jo_ab"] = np.flatnonzero(np.char.startswith(t.astype(str), "JO-A") | np.char.startswith(t.astype(str), "JO-B"))
    g["dnp11"] = types("DNp11")
    g["dng100"] = types("DNg100")
    g["sensory"] = np.flatnonzero(n["super_class"].astype(str).to_numpy() == "sensory")
    return g


@lru_cache(maxsize=1)
def resting() -> tuple:
    """Every ORN at its Hallem & Carlson 2006 spontaneous rate."""
    from robot.head import RestingOlfaction
    r = RestingOlfaction(connectome())
    return r.indices, r._rates


@lru_cache(maxsize=1)
def orn_space():
    from brain.sensory.olfaction import OlfactorySpace
    n = connectome().neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    orn = np.flatnonzero(np.char.startswith(t.astype(str), "ORN_"))
    types_ = sorted(set(t[orn]))
    osp = OlfactorySpace([x[4:] for x in types_])
    gi = np.array([types_.index(t[i]) for i in orn])
    from brain.sensory.orn_side import orn_sides
    side = orn_sides(connectome())[orn]
    return orn, gi, side, osp


def odour_inputs(conc_left: np.ndarray, conc_right: np.ndarray) -> list:
    """ORN rates for odorant concentrations at the left / right antenna."""
    orn, gi, side, osp = orn_space()
    rl, rr = osp.rates(conc_left), osp.rates(conc_right)
    rate = np.where(side == "left", rl[gi], rr[gi])
    return [(orn, rate)]


def readout():
    from brain.motor.descending import DescendingReadout
    return DescendingReadout(connectome())


def hz(counts: np.ndarray, idx: np.ndarray, ms: float) -> float:
    return float(counts[idx].mean() / (ms / 1000.0)) if len(idx) else float("nan")
