"""
Which parts of the fly brain are active, for the dashboard (robot/dashboard.py).

From the engine's cumulative spike counts (sampled ~5 times a second, in the
dashboard's background thread: nothing is added to the per-millisecond loop):

  systems   mean firing rate (Hz) of named systems, by cell type and class:
            the visual neurons the camera and lidar drive (LC4, LPLC2,
            LC10a), the optic lobes (driven with --eye: robot/eye.py),
            smell, taste, touch and hearing, the learning centre (mushroom
            body), navigation (central complex), descending commands, body
            signals, motor neurons, the nerve cord, the rest of the brain
  geometry  once: every neuron with a soma position (~85%; brain and nerve
            cord, MaleCNS / FlyWire coordinates), as uint16 x, y, z and its
            system (the page draws them in 3D, robot/dashboard_page)
  activity  each update: per positioned neuron a 4-bit level (two per byte),
            the log of its firing rate, decaying over ~0.5 s so a single
            spike shows as a brief flash
  flow      each update: the wiring the activity is travelling along: from
            the neurons that fired most, lines to their strongest synaptic
            targets (the connectome's synapse counts; - = inhibitory)
  circuit   once: every neuron's strongest inputs and outputs (K each, by
            synapse count), cell types and systems, for the dashboard server
            to answer "what is this neuron wired to?" when one is tapped
"""
from __future__ import annotations

import io
import struct
import time

import numpy as np

SYSTEMS = [
    # the visual neurons the camera and lidar actually drive (robot/head.py, robot/lidar.py):
    # looming detectors and the small-object / person pathway
    ("Camera & lidar inputs (LC4, LPLC2, LC10a)", lambda sc, cl, ty: np.isin(ty, ["LC4", "LPLC2", "LC10a"])),
    # the rest of vision: the optic lobes (camera and lidar through flyvis with --eye)
    ("Optic lobes", lambda sc, cl, ty: np.isin(sc, ["ol_intrinsic", "ol_sensory", "visual_projection",
                                                    "visual_centrifugal", "visual_projection_tbc"])
                                       | np.isin(cl, ["visual", "ocellar"])),
    ("Smell", lambda sc, cl, ty: np.isin(cl, ["olfactory", "ALPN", "ALLN", "LHLN", "LHCENT", "ALIN", "ALON", "mAL",
                                              "hygrosensory", "thermosensory"])),
    ("Taste", lambda sc, cl, ty: np.isin(cl, ["gustatory", "chemosensory"])),
    ("Touch & hearing", lambda sc, cl, ty: np.char.startswith(cl.astype(str), "mechanosensory")),
    ("Learning (mushroom body)", lambda sc, cl, ty: np.isin(cl, ["Kenyon_Cell", "MBON", "DAN", "MBIN"])),
    ("Navigation (central complex)", lambda sc, cl, ty: np.isin(cl, ["CX", "TuBu"])),
    ("Descending commands", lambda sc, cl, ty: sc == "descending_neuron"),
    ("Body signals (ascending)", lambda sc, cl, ty: np.isin(sc, ["ascending_neuron", "sensory_ascending"])),
    ("Motor neurons", lambda sc, cl, ty: np.isin(sc, ["vnc_motor", "cb_motor"]) | (cl == "brain_motor_neuron")),
    ("Nerve cord", lambda sc, cl, ty: np.char.startswith(sc.astype(str), "vnc")),
    ("Rest of the brain", lambda sc, cl, ty: np.ones(len(sc), bool)),
]
DECAY_S = 0.5                  # a level falls by half in this long
K_CIRCUIT = 16                 # strongest inputs / outputs kept per neuron
FLOW_SOURCES = 1200            # the most active neurons drawn with their outputs each update
FLOW_TARGETS = 3               # ... to their strongest targets


def top_k_rows(m, k):
    """Per row of a CSR matrix, the k entries largest in |value|: (index, value)
    arrays of shape (rows, k), -1 / 0 where a row has fewer."""
    n = m.shape[0]
    rows = np.repeat(np.arange(n), np.diff(m.indptr))
    order = np.lexsort((-np.abs(m.data), rows))
    rank = np.arange(len(order)) - m.indptr[rows[order]]
    keep = order[rank < k]
    r, c = rows[keep], rank[rank < k]
    idx = np.full((n, k), -1, np.int32)
    val = np.zeros((n, k), np.int32)
    idx[r, c] = m.indices[keep]
    val[r, c] = m.data[keep]
    return idx, val


class BrainActivity:
    def __init__(self, connectome):
        n = connectome.neurons
        sc = n["super_class"].fillna("").astype(str).to_numpy()
        cl = n["class"].fillna("").astype(str).to_numpy()
        ty = n["primary_type"].fillna("").astype(str).to_numpy()
        self.system = np.full(len(n), -1, np.int16)
        for k, (_, f) in enumerate(SYSTEMS):
            m = (self.system < 0) & f(sc, cl, ty)
            self.system[m] = k
        self.sys_n = np.bincount(self.system, minlength=len(SYSTEMS)).astype(float)
        xyz = np.stack([n[c].to_numpy(float) for c in ("pos_x_nm", "pos_y_nm", "pos_z_nm")], 1)
        has = np.isfinite(xyz).all(1)
        self.idx = np.flatnonzero(has)                       # the neurons drawn, in this order
        p = xyz[has]
        self.lo, self.hi = p.min(0), p.max(0)
        self.q = np.round((p - self.lo) / np.maximum(self.hi - self.lo, 1) * 65535).astype(np.uint16)
        self.level = np.zeros(len(self.idx), np.float32)
        self._prev = None
        self.pos_of = np.full(len(n), -1, np.int32)
        self.pos_of[self.idx] = np.arange(len(self.idx), dtype=np.int32)
        self.types = n["primary_type"].fillna("").astype(str).to_numpy()
        w = connectome.w.tocsr()
        self.out_idx, self.out_w = top_k_rows(w, K_CIRCUIT)
        self.in_idx, self.in_w = top_k_rows(w.T.tocsr(), K_CIRCUIT)
        self.flow = b""

    def geometry(self) -> bytes:
        """Header (b"FLYG", n, the box in micrometres as 6 float32) then n x
        (x, y, z uint16) then n system indices (uint8)."""
        head = b"FLYG" + struct.pack("<I6f", len(self.idx), *(self.lo / 1e3), *(self.hi / 1e3))
        return head + self.q.tobytes() + self.system[self.idx].astype(np.uint8).tobytes()

    def update(self, spike_counts):
        """(systems JSON, activity bytes) since the previous call, from the
        engine's cumulative counts; (None, None) on the first call."""
        now = time.monotonic()
        c = np.array(spike_counts, np.int64)                 # a snapshot (the engine keeps counting)
        prev, self._prev = self._prev, (now, c)
        if prev is None or now - prev[0] < 0.05:
            return None, None
        dt = now - prev[0]
        d = np.clip(c - prev[1], 0, None).astype(np.float32)   # a reset (new episode) never goes negative
        sys_rate = np.bincount(self.system, weights=d, minlength=len(SYSTEMS)) / np.maximum(self.sys_n, 1) / dt
        systems = {"systems": [{"name": name, "hz": round(float(sys_rate[k]), 2), "n": int(self.sys_n[k])}
                               for k, (name, _) in enumerate(SYSTEMS) if self.sys_n[k] > 0],
                   "spikes_per_s": int(d.sum() / dt)}
        # level: log2(1 + rate / 2 Hz) on a 0..15 scale (15 = ~200 Hz and up), held and decaying
        rate = d[self.idx] / dt
        now_level = np.minimum(15.0, 2.25 * np.log2(1.0 + rate / 2.0))
        self.level = np.maximum(self.level * np.float32(0.5 ** (dt / DECAY_S)), now_level)
        q = np.round(self.level).astype(np.uint8)
        if len(q) % 2:
            q = np.append(q, 0)
        self.flow = self._flow(d)
        return systems, (q[0::2] | (q[1::2] << 4)).tobytes()

    def _flow(self, d) -> bytes:
        """(source, target) pairs, as positions in the drawn order, int32; a
        negative target -(t + 1) is an inhibitory synapse."""
        fired = np.flatnonzero(d > 0)
        if not len(fired):
            return b""
        if len(fired) > FLOW_SOURCES:
            fired = fired[np.argpartition(-d[fired], FLOW_SOURCES)[:FLOW_SOURCES]]
        tgt = self.out_idx[fired, :FLOW_TARGETS]
        sgn = self.out_w[fired, :FLOW_TARGETS]
        src = np.repeat(fired, FLOW_TARGETS)
        tgt, sgn = tgt.ravel(), sgn.ravel()
        ok = tgt >= 0
        src, tgt, sgn = self.pos_of[src[ok]], self.pos_of[tgt[ok]], sgn[ok]
        ok = (src >= 0) & (tgt >= 0)
        src, tgt, sgn = src[ok], tgt[ok], sgn[ok]
        tgt = np.where(sgn < 0, -(tgt + 1), tgt)
        return np.stack([src, tgt], 1).astype(np.int32).tobytes()

    def circuit(self) -> bytes:
        """Everything the dashboard server needs to say what a neuron is wired
        to (an .npz: no code, loaded with allow_pickle=False)."""
        names, tid = np.unique(self.types.astype(str), return_inverse=True)
        names = names.astype(str)                 # text, not Python objects: loads without pickle
        b = io.BytesIO()
        np.savez_compressed(b, idx=self.idx.astype(np.int32), pos_of=self.pos_of, system=self.system.astype(np.int8),
                 systems=np.array([s for s, _ in SYSTEMS]), type_id=tid.astype(np.int32), types=names,
                 out_idx=self.out_idx, out_w=self.out_w.astype(np.int32), in_idx=self.in_idx,
                 in_w=self.in_w.astype(np.int32))
        return b.getvalue()
