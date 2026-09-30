"""
The mushroom body as a learning circuit: graded APL inhibition and
dopamine-gated plasticity of Kenyon cell -> MBON synapses.

PROVENANCE
----------
A. REAL DATA : every neuron and weight used here is from the loaded connectome:
   Kenyon cells (KC*), MBONs (MBON*), dopaminergic neurons (PAM*, PPL1*), the
   two APL neurons, and the synapse counts between them.
C. MODEL EXTENSIONS (not in Shiu et al. 2024; documented, parameterised):
   1. APL as a graded, non-spiking feedback inhibitor. APL releases GABA
      without spiking and normalises Kenyon cell activity so that only a sparse
      set (~5-10%) responds to an odour (Lin et al. 2014, Nat Neurosci
      17:559; Papadopoulou et al. 2011, Science 332:721). As one spiking LIF
      neuron per side it cannot do this: in the published model ~45% of all
      Kenyon cells respond to any odour, largely the same cells for every odour.
      Here each APL's activity is a leaky sum of its Kenyon cells' spikes,
      weighted by the real KC->APL synapse counts, and it inhibits each Kenyon
      cell in proportion to the real APL->KC synapse count. The spiking APL's
      own output is silenced so its inhibition is not counted twice.
   2. Dopamine-gated plasticity (Hige et al. 2015, Neuron 88:985; Cohn et al.
      2015, Cell 163:1742; Aso & Rubin 2016, eLife 5:e16135). A Kenyon cell's
      recent activity leaves an eligibility trace; dopamine arriving at an MBON's
      compartment depresses the synapses from eligible Kenyon cells onto it, and
      dopamine without Kenyon cell activity lets them recover. Which dopamine
      neurons teach which MBON comes from the real DAN->MBON synapse counts.
      Punishment neurons (PPL1) on approach MBONs and reward neurons (PAM) on
      avoidance MBONs give aversive and appetitive memories respectively.
   3. Dopamine in the mushroom body is modulatory (dan_modulatory=True): the
      model's convention treats dopamine as a fast excitatory transmitter, so
      odour-evoked DAN firing would excite MBONs directly, and DANs would
      excite each other (PPL101/PPL102 -> PPL101 is ~25% of PPL101's odour
      drive). Here the fast effect of DAN->KC, DAN->MBON and DAN->DAN synapses
      is removed; dopamine acts through the learning rule, which reads DAN
      spikes directly.
   4. KC->MBON gain (kc_mbon_gain): with sparse Kenyon cell codes (~5% of KCs,
      1-2 spikes each, as in real flies) the published per-synapse weight
      leaves MBONs nearly silent, whereas real MBONs respond to odours at tens
      of Hz. One multiplier on all KC->MBON synapses restores that; learning
      then scales each synapse within [0, gain].
   5. Dopamine threshold (da_threshold_hz): in the model dopamine neurons fire
      tens of Hz to any odour (spread excitation), so every odour would teach.
      Plasticity is driven only by DAN activity above a threshold rate, so
      weak odour-evoked firing does not teach and strong activation does. The
      rate is averaged over da_tau_ms = 200 ms so an odour-onset burst is not
      mistaken for sustained dopamine (with DAN->DAN removed, PPL101 reaches
      ~35-38 Hz to odours vs ~98 Hz with optogenetic-like 100 Hz drive).
   6. Tonic vs phasic dopamine (baseline_tau_ms): the threshold applies to
      each DAN's activity ABOVE ITS OWN SLOW BASELINE (running mean over
      ~10 s). DAN activity also tracks behavioural state (Cohn et al. 2015,
      Cell 163:1742), and with realistic resting sensory input some DANs
      (e.g. PPL107, driven by lateral-horn neurons) fire tonically at ~40 Hz
      with bursts above 60 Hz; without this they slowly depress their
      compartments with no reinforcement at all. Reinforcement (taste or
      optogenetic-like drive, ~100 Hz from a low baseline) still teaches.
      Ongoing dopamine-driven forgetting (Berry et al. 2012, Neuron 74:530)
      is therefore not modelled.

Requires the native engine (per-connection multipliers, lif_add_g).
"""
from __future__ import annotations

import numpy as np


class MushroomBody:
    def __init__(self, connectome, engine, *, apl_gain=0.0, apl_tau_ms=10.0,
                 plastic=True, elig_tau_ms=2000.0, lr_depress=6e-4, lr_recover=1.5e-4,
                 update_ms=10.0, dan_modulatory=True, kc_mbon_gain=1.0,
                 da_threshold_hz=60.0, da_tau_ms=200.0, baseline_tau_ms=10000.0):
        c, n = connectome, connectome.neurons
        t = n["primary_type"].fillna("").astype(str)
        side = n["side"].fillna("").astype(str).to_numpy()
        self.e, self.c = engine, c
        self.kc = n[t.str.match("^KC")]["idx"].to_numpy(np.int64)
        self.mbon = n[t.str.match("^MBON")]["idx"].to_numpy(np.int64)
        self.dan = n[t.str.match("^(PAM|PPL1)")]["idx"].to_numpy(np.int64)
        self.apl = n[t == "APL"]["idx"].to_numpy(np.int64)
        self.types = t.to_numpy()

        # --- graded APL -------------------------------------------------------
        self.apl_gain, self.apl_tau_ms = float(apl_gain), float(apl_tau_ms)
        W = abs(c.w)
        self.kc_to_apl = np.asarray(W[self.kc][:, self.apl].todense(), float)    # KC x APL
        self.apl_to_kc = np.asarray(W[self.apl][:, self.kc].todense(), float).T  # KC x APL
        # normalise: each APL's drive is a synapse-weighted fraction of its KCs;
        # its output per KC is relative to the mean APL->KC synapse count
        self.kc_to_apl /= np.maximum(self.kc_to_apl.sum(0, keepdims=True), 1)
        self.apl_to_kc /= np.maximum(self.apl_to_kc.mean(0, keepdims=True), 1e-9)
        self.apl_act = np.zeros(len(self.apl))
        self._kc_pos = np.full(c.n, -1, np.int64)
        self._kc_pos[self.kc] = np.arange(len(self.kc))
        self._dan_pos = np.full(c.n, -1, np.int64)
        self._dan_pos[self.dan] = np.arange(len(self.dan))
        if self.apl_gain > 0 and len(self.apl):
            engine.silence(self.apl)          # graded release replaces its spikes

        # --- plasticity ------------------------------------------------------
        self.plastic = plastic
        self.learning = True                 # False freezes the weights (e.g. while testing)
        self.elig_tau_ms, self.update_ms = float(elig_tau_ms), float(update_ms)
        self.lr_depress, self.lr_recover = float(lr_depress), float(lr_recover)
        self.elig = np.zeros(len(self.kc))
        self._dan_count = np.zeros(len(self.dan))
        # dopamine teaches only above a threshold rate (see extension 5)
        self.da_threshold_hz, self.da_tau_ms = float(da_threshold_hz), float(da_tau_ms)
        self.dan_rate = np.zeros(len(self.dan))                     # Hz, smoothed
        self.baseline_tau_ms = float(baseline_tau_ms)
        self.dan_base = np.zeros(len(self.dan))                     # Hz, slow baseline
        self._since_update = 0.0
        self.defer = None                    # a list: engine writes wait for Session to apply them
        if plastic:
            w = c.w.tocsr()
            # positions (CSR order, as the engine holds them) of KC->MBON edges
            is_mbon = np.zeros(c.n, bool); is_mbon[self.mbon] = True
            pos, pre, post = [], [], []
            for k, i in enumerate(self.kc):
                a, b = w.indptr[i], w.indptr[i + 1]
                cols = w.indices[a:b]
                sel = np.flatnonzero(is_mbon[cols])
                pos.append(a + sel); pre.append(np.full(len(sel), k)); post.append(cols[sel])
            self.edge_pos = np.concatenate(pos)
            self.edge_kc = np.concatenate(pre)                       # index into self.kc
            mpos = np.full(c.n, -1, np.int64); mpos[self.mbon] = np.arange(len(self.mbon))
            self.edge_mbon = mpos[np.concatenate(post)]              # index into self.mbon
            # teacher: fraction of each MBON's dopaminergic input from each DAN
            A = np.asarray(W[self.dan][:, self.mbon].todense(), float)   # DAN x MBON
            self.teacher = A / np.maximum(A.sum(0, keepdims=True), 1)
            self.mult = engine.plastic_multipliers()
            self.weights = np.ones(len(self.edge_pos))               # relative weights, in [0, 1]
            # KC->MBON transmission scale (see docstring, extension 4)
            self.kc_mbon_gain = float(kc_mbon_gain)
            self.mult[self.edge_pos] = np.float32(self.kc_mbon_gain)
        if dan_modulatory:
            mult = engine.plastic_multipliers()
            w = c.w.tocsr()
            target = np.zeros(c.n, bool)
            target[self.kc] = True; target[self.mbon] = True; target[self.dan] = True
            n_off = 0
            for i in self.dan:
                a, b = w.indptr[i], w.indptr[i + 1]
                sel = a + np.flatnonzero(target[w.indices[a:b]])
                mult[sel] = 0.0; n_off += len(sel)
            self.dan_edges_silenced = n_off

    # ------------------------------------------------------------------ step
    def step(self, spikes: np.ndarray, dt_ms: float = 1.0) -> None:
        """Call after every block of simulation with that block's spikes."""
        kpos = self._kc_pos[spikes]
        kc_spk = kpos[kpos >= 0]
        if self.apl_gain > 0 and len(self.apl):
            decay = np.exp(-dt_ms / self.apl_tau_ms)
            drive = self.kc_to_apl[kc_spk].sum(0) if kc_spk.size else 0.0
            self.apl_act = self.apl_act * decay + drive
            inh = -self.apl_gain * (self.apl_to_kc @ self.apl_act)       # per KC
            self._engine_write(lambda g=inh.astype(np.float32): self.e.add_g(self.kc, g))
        if self.plastic and self.learning:
            self.elig *= np.exp(-dt_ms / self.elig_tau_ms)
            np.add.at(self.elig, kc_spk, 1.0)
            self.dan_rate *= np.exp(-dt_ms / self.da_tau_ms)
            if self.baseline_tau_ms > 0:
                self.dan_base += (self.dan_rate - self.dan_base) * min(1.0, dt_ms / self.baseline_tau_ms)
            d = self._dan_pos[spikes]
            d = d[d >= 0]
            if d.size:
                np.add.at(self.dan_rate, d, 1000.0 / self.da_tau_ms)   # -> Hz
            self._since_update += dt_ms
            if self._since_update >= self.update_ms:
                self._learn()

    def _learn(self) -> None:
        # teaching signal per MBON: its DANs' rates above threshold (Hz),
        # weighted by their share of the MBON's dopaminergic input
        dopamine = np.maximum(self.dan_rate - self.dan_base - self.da_threshold_hz, 0.0) @ self.teacher
        self._since_update = 0.0
        if not dopamine.any():
            return
        da = dopamine[self.edge_mbon]
        el = np.minimum(self.elig[self.edge_kc], 1.0)      # saturating eligibility
        w = self.weights
        w += da * (self.lr_recover * (1.0 - w) * (1.0 - el) - self.lr_depress * el * w)
        np.clip(w, 0.0, 1.0, out=w)
        m = (self.kc_mbon_gain * w).astype(np.float32)

        def write():
            self.mult[self.edge_pos] = m
            commit = getattr(self.e, "commit_plastic", None)     # CUDA engine keeps a device copy
            if commit is not None:
                commit(self.edge_pos)
        self._engine_write(write)

    def _engine_write(self, fn) -> None:
        """Change the engine now, or, while the session pipelines blocks
        (Session.pipeline_plasticity), queue it for between blocks."""
        if self.defer is not None:
            self.defer.append(fn)
        else:
            fn()

    def summary(self) -> dict:
        """What has been learned: per MBON type, the mean relative KC->MBON weight."""
        if not self.plastic:
            return {}
        if not hasattr(self, "_mbon_type_of_edge"):
            self._mbon_type_of_edge = self.types[self.mbon][self.edge_mbon]
            self._mbon_types = sorted(set(self._mbon_type_of_edge))
            # each type's edges, once (the session reports this every step)
            self._type_edges = [np.flatnonzero(self._mbon_type_of_edge == mt) for mt in self._mbon_types]
        out = {}
        for mt, e in zip(self._mbon_types, self._type_edges):
            w = self.weights[e].mean()
            if w < 0.995:
                out[mt] = round(float(w), 3)
        return {"changed_mbons": out, "depressed_synapses": int((self.weights < 0.9).sum()),
                "total_synapses": int(len(self.weights))}

    def reset_activity(self) -> None:
        """Forget ongoing activity (APL state, eligibility), keep what was learned."""
        self.apl_act[:] = 0
        self.elig[:] = 0
        self.dan_rate[:] = 0
        self.dan_base[:] = 0
        self._since_update = 0.0
