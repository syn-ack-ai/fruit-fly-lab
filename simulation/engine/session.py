"""
A running simulation session: environment -> sensory encoders -> LIF engine over
the real FlyWire connectome -> descending-neuron readout -> body.

This is the object the interactive laboratory drives. It owns the closed loop
and the telemetry, and it does not contain any behavioural rule: it simply
advances the network and reports what the real neurons did.
"""
from __future__ import annotations

import os
import time
from collections import deque

import numpy as np

from brain.motor.descending import DescendingReadout
from brain.neuron_models.lif import LIFParams, DEFAULT
from simulation.engine.lif_engine import LIFEngine


def _calibrated_gain() -> float:
    """Synaptic gain for the active dataset: 1.0 for FlyWire FAFB (the published
    model); for the MaleCNS the value fitted in data/metadata/calibration_malecns.json.
    $FLY_GAIN overrides it."""
    import json
    import config
    if os.environ.get("FLY_GAIN"):
        return float(os.environ["FLY_GAIN"])
    if config.DATASET_KEY == "malecns":
        return float(json.loads((config.METADATA_DIR / "calibration_malecns.json").read_text())["gain"])
    return 1.0


def _edge_positions(connectome, pre_mask, post_mask) -> np.ndarray:
    """CSR positions (the engine's connection order) of pre_mask -> post_mask connections."""
    w = connectome.w.tocsr(); pos = []
    for i in np.flatnonzero(pre_mask):
        a, b = w.indptr[i], w.indptr[i + 1]
        pos.append(a + np.flatnonzero(post_mask[w.indices[a:b]]))
    return np.concatenate(pos) if pos else np.empty(0, np.int64)


def _positions_into(connectome, post: int, pre_mask=None) -> np.ndarray:
    """CSR positions (engine connection order) of connections onto neuron `post`."""
    wt = connectome.w.T.tocsr()
    pre = wt.indices[wt.indptr[post]:wt.indptr[post + 1]]
    if pre_mask is not None:
        pre = pre[pre_mask[pre]]
    w = connectome.w.tocsr()
    pos = []
    for i in pre:
        a, b = w.indptr[i], w.indptr[i + 1]
        k = np.searchsorted(w.indices[a:b], post)
        if k < b - a and w.indices[a + k] == post:
            pos.append(a + k)
    return np.array(pos, np.int64)


def giant_fibre_multipliers(connectome, cfg: dict) -> list:
    """[(positions, multiplier)] for the Giant Fibre's inputs (see dynamics json)."""
    n = connectome.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    jo = np.char.startswith(t.astype(str), "JO-")
    out = []
    for gf in np.flatnonzero(t == cfg.get("cell_type", "DNp01")):
        out.append((_positions_into(connectome, gf), float(cfg["chemical_input_scale"])))
        out.append((_positions_into(connectome, gf, jo),
                    float(cfg["johnstons_organ_input"]) / float(cfg["chemical_input_scale"])))
    return out


def orn_pn_compensation(connectome) -> list:
    """[(connection positions, multiplier)] per projection neuron, equalising its
    mean strength per ORN connection to its cell type's mean (Tobin et al. 2017)."""
    n = connectome.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    is_orn = np.char.startswith(t.astype(str), "ORN_")
    is_pn = (n["class"].fillna("").astype(str) == "ALPN").to_numpy()
    wt = connectome.w.T.tocsr()                 # rows: postsynaptic neuron
    per_pn = {}
    for j in np.flatnonzero(is_pn):
        a, b = wt.indptr[j], wt.indptr[j + 1]
        pre = wt.indices[a:b]
        m = is_orn[pre]
        if m.sum():
            per_pn[j] = (pre[m], float(np.abs(wt.data[a:b][m]).mean()))
    by_type = {}
    for j, (_, spc) in per_pn.items():
        by_type.setdefault(t[j], []).append(spc)
    w = connectome.w.tocsr()
    out = []
    for j, (pres, spc) in per_pn.items():
        target = float(np.mean(by_type[t[j]]))
        pos = []
        for i in pres:                          # CSR position of i -> j
            a, b = w.indptr[i], w.indptr[i + 1]
            k = np.searchsorted(w.indices[a:b], j)
            if k < b - a and w.indices[a + k] == j:
                pos.append(a + k)
        out.append((np.array(pos, np.int64), float(np.clip(target / spc, 0.5, 2.0))))
    return out


def apply_dynamics(engine, connectome, name: str | None = None) -> dict | None:
    """Optional dynamics beyond the published model (adaptation, antennal-lobe
    slow inhibition, excitatory-LN transmission), from
    data/metadata/dynamics_<name>.json. The name comes from $FLY_DYNAMICS
    ("published" = none, the default; "calibrated")."""
    import json
    import config
    name = name or os.environ.get("FLY_DYNAMICS", "published")
    if name == "published":
        return None
    cfg = json.loads((config.METADATA_DIR / f"dynamics_{name}.json").read_text())
    n = connectome.neurons
    al_ln = (n["class"].fillna("").astype(str) == "ALLN").to_numpy()
    sign = n["sign"].to_numpy()
    ad, sl = cfg["adaptation"], cfg["slow_inhibition"]
    adapt = np.full(connectome.n, ad["adapt_mV_per_spike"], np.float32)
    adapt[al_ln] += ad.get("extra_al_ln_adapt_mV_per_spike", 0.0)
    cc = cfg.get("central_complex") or {}
    ring = None
    if cc.get("ring_internal_gain", 1.0) != 1.0 or cc.get("ring_adapt_mV_per_spike") is not None:
        ring = _cx_ring_mask(connectome)
        if cc.get("ring_adapt_mV_per_spike") is not None:
            adapt[ring] = cc["ring_adapt_mV_per_spike"]
    engine.set_dynamics(tau_adapt_ms=ad["tau_ms"], adapt_mV=adapt,
                        tau_slow_ms=sl["tau_ms"],
                        slow_ratio=np.where(al_ln & (sign < 0), sl["slow_ratio"], 0.0).astype(np.float32))
    comp = cfg.get("orn_pn_compensation")
    if comp and comp.get("enabled"):
        mult = engine.plastic_multipliers()
        for pos, m in orn_pn_compensation(connectome):
            mult[pos] *= np.float32(m)
    gf = cfg.get("giant_fibre")
    if gf:
        mult = engine.plastic_multipliers()
        for pos, m in giant_fibre_multipliers(connectome, gf):
            mult[pos] *= np.float32(m)
    eln = cfg.get("al_excitatory_ln")
    if eln:
        is_eln = al_ln & (sign > 0)
        is_pn = (n["class"].fillna("").astype(str) == "ALPN").to_numpy()
        mult = engine.plastic_multipliers()
        mult[_edge_positions(connectome, is_eln, is_pn)] = np.float32(eln["to_pn_gain"])
        mult[_edge_positions(connectome, is_eln, is_eln)] = np.float32(eln["to_eln_gain"])
    if ring is not None and cc.get("ring_internal_gain", 1.0) != 1.0:
        mult = engine.plastic_multipliers()
        mult[_edge_positions(connectome, ring, ring)] *= np.float32(cc["ring_internal_gain"])
    return cfg


def _cx_ring_mask(connectome) -> np.ndarray:
    """E-PG, P-EN1, P-EN2, P-EG and Delta7 neurons (the PB/EB heading ring)."""
    t = connectome.neurons["primary_type"].fillna("").astype(str).to_numpy()
    return np.isin(t, ["EPG", "PEN_a/PEN1", "PEN_b/PEN2", "PEG", "Delta7"])


def make_engine(connectome, params: LIFParams = DEFAULT, seed: int = 0,
                kind: str | None = None):
    """
    The LIF engine for a session. "native" is the multi-core C engine
    (native/, bit-identical to this package's Python engine up to the PRNG);
    "python" is simulation/engine/lif_engine.py. Default: $FLY_ENGINE, else
    native when it has been built.
    """
    from native import lif_native
    kind = kind or os.environ.get("FLY_ENGINE") or (
        "native" if lif_native.available() else "python")
    if kind == "native":
        eng = lif_native.NativeLIFEngine.from_connectome(connectome, seed=seed, params=params)
        cal = _calibrated_gain()
        if cal != 1.0:
            eng.set_gain(cal)
        apply_dynamics(eng, connectome)
        return eng
    if kind == "python":
        if _calibrated_gain() != 1.0:
            raise ValueError("the Python engine implements gain 1.0 only; use the native engine")
        return LIFEngine(connectome, params, seed=seed)
    raise ValueError("unknown engine %r (expected 'native' or 'python')" % kind)


class SpikeRecorder:
    """
    Sliding-window spike statistics, kept cheap enough to run every frame.

    Per 1 ms block only what the closed loop reads is updated: spike totals for
    the tracked descending-neuron groups and the proboscis motor neurons, which
    cost in proportion to spike count. The full per-neuron window (all 139,255
    neurons) and the per-region rates are computed from the window's spikes
    only when something asks for them, i.e. once per displayed frame.
    """

    def __init__(self, connectome, window_frames: int = 50, readout=None):
        self.c = connectome
        self.n = connectome.n
        self.window_frames = window_frames
        self._ring: deque = deque()
        self.window_total = 0          # == window_sum.sum()
        self._ws = None                # cached window_sum

        # region codes for brain-region activity
        regions = connectome.neurons["primary_neuropil"].astype(str)
        self.region_names = sorted(regions.unique())
        code = {r: i for i, r in enumerate(self.region_names)}
        self.region_code = regions.map(code).to_numpy().astype(np.int64)
        self.region_sizes = np.bincount(self.region_code,
                                        minlength=len(self.region_names))

        # running window sums for the readout groups (see DescendingReadout)
        # (proboscis motor neurons are one extra group; they are not DNs)
        self._group_of = None
        if readout is not None:
            if np.intersect1d(readout.group_neurons, readout.proboscis_idx).size:
                raise ValueError("proboscis motor neurons overlap a DN group")
            self._group_of = np.full(self.n, -1, dtype=np.int64)
            self._group_of[readout.group_neurons] = readout.group_ids
            self._group_of[readout.proboscis_idx] = readout.n_groups
            self._sums = np.zeros(readout.n_groups + 1, dtype=np.int64)
            self.group_sum = self._sums[:-1]           # view

    @property
    def proboscis_sum(self) -> int:
        return int(self._sums[-1])

    def _apply(self, spike_idx: np.ndarray, sign: int) -> None:
        self.window_total += sign * int(spike_idx.size)
        if self._group_of is not None:
            g = self._group_of[spike_idx]
            g = g[g >= 0]
            if g.size:
                self._sums += sign * np.bincount(g, minlength=self._sums.size)

    def push(self, spike_idx: np.ndarray) -> None:
        if spike_idx.size:
            self._apply(spike_idx, +1)
        self._ring.append(spike_idx)
        while len(self._ring) > self.window_frames:
            old = self._ring.popleft()
            if old.size:
                self._apply(old, -1)
        self._ws = None

    def _window_spikes(self) -> np.ndarray:
        parts = [a for a in self._ring if a.size]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    @property
    def window_sum(self) -> np.ndarray:
        """Spikes per neuron over the window (all neurons). Read-only."""
        if self._ws is None:
            self._ws = np.bincount(self._window_spikes(),
                                   minlength=self.n).astype(np.int32)
        return self._ws

    def region_activity(self, window_ms: float) -> dict:
        """Mean firing rate (Hz) per brain region over the window."""
        tot = np.bincount(self.region_code[self._window_spikes()],
                          minlength=len(self.region_names))
        rate = tot / np.maximum(self.region_sizes, 1) / (window_ms * 1e-3)
        return {n: float(r) for n, r in zip(self.region_names, rate) if r > 0.01}

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.window_sum))


BODY_TRACK_FIELDS = ("t_ms", "x_mm", "y_mm", "z_mm", "heading_deg", "speed_mm_s",
                     "turn_rate_deg_s", "wing_angle_deg", "proboscis_extension",
                     "leg_extension", "airborne", "behaviour")


class Session:
    """One interactive experiment on the real FlyWire connectome."""

    #: how often the closed loop runs (sensory rates, motor readout, body,
    #: learning). 1 ms by default; FLY_BLOCK_MS=2 halves the Python work per
    #: simulated second (used on the Pi, where it competes with the engine for
    #: memory bandwidth). Must be a whole number of engine steps.
    RATE_UPDATE_MS = float(os.environ.get("FLY_BLOCK_MS", "1.0"))

    def __init__(self, connectome, params: LIFParams = DEFAULT, seed: int = 0,
                 window_ms: float = 50.0, engine: str | None = None,
                 pipelined: bool = True):
        self.c = connectome
        self.p = params
        self.engine = make_engine(connectome, params, seed=seed, kind=engine)
        self.p = getattr(self.engine, "p", params)      # the engine's dt (FLY_DT)
        self._native = hasattr(self.engine, "run_collect")
        # overlap Python readout with native compute (identical results;
        # tests/test_native_engine.py checks this against the serial order)
        self.pipelined = pipelined
        self._total_spikes = 0
        self.readout = DescendingReadout(connectome)
        self.window_ms = window_ms
        self.recorder = SpikeRecorder(connectome,
                                      window_frames=int(window_ms / self.RATE_UPDATE_MS),
                                      readout=self.readout)
        self.encoders = []          # list of (encoder, stimulus)
        self.paused = False
        self.history = []           # telemetry frames, for replay
        self._raster = deque(maxlen=20000)
        # body pose every 1 ms block, for smooth 3D playback (see BODY_TRACK_FIELDS)
        self.body_track = deque(maxlen=5000)

        from fly.body.fly_body import FlyBody
        self.body = FlyBody()
        self.world = None           # closed-loop world (fly/world), see set_world()
        self.world_senses = None
        self.mb = None              # mushroom-body learning (world mode)

        # Neurons whose spikes are always reported individually in the raster.
        n = connectome.neurons
        watch = n[n["super_class"].astype(str).isin(
            ["descending", "visual_projection", "sensory"])]
        self.watch_idx = watch["idx"].to_numpy(dtype=np.int64)
        self._is_watch = np.zeros(connectome.n, dtype=bool)
        self._is_watch[self.watch_idx] = True

    # ------------------------------------------------------------------ world
    def set_world(self, config=None, neural: bool = True, senses: dict | None = None,
                  seed: int = 0, learning: bool = True, navigation: bool = False) -> None:
        """Put the fly in a closed-loop world (config: fly.world.world.WorldConfig),
        or take it out (config=None, back to the open-loop stimulus lab).

        In the world the loop is closed: the body's position sets what the
        senses receive. Sensory rates for a block are computed from the body
        as it was two blocks earlier (the pipelined engine computes block k+1
        while block k is read out), i.e. a 1-2 ms sensory latency; real
        sensory latencies are longer (>= 5 ms)."""
        from fly.body.fly_body import FlyBody
        self.clear_stimuli()
        if self.mb is not None:                     # undo the mushroom body's multipliers
            self.mb = None
            self.engine.plastic_multipliers()[:] = 1.0
            apply_dynamics(self.engine, self.c)
        if config is None:
            self.world = self.world_senses = None
            self.nav = None
            self.body = FlyBody()
            return
        from fly.body.foraging_body import ForagingBody
        from fly.world.senses import WorldSenses
        from fly.world.world import World
        config.seed = seed
        self.world = World(config)
        self.body = ForagingBody(neural=neural, seed=seed)
        self.world_senses = WorldSenses(self.c, self.world, self.body)
        self.world_senses.dt_ms = self.RATE_UPDATE_MS
        for k, v in (senses or {}).items():
            self.world_senses.enabled[k] = bool(v)
        self.add_stimulus(self.world_senses, self.world_senses)
        if learning and self._native:
            # dopamine-gated KC->MBON plasticity, running while the fly explores
            from brain.plasticity.mushroom_body import MushroomBody
            self.mb = MushroomBody(self.c, self.engine, plastic=True, kc_mbon_gain=8.0,
                                   dan_modulatory=True)
        self.world_senses.reinforcement = bool(learning)
        self.nav = None
        if navigation and self.mb is not None:
            # heading compass + FC2 goal gated by learned odour valence
            # (brain/navigation/valence_goal.py)
            from brain.navigation.compass import Compass, CompassDrive
            from brain.navigation.goal import GoalCircuit, GoalDrive
            from brain.navigation.valence_goal import ValenceGoal
            cx = Compass(self.c)
            cd = CompassDrive(cx, heading_deg=self.body.state.heading_deg)
            gd = GoalDrive(GoalCircuit(self.c, cx))
            self.nav = ValenceGoal(self.mb, gd, cd)
            self.add_stimulus(cd, cd)
            self.add_stimulus(gd, gd)

    # ---------------------------------------------------------------- stimuli
    def clear_stimuli(self) -> None:
        self.encoders = []
        self.engine.clear_poisson()

    def add_stimulus(self, encoder, stimulus) -> None:
        self.encoders.append((encoder, stimulus))

    def add_modality(self, key: str, intensity: float = 1.0,
                     duration_ms: float = float("inf"),
                     delay_ms: float = 0.0):
        """
        Deliver one of the registered stimuli (see brain/sensory/modalities.py).

        Raises for a modality the connectome cannot support, rather than faking
        a response.
        """
        from brain.sensory.encoders import PopulationEncoder
        from brain.sensory.modalities import BY_KEY
        from simulation.stimuli.pulse import PulseStimulus

        m = BY_KEY.get(key)
        if m is None:
            raise KeyError("unknown modality %r" % key)
        if not m.supported:
            raise ValueError("%s is not currently modeled: %s"
                             % (m.label, m.unsupported_reason))
        enc = PopulationEncoder(self.c, m)
        stim = PulseStimulus(modality_key=key, intensity=intensity,
                             t_start_ms=self.engine.t_ms + delay_ms,
                             duration_ms=duration_ms)
        self.add_stimulus(enc, stim)
        return stim

    def add_looming(self, azimuth_deg: float = 45.0, elevation_deg: float = 0.0,
                    half_size_mm: float = 5.0, speed_mm_s: float = 250.0,
                    start_distance_mm: float = 50.0, delay_ms: float = 0.0):
        """Throw an object at the fly."""
        from brain.sensory.encoders import LoomingEncoder
        from brain.sensory.retinotopy import load_retinotopy
        from simulation.stimuli.looming import LoomingStimulus

        enc = LoomingEncoder(self.c, load_retinotopy(self.c))
        stim = LoomingStimulus(
            azimuth_deg=azimuth_deg, elevation_deg=elevation_deg,
            half_size_mm=half_size_mm, speed_mm_s=speed_mm_s,
            start_distance_mm=start_distance_mm,
            t_start_ms=self.engine.t_ms + delay_ms)
        self.add_stimulus(enc, stim)
        return stim

    def _refresh_rates(self) -> None:
        self._apply_rates(self._rates_at(self.engine.t_ms))

    def _rates_at(self, t_ms: float):
        """Poisson targets and rates for a block starting at t_ms, or None."""
        if not self.encoders:
            return None
        if len(self.encoders) == 1:
            # Same result as the general path below (unique neurons in
            # ascending order, which fixes how Poisson draws map to neurons),
            # with the sort computed once per encoder.
            enc, stim = self.encoders[0]
            order = getattr(enc, "_ascending", None)
            if order is None:
                order = np.argsort(enc.indices, kind="stable")
                if np.unique(enc.indices).size != enc.indices.size:
                    order = False                    # duplicates: use general path
                enc._ascending = order
            if order is not False:
                return enc.indices[order], enc.rates_hz(t_ms, stim)[order]
        idx_all, rate_all = [], []
        for enc, stim in self.encoders:
            idx_all.append(enc.indices)
            rate_all.append(enc.rates_hz(t_ms, stim))
        idx = np.concatenate(idx_all)
        rates = np.concatenate(rate_all)
        # A neuron driven by several stimuli takes the strongest drive.
        order = np.argsort(-rates)
        idx, rates = idx[order], rates[order]
        uniq, first = np.unique(idx, return_index=True)
        return uniq, rates[first]

    def _apply_rates(self, r) -> None:
        if r is not None:
            self.engine.set_poisson(*r)

    # -------------------------------------------------------------------- run
    #: full telemetry frames kept in .history (one per advance() call)
    HISTORY_MAX = 20000

    def advance(self, duration_ms: float) -> list:
        """
        Advance the simulation in 1 ms blocks. The closed loop (sensory rates,
        descending readout, body) runs every block; the full telemetry frame,
        which only displays and experiments read, is built for the last block.
        Returns [frame].
        """
        if self.paused:
            return []
        n_blocks = max(1, int(round(duration_ms / self.RATE_UPDATE_MS)))
        steps = int(round(self.RATE_UPDATE_MS / self.p.dt))

        t_start = time.perf_counter()
        if self._native and self.pipelined:
            # Pipelined. While C computes block k, Python prepares block k+1's
            # rates and then processes block k's spikes (readout, body) during
            # block k+1. Open loop (stimuli depend only on the clock) this is
            # identical to the serial order; in a closed-loop world, block
            # k+1's senses see the body as it was after block k-1.
            step0 = self.engine.step_count
            t_block = lambda k: (step0 + k * steps) * self.p.dt   # == engine.t_ms
            self._refresh_rates()
            self.engine.start(steps)
            for k in range(n_blocks):
                nxt = self._rates_at(t_block(k + 1)) if k + 1 < n_blocks else None
                spk = self.engine.wait()
                if k + 1 < n_blocks:
                    self._apply_rates(nxt)
                    self.engine.start(steps)
                self._post_block(spk, t_block(k + 1))
        else:
            for _ in range(n_blocks):
                self._refresh_rates()
                if self._native:
                    spk = self.engine.run_collect(steps)
                else:
                    block = [self.engine.step() for _ in range(steps)]
                    spk = (np.concatenate(block) if any(b.size for b in block)
                           else np.empty(0, dtype=np.int64))
                self._post_block(spk, self.engine.t_ms)
        wall = (time.perf_counter() - t_start) / n_blocks

        frame = self._telemetry(spk, wall)
        self.history.append(frame)
        if len(self.history) > self.HISTORY_MAX:
            del self.history[:len(self.history) - self.HISTORY_MAX]
        return [frame]

    def _post_block(self, spk: np.ndarray, t_ms: float) -> None:
        """Per-millisecond work: statistics, raster, descending readout, body.
        `t_ms` is the end of the block (the engine may already be past it)."""
        self._total_spikes += int(spk.size)
        self.recorder.push(spk)
        win = self.window_ms
        rec = self.recorder

        # raster entries for watched neurons only (keeps payload small)
        if spk.size:
            hit = spk[self._is_watch[spk]]
            if hit.size:
                rt = round(t_ms, 2)
                self._raster.extend((rt, i) for i in hit[:200].tolist())

        self._channels = self.readout.channels(None, win, sums=rec.group_sum)
        self._laterality = self.readout.escape_laterality(None, sums=rec.group_sum)
        self._prob = self.readout.proboscis_drive_from_total(rec.proboscis_sum, win)

        # The body is driven ONLY by these neural readouts.
        self.body.update(self.RATE_UPDATE_MS, self._channels, t_ms,
                         escape_laterality=self._laterality,
                         proboscis_drive=self._prob)
        if self.world is not None:
            self.world.step(self.RATE_UPDATE_MS, t_ms, self.body)
        if self.mb is not None:
            self.mb.step(spk, self.RATE_UPDATE_MS)
        if getattr(self, "nav", None) is not None:
            b = self.body.state
            self.nav.compass.set_heading(b.heading_deg)
            if getattr(self, "nav_inputs", None) is not None:      # e.g. the Habitat home
                od, af, asp = self.nav_inputs()
            else:
                sn = self.world.senses if self.world is not None else {}
                od = 0.5 * (sn.get("odour_L", 0.0) + sn.get("odour_R", 0.0))
                af, asp = sn.get("air_from_deg"), sn.get("airspeed_mm_s", 0.0)
            self.nav.step(spk, self.RATE_UPDATE_MS, b.heading_deg, odour=od,
                          air_from_deg=af, airspeed=asp)
        b = self.body.state
        self.body_track.append((t_ms, b.x_mm, b.y_mm, b.z_mm, b.heading_deg,
                                b.speed_mm_s, b.turn_rate_deg_s, b.wing_angle_deg,
                                b.proboscis_extension, b.leg_extension,
                                b.airborne, b.behaviour))

    def _telemetry(self, spk: np.ndarray, wall_s: float) -> dict:
        eng = self.engine
        win = self.window_ms
        ws = self.recorder.window_sum
        stim_state = [{"kind": type(s).__name__, **s.state(eng.t_ms)}
                      for _, s in self.encoders]
        return {
            "t_ms": round(eng.t_ms, 3),
            "n_spikes": int(spk.size),
            "active_neurons": self.recorder.active_count,
            "total_spikes": self._total_spikes,
            "mean_rate_hz": float(self.recorder.window_total / self.c.n / (win * 1e-3)),
            "regions": self.recorder.region_activity(win),
            "dn_rates": self.readout.rates(ws, win),
            "channels": self._channels,
            "proboscis_drive": self._prob,
            "escape_laterality": self._laterality,
            "stimuli": stim_state,
            "body": self.body.as_dict(),
            "world": None if self.world is None else self.world.state(),
            "learning": None if self.mb is None else self.mb.summary(),
            "navigation": None if getattr(self, "nav", None) is None else self.nav.state(),
            "wall_ms": round(wall_s * 1000.0, 3),     # mean wall time per 1 ms block
        }

    # ------------------------------------------------------------------ state
    def reset(self, seed: int = 0) -> None:
        if self._native:
            self.engine.reset(seed=seed)
        else:
            self.engine.reset()
            self.engine.rng = np.random.default_rng(seed)
        self._total_spikes = 0
        self.recorder = SpikeRecorder(
            self.c, window_frames=int(self.window_ms / self.RATE_UPDATE_MS),
            readout=self.readout)
        self.history.clear()
        self._raster.clear()
        self.body_track.clear()
        self.clear_stimuli()
        if self.world is not None:
            self.world.reset(seed)
            self.body._seed = seed
            self.body.reset()
            self.world_senses.body = self.body
            self.add_stimulus(self.world_senses, self.world_senses)
            if getattr(self, "nav", None) is not None:
                self.nav.compass.set_heading(self.body.state.heading_deg)
                self.add_stimulus(self.nav.compass, self.nav.compass)
                self.add_stimulus(self.nav.goal, self.nav.goal)

    def raster(self, last_ms: float = 200.0) -> list:
        t_min = self.engine.t_ms - last_ms
        return [[t, i] for t, i in self._raster if t >= t_min]

    @property
    def provenance(self) -> dict:
        return {
            "engine": self.engine.provenance,
            "motor": self.readout.provenance,
            "encoders": [e.provenance for e, _ in self.encoders],
        }
