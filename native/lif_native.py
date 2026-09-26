"""
ctypes binding for the native aarch64 LIF engine (native/lif_native.c).

A drop-in for simulation/engine/lif_engine.LIFEngine where the laboratory uses
it (step, set_poisson, silence, reset, spike_counts, t_ms, provenance), with
the hot loop running in C across several cores. Poisson draws use the shared
mulberry32 stream (tools/prng.py, web/js/engine.js) rather than numpy's
generator, so spike trains are comparable bit for bit with the browser engine;
native/verify_native.py checks that.

Build:  make -C native
"""
from __future__ import annotations

import ctypes as C
import json
import os
from pathlib import Path

import numpy as np

from brain.neuron_models.lif import DEFAULT, LIFParams

ROOT = Path(__file__).resolve().parent.parent
_LIB = Path(os.environ.get("FLY_NATIVE_LIB", Path(__file__).resolve().parent / "liblif.so"))
WEB_DATA = ROOT / "web" / "data"

#: Compute threads. The caller is one of them (pinned to core 0); on a Pi 5
#: that also runs other services, leaving one core free is faster than using 4.
DEFAULT_THREADS = int(os.environ.get("FLY_THREADS", "3"))

_i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
_i16p = np.ctypeslib.ndpointer(np.int16, flags="C_CONTIGUOUS")
_f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")


def available() -> bool:
    return _LIB.exists()


def _load_lib():
    lib = C.CDLL(str(_LIB))
    lib.lif_create.restype = C.c_void_p
    lib.lif_create.argtypes = [C.c_int, C.c_int, _i32p, _i32p, _i16p, C.c_uint32, C.c_int]
    for name in ("lif_destroy", "lif_reset", "lif_wake_all"):
        getattr(lib, name).argtypes = [C.c_void_p]
    lib.lif_set_seed.argtypes = [C.c_void_p, C.c_uint32]
    lib.lif_set_poisson.argtypes = [C.c_void_p, _i32p, _f64p, C.c_int]
    lib.lif_set_poisson_rates.argtypes = [C.c_void_p, _f64p, C.c_int]
    lib.lif_set_poisson_rates.restype = C.c_int
    lib.lif_set_gain.argtypes = [C.c_void_p, _i16p, C.c_double]
    _f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    lib.lif_set_dynamics.argtypes = [C.c_void_p, C.c_double, _f32p, C.c_double, _f32p]
    for name in ("lif_adapt", "lif_gs"):
        getattr(lib, name).argtypes = [C.c_void_p]
        getattr(lib, name).restype = C.POINTER(C.c_float)
    lib.lif_add_g.argtypes = [C.c_void_p, _i32p, np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"), C.c_int]
    lib.lif_plastic_enable.argtypes = [C.c_void_p]
    lib.lif_plastic_enable.restype = C.POINTER(C.c_float)
    lib.lif_silence.argtypes = [C.c_void_p, _i32p, C.c_int, C.c_int]
    lib.lif_step.argtypes = [C.c_void_p]
    lib.lif_step.restype = C.c_int
    lib.lif_run.argtypes = [C.c_void_p, C.c_int]
    lib.lif_run.restype = C.c_int
    lib.lif_run_collect.argtypes = [C.c_void_p, C.c_int]
    lib.lif_run_collect.restype = C.c_long
    lib.lif_start.argtypes = [C.c_void_p, C.c_int]
    lib.lif_wait.argtypes = [C.c_void_p]
    lib.lif_wait.restype = C.c_long
    for name, ctype in (("lif_spikes", C.c_int32), ("lif_collected", C.c_int32),
                        ("lif_v", C.c_float), ("lif_g", C.c_float),
                        ("lif_spike_counts", C.c_int32)):
        fn = getattr(lib, name)
        fn.argtypes = [C.c_void_p]
        fn.restype = C.POINTER(ctype)
    for name in ("lif_step_count", "lif_active_blocks", "lif_edges"):
        getattr(lib, name).argtypes = [C.c_void_p]
        getattr(lib, name).restype = C.c_long
    lib.lif_nblocks.argtypes = [C.c_void_p]
    lib.lif_nblocks.restype = C.c_int
    lib.lif_phase_times.argtypes = [C.c_void_p, _f64p]
    if hasattr(lib, "lif_sync_host"):          # CUDA engine: device -> host mirrors
        lib.lif_sync_host.argtypes = [C.c_void_p]
        lib.lif_sync_host.restype = None
    if hasattr(lib, "lif_plastic_commit"):     # CUDA engine: upload multipliers
        lib.lif_plastic_commit.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
        lib.lif_plastic_commit.restype = None
    if hasattr(lib, "lif_dt"):
        lib.lif_dt.argtypes = [C.c_void_p]
        lib.lif_dt.restype = C.c_double
    if hasattr(lib, "lif_set_quiesce_tol"):
        lib.lif_set_quiesce_tol.argtypes = [C.c_void_p, C.c_float]
        lib.lif_set_quiesce_tol.restype = None
    return lib


def load_web_connectome(data_dir: Path = WEB_DATA):
    """Decode web/data/connectome.bin (see tools/export_web_connectome.py)."""
    meta = json.loads((data_dir / "meta.json").read_text())
    n, nnz = meta["n"], meta["nnz"]
    buf = (data_dir / "connectome.bin").read_bytes()
    indptr = np.frombuffer(buf, np.int32, n + 1, 0).copy()
    delta = np.frombuffer(buf, np.int32, nnz, (n + 1) * 4).astype(np.int64)
    weights = np.frombuffer(buf, np.int16, nnz, (n + 1) * 4 + nnz * 4).copy()
    # undo per-row delta coding: cumulative sum restarted at each row start
    row = np.repeat(np.arange(n), np.diff(indptr))
    cs = np.cumsum(delta)
    starts = indptr[:-1].astype(np.int64)
    base = np.where(starts > 0, cs[np.maximum(starts - 1, 0)], 0)
    indices = (cs - base[row]).astype(np.int32)
    return meta, indptr, indices, weights


class NativeLIFEngine:
    """Steppable whole-brain LIF simulation, native multi-core backend."""

    def __init__(self, indptr, indices, weights, seed: int = 0,
                 threads: int = DEFAULT_THREADS, params: LIFParams = DEFAULT,
                 connectome=None):
        if params != DEFAULT:
            raise ValueError("the native engine implements the published "
                             "Shiu et al. 2024 parameters only")
        self._h = None
        self._lib = _load_lib()
        self._keep = (np.ascontiguousarray(indptr, np.int32),
                      np.ascontiguousarray(indices, np.int32),
                      np.ascontiguousarray(weights, np.int16))
        self.n = len(indptr) - 1
        self.p = params
        self.c = connectome
        self.threads = threads
        self._h = self._lib.lif_create(self.n, len(indices), *self._keep,
                                       seed & 0xFFFFFFFF, threads)
        if self._h and hasattr(self._lib, "lif_dt"):
            # FLY_DT (default 0.1 ms, the published value) is read by the engine
            import dataclasses
            dt = float(self._lib.lif_dt(self._h))
            if dt != self.p.dt:
                self.p = dataclasses.replace(self.p, dt=dt)
        if not self._h:
            raise ValueError("engine creation failed (FLY_DT must make 1.8 and 2.2 ms whole steps) or connectome too large for the packed format "
                             "(>262,144 neurons or |synapse count| >= 8192)")
        self._sync = getattr(self._lib, "lif_sync_host", None)
        self._v = np.ctypeslib.as_array(self._lib.lif_v(self._h), (self.n,))
        self._g = np.ctypeslib.as_array(self._lib.lif_g(self._h), (self.n,))
        self._adapt = self._gs = None
        self.spike_counts = np.ctypeslib.as_array(
            self._lib.lif_spike_counts(self._h), (self.n,))
        self._silenced = np.zeros(self.n, dtype=bool)   # mirror, for inspection
        self._poi_idx = np.empty(0, np.int32)             # current Poisson targets

    # State views. With the CUDA engine the state lives on the GPU and these
    # host mirrors are refreshed when read (and uploaded again by wake_all()).
    @property
    def v(self) -> np.ndarray:
        if self._sync:
            self._sync(self._h)
        return self._v

    @property
    def g(self) -> np.ndarray:
        if self._sync:
            self._sync(self._h)
        return self._g

    @property
    def adapt(self):
        if self._sync and self._adapt is not None:
            self._sync(self._h)
        return self._adapt

    @property
    def gs(self):
        if self._sync and self._gs is not None:
            self._sync(self._h)
        return self._gs

    @classmethod
    def from_connectome(cls, connectome, seed: int = 0,
                        threads: int = DEFAULT_THREADS, params: LIFParams = DEFAULT):
        w = connectome.w.tocsr()
        w.sort_indices()
        if np.abs(w.data).max() > 32767:
            raise ValueError("synapse counts exceed int16")
        return cls(w.indptr, w.indices, w.data.astype(np.int16), seed=seed,
                   threads=threads, params=params, connectome=connectome)

    @classmethod
    def from_web_data(cls, seed: int = 0, threads: int = DEFAULT_THREADS):
        meta, indptr, indices, weights = load_web_connectome()
        eng = cls(indptr, indices, weights, seed, threads)
        eng.meta = meta
        return eng

    def close(self):
        if self._h:
            self._lib.lif_destroy(self._h)
            self._h = None

    __del__ = close

    # ------------------------------------------------------------------ state
    def reset(self, seed: int | None = None):
        """Back to rest; clears stimuli and silencing, as LIFEngine.reset does."""
        self._lib.lif_reset(self._h)
        self._silenced[:] = False
        self._poi_idx = np.empty(0, np.int32)
        if seed is not None:
            self._lib.lif_set_seed(self._h, seed & 0xFFFFFFFF)

    def set_gain(self, gain: float):
        """Scale all recurrent synaptic weights by `gain` (sensory drive
        unchanged). 1.0 is the published model. For activity-matched controls."""
        self._lib.lif_set_gain(self._h, self._keep[2], float(gain))
        self.gain = float(gain)

    def set_dynamics(self, tau_adapt_ms=200.0, adapt_mV=0.0, tau_slow_ms=100.0, slow_ratio=0.0):
        """Optional dynamics beyond the published model: spike-frequency
        adaptation (+adapt_mV per spike, decaying with tau_adapt_ms) and slow
        (GABA-B-like) inhibition (an inhibitory neuron's synapses also drive a
        slow channel at slow_ratio x their weight, decaying with tau_slow_ms).
        adapt_mV (per neuron) and slow_ratio (per presynaptic neuron) are
        scalars or arrays of n. All zero restores the exact published model."""
        a = np.ascontiguousarray(np.broadcast_to(np.float32(adapt_mV) if np.isscalar(adapt_mV)
                                                 else np.asarray(adapt_mV, np.float32), (self.n,)))
        r = np.ascontiguousarray(np.broadcast_to(np.float32(slow_ratio) if np.isscalar(slow_ratio)
                                                 else np.asarray(slow_ratio, np.float32), (self.n,)))
        self._lib.lif_set_dynamics(self._h, tau_adapt_ms, a, tau_slow_ms, r)
        self.dynamics = dict(tau_adapt_ms=tau_adapt_ms, adapt_mV=adapt_mV,
                             tau_slow_ms=tau_slow_ms, slow_ratio=slow_ratio)
        on = bool((a > 0).any() or (r > 0).any())
        # views of the extra state (None when the published model is active)
        self._adapt = np.ctypeslib.as_array(self._lib.lif_adapt(self._h), (self.n,)) if on else None
        self._gs = np.ctypeslib.as_array(self._lib.lif_gs(self._h), (self.n,)) if on else None

    def add_g(self, indices, values):
        """Add to the synaptic drive g (mV) of the given neurons, between steps."""
        idx = np.ascontiguousarray(indices, np.int32)
        val = np.ascontiguousarray(np.broadcast_to(values, idx.shape), np.float32)
        self._lib.lif_add_g(self._h, idx, val, idx.size)

    def plastic_multipliers(self) -> np.ndarray:
        """Turn on per-connection plasticity and return the multipliers (a
        float32 view, one per connection in CSR order, initially 1.0). Modify
        them only between steps, never while a start()/wait() block runs."""
        ptr = self._lib.lif_plastic_enable(self._h)
        return np.ctypeslib.as_array(ptr, (len(self._keep[1]),))

    def commit_plastic(self, positions=None) -> None:
        """Tell the engine that plasticity multipliers changed (all of them, or
        the given connection positions). Needed by the CUDA engine, which keeps
        a device copy; a no-op for the CPU engine, which reads them in place."""
        fn = getattr(self._lib, "lif_plastic_commit", None)
        if fn is None:
            return
        if positions is None:
            fn(self._h, None, -1)
            return
        pos = np.ascontiguousarray(positions, np.int64)
        fn(self._h, pos.ctypes.data, int(pos.size))

    def wake_all(self):
        """Call after writing to .v or .g directly, so no block stays skipped."""
        self._lib.lif_wake_all(self._h)

    # -------------------------------------------------------------- stimulation
    def set_poisson(self, indices, rates_hz):
        idx = np.ascontiguousarray(np.atleast_1d(indices), np.int32)
        rates = np.ascontiguousarray(np.atleast_1d(np.asarray(rates_hz, np.float64)))
        if rates.size == 1 and idx.size > 1:
            rates = np.repeat(rates, idx.size)
        if idx.size != rates.size:
            raise ValueError("indices and rates_hz must have the same length")
        # The session refreshes rates every millisecond on unchanged targets.
        if (idx.size == self._poi_idx.size and idx.size
                and np.array_equal(idx, self._poi_idx)
                and self._lib.lif_set_poisson_rates(self._h, rates, idx.size) == 0):
            return
        self._lib.lif_set_poisson(self._h, idx, rates, idx.size)
        self._poi_idx = idx

    def clear_poisson(self):
        self._lib.lif_set_poisson(self._h, np.zeros(1, np.int32), np.zeros(1), 0)
        self._poi_idx = np.empty(0, np.int32)

    def silence(self, indices, on: bool = True):
        """Silence neurons by removing their output (Shiu et al. `silence()`)."""
        idx = np.ascontiguousarray(np.atleast_1d(indices), np.int32)
        self._lib.lif_silence(self._h, idx, idx.size, int(on))
        self._silenced[idx] = on

    def unsilence(self, indices):
        self.silence(indices, on=False)

    def unsilence_all(self):
        self.unsilence(np.flatnonzero(self._silenced))

    # -------------------------------------------------------------------- run
    def step(self) -> np.ndarray:
        """Advance one dt. Returns the indices of neurons that spiked."""
        k = self._lib.lif_step(self._h)
        if not k:
            return np.empty(0, np.int32)
        return np.ctypeslib.as_array(self._lib.lif_spikes(self._h), (k,)).copy()

    def run_collect(self, steps: int) -> np.ndarray:
        """Advance `steps` steps; all their spikes in one array, in step order."""
        k = self._lib.lif_run_collect(self._h, steps)
        if not k:
            return np.empty(0, np.int32)
        return np.ctypeslib.as_array(self._lib.lif_collected(self._h), (k,)).copy()

    def start(self, steps: int) -> None:
        """Begin `steps` steps on the engine's driver thread and return at once.
        Do not touch the engine again until wait()."""
        self._lib.lif_start(self._h, steps)

    def wait(self) -> np.ndarray:
        """Block (GIL released) until start()'s steps finish; their spikes."""
        k = self._lib.lif_wait(self._h)
        if not k:
            return np.empty(0, np.int32)
        return np.ctypeslib.as_array(self._lib.lif_collected(self._h), (k,)).copy()

    def run(self, duration_ms: float) -> int:
        """Run for a duration in C without collecting spikes. Returns the total."""
        return self._lib.lif_run(self._h, int(round(duration_ms / self.p.dt)))

    # ---------------------------------------------------------------- helpers
    @property
    def step_count(self) -> int:
        return self._lib.lif_step_count(self._h)

    @property
    def t_ms(self) -> float:
        return self.step_count * self.p.dt

    def firing_rates_hz(self, window_ms: float = None) -> np.ndarray:
        elapsed = self.t_ms if window_ms is None else window_ms
        if elapsed <= 0:
            return np.zeros(self.n, dtype=np.float64)
        return self.spike_counts / (elapsed * 1e-3)

    @property
    def active_fraction(self) -> float:
        """Mean fraction of 16-neuron blocks integrated per step since reset."""
        steps = max(1, self.step_count)
        return self._lib.lif_active_blocks(self._h) / (steps * self._lib.lif_nblocks(self._h))

    def set_quiesce_tolerance(self, tol_mV: float = 0.0) -> None:
        """Skip blocks whose neurons are within tol_mV of rest (and |g| <
        tol_mV), snapping them to rest. 0 (default) = exact published model,
        bit-identical to the reference engines; > 0 trades that for speed when
        activity is widespread and subthreshold (e.g. a closed-loop world)."""
        self._lib.lif_set_quiesce_tol(self._h, float(tol_mV))
        self.quiesce_tol = float(tol_mV)

    def phase_times_us(self) -> dict:
        """Mean microseconds per step in each phase since reset."""
        out = np.zeros(3)
        self._lib.lif_phase_times(self._h, out)
        out /= max(1, self.step_count)
        return {k: round(float(x), 1) for k, x in zip(("integrate", "serial", "scatter"), out)}

    @property
    def provenance(self) -> dict:
        d = {
            "n_neurons": self.n,
            "n_connections": int(len(self._keep[1])),
            "n_synapses": int(np.abs(self._keep[2].astype(np.int64)).sum()),
            "model": "Shiu et al. 2024 LIF (exact linear integration)",
            "backend": "native C, %d threads; bit-identical to the Python and "
                       "browser engines (native/verify_native.py) unless extra "
                       "dynamics are enabled" % self.threads,
            "gain": getattr(self, "gain", 1.0),
            "extra_dynamics": getattr(self, "dynamics", None) and {
                k: (v if np.isscalar(v) else "per-neuron") for k, v in self.dynamics.items()},
            "params": self.p.as_dict(),
        }
        if self.c is not None:
            d = {"connectome": self.c.dataset, **d}
        return d
