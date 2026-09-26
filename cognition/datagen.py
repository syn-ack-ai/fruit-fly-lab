"""
Labelled whole-brain spike data for training decoders ("artificial cortex").

Each trial: the brain starts at rest, one stimulus class drives its real
sensory neurons for 100 ms, and the whole brain's spikes are recorded in 10 ms
bins until 500 ms after the stimulus ends. Nothing else is recorded: no
membrane potentials, no hand-picked features, no labels inside the data.

Two facts are stored so decoders can be tested honestly:
  - which neurons each class drives directly (`driven_<class>`), so a decoder
    can be denied the input neurons and must read the rest of the brain;
  - stimulus onset/offset bins, so decoding can be measured as a function of
    time since the stimulus ended (how long the brain itself holds the
    information).

Stimulus strength, looming speed/size and onset time are randomised per trial
so a decoder cannot key on one fixed pattern.

Run (on a multi-core machine; each worker is a separate process):
    python -m cognition.datagen --out data/cognition/real --trials 4000 --workers 4 --threads 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

BIN_MS = 10
PRE_MS = 20            # rest before any stimulus
JITTER_MS = 30         # random extra delay before onset
STIM_MS = 100          # drive duration
POST_MS = 500          # recording after the stimulus ends
TRIAL_MS = PRE_MS + JITTER_MS + STIM_MS + POST_MS
N_BINS = TRIAL_MS // BIN_MS

MODALITIES = ["taste_sugar", "taste_bitter", "odor_vinegar", "odor_geosmin", "odor_cva",
              "odor_co2", "wind", "sound", "touch_head", "cold", "heat", "humidity"]
LOOMING = {"loom_left": -45.0, "loom_right": 45.0, "loom_front": 0.0}
CLASSES = ["none", *LOOMING, *MODALITIES]


class _Stim:
    """A stimulus whose drive is on only between t_on and t_off (sim ms)."""

    def __init__(self, enc, inner, t_on, t_off, gain=1.0):
        self.enc, self.inner, self.t_on, self.t_off, self.gain = enc, inner, t_on, t_off, gain

    def rates(self, t):
        if not (self.t_on <= t < self.t_off):
            return None
        return self.enc.rates_hz(t, self.inner) * self.gain


def _setup(connectome):
    from brain.sensory.encoders import LoomingEncoder, PopulationEncoder
    from brain.sensory.modalities import BY_KEY
    from brain.sensory.retinotopy import load_retinotopy
    loom = LoomingEncoder(connectome, load_retinotopy(connectome))
    pop = {k: PopulationEncoder(connectome, BY_KEY[k]) for k in MODALITIES}
    return loom, pop


def _stimulus(cls, rng, loom, pop, t_on):
    from simulation.stimuli.looming import LoomingStimulus
    from simulation.stimuli.pulse import PulseStimulus
    t_off = t_on + STIM_MS
    if cls == "none":
        return None
    if cls in LOOMING:
        speed = rng.uniform(150.0, 400.0)
        # collision just after the drive ends: the strongest part of the approach
        st = LoomingStimulus(azimuth_deg=LOOMING[cls] + rng.uniform(-10, 10),
                             elevation_deg=rng.uniform(-10, 10),
                             half_size_mm=rng.uniform(3.0, 8.0), speed_mm_s=speed,
                             start_distance_mm=speed * (STIM_MS + 10) / 1000.0, t_start_ms=t_on)
        return _Stim(loom, st, t_on, t_off)
    st = PulseStimulus(modality_key=cls, intensity=rng.uniform(0.5, 1.0),
                       t_start_ms=t_on, duration_ms=STIM_MS)
    return _Stim(pop[cls], st, t_on, t_off)


def run_trial(engine, stim, seed):
    """Simulate one trial; return (bins, neurons, counts) of all spikes."""
    engine.reset(seed=seed)
    steps_per_ms = int(round(1.0 / engine.p.dt))
    b_all, n_all = [], []
    idx = None
    for t in range(TRIAL_MS):
        r = stim.rates(float(t)) if stim is not None else None
        if r is not None:
            if idx is None:
                idx = np.sort(stim.enc.indices)
                order = np.argsort(stim.enc.indices, kind="stable")
            engine.set_poisson(idx, r[order])
        elif idx is not None:
            engine.clear_poisson()
            idx = None
        spk = engine.run_collect(steps_per_ms)
        if spk.size:
            n_all.append(spk)
            b_all.append(np.full(spk.size, t // BIN_MS, dtype=np.uint8))
    if not n_all:
        return (np.empty(0, np.uint8), np.empty(0, np.int32), np.empty(0, np.uint16))
    b = np.concatenate(b_all).astype(np.int64)
    n = np.concatenate(n_all).astype(np.int64)
    key, cnt = np.unique(b * engine.n + n, return_counts=True)
    return ((key // engine.n).astype(np.uint8), (key % engine.n).astype(np.int32),
            cnt.astype(np.uint16))


def _worker(args):
    wid, trial_ids, out, threads, seed0, wiring, gain = args
    os.environ["FLY_THREADS"] = str(threads)
    os.environ["FLY_CPU_BASE"] = str(wid * threads)          # separate cores per worker
    from brain.neurons.registry import load_connectome
    from native.lif_native import NativeLIFEngine
    from cognition.wiring import wiring_for

    c = load_connectome()
    indptr, indices, weights = wiring_for(c, wiring)
    engine = NativeLIFEngine(indptr, indices, weights, seed=0, threads=threads, connectome=c)
    if gain != 1.0:
        engine.set_gain(gain)
    loom, pop = _setup(c)

    meta = {"cls": [], "seed": [], "on_bin": [], "off_bin": []}
    ptr, B, N, C = [0], [], [], []
    t0 = time.time()
    for k, tid in enumerate(trial_ids):
        rng = np.random.default_rng([seed0, tid])
        cls = CLASSES[tid % len(CLASSES)]                    # balanced classes
        t_on = PRE_MS + int(rng.integers(0, JITTER_MS + 1))
        stim = _stimulus(cls, rng, loom, pop, t_on)
        b, n, cnt = run_trial(engine, stim, seed=int(rng.integers(1, 2**31 - 1)))
        B.append(b); N.append(n); C.append(cnt); ptr.append(ptr[-1] + len(n))
        meta["cls"].append(CLASSES.index(cls))
        meta["seed"].append(int(tid))
        meta["on_bin"].append(t_on // BIN_MS)
        meta["off_bin"].append((t_on + STIM_MS) // BIN_MS)
        if wid == 0 and (k + 1) % 25 == 0:
            rate = (k + 1) / (time.time() - t0)
            print(f"[worker 0] {k + 1}/{len(trial_ids)} trials, {rate:.2f}/s", flush=True)
    engine.close()
    path = Path(out) / f"shard_{wid:03d}.npz"
    np.savez(path, ptr=np.array(ptr, np.int64),
             bins=np.concatenate(B) if B else np.empty(0, np.uint8),
             neurons=np.concatenate(N) if N else np.empty(0, np.int32),
             counts=np.concatenate(C) if C else np.empty(0, np.uint16),
             **{k: np.array(v, np.int32) for k, v in meta.items()})
    return str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=1600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="engine threads per worker")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wiring", default="real", help="real | shuffled | random")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="recurrent synaptic gain (1.0 = published model)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from brain.neurons.registry import load_connectome
    c = load_connectome()
    loom, pop = _setup(c)
    driven = {"loom_left": loom.indices, "loom_right": loom.indices, "loom_front": loom.indices,
              **{k: pop[k].indices for k in MODALITIES}}
    np.savez(out / "driven.npz", **{k: np.asarray(v, np.int32) for k, v in driven.items()})
    (out / "meta.json").write_text(json.dumps({
        "classes": CLASSES, "bin_ms": BIN_MS, "n_bins": N_BINS, "trial_ms": TRIAL_MS,
        "stim_ms": STIM_MS, "post_ms": POST_MS, "n_neurons": int(c.n),
        "wiring": a.wiring, "gain": a.gain, "seed": a.seed, "trials": a.trials,
        "dataset": c.dataset}, indent=1))

    ids = np.arange(a.trials)
    shards = [(w, ids[w::a.workers].tolist(), str(out), a.threads, a.seed, a.wiring, a.gain)
              for w in range(a.workers)]
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        for p in pool.imap_unordered(_worker, shards):
            print("wrote", p, flush=True)
    print(f"{a.trials} trials in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
