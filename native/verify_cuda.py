"""Check the CUDA engine (native/liblif_cuda.so) against the CPU engine
(native/liblif.so): identical spike trains and final state.

    FLY_THREADS=8 python native/verify_cuda.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.neurons.registry import load_connectome            # noqa: E402
from native import lif_native                                # noqa: E402
from simulation.engine.session import apply_dynamics          # noqa: E402

HERE = Path(__file__).resolve().parent
CPU, GPU = HERE / "liblif.so", HERE / "liblif_cuda.so"


def make(c, lib, seed=7):
    lif_native._LIB = lib
    return lif_native.NativeLIFEngine.from_connectome(c, seed=seed, threads=int(os.environ.get("FLY_THREADS", "8")))


def scenario(c, name, e):
    n = c.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    orn = np.flatnonzero(np.char.startswith(t.astype(str), "ORN_"))
    loom = np.sort(c.by_cell_types(["LC4", "LPLC2"])["idx"].to_numpy())
    out = []
    if name == "published_looming":
        e.reset(seed=7); e.set_poisson(loom, 120.0)
        for _ in range(1000): out.append(e.run_collect(10))
    elif name in ("calibrated_odour", "dt02"):
        apply_dynamics(e, c, "calibrated")
        e.reset(seed=7)
        rate = np.where(np.isin(t[orn], ["ORN_DM1", "ORN_DM4", "ORN_VA2"]), 90.0, 12.0)
        e.set_poisson(orn, rate)
        for _ in range(1000): out.append(e.run_collect(int(round(1.0 / e.p.dt))))
    elif name == "silence_and_switch":
        apply_dynamics(e, c, "calibrated")
        e.reset(seed=7); e.set_poisson(loom, 120.0)
        for k in range(1200):
            if k == 300: e.silence(np.flatnonzero(t == "DNp01"))
            if k == 600: e.set_poisson(orn, 20.0)
            if k == 900: e.set_poisson(orn, np.linspace(0, 80, len(orn)))
            out.append(e.run_collect(10))
    elif name == "gain":
        e.set_gain(0.8); e.reset(seed=7); e.set_poisson(loom, 150.0)
        for _ in range(1000): out.append(e.run_collect(10))
    elif name == "pipelined":
        apply_dynamics(e, c, "calibrated")
        e.reset(seed=7); e.set_poisson(orn, 15.0)
        for _ in range(1000):
            e.start(10); out.append(e.wait())
    elif name == "orn_std":
        # v3 dynamics: ORN output depression, rate steps so depletion rises and recovers
        apply_dynamics(e, c, "calibrated_v3")
        e.reset(seed=7)
        for k in range(1500):
            if k == 0: e.set_poisson(orn, 8.0)
            if k == 500: e.set_poisson(orn, np.where(np.isin(t[orn], ["ORN_DM1", "ORN_VA2"]), 120.0, 8.0))
            if k == 1000: e.set_poisson(orn, 4.0)
            out.append(e.run_collect(10))
        out.append(e.std_depletion())
    elif name == "storm":
        # the whole brain driven hard at dt 0.2: more spikes per 1000-step chunk
        # than the old fixed 4M-spike device buffer held
        e.reset(seed=7); e.set_poisson(np.arange(len(n)), 150.0)
        for _ in range(3): out.append(e.run_collect(1000))
        # Poisson neurons have no refractory period: at 600 Hz one 1000-step
        # chunk would hold ~17M spikes, over the buffer; chunks must shorten
        e.set_poisson(np.arange(len(n)), 600.0)
        for _ in range(2): out.append(e.run_collect(1000))
    elif name == "host_edits":
        # state changed between runs: single steps, add_g (a neuron twice),
        # writes to v, reseeding, plasticity commits, depression switched off
        apply_dynamics(e, c, "calibrated_v3")
        e.reset(seed=7); e.set_poisson(orn, 15.0)
        rng = np.random.default_rng(3)
        pm = None
        for k in range(600):
            if k % 50 == 10:
                idx = rng.integers(0, len(n), 40)
                idx[1] = idx[0]
                e.add_g(idx, rng.uniform(-3, 3, 40).astype(np.float32))
            if k % 97 == 20:
                v = e.v
                sel = rng.integers(0, len(n), 200)
                v[sel] = v[sel] + np.float32(1.5)
                e.wake_all()
            if k == 150: e._lib.lif_set_seed(e._h, 11)
            if k == 200:
                pm = e.plastic_multipliers(); pm[::7] = 0.8; e.commit_plastic()
            if k in (250, 350):
                pos = rng.integers(0, len(pm), 1000); pm[pos] = rng.uniform(0.5, 1.5, 1000).astype(np.float32)
                e.commit_plastic(pos)
            if k == 400:
                e.set_std(0.0); out.append(np.array([e.std_depletion() is None]))
            out.append(e.step() if k % 3 == 0 else e.run_collect(10))
    elif name == "quiesce_tol":
        apply_dynamics(e, c, "calibrated"); e.set_quiesce_tolerance(1e-3)
        e.reset(seed=7); e.set_poisson(orn, 15.0)
        for _ in range(1000): out.append(e.run_collect(10))
    ad = e.adapt.copy() if e.adapt is not None else np.zeros(1)
    return out, e.v.copy(), e.g.copy(), ad, e.spike_counts.copy()


def main():
    names = sys.argv[1:] or ["published_looming", "calibrated_odour", "silence_and_switch", "gain",
                             "pipelined", "quiesce_tol", "orn_std", "host_edits", "dt02", "storm"]
    ok_all = True
    for name in names:
        if name in ("dt02", "storm"):
            os.environ["FLY_DT"] = "0.2"
        c = load_connectome()
        res = {}
        for tag, lib in (("cpu", CPU), ("gpu", GPU)):
            e = make(c, lib)
            t0 = time.time(); res[tag] = scenario(c, name, e); res[tag + "_s"] = time.time() - t0
            e.close()
        a, b = res["cpu"], res["gpu"]
        first = next((k for k, (x, y) in enumerate(zip(a[0], b[0])) if not np.array_equal(x, y)), None)
        same = first is None and all(np.array_equal(x, y) for x, y in zip(a[1:], b[1:]))
        ok_all &= same
        nspk = sum(len(x) for x in a[0])
        print(f"{name:20s} {'IDENTICAL' if same else 'DIFFERENT (first block %s)' % first}  spikes {nspk}"
              f"  cpu {res['cpu_s']:.1f}s gpu {res['gpu_s']:.1f}s", flush=True)
        os.environ.pop("FLY_DT", None)
    # lif_wait with no job started returns at once (it used to block forever)
    import threading
    e = make(load_connectome(), GPU)
    th = threading.Thread(target=e._lib.lif_wait, args=(e._h,), daemon=True)
    th.start(); th.join(10.0)
    print("wait without start  ", "returns" if not th.is_alive() else "HANGS")
    ok_all &= not th.is_alive()
    if not th.is_alive():
        e.close()
    print("ALL IDENTICAL" if ok_all else "MISMATCH")


if __name__ == "__main__":
    main()
