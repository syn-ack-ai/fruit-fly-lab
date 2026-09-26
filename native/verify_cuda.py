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
    elif name == "quiesce_tol":
        apply_dynamics(e, c, "calibrated"); e.set_quiesce_tolerance(1e-3)
        e.reset(seed=7); e.set_poisson(orn, 15.0)
        for _ in range(1000): out.append(e.run_collect(10))
    ad = e.adapt.copy() if e.adapt is not None else np.zeros(1)
    return out, e.v.copy(), e.g.copy(), ad, e.spike_counts.copy()


def main():
    names = sys.argv[1:] or ["published_looming", "calibrated_odour", "silence_and_switch", "gain",
                             "pipelined", "quiesce_tol", "orn_std", "dt02"]
    ok_all = True
    for name in names:
        if name == "dt02":
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
    print("ALL IDENTICAL" if ok_all else "MISMATCH")


if __name__ == "__main__":
    main()
