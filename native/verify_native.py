"""
Verify the native engine is bit-identical to the browser engine (web/js/engine.js),
which tools/verify_web_engine.py in turn verifies against the Python engine.

Compares, for every one of the 139,255 neurons: final membrane potential v,
final synaptic drive g (bitwise, as float32), total spike count, and the
number of spikes on every single timestep. Runs the native engine with 1 and 4
threads.

Run:  python -m native.verify_native
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from native.lif_native import NativeLIFEngine, load_web_connectome, ROOT


def scenarios(meta):
    drive = meta["lesionable"]["LC4"] + meta["lesionable"]["LPLC2"]
    return [
        ("deterministic propagation", {
            "seed": 3, "duration_ms": 120.0,
            "inject_g": [[int(i), 1600.0] for i in drive[:60]],
            "poisson_idx": [], "poisson_rate": 0.0, "silence": []}),
        ("poisson looming drive, 150 Hz", {
            "seed": 7, "duration_ms": 300.0,
            "poisson_idx": drive, "poisson_rate": 150.0, "silence": []}),
        ("looming drive with LC4+LPLC2 silenced", {
            "seed": 7, "duration_ms": 120.0,
            "poisson_idx": drive, "poisson_rate": 150.0, "silence": drive}),
        ("heavy drive, 400 Hz", {
            "seed": 11, "duration_ms": 300.0,
            "poisson_idx": drive, "poisson_rate": 400.0, "silence": []}),
    ]


def run_native(conn, sc, threads):
    e = NativeLIFEngine(*conn, seed=sc["seed"], threads=threads)
    if sc["silence"]:
        e.silence(sc["silence"])
    for i, val in sc.get("inject_g", []):
        e.g[i] = val
    e.wake_all()
    if sc["poisson_idx"]:
        e.set_poisson(sc["poisson_idx"], sc["poisson_rate"])
    steps = int(round(sc["duration_ms"] / 0.1))
    per = np.empty(steps, np.int32)
    t0 = time.perf_counter()
    for s in range(steps):
        per[s] = e._lib.lif_step(e._h)
    wall = time.perf_counter() - t0
    out = (e.v.copy(), e.g.copy(), e.spike_counts.copy(), per)
    e.close()
    return out, wall


def run_js(sc, n):
    with tempfile.TemporaryDirectory() as td:
        sp, op = Path(td) / "s.json", Path(td) / "o.bin"
        sp.write_text(json.dumps(sc))
        subprocess.run(["node", str(ROOT / "native" / "dump_js.mjs"), str(sp), str(op)], check=True)
        b = op.read_bytes()
    v = np.frombuffer(b, np.float32, n, 0)
    g = np.frombuffer(b, np.float32, n, 4 * n)
    c = np.frombuffer(b, np.int32, n, 8 * n)
    per = np.frombuffer(b, np.int32, offset=12 * n)
    return v, g, c, per


def main():
    meta, *arrs = load_web_connectome()
    conn = tuple(arrs)
    n = meta["n"]
    ok = True
    for name, sc in scenarios(meta):
        print(f"\n== {name} ({sc['duration_ms']:.0f} ms)")
        js = run_js(sc, n)
        for threads in (1, 2, 3, 4):
            nat, wall = run_native(conn, sc, threads)
            same = [np.array_equal(a.view(np.uint32) if a.dtype == np.float32 else a,
                                   b.view(np.uint32) if b.dtype == np.float32 else b)
                    for a, b in zip(nat, js)]
            steps = len(nat[3])
            status = "MATCH" if all(same) else "MISMATCH " + str(dict(zip("v g counts per_step".split(), same)))
            ok &= all(same)
            print(f"  native x{threads}: {status}  | spikes={int(nat[2].sum())} active={int((nat[2] > 0).sum())}"
                  f" | {wall * 1e6 / steps:.1f} us/step = {sc['duration_ms'] / (wall * 1e3):.2f}x real-time speed")
    print("\nALL MATCH" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
