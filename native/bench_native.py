"""
Real-time benchmark of the native engine on the whole 139,255-neuron brain.
Times lif_run() entirely in C, so no Python per-step overhead is included.

Run:  python -m native.bench_native
"""
import time

import numpy as np

from native.lif_native import NativeLIFEngine, load_web_connectome


def main():
    meta, *conn = load_web_connectome()
    drive = meta["lesionable"]["LC4"] + meta["lesionable"]["LPLC2"]
    cases = [("silent brain", 0, 300), ("looming 150 Hz", 150, 300),
             ("looming 400 Hz", 400, 300), ("looming 150 Hz, 2 s", 150, 2000)]
    print(f"{'case':22s} {'thr':>3s} {'us/step':>8s} {'x realtime':>10s} {'spikes':>8s} {'active blk':>10s}")
    for threads in (1, 2, 3, 4):
        e = NativeLIFEngine(*conn, seed=7, threads=threads)
        for name, rate, ms in cases:
            e.reset(seed=7)
            if rate:
                e.set_poisson(drive, rate)
            t0 = time.perf_counter()
            spikes = e.run(ms)
            wall = time.perf_counter() - t0
            steps = e.step_count
            print(f"{name:22s} {threads:3d} {wall * 1e6 / steps:8.1f} {ms / 1e3 / wall:10.2f} "
                  f"{spikes:8d} {e.active_fraction:10.1%}  edges/step={e._lib.lif_edges(e._h) / steps:7.0f}  {e.phase_times_us()}")
        e.close()
    print("\n'x realtime' > 1 means faster than real time (0.1 ms step in < 100 us).")


if __name__ == "__main__":
    main()
