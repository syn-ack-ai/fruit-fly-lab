"""
Activity-matched control: does the real wiring hold information longer than a
scrambled one *at the same level of activity*?

For each wiring and each recurrent synaptic gain (sensory drive unchanged),
generate trials (cognition/datagen.py), measure how active the brain is, and
how well the stimulus can be decoded after it ends (cognition/decode.py).
Comparing memory against activity across wirings separates "the real wiring
is special" from "the real wiring is merely more excitable".

Activity measures, per trial, averaged over all non-empty classes:
  evoked   spikes during the 100 ms stimulus (driven sensory neurons excluded)
  after    spikes 50-500 ms after the stimulus ended (driven neurons excluded)
  latched  fraction of trials still spiking (>1000 spikes) 50-500 ms after

  python -m cognition.sweep --out ~/fly-lab/sweep --trials 1600 \\
      --conditions real:0.8,1.0 shuffled:1.0,2.0,3.0 random:1.0,2.0,3.0
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def activity(dataset: Path) -> dict:
    meta = json.loads((dataset / "meta.json").read_text())
    driven = np.load(dataset / "driven.npz")
    drv = np.zeros(meta["n_neurons"], bool)
    for k in driven.files:
        drv[driven[k]] = True
    ev, af = [], []
    for f in sorted(glob.glob(str(dataset / "shard_*.npz"))):
        with np.load(f) as z:
            ptr, cls, off, bins, neu, cnt = (z[k] for k in ("ptr", "cls", "off_bin", "bins", "neurons", "counts"))
        keep = ~drv[neu]
        for k in range(len(cls)):
            if meta["classes"][cls[k]] == "none":
                continue
            a, b = ptr[k], ptr[k + 1]
            bb, cc, kk = bins[a:b], cnt[a:b].astype(np.int64), keep[a:b]
            ev.append(cc[kk & (bb < off[k])].sum())
            af.append(cc[kk & (bb >= off[k] + 5)].sum())
    ev, af = np.array(ev), np.array(af)
    return {"evoked": float(ev.mean()), "after": float(af.mean()),
            "latched": float((af > 1000).mean()), "trials": int(len(ev))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=1600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="wiring:gain,gain,...  e.g. shuffled:1,2,3")
    ap.add_argument("--probe", action="store_true", help="activity only, no decoding")
    a = ap.parse_args()
    root = Path(a.out).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / ("probe.json" if a.probe else "summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for cond in a.conditions:
        wiring, gains = cond.split(":")
        for g in [float(x) for x in gains.split(",")]:
            key = f"{wiring}@{g:g}"
            ds = root / f"{wiring}_g{g:g}_n{a.trials}"
            if not (ds / "shard_000.npz").exists():
                t0 = time.time()
                subprocess.run([sys.executable, "-m", "cognition.datagen", "--out", str(ds),
                                "--trials", str(a.trials), "--workers", str(a.workers),
                                "--threads", str(a.threads), "--seed", "1",
                                "--wiring", wiring, "--gain", str(g)],
                               check=True, stdout=subprocess.DEVNULL)
                gen_s = time.time() - t0
            else:
                gen_s = None
            act = activity(ds)
            row = {"wiring": wiring, "gain": g, **act, "generate_s": gen_s}
            if not a.probe:
                res_path = ds / "decode.json"
                subprocess.run([sys.executable, "-m", "cognition.decode", str(ds),
                                "--out", str(res_path)], check=True, stdout=subprocess.DEVNULL)
                res = json.loads(res_path.read_text())
                r = next(iter(res.values()))
                for mode in ("brain", "input"):
                    accs = {w["start_ms"]: w["acc"] for w in r["modes"][mode]}
                    row[f"{mode}_acc"] = accs
                    row[f"{mode}_memory"] = float(np.mean([accs[t] for t in (100, 150, 200, 300, 400)]))
            summary[key] = row
            summary_path.write_text(json.dumps(summary, indent=1))
            line = (f"{key:14s} evoked {act['evoked']:9.0f}  after {act['after']:9.0f}  "
                    f"latched {act['latched'] * 100:5.1f}%")
            if not a.probe:
                line += (f"  | memory (mean acc 100-450 ms after): brain {row['brain_memory'] * 100:5.1f}%"
                         f"  input {row['input_memory'] * 100:5.1f}%")
            print(line, flush=True)


if __name__ == "__main__":
    main()
