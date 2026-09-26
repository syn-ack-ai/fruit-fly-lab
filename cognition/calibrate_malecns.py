"""
Calibrate the MaleCNS synaptic gain against the published FlyWire model.

The LIF parameters of Shiu et al. (2024) were fitted on FlyWire FAFB. The
MaleCNS counts synapses independently (more per connection), so the same
w_syn may not give the same activity. Here each sensory modality is driven
identically in both flies (the same cell types, via brain/sensory/crossmap.py,
at the same Poisson rate) and the firing rate of every cell type present in
both connectomes is compared (FlyWire naming; directly driven types excluded).
The best gain makes the MaleCNS brain respond like the validated FAFB model.

Per gain and stimulus:  ratio  = total rate over shared types, MaleCNS / FAFB
                        r_log  = Pearson r of log(1 + rate) across shared types
Spot checks (published results): sugar -> MN9 (proboscis motor neuron) fires,
bitter -> MN9 does not, looming -> DNp01 (Giant Fibre) fires.

Run:  python -m cognition.calibrate_malecns --gains 0.4,0.5,0.6,0.7,0.8,1.0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STIMULI = ["taste_sugar", "taste_bitter", "looming", "odor_vinegar", "odor_co2", "wind",
           "sound", "touch_head", "humidity", "heat"]
RATE_HZ = 150.0
DURATION_MS = 500.0
SEEDS = (1, 2, 3)
CHECKS = {"taste_sugar": "MN9", "taste_bitter": "MN9", "looming": "DNp01"}


def _drive_idx(c, key, dataset):
    if dataset == "malecns":
        import config
        m = json.loads((config.DERIVED_DIR / "malecns" / "modality_map.json").read_text())
        return np.array(m[key]["idx"], dtype=np.int64)
    from brain.sensory.modalities import BY_KEY, resolve_neurons
    if key == "looming":
        return c.by_cell_types(["LC4", "LPLC2"])["idx"].to_numpy()
    return resolve_neurons(BY_KEY[key], c)


def _fw_types(c, dataset):
    n = c.neurons
    if dataset == "malecns":
        t = n["flywire_type"].fillna(n["primary_type"])
    else:
        t = n["primary_type"]
    return t.fillna("").astype(str).to_numpy()


def run_dataset(dataset: str, gains, threads: int) -> dict:
    """Per stimulus and gain: mean firing rate (Hz) per FlyWire-named type."""
    from brain.neurons.registry import load_connectome
    from native.lif_native import NativeLIFEngine
    c = load_connectome()
    types = _fw_types(c, dataset)
    e = NativeLIFEngine.from_connectome(c, seed=1, threads=threads)
    out = {}
    for g in gains:
        e.set_gain(g)
        for key in STIMULI:
            idx = np.sort(_drive_idx(c, key, dataset))
            counts = np.zeros(c.n)
            for s in SEEDS:
                e.reset(seed=s)
                e.set_poisson(idx, RATE_HZ)
                e.run(DURATION_MS)
                counts += e.spike_counts
            rate = counts / len(SEEDS) / (DURATION_MS / 1000.0)
            driven = set(types[idx])
            df = pd.DataFrame({"t": types, "r": rate})
            df = df[(df.t != "") & ~df.t.isin(driven)]
            out[f"{key}@{g:g}"] = df.groupby("t")["r"].mean().round(4).to_dict()
            print(f"[{dataset}] gain {g:g} {key:14s} driven {len(idx):4d}  mean rate "
                  f"{rate.mean():7.3f} Hz  active {int((rate > 0).sum())}", file=sys.stderr, flush=True)
    e.close()
    return out


def compare(fafb: dict, mc: dict, gains) -> pd.DataFrame:
    rows = []
    for g in gains:
        for key in STIMULI:
            a = pd.Series(fafb[f"{key}@1"]); b = pd.Series(mc[f"{key}@{g:g}"])
            shared = a.index.intersection(b.index)
            a, b = a[shared], b[shared]
            act = (a > 0) | (b > 0)
            r = float(np.corrcoef(np.log1p(a[act]), np.log1p(b[act]))[0, 1]) if act.sum() > 2 else np.nan
            row = {"gain": g, "stimulus": key, "shared_types": int(len(shared)),
                   "ratio": float(b.sum() / max(a.sum(), 1e-9)), "r_log": r}
            if key in CHECKS:
                t = CHECKS[key]
                row["check"] = f"{t} {a.get(t, np.nan):.0f} vs {b.get(t, np.nan):.0f} Hz"
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gains", default="0.4,0.5,0.6,0.7,0.8,1.0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="calibration_malecns.json")
    ap.add_argument("--dataset", default=None, help=argparse.SUPPRESS)   # worker mode
    a = ap.parse_args()
    gains = [float(x) for x in a.gains.split(",")]
    if a.dataset:                                       # one dataset per process
        json.dump(run_dataset(a.dataset, gains, a.threads), sys.stdout)
        return
    res = {}
    for ds, gs in (("fafb", "1"), ("malecns", a.gains)):
        env = {**os.environ, "FLY_DATASET": ds}
        p = subprocess.run([sys.executable, "-m", "cognition.calibrate_malecns", "--dataset", ds,
                            "--gains", gs, "--threads", str(a.threads)],
                           env=env, check=True, capture_output=True, text=True)
        sys.stderr.write(p.stderr)
        res[ds] = json.loads(p.stdout.splitlines()[-1])
    Path(a.out).write_text(json.dumps(res))
    df = compare(res["fafb"], res["malecns"], gains)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    summ = df.groupby("gain").agg(ratio_median=("ratio", "median"), r_log_mean=("r_log", "mean"))
    print("\n" + summ.to_string(float_format=lambda x: f"{x:.3f}"))
    best = summ.assign(score=summ.r_log_mean - (np.log(summ.ratio_median).abs() * 0.5)).score.idxmax()
    print(f"\nbest gain: {best:g}  (highest type-rate correlation with activity ratio near 1)")


if __name__ == "__main__":
    main()
