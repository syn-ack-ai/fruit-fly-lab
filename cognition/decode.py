"""
How long does the brain hold on to what happened? And does the real wiring
hold it better than shuffled or random wiring?

For every trial (cognition/datagen.py) and every 50 ms window from the last
half of the stimulus to 450 ms after it ended, a decoder is asked which of the
16 stimulus classes occurred, from the spikes in that window alone.

Decoder: a sparse linear readout over every neuron (multinomial logistic
regression on log(1 + spike count)), the standard readout for a recurrent
network. It is trained on the GPU as an EmbeddingBag, so its cost scales with
the number of spikes, not with the 139,255 neurons.

Modes:
  brain   every neuron EXCEPT those any stimulus drives directly: what the rest
          of the brain carries (the decoder cannot just read the sensors)
  input   ONLY the directly driven sensory neurons: the "no brain" baseline

Train/test split is by trial number (every 5th trial is test), so it is the
same trials, with the same stimuli, in every wiring. Chance is 1/16.

Run:  python -m cognition.decode ~/fly-lab/data/real ~/fly-lab/data/shuffled ...
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

WINDOWS_MS = [-50, 0, 50, 100, 150, 200, 300, 400]   # window start, relative to offset
WIDTH_MS = 50


def load(dataset: Path):
    meta = json.loads((dataset / "meta.json").read_text())
    driven = np.load(dataset / "driven.npz")
    drv = np.zeros(meta["n_neurons"], bool)
    for k in driven.files:
        drv[driven[k]] = True
    trials = []
    for f in sorted(glob.glob(str(dataset / "shard_*.npz"))):
        with np.load(f) as z:           # read each array once; trials are views into it
            arr = {k: z[k] for k in ("ptr", "seed", "cls", "off_bin", "bins", "neurons", "counts")}
        ptr = arr["ptr"]
        for k in range(len(arr["cls"])):
            a, b = ptr[k], ptr[k + 1]
            trials.append({"id": int(arr["seed"][k]), "cls": int(arr["cls"][k]),
                           "off": int(arr["off_bin"][k]), "bins": arr["bins"][a:b],
                           "neurons": arr["neurons"][a:b], "counts": arr["counts"][a:b]})
    trials.sort(key=lambda t: t["id"])
    return meta, drv, trials


def window_bags(trials, drv, start_ms, bin_ms, mode):
    """Per trial: (neuron ids, log1p counts) summed over the window."""
    keep = ~drv if mode == "brain" else drv
    w0, w1 = start_ms // bin_ms, (start_ms + WIDTH_MS) // bin_ms
    idx, wts, offsets = [], [], [0]
    for t in trials:
        sel = (t["bins"] >= t["off"] + w0) & (t["bins"] < t["off"] + w1)
        n, c = t["neurons"][sel], t["counts"][sel].astype(np.int64)
        m = keep[n]
        n, c = n[m], c[m]
        if n.size:
            u, inv = np.unique(n, return_inverse=True)
            cnt = np.bincount(inv, weights=c)
            idx.append(u); wts.append(np.log1p(cnt))
            offsets.append(offsets[-1] + len(u))
        else:
            offsets.append(offsets[-1])
    idx = np.concatenate(idx) if idx else np.empty(0, np.int64)
    wts = np.concatenate(wts) if wts else np.empty(0)
    return idx, wts, np.array(offsets[:-1])


def fit_eval(bags, y, train, test, n_neurons, n_classes, dev, epochs=300, wd=1e-4):
    idx, wts, off = bags
    lens = np.diff(np.append(off, len(idx)))

    def subset(sel):
        o, i, w = [0], [], []
        for k in np.flatnonzero(sel):
            a, b = off[k], off[k] + lens[k]
            i.append(idx[a:b]); w.append(wts[a:b]); o.append(o[-1] + (b - a))
        cat = lambda xs, dt: np.concatenate(xs) if xs else np.empty(0, dt)
        return (torch.tensor(cat(i, np.int64), device=dev),
                torch.tensor(cat(w, np.float32), dtype=torch.float32, device=dev),
                torch.tensor(o[:-1], dtype=torch.int64, device=dev),
                torch.tensor(y[sel], device=dev))

    tr, te = subset(train), subset(test)
    torch.manual_seed(0)
    emb = torch.nn.EmbeddingBag(n_neurons, n_classes, mode="sum").to(dev)
    torch.nn.init.zeros_(emb.weight)
    bias = torch.zeros(n_classes, device=dev, requires_grad=True)
    opt = torch.optim.AdamW(list(emb.parameters()) + [bias], lr=0.02, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        logits = emb(tr[0], tr[2], per_sample_weights=tr[1]) + bias
        loss = torch.nn.functional.cross_entropy(logits, tr[3])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (emb(te[0], te[2], per_sample_weights=te[1]) + bias).argmax(1)
        acc = (pred == te[3]).float().mean().item()
        per_class = [((pred == te[3]) & (te[3] == c)).sum().item() / max(1, (te[3] == c).sum().item())
                     for c in range(n_classes)]
    return acc, per_class, pred.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    results = {}
    for ds in a.datasets:
        ds = Path(ds).expanduser()
        t0 = time.time()
        meta, drv, trials = load(ds)
        y = np.array([t["cls"] for t in trials])
        ids = np.array([t["id"] for t in trials])
        test = ids % 5 == 0
        train = ~test
        C = meta["classes"]
        name = meta["wiring"] + ("" if meta.get("gain", 1.0) == 1.0 else "@%g" % meta["gain"])
        results[name] = {"classes": C, "n_trials": len(trials), "windows_ms": WINDOWS_MS, "modes": {}}
        for mode in ("brain", "input"):
            row = []
            for w in WINDOWS_MS:
                bags = window_bags(trials, drv, w, meta["bin_ms"], mode)
                acc, per_class, _ = fit_eval(bags, y, train, test, meta["n_neurons"], len(C), dev)
                row.append({"start_ms": w, "acc": acc, "per_class": per_class})
            results[name]["modes"][mode] = row
            print(f"{name:9s} {mode:5s} " + " ".join(
                f"{r['start_ms']:+4d}:{r['acc'] * 100:5.1f}%" for r in row), flush=True)
        print(f"  ({len(trials)} trials, {time.time() - t0:.0f} s)", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"chance = {100 / len(C):.1f}%   windows are {WIDTH_MS} ms, start relative to stimulus offset")


if __name__ == "__main__":
    main()
