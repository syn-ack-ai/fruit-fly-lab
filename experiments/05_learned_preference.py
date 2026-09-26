"""
Experiment 05 - does learned odour valence change where the fly goes?

Classical conditioning (as in fly labs): odour A is presented with the reward
dopamine neurons (PAM05-08, 100 Hz), odour B with the punishment ones
(PPL101/PPL103, 100 Hz) -- or the reciprocal, or no training. Then the fly is
released midway between a fruit smelling of A and one smelling of B, in still
air, with taste and learning switched off, and we measure where it spends its
time: preference index PI = (t_A - t_B) / (t_A + t_B) within 40 mm of each.

Navigation ON = the central-complex goal gated by learned valence
(brain/navigation/valence_goal.py); OFF = the same fly without that coupling.

    python -m experiments.05_learned_preference --seeds 1 2 3 4 5 6 --secs 60
"""
from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ODOUR_A, ODOUR_B = "ripe", "citrus"          # fruit kinds (fly/world/world.py)
SEP_MM = 60.0
NEAR_MM = 40.0
TRIALS = 4
TRIAL_MS = 500


def run_one(args: dict) -> dict:
    os.environ.setdefault("FLY_DYNAMICS", "calibrated")
    os.environ["FLY_THREADS"] = str(args["threads"])
    os.environ["FLY_CPU_BASE"] = str(args["cpu_base"])
    from brain.neurons.registry import load_connectome
    from fly.world.world import FRUIT_KINDS, Fruit, WorldConfig
    from simulation.engine.session import Session

    c = load_connectome()
    t = c.neurons["primary_type"].fillna("").astype(str).to_numpy()
    ses = Session(c, seed=args["seed"])
    cfg = WorldConfig(n_fruit=0, wind_speed_mm_s=0.0, predators=False)
    ses.set_world(cfg, seed=args["seed"], learning=True, navigation=args["nav"],
                  senses={"taste": False})
    ses.reset(seed=args["seed"])
    ses.body.spont_takeoff = 0.0
    w = ses.world
    w.fruits = [Fruit(-SEP_MM, 0.0, kind=ODOUR_A), Fruit(SEP_MM, 0.0, kind=ODOUR_B)]
    ses.world_senses._B = None                          # fruit blends changed
    mb, e = ses.mb, ses.engine
    osp = ses.world_senses.olf
    orn_types = sorted({x for x in t if x.startswith("ORN_")})
    gi = {ty: i for i, ty in enumerate(orn_types)}
    orn = np.flatnonzero(np.char.startswith(t.astype(str), "ORN_"))

    def orn_rates(kind):
        r = osp.rates(osp.sources[FRUIT_KINDS[kind]["source"]] * 0.5)
        return np.array([r[gi[t[i]]] for i in orn])

    reward = np.flatnonzero(np.isin(t, ["PAM05", "PAM06", "PAM07", "PAM08"]))
    punish = np.flatnonzero(np.isin(t, ["PPL101", "PPL103"]))

    def pair(kind, dans):
        idx = np.concatenate([orn, dans]); r = np.concatenate([orn_rates(kind), np.full(len(dans), 100.0)])
        o = np.argsort(idx)
        e.reset(seed=args["seed"] + 1000); mb.reset_activity(); mb.learning = True
        e.set_poisson(idx[o], r[o])
        steps = int(round(1.0 / ses.p.dt))
        for _ in range(TRIAL_MS):
            mb.step(e.run_collect(steps))

    plus, minus = {"naive": (None, None), "A+B-": (ODOUR_A, ODOUR_B),
                   "A-B+": (ODOUR_B, ODOUR_A)}[args["train"]]
    for _ in range(TRIALS if plus else 0):
        pair(plus, reward)
        pair(minus, punish)
    learned = int((mb.weights < 0.9).sum())

    # test: fresh start between the fruits, no taste, no learning
    ses.reset(seed=args["seed"])
    ses.body.spont_takeoff = 0.0
    w.fruits = [Fruit(-SEP_MM, 0.0, kind=ODOUR_A), Fruit(SEP_MM, 0.0, kind=ODOUR_B)]
    ses.world_senses._B = None
    rng = np.random.default_rng(args["seed"])
    ses.body.state.heading_deg = float(rng.uniform(0, 360))
    mb.learning = False
    tA = tB = 0.0; first = None; vals = []; on = []
    steps_ms = 10.0
    while ses.engine.t_ms < args["secs"] * 1000.0:
        f = ses.advance(steps_ms)[-1]
        b = f["body"]
        dA = math.hypot(b["x_mm"] + SEP_MM, b["y_mm"]); dB = math.hypot(b["x_mm"] - SEP_MM, b["y_mm"])
        if dA < NEAR_MM: tA += steps_ms / 1000.0
        if dB < NEAR_MM: tB += steps_ms / 1000.0
        if first is None and min(dA, dB) < 15.0:
            first = "A" if dA < dB else "B"
        if f.get("navigation"):
            vals.append(f["navigation"]["valence"]); on.append(f["navigation"]["goal_on"])
    pi = (tA - tB) / (tA + tB) if tA + tB > 0 else 0.0
    return {**{k: args[k] for k in ("seed", "train", "nav")}, "tA": round(tA, 1), "tB": round(tB, 1),
            "PI": round(pi, 3), "first": first, "learned_synapses": learned,
            "valence_mean": round(float(np.mean(vals)), 4) if vals else None,
            "goal_on_frac": round(float(np.mean(on)), 3) if on else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="simulation/outputs/experiment_05_learned_preference.json")
    a = ap.parse_args()
    ncpu = os.cpu_count() or 4
    jobs = []
    for nav in (True, False):
        for train in ("naive", "A+B-", "A-B+"):
            if not nav and train == "naive":
                continue
            for s in a.seeds:
                jobs.append({"seed": s, "train": train, "nav": nav, "secs": a.secs, "threads": a.threads})
    for i, j in enumerate(jobs):
        j["cpu_base"] = (i * a.threads) % ncpu
    res = []
    with get_context("spawn").Pool(max(1, ncpu // a.threads)) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            res.append(r); print(r, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("\ncondition        n   mean PI   first A/B   mean valence")
    for nav in (True, False):
        for train in ("naive", "A+B-", "A-B+"):
            rs = [r for r in res if r["nav"] == nav and r["train"] == train]
            if not rs:
                continue
            pis = [r["PI"] for r in rs]
            fa = sum(r["first"] == "A" for r in rs); fb = sum(r["first"] == "B" for r in rs)
            vv = [r["valence_mean"] for r in rs if r["valence_mean"] is not None]
            print(f"nav={'on ' if nav else 'off'} {train:6s} {len(rs):3d}  {np.mean(pis):+.2f} +- {np.std(pis):.2f}   {fa}/{fb}"
                  f"   {np.mean(vv) if vv else float('nan'):+.3f}")


if __name__ == "__main__":
    main()
