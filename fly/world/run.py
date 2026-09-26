"""
Run the closed-loop world headless (no browser), e.g. on the Mac, and record it.

    python -m fly.world.run --secs 600 --seeds 1 2 3 4 --threads 3 --out simulation/outputs/world_runs
    python -m fly.world.run --secs 300 --seeds 1 --brain off        # control: body rhythm only

Each run is saved as <out>/<name>_seed<k>.json.gz: the configuration, the
fly's path every 10 ms (t, x, y, z, heading, behaviour), world snapshots every
250 ms (fruit, energy, statistics), descending-neuron channels every 50 ms,
and every event. Several seeds run in parallel, each pinned to its own cores.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path


def run_one(args: dict) -> dict:
    os.environ.setdefault("FLY_DYNAMICS", "calibrated")
    os.environ["FLY_THREADS"] = str(args["threads"])
    os.environ["FLY_CPU_BASE"] = str(args["cpu_base"])
    from brain.neurons.registry import load_connectome
    from fly.world.world import WorldConfig
    from simulation.engine.session import Session

    c = load_connectome()
    ses = Session(c, seed=args["seed"])
    cfg = WorldConfig(**args.get("world", {}))
    ses.set_world(cfg, neural=args["neural"], senses=args.get("senses"), seed=args["seed"],
                  learning=args.get("learning", True))
    ses.reset(seed=args["seed"])
    rec = {"config": {**args, "world": vars(cfg)}, "track": [], "world": [], "channels": []}
    t0 = time.time()
    total_ms = args["secs"] * 1000.0
    while ses.engine.t_ms < total_ms:
        f = ses.advance(10.0)[-1]
        b = f["body"]
        rec["track"].append([round(f["t_ms"], 1), round(b["x_mm"], 2), round(b["y_mm"], 2),
                             round(b["z_mm"], 2), round(b["heading_deg"], 1), b["behaviour"]])
        tm = int(round(f["t_ms"]))
        if tm % 50 == 0:
            rec["channels"].append([tm, {k: round(v, 3) for k, v in f["channels"].items() if abs(v) > 0.01}])
        if tm % 250 == 0:
            w = f["world"]
            rec["world"].append([tm, {"fruits": w["fruits"], "stats": w["stats"], "energy": w["energy"]}])
    w = ses.world.state()
    rec["events"] = {"world": ses.world.events, "body": ses.body.events}
    rec["learning"] = ses.mb.summary() if ses.mb is not None else None
    rec["final"] = {"stats": w["stats"], "energy": w["energy"], "fruits": w["fruits"],
                    "wall_s": round(time.time() - t0, 1),
                    "realtime_factor": round(args["secs"] / (time.time() - t0), 2)}
    out = Path(args["out"]) / ("%s_seed%d.json.gz" % (args["name"], args["seed"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as fh:
        json.dump(rec, fh)
    return {"seed": args["seed"], "file": str(out), "learning": rec["learning"], **rec["final"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--secs", type=float, default=300.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--threads", type=int, default=3, help="engine threads per run")
    ap.add_argument("--brain", choices=("on", "off"), default="on")
    ap.add_argument("--learning", choices=("on", "off"), default="on")
    ap.add_argument("--senses", default="{}", help='JSON, e.g. {"smell": 0}')
    ap.add_argument("--world", default="{}", help='JSON WorldConfig overrides, e.g. {"n_fruit": 5}')
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="simulation/outputs/world_runs")
    a = ap.parse_args()
    name = a.name or ("brain_%s_learning_%s" % (a.brain, a.learning))
    ncpu = os.cpu_count() or 4
    jobs = [{"secs": a.secs, "seed": s, "threads": a.threads, "cpu_base": (i * a.threads) % ncpu,
             "neural": a.brain == "on", "learning": a.learning == "on",
             "senses": json.loads(a.senses),
             "world": json.loads(a.world), "name": name, "out": a.out}
            for i, s in enumerate(a.seeds)]
    workers = max(1, min(len(jobs), ncpu // a.threads))
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            st = r["stats"]
            print(f"seed {r['seed']}: {r['realtime_factor']}x real time | visits {st['fruit_visits']} "
                  f"ate sweet {st['eaten_sweet']:.3f} bitter {st['eaten_bitter']:.3f} "
                  f"first meal {st['first_food_ms']} | flown {st['airborne_ms'] / 1000:.0f}s "
                  f"| travelled {st['distance_mm'] / 1000:.1f} m | {r['file']}", flush=True)


if __name__ == "__main__":
    main()
