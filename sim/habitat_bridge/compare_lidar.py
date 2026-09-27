"""
Compare battery-pet lifetimes (run_cortex_life.sh: pet, petlidar, petobst,
petsense, petttc ...) on bumps, walking and the pet's own life.

    .venv/bin/python -m sim.habitat_bridge.compare_lidar simulation/outputs/habitat/lidar_v2

Per condition, over all seeds x days:
  person bumps     contacts with the person; "pet-caused" = the pet was closing
                   faster than the person (bump_analysis.classify)
  scene            Spot body: Habitat's contact count (every contact point, every
                   1/120 s step: a frame count, not separate bumps); rover body:
                   contact steps at 10 Hz and separate bumps (habitat_server)
  blocked          rover only: seconds the navmesh stopped the base (pressing on something)
  brake            s/day the lidar safety layer cut the brain's speed by > 0.05 m/s
                   (an organic pet should rarely need it)
  pivot            s/day the body turned in place because it was pinned (robot/avoid.py)
  walked           path length (m/day)
  dock / meals     days it reached its dock; meals (robot/battery.py)
  rest             s/day napping (the neocortex's rest time, cortex/v0.py)
  low, flat        s/day below 20%; days that ended flat (rescued next morning)
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from sim.habitat_bridge.bump_analysis import classify


def summarise(root: str) -> dict:
    per = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(root, "**", "brain_day*_ep*.json"), recursive=True)):
        cond = os.path.basename(os.path.dirname(f))
        r = json.load(open(f))
        k = per[cond]
        log = r["log"]
        pet_fault = sum(classify(log, t)["pet_fault"] for (t, *_rest) in r.get("bumps", {}).get("list", []))
        xy = np.array([p["robot"][:2] for p in log]) if log else np.zeros((1, 2))
        h = r["home"]
        b = h.get("battery") or {}
        k["person_bumps"].append(r["collisions"])
        k["pet_caused"].append(pet_fault)
        k["scene"].append(r.get("scene_contacts") or 0)
        k["scene_bumps"].append(r.get("scene_bumps"))
        k["blocked_s"].append(r.get("blocked_s"))
        k["walked_m"].append(float(np.hypot(*np.diff(xy, axis=0).T).sum()))
        k["dock"].append(h["first_bowl_s"] is not None)
        k["meals"].append(b.get("meals", 0))
        k["rest_s"].append((r.get("cortex") or {}).get("rest_s", 0.0))
        k["low_s"].append(b.get("low_s", 0.0))
        k["flat"].append(bool(b.get("flat_s", 0.0) > 0))
        k["pivot_s"].append(0.1 * sum(bool(((p.get("lidar") or {}).get("avoid") or {}).get("pivot")) for p in log))
        k["brake_s"].append(0.1 * sum(((p.get("lidar") or {}).get("brake") or 0) > 0.05 for p in log))
        k["recoveries"].append((log[-1].get("lidar") or {}).get("recoveries") if log else None)
    return per


def main(root: str) -> None:
    per = summarise(root)
    print(f"{'condition':10s} {'days':>4s} {'bumps':>5s} {'pet':>4s} {'scene/d':>8s} {'sbumps/d':>8s} "
          f"{'blk s/d':>7s} {'brk s/d':>7s} {'piv s/d':>7s} {'m/d':>5s} {'dock':>6s} {'meals':>5s} {'rest/d':>6s} {'low/d':>6s} {'flat':>4s} {'recov':>5s}")
    for cond, k in sorted(per.items()):
        n = len(k["walked_m"])
        sb = [x for x in k["scene_bumps"] if x is not None]
        bl = [x for x in k["blocked_s"] if x is not None]
        rc = [x for x in k["recoveries"] if x is not None]
        print(f"{cond:10s} {n:4d} {sum(k['person_bumps']):5d} {sum(k['pet_caused']):4d} "
              f"{np.mean(k['scene']):8.0f} {(np.mean(sb) if sb else math.nan):8.1f} "
              f"{(np.mean(bl) if bl else math.nan):7.1f} {np.mean(k['brake_s']):7.1f} {np.mean(k['pivot_s']):7.1f} {np.mean(k['walked_m']):5.1f} "
              f"{sum(k['dock']):3d}/{n:<2d} {sum(k['meals']):5d} {np.mean(k['rest_s']):6.1f} "
              f"{np.mean(k['low_s']):6.1f} {sum(k['flat']):4d} {(sum(rc) if rc else '-')!s:>5s}")


if __name__ == "__main__":
    main(sys.argv[1])
