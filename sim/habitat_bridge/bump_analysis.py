"""Who bumped whom? For every pet-person contact in a lifetime run, compare how
fast the pet was closing on the person (its speed toward them) with how fast
the person was closing on the pet, in the 0.1 s step before contact (ground
truth from the Habitat log; evaluation only). The pet is counted as the one
that bumped only if it was moving toward the person (> 0.02 m/s) at least as
fast as the person was moving toward it.

    python -m sim.habitat_bridge.bump_analysis simulation/outputs/habitat/cortex_life_manners
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict


def classify(log, t):
    i = min(range(len(log)), key=lambda k: abs(log[k]["t"] - t))
    a, b = log[max(0, i - 1)], log[i]
    dt = max(b["t"] - a["t"], 1e-3)
    rx, rz = a["robot"][0] - a["human"][0], a["robot"][1] - a["human"][1]
    d = math.hypot(rx, rz) or 1e-6
    ux, uz = rx / d, rz / d                                   # person -> pet
    hvx, hvz = (b["human"][0] - a["human"][0]) / dt, (b["human"][1] - a["human"][1]) / dt
    person_closing = hvx * ux + hvz * uz
    yaw = math.radians(a["robot"][2])                         # CCW from +x in the (x, -z) plane
    pvx, pvz = a["v"] * math.cos(yaw), -a["v"] * math.sin(yaw)
    pet_closing = -(pvx * ux + pvz * uz)
    return {"pet_closing": pet_closing, "person_closing": person_closing,
            "pet_fault": pet_closing > 0.02 and pet_closing >= person_closing, "behind": abs(a["az"]) > 90,
            "manner": (a.get("cortex") or {}).get("manner")}


def main(root):
    tot = defaultdict(lambda: defaultdict(int))
    for f in sorted(glob.glob(os.path.join(root, "**", "brain_day*_ep*.json"), recursive=True)):
        cond = os.path.basename(os.path.dirname(f))
        r = json.load(open(f))
        for (t, _hs, _v, wanted) in r.get("bumps", {}).get("list", []):
            c = classify(r["log"], t)
            k = tot[cond]
            k["bumps"] += 1
            if c["pet_fault"]:
                k["pet ran into person"] += 1
                k["  ...at > 0.15 m/s"] += c["pet_closing"] > 0.15
                k["  ...when it did not want company"] += wanted is False
            else:
                k["person walked into pet"] += 1
                k["  ...from behind the pet"] += c["behind"]
    for cond, k in sorted(tot.items()):
        print(cond, dict(k))


if __name__ == "__main__":
    main(sys.argv[1])
