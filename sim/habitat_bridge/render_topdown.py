"""
Top-down videos of the pet's days in the Habitat house (drawn from the logs).

    ~/miniforge3/envs/habitat/bin/python -m sim.habitat_bridge.render_topdown \
        simulation/outputs/habitat/home/topdown.npz simulation/outputs/habitat/home/on/brain_day0_ep0.json out.mp4

Grey = floor the robot can reach, dark = walls/furniture; orange = food bowl,
green = bitter plant, blue = the person, red = the pet (triangle = heading),
thin lines = where they have been. Pats, treats and eating are written on the
frame when they happen.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

SCALE = 2          # pixels per map cell


def render(topdown_npz: str, log_json: str, out_mp4: str, fps: float = 10.0, every: int = 2) -> str:
    import cv2
    import imageio
    from sim.habitat_bridge.home import BOWL_XZ, EAT_R_M, PLANT_XZ
    td = np.load(topdown_npz)
    m, xmin, zmin, mpp = td["map"], float(td["xmin"]), float(td["zmin"]), float(td["mpp"])
    base = np.full(m.shape + (3,), 40, np.uint8)
    base[m == 1] = (200, 200, 200)
    base[m == 2] = (90, 90, 90)
    base = cv2.resize(base, (m.shape[1] * SCALE, m.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST)

    def px(x, z):
        return int(round((x - xmin) / mpp * SCALE)), int(round((z - zmin) / mpp * SCALE))

    rad = int(EAT_R_M / mpp * SCALE)
    cv2.circle(base, px(*BOWL_XZ), rad, (255, 150, 30), 2)
    cv2.circle(base, px(*BOWL_XZ), 5, (255, 150, 30), -1)
    cv2.circle(base, px(*PLANT_XZ), rad, (60, 170, 60), 2)
    cv2.circle(base, px(*PLANT_XZ), 5, (60, 170, 60), -1)
    d = json.load(open(log_json))
    log = d["log"]
    frames, rtrail, htrail = [], [], []
    for k, r in enumerate(log):
        rtrail.append(px(r["robot"][0], r["robot"][1]))
        htrail.append(px(r["human"][0], r["human"][1]))
        if k % every:
            continue
        img = base.copy()
        if len(htrail) > 1:
            cv2.polylines(img, [np.array(htrail, np.int32)], False, (120, 140, 255), 1)
        if len(rtrail) > 1:
            cv2.polylines(img, [np.array(rtrail, np.int32)], False, (255, 90, 90), 1)
        cv2.circle(img, htrail[-1], 7, (40, 60, 230), -1)
        x, z, yaw = r["robot"]
        a = math.radians(yaw)
        f = np.array([math.cos(a), -math.sin(a)])          # heading in (x, z)
        l = np.array([-math.sin(a), -math.cos(a)])
        c = np.array([x, z])
        tri = [c + 0.35 * f, c - 0.2 * f + 0.18 * l, c - 0.2 * f - 0.18 * l]
        cv2.fillPoly(img, [np.array([px(*p) for p in tri], np.int32)], (230, 40, 40))
        txt = ["t = %4.1f s" % r["t"], r.get("behaviour", "")[:28]]
        if r.get("taste"):
            txt.append("tasting %s" % r["taste"])
        if r.get("pet"):
            txt.append("being petted")
        if r.get("treat"):
            txt.append("treat!")
        if r.get("proboscis", 0) > 0.5:
            txt.append("eating")
        for i, s in enumerate(txt):
            cv2.putText(img, s, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        frames.append(img)
    imageio.mimsave(out_mp4, frames, fps=fps)
    return out_mp4


if __name__ == "__main__":
    print(render(sys.argv[1], sys.argv[2], sys.argv[3]))
