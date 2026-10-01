"""
Option A (robot/retina.py: the camera drives the connectome's photoreceptors)
against option B (robot/flyvis_eye.py: the published flyvis eye model computes
the optic lobe, its outputs drive the connectome) on the same stimuli and
read-outs. Both get the visual field as a panorama (azimuth -180..180,
elevation 90..-90, brightness 0..1; robot/retina.py PANO_*).

Stimuli (the fly's view; + azimuth = right):
  grating_right / grating_left   vertical stripes moving to the right / left (yaw)
  grating_up / grating_down      horizontal stripes moving up / down
  loom_front / recede_front      a dark disc approaching / receding straight ahead
  dot_left / dot_right           a small dark dot moving across one side
  blank                          uniform grey

Read-outs, per side: T4a-d, T5a-d (motion detectors; a = front-to-back,
b = back-to-front, c = up, d = down: Maisak et al. 2013), HS (HSE/HSN/HSS),
VS, H1, H2, LPLC2, LC4, LC10a, the giant fibre (DNp01), DNa02 (steering),
photoreceptors and the optic lobe as a whole.

Scores (what a fly's visual system does):
  T4/T5 direction selectivity  DSI = (pref - null) / (pref + null), per
                               subtype and eye, preferred direction from biology
  HS                            front-to-back > back-to-front on its own eye
  looming                       LPLC2, LC4, GF: approach > recede
  small object                  LC10a: dot on its side > dot on the other side
  cost                          wall time per simulated second

    PYTHONPATH=. FLY_DATASET=merged FLY_DYNAMICS=calibrated \\
        python -m experiments.vision_ab.harness --kinds none A B --out results/vision_ab
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from robot.retina import PANO_H, PANO_W

AZ = -180.0 + (np.arange(PANO_W) + 0.5) * 360.0 / PANO_W          # per column, + = right
EL = 90.0 - (np.arange(PANO_H) + 0.5) * 180.0 / PANO_H            # per row, + = up
PRE_MS, STIM_MS = 400.0, 1000.0


# ------------------------------------------------------------------ stimuli
class Stimulus:
    name = "blank"
    background = 0.5                                                # also the lead-in's: no step at onset

    def frame(self, t_s: float) -> np.ndarray:                     # t_s since stimulus onset
        return np.full((PANO_H, PANO_W), 0.5, np.float32)


class Grating(Stimulus):
    def __init__(self, name, axis, sign, speed=60.0, wavelength=30.0):
        self.name, self.axis, self.sign, self.speed, self.wl = name, axis, sign, speed, wavelength

    def frame(self, t_s):
        if self.axis == "az":                                       # vertical stripes moving in azimuth
            ph = (AZ[None, :] - self.sign * self.speed * t_s) / self.wl
            img = np.broadcast_to(ph, (PANO_H, PANO_W))
        else:                                                       # horizontal stripes moving in elevation
            ph = (EL[:, None] - self.sign * self.speed * t_s) / self.wl
            img = np.broadcast_to(ph, (PANO_H, PANO_W))
        return (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * img))).astype(np.float32)


class Looming(Stimulus):
    """A dark disc straight ahead whose angular radius grows (approach) or
    shrinks (recede): an object of half-size 0.1 m at 1.5 m -> 0.15 m."""

    def __init__(self, name, receding=False, az=0.0, el=0.0):
        self.name, self.recede, self.az0, self.el0 = name, receding, az, el
        self.background = 0.8

    def frame(self, t_s):
        f = min(t_s / (STIM_MS / 1000.0), 1.0)
        d = 1.5 - 1.35 * f if not self.recede else 0.15 + 1.35 * f
        theta = math.degrees(math.atan2(0.1, d))
        dist = np.hypot((AZ[None, :] - self.az0) * np.cos(np.radians(EL[:, None])), EL[:, None] - self.el0)
        img = np.full((PANO_H, PANO_W), 0.8, np.float32)
        img[dist <= theta] = 0.05
        return img


class Dot(Stimulus):
    """A small dark dot (radius 4 deg) moving front-to-back across one side."""

    def __init__(self, name, side):
        self.name, self.side = name, side
        self.background = 0.8

    def frame(self, t_s):
        az = self.side * (20.0 + 60.0 * t_s)
        dist = np.hypot((AZ[None, :] - az), EL[:, None] - 5.0)
        img = np.full((PANO_H, PANO_W), 0.8, np.float32)
        img[dist <= 4.0] = 0.05
        return img


STIMULI = [Stimulus(), Grating("grating_right", "az", +1), Grating("grating_left", "az", -1),
           Grating("grating_up", "el", +1), Grating("grating_down", "el", -1),
           Looming("loom_front"), Looming("recede_front", receding=True),
           Dot("dot_left", -1), Dot("dot_right", +1)]


class PanoramaSource:
    """What the encoders sample: the stimulus's background before it starts
    (so its onset is the object, not a change of brightness), then its frames
    (rendered at 100 Hz)."""

    def __init__(self, stim, t0_ms):
        self.stim, self.t0 = stim, t0_ms

    def __call__(self, t_ms):
        k = int((t_ms - self.t0) // 10.0)                           # frame number at 100 Hz
        if k < 0:
            return ("pre", np.full((PANO_H, PANO_W), self.stim.background, np.float32))
        return ((self.stim.name, k), self.stim.frame(k / 100.0))


# ------------------------------------------------------------------ read-outs
GROUPS = {**{t: [t] for t in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "VS", "H1", "H2",
                              "LPLC2", "LC4", "LC10a", "DNa02")},
          "HS": ["HSE", "HSN", "HSS"], "GF": ["DNp01"], "photoreceptors": ["R1-R6", "R7y", "R7p", "R8y", "R8p"],
          "L1": ["L1"], "L2": ["L2"], "Mi1": ["Mi1"], "Tm1": ["Tm1"]}


def group_index(c):
    t = c.neurons["primary_type"].fillna("").astype(str).to_numpy()
    side = c.neurons["side"].fillna("").astype(str).to_numpy()
    sc = c.neurons["super_class"].fillna("").astype(str).to_numpy()
    out = {}
    for g, types in GROUPS.items():
        for s in ("left", "right"):
            out[f"{g}_{s}"] = np.flatnonzero(np.isin(t, types) & (side == s))
    out["optic_lobe"] = np.flatnonzero(np.isin(sc, ["ol_intrinsic", "visual_projection", "visual_centrifugal"]))
    return out


# ------------------------------------------------------------------ runs
def make_encoders(kind, c, rt, source):
    if kind == "none":
        return []
    if kind == "A":
        from robot.retina import RetinaEncoder
        e = RetinaEncoder(c, rt, source)
        return [(e, e)]
    if kind in ("A2", "A3"):
        from robot.retina import LaminaEncoder
        e = LaminaEncoder(c, rt, source, rest_hz=float(os.environ.get("FLY_LAMINA_REST", 30.0)),
                          gain_hz=float(os.environ.get("FLY_LAMINA_GAIN", 60.0)),
                          tonic_hz=0.0 if kind == "A2" else float(os.environ.get("FLY_TONIC_HZ", 8.0)))
        return [(e, e)]
    if kind == "B":
        from robot.flyvis_eye import FlyvisEncoder
        e = FlyvisEncoder(c, rt, source)
        return [(e, e)]
    raise ValueError(kind)


def run(kind, stim, c, rt, seed=1):
    from robot.head import RestingOlfaction
    from simulation.engine.session import Session
    ses = Session(c, seed=seed)
    ses.reset(seed=seed)
    rest = RestingOlfaction(c)
    ses.add_stimulus(rest, rest)                                     # the robot's resting background
    src = PanoramaSource(stim, ses.engine.t_ms + PRE_MS)
    for enc, st in make_encoders(kind, c, rt, src):
        ses.add_stimulus(enc, st)
    groups = group_index(c)
    ses.advance(PRE_MS)
    c0 = np.array(ses.engine.spike_counts, np.int64)
    t0 = time.perf_counter()
    ses.advance(STIM_MS)
    wall = time.perf_counter() - t0
    c1 = np.array(ses.engine.spike_counts, np.int64)
    d = (c1 - c0) / (STIM_MS / 1000.0)
    return {g: round(float(d[ix].mean()), 3) if len(ix) else None for g, ix in groups.items()}, wall


def dsi(p, n):
    return None if p is None or n is None or p + n <= 0 else round((p - n) / (p + n), 3)


def scores(r: dict) -> dict:
    """r: {stimulus: {group: Hz}}."""
    s = {}
    # front-to-back is rightward on the right eye, leftward on the left eye
    ftb = {"right": "grating_right", "left": "grating_left"}
    btf = {"right": "grating_left", "left": "grating_right"}
    pref = {"a": ftb, "b": btf, "c": {"left": "grating_up", "right": "grating_up"},
            "d": {"left": "grating_down", "right": "grating_down"}}
    null = {"a": btf, "b": ftb, "c": {"left": "grating_down", "right": "grating_down"},
            "d": {"left": "grating_up", "right": "grating_up"}}
    for T in ("T4", "T5"):
        for sub in "abcd":
            for side in ("left", "right"):
                g = f"{T}{sub}_{side}"
                s[f"DSI {g}"] = dsi(r[pref[sub][side]][g], r[null[sub][side]][g])
    for side in ("left", "right"):
        s[f"HS ftb-btf {side}"] = dsi(r[ftb[side]][f"HS_{side}"], r[btf[side]][f"HS_{side}"])
        for g in ("LPLC2", "LC4", "GF"):
            s[f"loom/recede {g}_{side}"] = dsi(r["loom_front"][f"{g}_{side}"], r["recede_front"][f"{g}_{side}"])
        other = "dot_left" if side == "right" else "dot_right"
        s[f"LC10a own-side dot {side}"] = dsi(r[f"dot_{side}"][f"LC10a_{side}"], r[other][f"LC10a_{side}"])
    s["optic lobe Hz (grating_right)"] = r["grating_right"]["optic_lobe"]
    s["optic lobe Hz (blank)"] = r["blank"]["optic_lobe"]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["none", "A"])
    ap.add_argument("--out", default="results/vision_ab")
    ap.add_argument("--only", nargs="*", help="stimulus names")
    a = ap.parse_args()
    from brain.neurons.registry import load_connectome
    from brain.sensory.retinotopy import load_retinotopy
    c = load_connectome()
    rt = load_retinotopy(c)
    os.makedirs(a.out, exist_ok=True)
    for kind in a.kinds:
        res, walls = {}, []
        for stim in STIMULI:
            if a.only and stim.name not in a.only:
                continue
            res[stim.name], w = run(kind, stim, c, rt)
            walls.append(w)
            print(kind, stim.name, {k: v for k, v in res[stim.name].items()
                                    if k.split("_")[0] in ("T4a", "HS", "LPLC2", "GF", "LC10a", "optic")}, flush=True)
        out = {"kind": kind, "rates": res, "wall_s_per_sim_s": round(float(np.mean(walls)) / (STIM_MS / 1000.0), 3)}
        if not a.only:
            out["scores"] = scores(res)
        with open(os.path.join(a.out, f"{kind}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        print(kind, "scores", json.dumps(out.get("scores"), indent=1), "cost", out["wall_s_per_sim_s"], flush=True)


if __name__ == "__main__":
    main()
