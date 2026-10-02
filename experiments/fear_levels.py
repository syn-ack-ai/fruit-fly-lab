"""
Fear level -> the fly's escape command (robot/threat.py's calibration).

A held fear appraisal (robot/threat.Fear: a virtual looming disc whose size
and expansion grow with the level) drives the real LC4 / LPLC2 neurons; the
escape channels (DNp01 takeoff; DNp02 / DNp04 / DNp11 long mode) are read
over 1 s. Wanted: level 1 crosses the escape threshold (0.5, the startle in
robot/head.py and the personality's "startled"), level <= 0.35 ("wary")
does not.

Run:  FLY_TRIM=robot python -m experiments.fear_levels [--seeds 3]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import config
from brain.neurons.registry import load_connectome
from brain.sensory.encoders import LoomingEncoder
from brain.sensory.retinotopy import load_retinotopy
from robot.threat import Fear
from simulation.engine.session import Session

LEVELS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
AZIMUTHS = (0.0, 45.0, -45.0)
PRE_MS, ON_MS, WIN_MS = 300.0, 1000.0, 50.0


def run(c, retino, level, az, seed):
    ses = Session(c, seed=seed)
    fear = Fear()
    ses.add_stimulus(LoomingEncoder(c, retino), fear)
    out = {"takeoff": [], "long": []}
    t = 0.0
    while t < PRE_MS + ON_MS:
        if t >= PRE_MS and level > 0:
            fear.appraise("test", level, t / 1e3, azimuth_deg=az, fade_s=1e9)
        fear.update(t / 1e3)
        ses.advance(WIN_MS)
        t += WIN_MS
        if t > PRE_MS:
            ch = ses.readout.channels(ses.recorder.window_sum, ses.window_ms)
            out["takeoff"].append(ch.get("escape_takeoff", 0.0))
            out["long"].append(ch.get("escape_long_mode", 0.0))
    esc = np.maximum(out["takeoff"], out["long"])
    return {"max_takeoff": float(np.max(out["takeoff"])), "max_long": float(np.max(out["long"])),
            "frac_over_0.5": float((esc > 0.5).mean()),
            "first_over_ms": float(WIN_MS * (int(np.argmax(esc > 0.5)) + 1)) if (esc > 0.5).any() else None}


# The knife test on the robot (2026-10-01, robot/appraise.py live): Gemma's
# answers once a second from t = 64 s: danger, wary, none, wary, wary, none, wary
KNIFE = (2, 1, 0, 1, 1, 0, 1)


def replay(c, retino, looks, seed, every_s=1.0, tail_s=4.0):
    """The escape command (max of takeoff and long mode) per 0.25 s while the
    looks arrive as the robot got them."""
    from robot.threat import LEVELS
    from fly.body.foraging_body import ForagingBody
    ses = Session(c, seed=seed)
    ses.body = ForagingBody(neural=True, seed=seed, spontaneous_takeoff_per_s=0.0, wheeled=True)
    fear = Fear()
    ses.add_stimulus(LoomingEncoder(c, retino), fear)
    ses.advance(PRE_MS)
    trace, t = [], 0.0
    while t < len(looks) * every_s + tail_s - 1e-9:
        k = int(round(t / every_s, 6)) if abs(t / every_s - round(t / every_s)) < 1e-6 else None
        if k is not None and k < len(looks):
            fear.appraise("vision", LEVELS[looks[k]], t)
        L = fear.update(t)
        ses.advance(WIN_MS)
        t = round(t + WIN_MS / 1e3, 6)
        ch = ses.readout.channels(ses.recorder.window_sum, ses.window_ms)
        trace.append((t, L.get("level", 0.0), max(ch.get("escape_takeoff", 0.0), ch.get("escape_long_mode", 0.0)),
                      ses.body.state.behaviour))
    return trace, ses.body.startles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--knife", action="store_true", help="only replay the knife test")
    a = ap.parse_args()
    c = load_connectome()
    retino = load_retinotopy(c)
    if a.knife:
        runs = [replay(c, retino, KNIFE, s) for s in range(a.seeds)]
        traces = [r[0] for r in runs]
        print(f"brain {config.BRAIN_KEY}: the knife test's looks {KNIFE}, one a second")
        print("   t (s)  look  fear level  escape command (mean of seeds; > 0.5 = escape)   body (seed 0, wheeled)")
        for i in range(0, len(traces[0]), 5):
            t, lv = traces[0][i][0], traces[0][i][1]
            esc = np.mean([tr[i][2] for tr in traces])
            look = KNIFE[int(t - 0.05)] if t - 0.05 < len(KNIFE) and abs((t - 0.05) % 1.0) < 1e-6 else ""
            print(f"  {t:6.2f}  {look!s:4}  {lv:10.2f}  {esc:5.2f} {'#' * int(round(esc * 20)):20}  {traces[0][i][3]}")
        over = np.mean([[x[2] > 0.5 for x in tr] for tr in traces], axis=0)
        print(f"escape (> 0.5) in {100 * over.mean():.0f}% of 50 ms windows over {traces[0][-1][0]:.0f} s; "
              f"startles per seed: {[r[1] for r in runs]}")
        return
    print(f"brain {config.BRAIN_KEY}: {c.n} neurons")
    print(" level    az | max takeoff  max long | windows > 0.5 | first > 0.5 (ms)")
    rows = []
    for lv in LEVELS:
        for az in AZIMUTHS if lv > 0 else (0.0,):
            rs = [run(c, retino, lv, az, s) for s in range(a.seeds)]
            r = {k: float(np.mean([x[k] for x in rs])) for k in ("max_takeoff", "max_long", "frac_over_0.5")}
            firsts = [x["first_over_ms"] for x in rs if x["first_over_ms"] is not None]
            r.update(level=lv, azimuth=az, seeds_over=len(firsts),
                     first_ms=float(np.mean(firsts)) if firsts else None)
            rows.append(r)
            print(f" {lv:5.2f} {az:+5.0f} | {r['max_takeoff']:11.2f} {r['max_long']:9.2f} | "
                  f"{r['frac_over_0.5']:13.2f} | {r['first_ms'] or '-'} ({len(firsts)}/{a.seeds} seeds)")
    out = config.OUTPUT_DIR / f"fear_levels_{config.BRAIN_KEY}.json"
    out.write_text(json.dumps(rows, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
