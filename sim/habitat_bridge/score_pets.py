"""
Score battery-pet lifetimes (run_cortex_life.sh) from 0 to 100 on how good a
pet the robot would be. The anchors are absolute -- not relative to the other
brain -- so two brains can both score well, and were fixed on 2026-09-27
BEFORE the complete brain's real-world run had finished (they are judgement
calls; each metric's score is printed, so they can be reweighted).

    .venv/bin/python -m sim.habitat_bridge.score_pets fafb=simulation/outputs/habitat/real_v1_fafb \\
        merged=simulation/outputs/habitat/real_v3_merged

Per day (120 s simulated), linear between the anchors (100 at `best`, 0 at `worst`):

  SAFETY (40)
    furniture bumps / day          0 .. 5          15
    pet-caused person bumps / day  0 .. 0.5        15
    safety-layer brake s / day     0 .. 60          5   (lidar conditions only:
                                                        an organic pet rarely needs it)
    pinned / pressing s / day      0 .. 30          5   (navmesh-blocked + pivot)
  SELF-CARE (30)
    s / day below 20% battery      0 .. 60         15
    days with a flat battery       0 .. 100%       10
    s / day eating when full       0 .. 10          5
  LIFE AND COMPANY (30)
    walked m / day                 20 .. 0         15   (a pet that stays put scores 0)
    s / day near the person        30 .. 0         15
  ALIVENESS (added 2026-09-27 at the user's request -- "does it feel alive,
  think Johnny 5"; also fixed before the complete brain's run had finished).
  Things are seen as alive when they change speed and direction by
  themselves (Heider & Simmel 1944; Tremoulet & Feldman 2000, Perception
  29:943), smoothly, in bouts, and in response to others. From the 10 Hz pose:
    speed variation while moving   CV 0.3-0.7 ideal; 0 at CV 0 (clockwork) or >= 1.2   6
    smoothness: heading reversals  0 .. 20 per active minute (twitching/dithering)     8
    move/pause bouts per minute    2-8 ideal; 0 at 0 (never stops/starts) or >= 20     5
    irregular bouts: CV of bout    0.5+ .. 0 (metronome-like bouts)                    3
      durations
    turns toward the person        fraction of appearances (in view after >= 1 s out    8
      when they appear             of view, not already faced, not eating, and NOT
                                   brought into view by its own turn) followed within
                                   3 s by its own rotation of >= 20 deg toward them: 1 .. 0

Speed and walked distance are measured over 0.5 s of displacement: logs before
2026-09-27 (evening) round the pose to 1 cm, which at 10 Hz is a 0.1 m/s speed
step and inflated the speed variation of slow movers (review 2026-09-27).

The total weighs the categories 28/21/21/30 (the first three keep their
original 40/30/30 proportions); "without aliveness" gives the original total.
A category is scored on the metrics available in a condition (weights
renormalised). Scores are per seed (all its days), then mean and range over seeds.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

from sim.habitat_bridge.bump_analysis import classify

# key: (category, weight, best, worst, label)
METRICS = {
    "furniture": ("safety", 15, 0.0, 5.0, "furniture bumps/day"),
    "person": ("safety", 15, 0.0, 0.5, "pet-caused person bumps/day"),
    "brake": ("safety", 5, 0.0, 60.0, "safety-layer brake s/day"),
    "pinned": ("safety", 5, 0.0, 30.0, "pinned/pressing s/day"),
    "low": ("self-care", 15, 0.0, 60.0, "s/day below 20% battery"),
    "flat": ("self-care", 10, 0.0, 1.0, "fraction of days with a flat battery"),
    "full_eat": ("self-care", 5, 0.0, 10.0, "s/day eating when full"),
    "walked": ("life", 15, 20.0, 0.0, "walked m/day"),
    "company": ("life", 15, 30.0, 0.0, "s/day near the person"),
    # (lo, hi) best = a band scored 100; (lo, hi) worst = 0 below lo / above hi
    "speed_cv": ("aliveness", 6, (0.3, 0.7), (0.0, 1.2), "speed variation (CV) while moving"),
    "reversals": ("aliveness", 8, 0.0, 20.0, "heading reversals / active min"),
    "bouts": ("aliveness", 5, (2.0, 8.0), (0.0, 20.0), "move/pause bouts / min"),
    "bout_cv": ("aliveness", 3, 0.5, 0.0, "irregularity of bouts (CV)"),
    "orient": ("aliveness", 8, 1.0, 0.0, "turns toward person on appearance"),
}
CATS = {"safety": 28, "self-care": 21, "life": 21, "aliveness": 30}
ORIGINAL = {"safety": 40, "self-care": 30, "life": 30}
MOVING = 0.03          # m/s
TURNING = 15.0         # deg/s
LIDAR = {"petlidar", "petsteer", "petobst", "petsense", "petroute", "petttc"}


def _day(r: dict, cond: str) -> dict:
    log = r["log"]
    h = r["home"]
    b = h.get("battery") or {}
    lid = [p.get("lidar") or {} for p in log]
    d = {
        "furniture": r.get("scene_bumps") or 0,
        "person": sum(classify(log, t)["pet_fault"] for (t, *_x) in r.get("bumps", {}).get("list", [])),
        "pinned": (r.get("blocked_s") or 0.0) + 0.1 * sum(bool((x.get("avoid") or {}).get("pivot")) for x in lid),
        "low": b.get("low_s", 0.0),
        "flat": float(b.get("flat_s", 0.0) > 0),
        "full_eat": h.get("eating_when_full_s", 0.0),
        "walked": float(np.hypot(*np.diff(np.array([p["robot"][:2] for p in log])[::5], axis=0).T).sum()) if len(log) > 5 else 0.0,
        "company": h.get("near_person_s", 0.0),
        # reported, not scored
        "stuck": stuck_s(log),
        "song_bouts": (r.get("song") or {}).get("bouts", 0),
        "dock": float(h["first_bowl_s"] is not None),
        "meals": b.get("meals", 0),
        "eating_s": h.get("eating_s", 0.0),
        "battery_end": b.get("end"),
    }
    if cond in LIDAR:
        d["brake"] = 0.1 * sum((x.get("brake") or 0) > 0.05 for x in lid)
    d.update(motion(log))
    return d


def stuck_s(log: list, min_s: float = 5.0, near_m: float = 0.15, move_m: float = 0.10, window_s: float = 3.0):
    """Seconds in runs of >= min_s in which an obstacle is within near_m of the
    body (the logged lidar min_clear) AND the body stayed within a move_m box
    over the last window_s -- robot/avoid.Unstick's own test (max - min extent,
    so shuttling back and forth at a wall counts). Reported, not scored (added
    2026-09-28; the "pinned" metric -- navmesh-blocked time -- did not show it).
    None without lidar clearance in the log."""
    if len(log) < 2 or not any((p.get("lidar") or {}).get("min_clear") is not None for p in log):
        return None
    t = np.array([p["t"] for p in log], float)
    clear = np.array([(p.get("lidar") or {}).get("min_clear") for p in log], dtype=object)
    clear = np.array([np.inf if c is None else float(c) for c in clear])
    xy = np.array([p["robot"][:2] for p in log], float)
    still = np.zeros(len(log), bool)
    j0 = 0
    for i in range(len(log)):
        while t[i] - t[j0] > window_s:
            j0 += 1
        if t[i] - t[j0] >= window_s - 0.15:
            w = xy[j0:i + 1]
            ext = float(np.hypot(*(w.max(0) - w.min(0))))
            still[i] = clear[j0:i + 1].max() <= near_m and ext < move_m
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    return float(sum(ln for _s, ln in _runs(still) if ln * dt >= min_s) * dt)


def _runs(mask: np.ndarray) -> list:
    """(start, length) of True runs."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    e = np.flatnonzero(np.diff(m))
    return list(zip(e[::2], e[1::2] - e[::2]))


def motion(log: list, dt: float = 0.1) -> dict:
    """Aliveness measures from the logged pose (x, y, yaw deg) at 10 Hz. Per-day
    values; a measure with nothing to measure (never moved, nobody appeared) is
    left out of that day's mean."""
    if len(log) < 20:
        return {}
    P = np.array([p["robot"][:3] for p in log], float)
    k = 5                                                               # 0.5 s of displacement
    sp = np.hypot(*(P[k:, :2] - P[:-k, :2]).T) / (k * dt)
    sp = np.concatenate([np.full(k // 2, sp[0]), sp, np.full(len(P) - 1 - len(sp) - k // 2, sp[-1])])
    yaw = np.unwrap(np.radians(P[:, 2])) * 180.0 / np.pi
    yr = np.diff(yaw) / dt
    moving = sp > MOVING
    active = moving | (np.abs(yr) > TURNING)
    out = {}
    if moving.sum() >= 10:
        out["speed_cv"] = float(sp[moving].std() / sp[moving].mean())
    # heading reversals: a turn one way, then the other within 0.5 s
    turn = np.where(yr > TURNING, 1, np.where(yr < -TURNING, -1, 0))
    rev, last, last_i = 0, 0, -99
    for i, t in enumerate(turn):
        if t:
            if last and t != last and i - last_i <= 5:
                rev += 1
            last, last_i = t, i
    act_min = active.sum() * dt / 60.0
    if act_min > 0.1:
        out["reversals"] = rev / act_min
    # bouts: moving runs, gaps < 0.5 s merged, >= 0.5 s long
    m = moving.copy()
    for st, ln in _runs(~m):
        if ln < 5 and st > 0 and st + ln < len(m):
            m[st:st + ln] = True
    bouts = [ln for _st, ln in _runs(m) if ln >= 5]
    out["bouts"] = len(bouts) / (len(sp) * dt / 60.0)
    if len(bouts) >= 3:
        out["bout_cv"] = float(np.std(bouts) / np.mean(bouts))
    # turning toward the person when they come into view: the robot's OWN
    # rotation (review 2026-09-27: a shrinking bearing also counted the person
    # drifting to the centre, and most appearances were made by the robot's own
    # turning, so a robot spinning in place scored 1.0). The bearing az (+ right)
    # changes with the heading as d(az) = +d(yaw) (checked on the logs: slope
    # 1.03), so a turn toward the person is -d(yaw) * sign(az).
    vis = np.array([bool(p.get("visible")) for p in log])
    az = np.array([float(p.get("az") or 0.0) for p in log])
    eating = np.array(["feeding" in str(p.get("behaviour", "")) for p in log])
    hits = n = 0
    for st, ln in _runs(~vis):
        i = st + ln                                    # first visible step
        if ln < 10 or i < 5 or i >= len(log) - 30 or abs(az[i]) <= 30.0 or eating[i]:
            continue
        toward = -np.sign(az[i]) * (yaw - yaw[i])      # own rotation toward them since i (deg)
        if toward[i - 5] < -7.5:                       # it was already turning toward them (> 15 deg/s):
            continue                                   # its own turn brought them into view
        n += 1
        hits += bool(toward[i + 1:i + 31].max() >= 20.0)
    if n:
        out["orient"] = hits / n
        out["orient_events"] = n
    return out


def load(root: str) -> dict:
    """{condition: {seed: [day dicts]}}"""
    out = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(root, "seed*", "*", "brain_day*_ep*.json"))):
        cond = os.path.basename(os.path.dirname(f))
        seed = os.path.basename(os.path.dirname(os.path.dirname(f)))
        out[cond][seed].append(_day(json.load(open(f)), cond))
    return out


def _metric_score(key: str, v: float) -> float:
    _c, _w, best, worst, _l = METRICS[key]
    if isinstance(best, tuple):                        # a band: 100 inside, 0 at the outer limits
        (b0, b1), (w0, w1) = best, worst
        f = (v - w0) / (b0 - w0) if v < b0 else (w1 - v) / (w1 - b1) if v > b1 else 1.0
        return float(100.0 * np.clip(f, 0.0, 1.0))
    return float(100.0 * np.clip((v - worst) / (best - worst), 0.0, 1.0))


def score(days: list) -> dict:
    """Mean per-day metrics of one seed -> metric, category and total scores."""
    m = {}
    for k in METRICS:
        v = [d[k] for d in days if k in d]
        if v:
            m[k] = float(np.mean(v))
    ms = {k: _metric_score(k, v) for k, v in m.items()}
    cs = {}
    for cat in CATS:
        ks = [k for k in ms if METRICS[k][0] == cat]
        w = np.array([METRICS[k][1] for k in ks], float)
        cs[cat] = float(np.dot(w, [ms[k] for k in ks]) / w.sum())
    total = sum(CATS[c] * cs[c] for c in CATS) / sum(CATS.values())
    original = sum(ORIGINAL[c] * cs[c] for c in ORIGINAL) / sum(ORIGINAL.values())
    return {"metrics": m, "metric_scores": ms, "categories": cs, "total": total, "original": original}


def main(args: list) -> None:
    brains = dict(a.split("=", 1) for a in args)
    data = {b: load(p) for b, p in brains.items()}
    conds = sorted(set().union(*[set(d) for d in data.values()]))
    totals, originals = defaultdict(dict), defaultdict(dict)
    for cond in conds:
        print(f"\n=== {cond}" + ("  (lidar on)" if cond in LIDAR else "  (no lidar)"))
        rows = {}
        for b, d in data.items():
            seeds = d.get(cond, {})
            if not seeds:
                continue
            per_seed = [score(days) for days in seeds.values()]
            rows[b] = (per_seed, sum(len(x) for x in seeds.values()))
        print(f"  {'':34s}" + "".join(f"{b + ' (%d d)' % n:>26s}" for b, (_p, n) in rows.items()))
        for k, (_c, _w, best, worst, label) in METRICS.items():
            if not all(any(k in s["metrics"] for s in p) for p, _n in rows.values()):
                continue
            cells = []
            for p, _n in rows.values():
                v = np.mean([s["metrics"][k] for s in p if k in s["metrics"]])
                sc = np.mean([s["metric_scores"][k] for s in p if k in s["metric_scores"]])
                cells.append(f"{v:9.2f} -> {sc:5.1f}%")
            print(f"  {label:34s}" + "".join(f"{c:>26s}" for c in cells))
        for cat in CATS:
            print(f"  {cat.upper() + ' (%d)' % CATS[cat]:34s}" + "".join(
                f"{np.mean([s['categories'][cat] for s in p]):>25.1f}%" for p, _n in rows.values()))
        cells = []
        for b, (p, _n) in rows.items():
            t = [s["total"] for s in p]
            totals[b][cond] = float(np.mean(t))
            originals[b][cond] = float(np.mean([s["original"] for s in p]))
            cells.append(f"{np.mean(t):5.1f}% [{min(t):.0f}-{max(t):.0f}]")
        print(f"  {'TOTAL (range over seeds)':34s}" + "".join(f"{c:>26s}" for c in cells))
        print(f"  {'  without aliveness':34s}" + "".join(f"{originals[b][cond]:>25.1f}%" for b in rows))
        for b, (p, _n) in rows.items():
            days = [x for s in data[b][cond].values() for x in s]
            st = [x["stuck"] for x in days if x["stuck"] is not None]
            print(f"    {b}: stuck against something " + (f"{np.mean(st):.1f} s/day" if st else "n/a (no lidar)") + ", "
                  f"song bouts {np.mean([x['song_bouts'] for x in days]):.1f}/day")
            print(f"    {b}: dock {int(sum(x['dock'] for x in days))}/{len(days)} days, meals {sum(x['meals'] for x in days)}, "
                  f"eating {np.mean([x['eating_s'] for x in days]):.1f} s/day, battery at day end "
                  f"{np.mean([x['battery_end'] for x in days if x['battery_end'] is not None]):.2f}")
    print("\n=== overall")
    for name, tt in (("total", totals), ("without aliveness", originals)):
        for b, t in tt.items():
            lid = [v for c, v in t.items() if c in LIDAR]
            print(f"  {name:18s} {b:10s} all conditions {np.mean(list(t.values())):5.1f}%   "
                  f"with lidar (the rover's setup) {np.mean(lid) if lid else float('nan'):5.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
