"""
Which top-down inputs drive forward walking above rest? (ROADMAP: walking gap)

In the closed loop the male brain's walking command DNg100 runs at ~1.2x its
resting rate and FAFB's at ~1.6x (results/touch_subtypes_2026-09-28/). This
drives each neocortex channel alone on top of the resting input (calibrated
dynamics, 1 s warm-up on the resting input, then 2 s; 6 seeds) and reports
DNg100 and MDN relative to rest, plus the steering DNs.

    FLY_DATASET=merged python -m experiments.walking_drive_test
    FLY_DATASET=fafb   python -m experiments.walking_drive_test
"""
import numpy as np

import config
import robot.head as rh
from brain.navigation.compass import Compass, CompassDrive
from brain.navigation.goal import GoalCircuit, GoalDrive
from cognition.exam import core
from cortex.topdown import AttendEncoder, ExciteEncoder, RestEncoder

SEEDS = (1, 2, 3, 4, 5, 6)
c = core.connectome()
t = c.neurons.primary_type.fillna("").astype(str).to_numpy()
g = {k: np.flatnonzero(t == k) for k in ("DNg100", "MDN", "DNa01", "DNa02")}
ctx = core.Ctx({"dynamics": "calibrated"})
ri, rr = core.resting()
cx = Compass(c)
comp = CompassDrive(cx)
goal = GoalDrive(GoalCircuit(c, cx))
att = AttendEncoder(rh.ObjectEncoder(c, None))
exc = ExciteEncoder(c) if config.MALE_CNS else ExciteEncoder.empty()
rest = RestEncoder(c)


def drive(enc):
    return (enc.indices, enc.rates_hz())


def conds():
    comp.set_heading(0.0)
    yield "rest", []
    yield "compass", [drive(comp)]
    for az in (0.0, 60.0, 180.0):
        goal.set_goal(az)          # CCW: +60 = left
        yield f"compass + goal {az:+.0f}", [drive(comp), drive(goal)]
    for az in (0.0, 30.0):
        att.set(az, 1.0)
        yield f"attend {az:+.0f} (LC10a)", [drive(att)]
    goal.set_goal(0.0)
    att.set(0.0, 1.0)
    yield "compass + goal 0 + attend 0", [drive(comp), drive(goal), drive(att)]
    if len(exc.indices):
        for lv in (0.3, 1.0):
            exc.set(lv)
            yield f"excite {lv} (P1)", [drive(exc)]
    rest.set(1.0)
    yield "rest 1 (ER5)", [drive(rest)]


def main():
    print(f"{config.DATASET_KEY}: {', '.join(f'{k} {len(v)}' for k, v in g.items())}")
    base = None
    for name, inp in conds():
        r = ctx.rates([(ri, rr)] + inp, 2000.0, SEEDS, warm_inputs=[(ri, rr)], warm_ms=1000.0)
        m = {k: float(r[v].mean()) for k, v in g.items()}
        base = base or m
        print(f"{name:28s} DNg100 {m['DNg100']:5.1f} ({m['DNg100'] / base['DNg100']:4.2f}x)  "
              f"MDN {m['MDN']:5.1f}  DNa01 {m['DNa01']:5.1f}  DNa02 {m['DNa02']:5.1f}", flush=True)


if __name__ == "__main__":
    main()
