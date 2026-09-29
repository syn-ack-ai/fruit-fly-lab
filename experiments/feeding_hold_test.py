"""
Does the body hold the proboscis out while the brain tastes sugar?
(results/body_readout_2026-09-29/)

Records the proboscis motor neurons' spikes every 1 ms (calibrated dynamics,
1 s warm-up on the resting input, then 2 s into the condition and 8 s
recorded; 3 seeds) under sustained sugar (120 Hz, as at the Habitat dock), the
resting input and, on male brains, the P1 excitement channel. It replays them
through the readout (50 ms window, proboscis_gain) into ForagingBody and
reports the fraction of the last 7 s the proboscis is out, with the current
readout and with the one before 2026-09-29 (FLY_PROBOSCIS_HOLD=0). Then how
long the proboscis stays out after the sugar ends, when the next condition
follows directly.

    FLY_DATASET=merged python -m experiments.feeding_hold_test
    FLY_DATASET=fafb   python -m experiments.feeding_hold_test
"""
import os

import numpy as np

import config
from brain.motor.descending import DescendingReadout, proboscis_gain, proboscis_motor_indices
from brain.sensory.modalities import BY_KEY, resolve_neurons
from cognition.exam import core
from cortex.topdown import ExciteEncoder

SEEDS = (1, 2, 3)
WIN = 50


def record(ctx, rest, inp, seed, ms=8000):
    ctx.e.reset(seed=seed)
    ctx.set_inputs(rest)
    ctx.e.run(1000.0)
    ctx.set_inputs(rest + inp)
    ctx.e.run(2000.0)
    return np.array([ctx.window(1.0)[PM].sum() for _ in range(ms)])


def replay(train, hold: bool) -> np.ndarray:
    """Proboscis out (bool) per ms for a spike train through the readout and body."""
    os.environ["FLY_PROBOSCIS_HOLD"] = "1" if hold else "0"
    from fly.body.foraging_body import ForagingBody
    gain = proboscis_gain()
    b = ForagingBody(neural=True, seed=0, spontaneous_takeoff_per_s=0.0)      # as in Habitat
    win = np.convolve(train, np.ones(WIN), "full")[:len(train)]
    out = []
    for k, w in enumerate(win):
        hz = gain * w / len(PM) / (WIN * 1e-3)
        b.update(1.0, {}, float(k), proboscis_drive=DescendingReadout.proboscis_drive_from_hz(hz))
        out.append(b.state.proboscis_extension > 0.5)
    return np.array(out)


def extended(train, hold: bool) -> float:
    return float(replay(train, hold)[1000:].mean())


c = core.connectome()
PM = proboscis_motor_indices(c)


def main():
    ctx = core.Ctx({"dynamics": "calibrated"})
    ri, rr = core.resting()
    rest = [(ri, rr)]
    sugar = np.asarray(resolve_neurons(BY_KEY["taste_sugar"], c), np.int64)
    conds = [("sugar 120 Hz", [(sugar, 120.0)]), ("resting input", [])]
    if config.MALE_CNS:
        for lv in (0.3, 1.0):
            ex = ExciteEncoder(c)
            ex.set(lv)
            conds.append((f"P1 excitement {lv}", [(ex.indices, ex.rates_hz())]))
    os.environ["FLY_PROBOSCIS_HOLD"] = "1"
    print(f"{config.DATASET_KEY}: {len(PM)} proboscis motor neurons, gain {proboscis_gain():.2f}")
    trains = {}
    for name, inp in conds:
        trains[name] = [record(ctx, rest, inp, s) for s in SEEDS]
        hz = np.mean([t.sum() / len(PM) / 8.0 for t in trains[name]])
        now = np.mean([extended(t, True) for t in trains[name]])
        old = np.mean([extended(t, False) for t in trains[name]])
        print(f"{name:22s} {hz:5.1f} Hz   extended: now {now:5.1%}   before {old:5.1%}", flush=True)
    sugar = trains.pop("sugar 120 Hz")
    for name, after in trains.items():
        # (spliced recordings: the brain's own after-effects of sugar are not included)
        ms = []
        for s in sugar:
            for a in after:
                out = replay(np.r_[s, a], True)[len(s):]
                ms.append(int(np.argmax(~out)) if (~out).any() else len(out))
        print(f"sugar then {name:22s} retracted after {np.median(ms):5.0f} ms (median; max {max(ms)})", flush=True)


if __name__ == "__main__":
    main()
