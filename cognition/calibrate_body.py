"""
The body's walking-speed readout, per dataset: DNg100's rate under the
resting sensory input with the calibrated dynamics. The body walks at
WALK_SPEED_MM_S (12 mm/s, a fly's typical pace) when DNg100 fires at this
rate and scales with it (fly/body/foraging_body.py). The constant was measured
on FlyWire FAFB only (14.6 Hz, 2026-09-25); the complete male brain rests at a
quarter of that, so it walked at a quarter of the speed -- the whole "walks 3x
less than FAFB" gap in Habitat (2026-09-28).

    FLY_DATASET=merged python -m cognition.calibrate_body          # print
    FLY_DATASET=merged python -m cognition.calibrate_body --write  # data/metadata/body_readout_<dataset>.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def resting_dng100_hz(seeds=tuple(range(1, 13)), ms: float = 2000.0, warm_ms: float = 1000.0) -> tuple:
    """(mean, standard error) of DNg100's rate after a warm-up (the robot runs
    continuously; the first second is a start-up transient -- review 2026-09-28)."""
    from cognition.exam import core
    c = core.connectome()
    t = c.neurons["primary_type"].fillna("").astype(str).to_numpy()
    g = np.flatnonzero(t == "DNg100")
    ctx = core.Ctx({"dynamics": "calibrated"})
    ri, rr = core.resting()
    per = [ctx.trial([(ri, rr)], ms, s, warm_inputs=[(ri, rr)], warm_ms=warm_ms)[g].mean() / (ms / 1000.0) for s in seeds]
    return float(np.mean(per)), float(np.std(per, ddof=1) / np.sqrt(len(per)))


# prose and history in the dynamics file: editing them changes no dynamics
_TEXT_KEYS = {"rationale", "note", "note_merged", "summary", "result", "threshold_rationale", "calibration",
              "calibration_merged", "alternatives_tested", "fitted", "name", "version", "dataset"}


def _settings(x):
    if isinstance(x, dict):
        return {k: _settings(v) for k, v in sorted(x.items()) if k not in _TEXT_KEYS}
    if isinstance(x, list):
        return [_settings(v) for v in x]
    return x


def dynamics_hash() -> str:
    """What this rate was measured with -- the calibrated dynamics' settings
    (not their prose), the synaptic gain and the time step; it goes stale
    when any of them changes (review 2026-09-28: the file's bytes alone
    missed the gain and dt, and flagged edits to a rationale)."""
    import hashlib
    import json
    import config
    from simulation.engine.session import _calibrated_gain
    p = config.METADATA_DIR / f"dynamics_calibrated_{config.DATASET_KEY}.json"
    if not p.exists():
        p = config.METADATA_DIR / "dynamics_calibrated.json"
    key = {"dynamics": _settings(json.loads(p.read_text())), "gain": _calibrated_gain(),
           "dt_ms": float(os.environ.get("FLY_DT", "0.1"))}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def main():
    import config
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    hz, se = resting_dng100_hz()
    print(f"{config.DATASET_KEY}: DNg100 at rest {hz:.2f} +- {se:.2f} Hz (SE)")
    if a.write:
        p = config.METADATA_DIR / f"body_readout_{config.DATASET_KEY}.json"
        p.write_text(json.dumps({"dataset": config.DATASET_KEY, "dng100_rest_hz": round(hz, 2), "se_hz": round(se, 2),
                                 "dynamics_sha256_16": dynamics_hash(),
                                 "method": "cognition/calibrate_body.py: DNg100 under the resting receptor input, "
                                           "calibrated dynamics, 12 seeds x 2 s after a 1 s warm-up"}, indent=2))
        print("wrote", p)


if __name__ == "__main__":
    main()
