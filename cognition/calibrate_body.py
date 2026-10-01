"""
The body's readouts, per dataset (results/body_readout_2026-09-29/):

- Walking speed: DNg100's rate under the resting sensory input with the
  calibrated dynamics. The body walks at WALK_SPEED_MM_S (12 mm/s, a fly's
  typical pace) when DNg100 fires at this rate and scales with it
  (fly/body/foraging_body.py). It used to be one constant measured on FAFB
  (14.6 Hz, 2026-09-25); the complete male brain rests at a quarter of that,
  so it walked at a quarter of the speed (2026-09-28). Measured this way FAFB
  rests at 18.5 Hz, not 14.6 (the first measurement had no warm-up), so FAFB
  had walked 27% faster than a fly's pace (2026-09-29).
- Proboscis: the proboscis motor neurons' sustained rate under sugar
  (sugar_proboscis_hz), which the readout scales to FAFB's
  (brain/motor/descending.proboscis_gain).

    FLY_DATASET=merged python -m cognition.calibrate_body          # print
    FLY_DATASET=merged python -m cognition.calibrate_body --write  # data/metadata/body_readout_<dataset>.json

FAFB's values are also the references in the code (foraging_body.
FAFB_DNG100_REST_HZ, descending.PROBOSCIS_REF_SUGAR_HZ; tests check they
agree): after re-measuring FAFB, update them.
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


def sugar_proboscis_hz(seeds=tuple(range(1, 7)), ms: float = 8000.0, onset_ms: float = 2000.0,
                       warm_ms: float = 1000.0) -> tuple:
    """(mean, standard error) of the proboscis motor neurons' sustained rate
    under sugar (120 Hz on the sugar GRNs, as at the Habitat dock and in the
    exam) on the resting input: seconds 2-10 after onset. The proboscis
    readout scales each brain to FAFB's value (brain/motor/descending.
    PROBOSCIS_REF_SUGAR_HZ): a robot at its dock holds the sugar for tens of
    seconds, not the exam's first second (male MN9: 16 Hz in the first
    second, 8.6 sustained)."""
    from brain.motor.descending import proboscis_motor_indices
    from brain.sensory.modalities import BY_KEY, resolve_neurons
    from cognition.exam import core
    c = core.connectome()
    pm = proboscis_motor_indices(c)
    sugar = np.asarray(resolve_neurons(BY_KEY["taste_sugar"], c), np.int64)
    ctx = core.Ctx({"dynamics": "calibrated"})
    ri, rr = core.resting()
    per = []
    for s in seeds:
        ctx.trial([(ri, rr), (sugar, 120.0)], onset_ms, s, warm_inputs=[(ri, rr)], warm_ms=warm_ms)
        per.append(ctx.window(ms)[pm].sum() / len(pm) / (ms / 1000.0))
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
    phz, pse = sugar_proboscis_hz()
    print(f"{config.DATASET_KEY}: proboscis motor neurons under sustained sugar {phz:.2f} +- {pse:.2f} Hz (SE)")
    if a.write:
        p = config.METADATA_DIR / f"body_readout_{config.BRAIN_KEY}.json"
        p.write_text(json.dumps({"dataset": config.DATASET_KEY, "dng100_rest_hz": round(hz, 2), "se_hz": round(se, 2),
                                 "proboscis_sugar_hz": round(phz, 2), "proboscis_se_hz": round(pse, 2),
                                 "dynamics_sha256_16": dynamics_hash(),
                                 "method": "cognition/calibrate_body.py: DNg100 under the resting receptor input, "
                                           "calibrated dynamics, 12 seeds x 2 s after a 1 s warm-up; proboscis "
                                           "motor neurons under 120 Hz sugar, 6 seeds, seconds 2-10 after onset"},
                                indent=2))
        print("wrote", p)


if __name__ == "__main__":
    main()
