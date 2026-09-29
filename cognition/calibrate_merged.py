"""
Calibrate the optional dynamics for OUR brain (FLY_DATASET=merged: the male
CNS with its gaps filled and made left/right symmetric, brain/connectivity/
merge.py) against the fly exam (cognition/exam), as the FAFB v3 calibration
was: the same parameters, the same tests, fitted anew because the wiring and
the synaptic gain differ.

Each candidate = the current data/metadata/dynamics_calibrated_merged.json
with some values overridden (dotted keys, e.g. "adaptation.adapt_mV_per_spike").
It is written to dynamics_cand<k>_merged.json, the whole exam runs on it
(without the robustness sweep), and candidates are ranked by tests passed,
then by exam score. Several candidates run at once.

    FLY_DATASET=merged python -m cognition.calibrate_merged --stage rest --parallel 5
    FLY_DATASET=merged python -m cognition.calibrate_merged --apply BEST.json   (writes the winner)

Roles stay as in the exam: "fit" tests may be used to choose values, "held_out"
tests are only checked, constraints must keep passing.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import config

BASE = config.METADATA_DIR / "dynamics_calibrated_merged.json"

STAGES = {
    # the synaptic gain together with the settings that most change overall drive
    # (the MaleCNS gain 0.62 was fixed with the PUBLISHED dynamics, only to stop
    # a runaway; with adaptation and depression the right gain can differ)
    "gain": {"gain": [0.62, 0.7, 0.8, 0.9, 1.0],
             "orn_short_term_depression.release_f": [0.96, 0.85],
             "adaptation.extra_al_ln_adapt_mV_per_spike": [1.0, 2.0]},
    # resting antennal-lobe rates and activity after odour offset
    "rest": {"orn_short_term_depression.release_f": [0.96, 0.9, 0.8, 0.7],
             "adaptation.adapt_mV_per_spike": [0.2, 0.4],
             "adaptation.extra_al_ln_adapt_mV_per_spike": [1.0, 2.0, 4.0]},
    # after "gain": latching, the ORN->PN transform and lateralisation
    "fine": {"gain": [0.85, 0.9, 0.95],
             "orn_short_term_depression.release_f": [0.9, 0.96],
             "orn_short_term_depression.full_strength_gain": [1.0, 1.5, 2.0],
             "adaptation.adapt_mV_per_spike": [0.2, 0.5, 1.0]},
    # receptor synapses: FAFB-like depression (fitted to the measured ORN->PN
    # transform), weaker full strength (male ORN->PN connections are stronger)
    "orn": {"gain": [0.85, 0.9],
            "orn_short_term_depression.release_f": [0.94, 0.96, 0.97],
            "orn_short_term_depression.full_strength_gain": [0.5, 0.7, 0.85, 1.0]},
    # activity that outlasts a stimulus: mostly the nerve cord (flight and
    # abdominal motor circuits), which FAFB does not have
    "vnc": {"vnc.extra_adapt_mV_per_spike": [0.5, 1.0, 2.0, 4.0],
            "adaptation.adapt_mV_per_spike": [0.2, 0.3]},
    # latching (nerve cord) together with odour lateralisation (ipsi/contra release)
    "vnc2": {"vnc.extra_adapt_mV_per_spike": [1.0, 2.0],
             "adaptation.adapt_mV_per_spike": [0.2, 0.3],
             "orn_pn_lateral_release.ipsi_contra_ratio": [1.4, 2.0, 3.0]},
    # stronger ipsilateral release for odour steering, weaker receptor synapses
    # to keep resting rates, nerve-cord adaptation against latching
    "joint": {"orn_pn_lateral_release.ipsi_contra_ratio": [2.5, 3.0],
              "orn_short_term_depression.full_strength_gain": [0.55, 0.6, 0.65],
              "vnc.extra_adapt_mV_per_spike": [1.5, 2.0],
              "adaptation.adapt_mV_per_spike": [0.25, 0.3]},
    # ORN->PN input normalisation (Tobin et al. 2017, generalised) with receptor
    # strength and ipsi/contra release: the male's VA1v PNs rest at ~100 Hz without it
    "norm": {"orn_pn_input_normalization.exponent": [0.5, 0.75, 1.0],
             "orn_short_term_depression.full_strength_gain": [0.55, 0.7, 0.85, 1.0],
             "orn_pn_lateral_release.ipsi_contra_ratio": [1.4, 2.5]},
    # after review 2 (no side bug): DNa02 sits near threshold (gain), weak
    # receptor strength starves attractive channels, VA1v needs normalisation
    # (now uniglomerular only, clipped), release ratio back to FAFB's 1.4
    "final": {"gain": [0.9, 0.95, 1.0],
              "orn_short_term_depression.full_strength_gain": [0.7, 0.85, 1.0],
              "orn_pn_input_normalization.exponent": [0.5, 1.0],
              "orn_pn_lateral_release.ipsi_contra_ratio": [1.4]},
    # glomerulus specificity and odour lateralisation
    "odour": {"al_excitatory_ln.to_pn_gain": [0.1, 0.05, 0.0],
              "orn_pn_lateral_release.ipsi_contra_ratio": [1.4, 2.0, 3.0],
              "slow_inhibition.slow_ratio": [0.0, 0.3]},
}


# random search over all interacting settings at once (grids of 3-4 values
# kept trading one test for another); each value uniform in its range
RANGES = {
    "gain": (0.85, 1.05),
    "orn_short_term_depression.release_f": (0.94, 0.975),
    "orn_short_term_depression.full_strength_gain": (0.55, 1.0),
    "orn_pn_input_normalization.exponent": (0.0, 1.0),
    "orn_pn_lateral_release.ipsi_contra_ratio": (1.2, 2.5),
    "adaptation.adapt_mV_per_spike": (0.15, 0.4),
    "adaptation.extra_al_ln_adapt_mV_per_spike": (0.5, 2.0),
    "vnc.extra_adapt_mV_per_spike": (0.5, 3.0),
    "al_excitatory_ln.to_pn_gain": (0.05, 0.15),
    "giant_fibre.threshold_correction": (0.3, 0.8),   # the applied 0.40 lay outside the first range
    # descending neurons' own adaptation (session.py): the DN-DN and brain <->
    # nerve-cord loops (DNg33, DNg12, DNge019 ...) latch after an odour on some
    # seeds; which build passes no_latching was a matter of chance (2026-09-27)
    "vnc.descending_adapt_mV_per_spike": (0.0, 0.5),
}


def random_candidates(n: int, seed: int, around: dict | None = None, shrink: float = 1.0) -> list:
    """n candidates uniform in RANGES, or (around given, shrink < 1) in a box of
    shrink x the range centred on `around`."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = {}
        for k, (lo, hi) in RANGES.items():
            if around is not None and k in around:
                half = 0.5 * shrink * (hi - lo)
                lo, hi = max(lo, around[k] - half), min(hi, around[k] + half)
            c[k] = round(float(rng.uniform(lo, hi)), 3)
        out.append(c)
    return out


def _set(d, dotted, v):
    ks = dotted.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = v


def run_candidate(k: int, overrides: dict, workers: int, out_dir: Path, seed_offset: int = 0) -> dict:
    cfg = copy.deepcopy(json.loads(BASE.read_text()))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "FLY_EXAM_SEED_OFFSET": str(seed_offset)}
    for key, v in overrides.items():
        if key == "gain":
            env["FLY_GAIN"] = str(v)                 # read by session._calibrated_gain
        else:
            _set(cfg, key, v)
    name = f"cand{os.getpid()}_{seed_offset}_{k}"
    path = config.METADATA_DIR / f"dynamics_{name}_{config.DATASET_KEY}.json"
    path.write_text(json.dumps(cfg, indent=1))
    out = out_dir / name
    t0 = time.time()
    try:
        subprocess.run([sys.executable, "-m", "cognition.exam", "--dynamics", name, "--workers", str(workers),
                        "--no-robustness", "--out", str(out)], check=True, capture_output=True, env=env)
        d = json.loads((out / f"exam_{name}_real.json").read_text())
        from cognition.exam.tests import ROLES
        base = [r for r in d["results"] if r["cfg"].get("wiring") == "real" and len(r["cfg"]) <= 2]
        res = {r["test"]: (r["pass"], r["value"], r.get("score", float(r["pass"]))) for r in base}
        # selection uses ONLY the tests calibration may use (fit, constraint);
        # held-out tests are reported, never optimised (review 2026-09-27)
        usable = [t for t in res if ROLES.get(t) in ("fit", "constraint")]
        return {"k": k, "overrides": overrides, "passed": d["card"]["passed"], "of": d["card"]["of"],
                "fc_passed": sum(res[t][0] for t in usable), "fc_of": len(usable),
                "fc_score": float(np.mean([res[t][2] for t in usable])) if usable else 0.0,
                "score": d["card"]["overall"], "roles": d["card"]["roles"],
                "failed": sorted(t for t, (p, _, _) in res.items() if not p),
                "values": {t: v for t, (_, v, _) in res.items()}, "seconds": round(time.time() - t0)}
    finally:
        path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES) + ["random"])
    ap.add_argument("--n", type=int, default=96, help="random stage: candidates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--around", default=None, help="random stage: JSON file of values to search around")
    ap.add_argument("--shrink", type=float, default=1.0)
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="simulation/outputs/calibrate_merged")
    ap.add_argument("--apply", default=None, help="a JSON file of overrides to write into the calibration")
    ap.add_argument("--exempt", default="",
                    help="comma-separated tests left out of the held-out ranking (documented limitations); still reported")
    ap.add_argument("--choose-offsets", default="100",
                    help="extra seed sets (FLY_EXAM_SEED_OFFSET) the top candidates must also pass; never 200 "
                         "(the report set). 2026-09-27: 100,300 -- one extra set let chance decide no_latching")
    ap.add_argument("--validate", type=int, default=5,
                    help="re-run the best N on the --choose-offsets seed sets and rank by the worst; +200 is the report set, never used here")
    a = ap.parse_args()
    if config.DATASET_KEY != "merged":
        raise SystemExit("run with FLY_DATASET=merged")
    if a.apply:
        cfg = json.loads(BASE.read_text())
        ov = json.loads(Path(a.apply).read_text())
        for key, v in ov.items():
            if key == "gain":                        # the dataset's gain file (session._calibrated_gain)
                gp = config.METADATA_DIR / "calibration_merged.json"
                gp.write_text(json.dumps({"dataset": "merged", "gain": v, "method":
                                          "cognition/calibrate_merged.py: fitted with the dynamics against the fly exam"}, indent=2))
            else:
                _set(cfg, key, v)
        cfg.setdefault("calibration_merged", []).append({"applied": ov, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        BASE.write_text(json.dumps(cfg, indent=2))
        print("applied", ov)
        return
    if a.stage == "random":
        around = json.loads(Path(a.around).read_text()) if a.around else None
        cands = random_candidates(a.n, a.seed, around, a.shrink)
        out = Path(a.out) / f"random_s{a.seed}"
    else:
        grid = STAGES[a.stage]
        keys = list(grid)
        cands = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
        out = Path(a.out) / a.stage
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(cands)} candidates, {a.parallel} at a time", flush=True)
    results = []
    with ThreadPoolExecutor(a.parallel) as ex:
        for r in ex.map(lambda kv: run_candidate(kv[0], kv[1], a.workers, out), enumerate(cands)):
            results.append(r)
            print(f"{r['passed']}/{r['of']} score {r['score']:.3f} {r['overrides']} failed {r['failed']}", flush=True)
    exempt = set(x for x in a.exempt.split(",") if x)
    def fc_key(r):                                  # fit + constraint tests only, exempt ones aside
        fails = [f for f in r["failed"] if f in FC_TESTS and f not in exempt]
        return (len(fails), -r["fc_score"])
    from cognition.exam.tests import ROLES
    FC_TESTS = {t for t, role in ROLES.items() if role in ("fit", "constraint")}
    results.sort(key=fc_key)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\nbest on the fitting seeds:", json.dumps(results[0]["overrides"]), results[0]["passed"], results[0]["failed"])
    # further seed sets (--choose-offsets, default +100) for CHOOSING; +200 is
    # never used here -- it is the report set (run the exam with
    # FLY_EXAM_SEED_OFFSET=200 afterwards)
    top = results[:a.validate]
    held = []
    with ThreadPoolExecutor(a.parallel) as ex:
        offs = [int(x) for x in a.choose_offsets.split(",") if x.strip()]
        if 200 in offs:
            raise SystemExit("+200 is the report set; it must not be used for choosing")
        jobs = [(r, off) for r in top for off in offs]
        for (r, off), v in zip(jobs, ex.map(lambda ro: run_candidate(ro[0]["k"], ro[0]["overrides"],
                                                                    a.workers, out / f"heldout{ro[1]}", ro[1]), jobs)):
            held.append({"k": r["k"], "offset": off, "passed": v["passed"], "failed": v["failed"]})
    for r in top:
        hv = [h for h in held if h["k"] == r["k"]]
        r["heldout_min_passed"] = min([r["passed"]] + [h["passed"] for h in hv])
        r["heldout_failed"] = sorted(set(f for h in hv for f in h["failed"]) | set(r["failed"]))
        # required tests failed on ANY seed set (the exempt ones reported, not ranked)
        r["required_failed"] = sorted((set(r["heldout_failed"]) & FC_TESTS) - exempt)
    top.sort(key=lambda r: (len(r["required_failed"]), -r["fc_score"]))
    (out / "results_heldout.json").write_text(json.dumps(top, indent=1, default=float))
    for r in top:
        print(f"held-out: required failed {r['required_failed']} | worst {r['heldout_min_passed']}/{r['of']}, any-seed fails {r['heldout_failed']} {r['overrides']}", flush=True)
    print("\nbest (held-out):", json.dumps(top[0]["overrides"]), top[0]["heldout_min_passed"])


if __name__ == "__main__":
    main()
