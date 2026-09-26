"""
Run the fly exam.

    python -m cognition.exam --dynamics calibrated --quick --out simulation/outputs/exam
    python -m cognition.exam --dynamics published --no-robustness
    python -m cognition.exam --tests sugar_activates_MN9,pursuit_sign --wiring shuffled

Writes <out>/exam_<dynamics>_<wiring>[_quick].json and a text report card.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from cognition.exam.worker import run_chunk as _chunk

ROBUSTNESS = {
    "wsyn":        {"quick": [0.7, 1.3], "full": [0.7, 0.85, 1.15, 1.3]},
    "edge_drop":   {"quick": [0.1], "full": [0.05, 0.1, 0.2]},
    "nt_flip":     {"quick": [0.07], "full": [0.03, 0.07, 0.13]},
    "lesion":      {"quick": [0.03], "full": [0.01, 0.03, 0.05]},
    "sensor_drop": {"quick": [0.5], "full": [0.2, 0.5, 0.8]},
    "sensor_gain": {"quick": [0.5], "full": [0.3, 0.5, 2.0]},
}


def _jobs(cfg, names, quick, per_chunk):
    return [(cfg, names[i:i + per_chunk], quick) for i in range(0, len(names), per_chunk)]


def main():
    ap = argparse.ArgumentParser(description="fly exam")
    ap.add_argument("--dynamics", default="calibrated")
    ap.add_argument("--wiring", default="real")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tests", default=None, help="comma-separated subset")
    ap.add_argument("--no-robustness", action="store_true")
    ap.add_argument("--controls", action="store_true", help="also run the tests on shuffled wiring")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="simulation/outputs/exam")
    a = ap.parse_args()
    from cognition.exam import tests
    names = a.tests.split(",") if a.tests else [t["name"] for t in tests.TESTS if (t["quick"] or not a.quick)]
    base = {"dynamics": a.dynamics, "wiring": a.wiring}
    jobs = _jobs(base, names, a.quick, 3)
    if a.controls:
        jobs += _jobs({**base, "wiring": "shuffled"}, names, a.quick, 3)
    if not a.no_robustness:
        for kind, lv in ROBUSTNESS.items():
            for x in lv["quick" if a.quick else "full"]:
                jobs += _jobs({**base, kind: x}, tests.CORE, a.quick, len(tests.CORE))
    t0 = time.time()
    results = []
    with get_context("spawn").Pool(a.workers) as pool:
        for out, secs in pool.imap_unordered(_chunk, jobs):
            results += out
            for r in out:
                print(f"{r['test']:28s} {'PASS' if r['pass'] else 'fail'} {r.get('value')!s:>10.10} "
                      f"{ {k: v for k, v in r['cfg'].items() if k not in ('dynamics',)} }", flush=True)
    card = report(results, base, names, a.quick)
    card["wall_s"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"exam_{a.dynamics}_{a.wiring}{'_quick' if a.quick else ''}"
    (out / (stem + ".json")).write_text(json.dumps({"card": card, "results": results}, indent=1, default=str))
    (out / (stem + ".txt")).write_text(card["text"])
    print(card["text"])


def report(results, base, names, quick):
    from cognition.exam import tests
    main_ = [r for r in results if r["cfg"] == base]
    ctrl = [r for r in results if r["cfg"] == {**base, "wiring": "shuffled"}]
    lines = [f"FLY EXAM  dynamics={base['dynamics']} wiring={base['wiring']}{' (quick)' if quick else ''}", ""]
    groups = {}
    for r in sorted(main_, key=lambda r: names.index(r["test"])):
        te = tests.BY_NAME[r["test"]]
        groups.setdefault(te["group"], []).append(r)
        val = r.get("value")
        vs = "ERROR" if r.get("error") else (f"{val:.3g}" if isinstance(val, (int, float)) and val == val else str(val))
        c = next((x for x in ctrl if x["test"] == r["test"]), None)
        cs = ""
        if c is not None:
            cv = c.get("value")
            cs = f"  | shuffled {'PASS' if c['pass'] else 'fail'} {cv:.3g}" if isinstance(cv, (int, float)) and cv == cv else "  | shuffled -"
        lines.append(f"  [{'x' if r['pass'] else ' '}] {r['test']:28s} {vs:>9s}  target: {te['target']}{cs}")
    card = {"groups": {}, "overall": None}
    lines.append("")
    for gname, rs in groups.items():
        p = sum(r["pass"] for r in rs)
        card["groups"][gname] = {"passed": p, "of": len(rs), "score": round(float(np.mean([r["score"] for r in rs])), 3)}
        lines.append(f"  {gname:9s} {p}/{len(rs)} passed, score {card['groups'][gname]['score']:.2f}")
    if main_:
        card["overall"] = round(float(np.mean([r["score"] for r in main_])), 3)
        card["passed"] = int(sum(r["pass"] for r in main_))
        card["of"] = len(main_)
        lines.append(f"  OVERALL   {card['passed']}/{card['of']} passed, score {card['overall']:.2f}")
    if ctrl:
        card["shuffled"] = {"passed": int(sum(r["pass"] for r in ctrl)), "of": len(ctrl),
                            "score": round(float(np.mean([r["score"] for r in ctrl])), 3)}
        lines.append(f"  SHUFFLED  {card['shuffled']['passed']}/{card['shuffled']['of']} passed, score {card['shuffled']['score']:.2f}")
    # robustness: pass rate of the core tests per perturbation level; AUC = mean over levels
    rob = {}
    for r in results:
        cfg = r["cfg"]
        pert = [k for k in ROBUSTNESS if k in cfg]
        if not pert or r["test"] not in tests.CORE:
            continue
        k = pert[0]
        rob.setdefault(k, {}).setdefault(cfg[k], []).append(r["pass"])
    if rob:
        lines.append("")
        lines.append("  ROBUSTNESS (core tests pass rate per level; AUC = mean)")
        card["robustness"] = {}
        for k, lv in rob.items():
            curve = {str(x): round(float(np.mean(v)), 2) for x, v in sorted(lv.items())}
            auc = round(float(np.mean(list(curve.values()))), 2)
            card["robustness"][k] = {"curve": curve, "auc": auc}
            lines.append(f"  {k:12s} AUC {auc:.2f}  {curve}")
        card["robustness_auc"] = round(float(np.mean([v["auc"] for v in card["robustness"].values()])), 2)
        lines.append(f"  mean robustness AUC {card['robustness_auc']:.2f}")
    errs = [r for r in results if r.get("error")]
    if errs:
        lines.append("")
        lines.append(f"  {len(errs)} tests raised errors, e.g. {errs[0]['test']}: {errs[0]['error']}")
    card["text"] = "\n".join(lines)
    return card


if __name__ == "__main__":
    main()
