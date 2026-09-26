"""Pool worker for the fly exam (importable, so spawn-started workers can find it)."""
from __future__ import annotations

import os
import time


def run_chunk(job):
    cfg, names, quick = job
    os.environ.setdefault("FLY_THREADS", "1")
    from cognition.exam import core, tests
    t0 = time.time()
    ctx = core.Ctx(cfg)
    out = []
    for nm in names:
        te = tests.BY_NAME[nm]
        t1 = time.time()
        try:
            r = te["fn"](ctx, quick)
            ok, sc = tests.score(te, r["value"])
            out.append({"cfg": cfg, "test": nm, "pass": ok, "score": sc, **r,
                        "seconds": round(time.time() - t1, 1)})
        except Exception as ex:                          # keep the exam going
            import traceback
            out.append({"cfg": cfg, "test": nm, "pass": False, "score": 0.0, "value": None,
                        "error": repr(ex), "trace": traceback.format_exc()[-600:]})
    return out, round(time.time() - t0, 1)


