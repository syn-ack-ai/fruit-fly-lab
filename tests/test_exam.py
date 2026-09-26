"""
The fly exam (cognition/exam): catalogue integrity, scoring, and one live
test on the whole-brain model (bitter taste must not drive MN9).
"""
from __future__ import annotations

import math

import pytest

from native import lif_native


def test_catalogue_entries_are_complete():
    from cognition.exam import tests
    names = [t["name"] for t in tests.TESTS]
    assert len(names) == len(set(names))
    for t in tests.TESTS:
        assert callable(t["fn"]) and t["cite"] and t["target"]
        assert t["lo"] <= t["hi"]
    assert set(tests.CORE) <= set(names)


def test_scoring_is_one_inside_the_band_and_decays_outside():
    from cognition.exam import tests
    t = dict(lo=1.0, hi=2.0, scale=1.0)
    assert tests.score(t, 1.5) == (True, 1.0)
    ok, s = tests.score(t, 2.5)
    assert not ok and math.isclose(s, 0.5)
    assert tests.score(t, 10.0) == (False, 0.0)
    assert tests.score(t, float("nan")) == (False, 0.0)


@pytest.mark.skipif(not lif_native.available(), reason="native engine not built")
def test_bitter_does_not_drive_mn9_in_the_exam():
    from cognition.exam import core, tests
    ctx = core.Ctx({"dynamics": "calibrated"})
    r = tests.bitter_mn9(ctx, quick=True)
    ok, _ = tests.score(tests.BY_NAME["bitter_no_MN9"], r["value"])
    assert ok
