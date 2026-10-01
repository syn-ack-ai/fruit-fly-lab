"""robot/dashboard.py: the brain client shares telemetry and spike counts
through memory; the server's reader computes the brain map itself."""
import time

import numpy as np

import robot.dashboard as D
from robot.brainmap import BrainActivity
from tests.test_brainmap import C


def test_share_to_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "STATIC_DIR", str(tmp_path))
    share = D.DashboardShare(n_counts=5, key="k", start_server=False)
    try:
        bm = BrainActivity(C())
        share.put_static(bm.geometry(), bm.circuit())
        share.set_head(None)
        state = {"telemetry": None, "tv": 0, "jpeg": None, "jv": 0, "geometry": None, "gv": 0,
                 "activity": None, "av": 0, "flow": None, "fv": 0, "circuit": None}
        r = D.BrainReader(share.shm.name, state)
        share.publish({"t": 1.0, "x": float("nan"), "n": np.int64(3)}, np.zeros(5, np.int32))
        r._poll()
        assert state["geometry"].startswith(b"FLYG") and state["circuit"] is not None
        assert state["telemetry"] == {"t": 1.0, "x": None, "n": 3}         # NaN cleaned for the browser
        assert state["activity"] is None                                    # one snapshot: no rate yet
        tv = state["tv"]
        r._poll()
        assert state["tv"] == tv                                            # nothing new: nothing redone
        time.sleep(0.06)
        share.publish({"t": 2.0}, np.array([0, 20, 0, 0, 0], np.int32))
        r._poll()
        sysrates = {s["name"]: s["hz"] for s in state["telemetry"]["activity"]["systems"]}
        assert sysrates["Camera & lidar inputs (LC4, LPLC2, LC10a)"] > 0
        pairs = np.frombuffer(state["flow"], np.int32).reshape(-1, 2)
        assert [tuple(map(int, p)) for p in pairs] == [(1, 2), (1, 3)]      # as BrainActivity's own flow
        assert state["jpeg"] is None
    finally:
        share.close()
