"""Milo's voice from the fly brain (2026-09-28): excitement -> P1 -> the song
command pIP10 -> the robot's "song"; the personality speaks on a vocal urge
and describes the fly brain's live state."""
import numpy as np
import pytest

import config


def test_song_channel_exists_on_male_brains_only():
    from brain.neurons.registry import load_connectome
    from brain.motor.descending import DescendingReadout
    from cortex.topdown import p1_indices
    c = load_connectome()
    L, R = DescendingReadout(c)._chan_groups["song"]
    if config.MALE_CNS:
        assert L and R and len(p1_indices(c)) > 50
    else:
        assert not L and not R and len(p1_indices(c)) == 0     # P1 and pIP10 are male


def test_excite_drives_p1_at_its_level():
    from brain.neurons.registry import load_connectome
    from cortex.topdown import ExciteEncoder, P1_MAX_HZ
    e = ExciteEncoder(load_connectome())
    e.set(0.5)
    assert np.all(e.rates_hz() == 0.5 * P1_MAX_HZ)
    e.set(3.0)
    assert np.all(e.rates_hz() == P1_MAX_HZ)                    # clipped


def _obs(vis):
    return {"robot": [0.0, 0.0, 0.0], "visible": vis, "az": 10.0, "half": 5.0, "human": [1.0, 1.0], "dist": 2.0}


def test_excitement_rises_when_the_person_reappears_and_decays():
    from cortex.v0 import CortexV0, EXCITE_TAU_S
    cx = CortexV0(seed=0, manners=True, naps=True, orient=True, voice=True)     # as the pet runs
    cx.reset(0, None, None)
    cx.social = 1.0
    ex = [cx.act(_obs(k >= 20 and k < 22), None, 0.1 * k, None).get("excite", 0.0) for k in range(20 + int(3 * EXCITE_TAU_S / 0.1))]
    assert max(ex[:20]) == 0.0 and ex[20] > 0.4                  # reappearing after > 1 s away
    assert ex[-1] < 0.1 * ex[20]                                  # decays
    import types
    meal = types.SimpleNamespace(battery=types.SimpleNamespace(meal=True, sense_hunger=1.0, hunger=0.5, soc=0.5),
                                 taste=None, stats={"pets": 0, "treats": 0}, conc={"L": np.zeros(1), "R": np.zeros(1)},
                                 petting=False, hunger=0.5)
    cx.reset(1, None, None)
    cx.social = 1.0
    ex = [cx.act(_obs(k >= 5), meal, 0.1 * k, None)["excite"] for k in range(10)]
    assert max(ex) == 0.0 and cx.excite > 0.4                     # excited, but quiet mid-meal


def test_song_bouts_need_duration_and_spacing():
    from sim.habitat_bridge.brain_client import _song, SONG_GAP_S
    s = {"bouts": 0, "s": 0.0, "on": False, "off_t": -1e9, "onset": False, "peak_hz": 0.0, "start_t": 0.0, "counted": False}
    def run(seq, t0):
        on = []
        for k, act in enumerate(seq):
            _song(s, {"channels": {"song": act, "hz_pIP10": 100 * act}}, t0 + 0.1 * k, 0.1)
            on.append(s["onset"])
        return on
    assert not any(run([0, 0.5, 0, 0.6, 0], 0.0))                  # 100 ms blips: no song
    on = run([0.5, 0.2, 0.4, 0.1, 0], 1.0)                        # 0.3 s held (0.2 keeps it on): one bout
    assert on.count(True) == 1 and s["bouts"] == 1
    assert s["s"] == pytest.approx(0.3)
    assert not any(run([0.5] * 5 + [0], 1.6))                     # too soon after: merged, no new song
    assert s["s"] == pytest.approx(0.8)                           # ... but its time counts (review)
    assert not any(run([0, 0.5, 0], 2.5))                         # a blip soon after: nothing (review)
    assert s["s"] == pytest.approx(0.8) and s["bouts"] == 1
    assert run([0] + [0.5] * 5, 1.6 + SONG_GAP_S + 1.0).count(True) == 1 and s["bouts"] == 2
    assert s["s"] == pytest.approx(1.3)


def test_words_need_a_vocal_urge_when_the_call_is_made():
    """Judged at ASK time: the reply lands after the model's latency (review
    2026-09-28: an urge arriving while a call was in flight unlocked the wrong
    reply and gated the right one)."""
    from cortex.personality import Personality
    p = Personality(url="http://127.0.0.1:9/none", urge_gate=True)
    r = lambda: {"intent": "none", "sound": "beep", "say": "hello", "mood": "happy", "feedback": 0, "note": None}
    p._reply = (9.0, r(), 3.0, True, False)            # asked at 9 with no urge; lands at 12
    p.event(10.0, "your person petted you", urge=True)  # an urge while it was in flight
    out = p.step(12.0, {})
    assert out["say"] is None and out["gated_say"] == "hello" and p.stats["gated"] == 1
    p._reply = (12.0, r(), 5.0, True, True)            # the call asked after the urge
    assert p.step(17.0, {})["say"] == "hello"          # said, although 7 s after the urge
    asked = []
    p._pending = None
    p._ask = lambda t, msg, gen, urge=True: asked.append(urge)
    p.step(50.0, {})                                    # an idle call (IDLE_S), long after the urge
    assert asked == [False]


def test_brain_readout_says_what_fires():
    from sim.habitat_bridge.brain_client import brain_readout
    fr = {"channels": {"escape_takeoff": 0.6, "hz_pIP10": 80.0, "song": 0.6, "turn_bias": -0.2, "forward_walk": 0.5}}
    txt = brain_readout(fr)
    assert "escape neurons FIRING" in txt and "pIP10 80 Hz (singing)" in txt and "pull left" in txt and "strong" in txt


def test_brain_readout_without_song_neurons_and_with_a_surprise():
    from sim.habitat_bridge.brain_client import brain_readout
    import types
    txt = brain_readout({"channels": {"escape_takeoff": 0.0, "turn_bias": 0.0}},    # FAFB: no pIP10
                        types.SimpleNamespace(last_rpe=0.8))
    assert "pIP10" not in txt and "quiet" in txt and "better than expected" in txt


def test_excite_channel_attaches_only_with_p1():
    from brain.neurons.registry import load_connectome
    from cortex.topdown import TopDown
    c = load_connectome()
    class Obj:
        indices = np.empty(0, np.int64); _az = _el = _sigma = np.empty(0); MAX_HZ = 150.0
    added = []
    ses = type("S", (), {"add_stimulus": lambda self, a, b: added.append(a)})()
    td = TopDown(c, Obj(), channels=("excite",))
    td.attach(ses)
    assert (td.excite in added) == bool(config.MALE_CNS)
    assert len(TopDown(c, Obj(), channels=("goal",)).excite.indices) == 0          # not loaded when unused


def test_session_reports_the_song_mean_over_the_step():
    """Review 2026-09-28: one 50 ms window of one cell per side moves in
    steps of a spike; the step mean over every 1 ms block is smooth."""
    import os
    os.environ.setdefault("FLY_DYNAMICS", "calibrated")
    from sim.habitat_bridge.brain_client import build_brain
    if not config.MALE_CNS:
        pytest.skip("song is male")
    ses, *_ = build_brain(1, topdown=("excite",))
    ses.topdown.attach(ses)
    means = []
    for _ in range(15):
        ses.topdown.apply(0.0, {"excite": 0.6})
        fr = ses.advance(100.0)[-1]
        means.append(round(fr["step_means"]["song"], 3))
    assert len(set(means)) > 6 and max(means) > 0.2


def test_excitement_fades_close_to_the_person():
    """P1 also drives courtship pursuit: close up, excitement must not drive it
    (2026-09-28: excitement held by greeting made the robot chase into legs)."""
    import math
    from cortex.v0 import CortexV0
    from robot.safety import person_distance
    half_near = next(h for h in np.arange(1.0, 90.0, 0.5) if person_distance(h) < 0.25)
    def run(half):
        cx = CortexV0(seed=0, manners=True, naps=True, orient=True, voice=True)
        cx.reset(0, None, None)
        cx.social = 1.0
        obs = lambda vis: {"robot": [0.0, 0.0, 0.0], "visible": vis, "az": 0.0, "half": half, "human": [0.0, 0.0], "dist": 1.0}
        return [cx.act(obs(k >= 20), None, 0.1 * k, None)["excite"] for k in range(26)][20:]
    assert max(run(half_near)) == 0.0                       # right beside the person: no P1 drive
    assert max(run(2.0)) > 0.3                              # a few metres away: sings


def test_stuck_metric_matches_the_unstick_definition():
    from sim.habitat_bridge.score_pets import stuck_s
    def log(xs, clear=0.05):
        return [{"t": 0.1 * k, "robot": [x, 0.0, 0.0], "lidar": {"min_clear": clear}} for k, x in enumerate(xs)]
    n = 150                                                   # 15 s
    assert stuck_s(log([0.0] * n)) >= 9.0                     # pressed on a wall, still
    shuttle = [0.1 * abs(((k % 30) / 15.0) - 1.0) for k in range(n)]        # 0-0.1 m back and forth
    assert stuck_s(log([0.02 * v for v in shuttle])) >= 9.0   # tiny shuttling: still stuck
    assert stuck_s(log([0.02 * k for k in range(n)])) == 0.0  # driving along the wall: not stuck
    assert stuck_s(log([0.0] * n, clear=0.5)) == 0.0          # standing in the open: not stuck
    assert stuck_s([{"t": 0.1 * k, "robot": [0, 0, 0]} for k in range(n)]) is None   # no lidar: n/a


def test_navigation_goals_skip_the_pursuit_neurons_on_male_brains(monkeypatch):
    """2026-09-28: in the male CNS the pursuit neurons (LC10a) made the robot
    walk backward (MDN); navigation goals go through the central complex only.
    Explicit attention (the orienting / unstick reflexes) still applies."""
    from brain.neurons.registry import load_connectome
    from cortex.topdown import TopDown
    class Obj:
        indices = np.empty(0, np.int64); _az = _el = _sigma = np.empty(0); MAX_HZ = 150.0; arousal = 1.0
    monkeypatch.delenv("FLY_ATTEND_FROM_GOAL", raising=False)
    td = TopDown(load_connectome(), Obj(), channels=("goal", "attend"))
    assert td.attend_from_goal == (not config.MALE_CNS)
    td.apply(90.0, {"goal_deg": 45.0, "goal_gain": 1.0})
    assert (td.attend.gain > 0) == (not config.MALE_CNS)
    td.apply(90.0, {"goal_deg": 45.0, "goal_gain": 1.0, "attend_az": 45.0, "attend_gain": 0.8})   # e.g. avoid's bend
    assert (td.attend.gain > 0) == (not config.MALE_CNS)
    td.apply(90.0, {"attend_az": 30.0, "attend_gain": 1.0, "attend_explicit": True})
    assert td.attend.gain == 1.0 and td.attend.az == 30.0


def test_body_walks_at_a_flys_pace_on_its_own_brains_resting_rate():
    """2026-09-28: the body's DNg100 -> speed constant was FAFB's resting rate;
    the male brain rests at ~3 Hz and walked at a quarter of a fly's pace."""
    import json
    import fly.body.foraging_body as fb
    p = config.METADATA_DIR / f"body_readout_{config.DATASET_KEY}.json"
    if p.exists():
        assert fb.DNG100_REST_HZ == json.loads(p.read_text())["dng100_rest_hz"]
        if config.MALE_CNS:
            assert 2.0 < fb.DNG100_REST_HZ < 8.0
    else:
        assert fb.DNG100_REST_HZ == 14.6                     # FAFB's constant (FAFB unchanged)
    assert abs(fb.DNG100_MM_PER_HZ * fb.DNG100_REST_HZ - fb.WALK_SPEED_MM_S) < 1e-9
