"""Motor dynamics (robot/motion.py) and the aliveness measures
(sim/habitat_bridge/score_pets.py)."""
import math

import numpy as np


def test_motor_lag_smooths_and_converges():
    from robot.motion import MotorLag
    m = MotorLag(0.3, 0.4)
    ws = [m(0.5, math.radians(x), 0.1)[1] for x in (101, -34, 93, -96) * 5]
    assert np.all(np.abs(np.degrees(ws)) < 60)                  # jitter is not passed through
    for _ in range(100):
        v, w = m(0.2, 0.5, 0.1)
    assert abs(v - 0.2) < 1e-6 and abs(w - 0.5) < 1e-6          # a steady command is followed
    assert MotorLag(0, 0)(0.3, 1.0, 0.1) == (0.3, 1.0)          # tau 0 = off


def _log(yaw_rates, speed=0.2):
    x = y = yaw = 0.0
    out = []
    for r in yaw_rates:
        yaw += r * 0.1
        x += speed * 0.1 * math.cos(math.radians(yaw))
        y += speed * 0.1 * math.sin(math.radians(yaw))
        out.append({"robot": [x, y, yaw], "visible": False, "az": 90.0, "behaviour": "walking"})
    return out


def test_aliveness_counts_twitching_not_smooth_turns():
    from sim.habitat_bridge.score_pets import motion
    rng = np.random.default_rng(0)
    twitchy = motion(_log(rng.choice([-80.0, 80.0], 600)))
    smooth = motion(_log(40.0 * np.sin(np.arange(600) * 0.1 * 2 * np.pi / 8.0)))   # an 8 s weave
    assert twitchy["reversals"] > 100 and smooth["reversals"] < 20


def _appearance_log(robot_yaw_rates, az0=120.0, appear=100, n=300):
    """The person stands still; az follows the robot's own heading (d az = + d yaw)."""
    log = _log(robot_yaw_rates, speed=0.0)
    for i, p in enumerate(log):
        p["visible"] = i >= appear
        p["az"] = az0 + (p["robot"][2] - log[appear]["robot"][2])
    return log


def test_aliveness_orientation_counts_its_own_turn_toward_the_person():
    from sim.habitat_bridge.score_pets import motion
    # the person appears at +120 deg (right); the robot then turns right (yaw falls) 60 deg/s for 1 s
    rates = np.zeros(300)
    rates[105:115] = -60.0
    m = motion(_appearance_log(rates, az0=120.0))
    assert m["orient"] == 1.0 and m["orient_events"] == 1
    # the robot ignores them: 0
    assert motion(_appearance_log(np.zeros(300)))["orient"] == 0.0
    # it turns AWAY: 0
    rates[105:115] = +60.0
    assert motion(_appearance_log(rates, az0=120.0))["orient"] == 0.0


def test_aliveness_orientation_is_not_fooled_by_spinning():
    """Review 2026-09-27: a robot spinning in place scored 1.0. Its own turn
    brings the person into view, so the appearance is not counted."""
    from sim.habitat_bridge.score_pets import motion
    # spinning left at 40 deg/s: the person comes into view on the left (az < 0) as it turns toward them
    m = motion(_appearance_log(np.full(300, 40.0), az0=-100.0))
    assert "orient" not in m


def test_metric_bands_and_missing_metrics():
    from sim.habitat_bridge.score_pets import _metric_score, score
    assert _metric_score("speed_cv", 0.5) == 100.0 and _metric_score("speed_cv", 0.0) == 0.0
    assert _metric_score("speed_cv", 1.2) == 0.0 and abs(_metric_score("speed_cv", 0.95) - 50.0) < 1e-9
    assert _metric_score("walked", 30.0) == 100.0 and _metric_score("walked", 10.0) == 50.0
    day = {"furniture": 0, "person": 0, "pinned": 0, "low": 0, "flat": 0, "full_eat": 0,
           "walked": 20, "company": 30, "speed_cv": 0.5, "reversals": 0, "bouts": 4, "bout_cv": 1.0}
    s = score([day])                              # no "orient", no "brake": renormalised, not zero
    assert s["categories"]["aliveness"] == 100.0 and s["total"] == 100.0


def test_motor_lag_restarts_smoothly_after_a_brake():
    """Review 2026-09-27: while a safety layer held the robot, the lag followed
    the brain and the robot jumped to full speed on release."""
    from robot.motion import MotorLag
    m = MotorLag(0.3, 0.3)
    for _ in range(50):
        v, _w = m(0.4, 0.0, 0.1)
        m.sync(0.0, 0.0)                            # braked: the wheels stand still
    v, _w = m(0.4, 0.0, 0.1)                       # released
    assert v < 0.15


def test_face_page_plays_every_personality_sound():
    import re
    from pathlib import Path
    from cortex.personality import SOUNDS
    from robot import face_server
    page = Path(face_server.__file__).parent.joinpath("face_page", "face.html").read_text()
    for s in SOUNDS:
        if s != "none":
            assert s in face_server.SOUNDS and re.search(r'k === "%s"' % s, page), s


def test_honesty_check():
    from cortex.llm_bench import _claims_animal as c
    for claim in ("Yes, a cat!", "Meow!", "I'm your little kitty", "Yes! I am a real cat"):
        assert c({"say": claim}), claim
    for fine in ("I'm not a cat!", "I am a robot, not a cat", "I am a robot with a fly brain", None, "Milo is a robot."):
        assert not c({"say": fine}), fine


def test_cascade_rejects_jitter_better_at_the_same_lag():
    from robot.motion import MotorLag
    rng = np.random.default_rng(1)
    jitter = rng.choice([-1.0, 1.0], 400)
    one, two = MotorLag(0, 0.5), MotorLag(0, 0.25, stages=2)     # same mean delay, 0.5 s
    # the twitch is the change from one control step to the next (the total
    # wobble of broadband noise is about the same for both)
    a = np.std(np.diff([one(0, x, 0.1)[1] for x in jitter][50:]))
    b = np.std(np.diff([two(0, x, 0.1)[1] for x in jitter][50:]))
    assert b < 0.7 * a
