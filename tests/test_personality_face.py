"""The LLM personality layer's reply parsing and latency, the face model, and
the scripted person (no language model or server needed)."""
from cortex.personality import Personality, parse
from robot.face import FaceModel
from sim.habitat_bridge.speech import ScriptedPerson


def test_parse_keeps_only_allowed_values():
    r = parse('sure! {"intent": "eat", "sound": "purr", "say": "yum", "mood": "happy", "feedback": 3, "note": null}')
    assert r == {"intent": "eat", "sound": "purr", "say": "yum", "mood": "happy", "feedback": 1, "note": None}
    r = parse('{"intent": "drive_to_kitchen_at_full_speed", "sound": "roar", "mood": "evil"}')
    assert r["intent"] == "none" and r["sound"] == "none" and r["mood"] == "calm"
    assert parse("no json here") is None


def test_reply_waits_for_its_latency_in_simulated_time():
    p = Personality(url="http://127.0.0.1:9/none")
    p.last_call_t = 0.0
    p._reply = (10.0, parse('{"intent": "rest", "mood": "sleepy"}'), 2.0, True)
    assert p.step(11.0, {})["new"] is False           # asked at 10 s, takes 2 s
    out = p.step(12.0, {})
    assert out["new"] and out["intent"] == "rest"
    assert p.take_fresh()["mood"] == "sleepy" and p.take_fresh() is None


def test_face_follows_brain_state():
    f = FaceModel()
    s = f.update(0.1, {"person_visible": True, "person_az": 30, "person_el": 0})
    assert s["gaze"][0] == 0.5
    s = f.update(0.1, {"startle": 1.0})
    assert s["open"] == 1.0 and s["mouth"] == "o" and s["pupil"] > 0.9
    s = f.update(0.1, {"eating": True, "mood": "happy"})
    assert s["mouth"] == "chew"
    e0 = s["event"]["id"]
    assert f.update(0.1, {"say": "hi", "sound": "trill"})["event"]["id"] == e0 + 1
    assert f.update(0.1, {})["event"]["id"] == e0 + 1   # an event is not repeated


def test_scripted_person_praises_an_answered_call():
    p = ScriptedPerson(name="Mote", seed=0)
    p.call_t = 5.0
    far = {"dist": 3.0, "robot": [0, 0, 0], "human": [3, 0]}
    near = {"dist": 1.0, "robot": [2, 0, 0], "human": [3, 0]}
    assert p.step(6.0, 0.1, far, far, False, 0.0) is None or p.call_t is not None
    assert p.step(9.0, 0.1, near, near, False, 0.0) == "good Mote!"
    assert p.summary()["answered"] == 1
    # the pet walking into a standing person is scolded
    a = {"dist": 0.7, "robot": [0, 0, 0], "human": [0.7, 0]}
    assert p.step(10.0, 0.1, a, a, True, 0.1) == "ouch, careful!"
