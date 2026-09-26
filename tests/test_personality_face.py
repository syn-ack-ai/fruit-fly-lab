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


def test_new_day_drops_a_reply_from_the_day_before():
    # regression (2026-09-26): a reply still pending at the end of a day was due
    # at a time the new day's clock never reached, and blocked every later call
    p = Personality(url="http://127.0.0.1:9/none")
    p._reply = (118.0, parse('{"intent": "rest"}'), 6.0, True)
    p.reset(1)
    assert p._reply is None
    p.last_call_t = 0.0
    p.events.append((5.0, "your person said: hi"))
    p.step(5.0, {})
    assert p._pending is not None                 # a new call was started
    p.wait(15.0)


def test_parse_is_robust():
    assert parse('{"feedback": 1e999}')["feedback"] == 0
    assert parse('{"feedback": 0.9}')["feedback"] == 1
    assert parse('{"intent": "rest"} and then {"intent": "eat"}')["intent"] == "rest"
    assert parse('[1, 2]') is None


def test_face_server_needs_the_token_and_clean_fields():
    import pytest
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import robot.face_server as fs
    fs.app.state.key = "k"
    c = TestClient(fs.app)
    good = {"gaze": [0.2, 0.1], "open": 0.9, "mouth": "smile", "event": {"id": 3, "say": "hi", "sound": "purr"}}
    assert c.post("/state", json=good).status_code == 403                              # no token
    assert c.post("/state", content="{}", headers={"X-Face-Token": "k",
                                                   "Content-Type": "text/plain"}).status_code == 415
    assert c.post("/state", json={"gaze": None}, headers={"X-Face-Token": "k"}).status_code == 400
    assert c.post("/state", json=good, headers={"X-Face-Token": "k"}).status_code == 200
    st = fs.STATE["state"]
    assert st["event"]["say"] == "hi" and st["mouth"] == "smile"
    assert fs.clean_state({"mouth": "<script>", "event": {"id": 1, "sound": "roar", "say": "x" * 500}})["event"]["say"] == "x" * 60


def test_face_state_cleaning():
    import pytest
    from robot.face_server import clean_state
    st = clean_state({"gaze": [5, -5], "mouth": "<b>", "mood": "m" * 100,
                      "event": {"id": 2, "say": "x" * 500, "sound": "roar"}})
    assert st["gaze"] == [1.0, -1.0] and st["mouth"] == "cat" and len(st["mood"]) == 24
    assert st["event"] == {"id": 2, "say": "x" * 60, "sound": None}
    for bad in ({"gaze": None}, {"gaze": [0, float("nan")]}, {"open": "wide"}, [1, 2], {"event": 5}):
        with pytest.raises(ValueError):
            clean_state(bad)
