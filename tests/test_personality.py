"""The LLM personality layer's reply parsing and latency, and the scripted
person (no language model needed)."""
from cortex.personality import Personality, parse
from sim.habitat_bridge.speech import ScriptedPerson


def test_parse_keeps_only_allowed_values():
    r = parse('sure! {"intent": "eat", "sound": "whirr", "say": "yum", "mood": "happy", "feedback": 3, "note": null}')
    assert r == {"intent": "eat", "sound": "whirr", "say": "yum", "mood": "happy", "feedback": 1, "fear": 0, "note": None}
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


def test_scripted_person_praises_an_answered_call():
    p = ScriptedPerson(name="Milo", seed=0)
    p.call_t = 5.0
    far = {"dist": 3.0, "robot": [0, 0, 0], "human": [3, 0]}
    near = {"dist": 1.0, "robot": [2, 0, 0], "human": [3, 0]}
    assert p.step(6.0, 0.1, far, far, False, 0.0) is None or p.call_t is not None
    assert p.step(9.0, 0.1, near, near, False, 0.0) == "good Milo!"
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
