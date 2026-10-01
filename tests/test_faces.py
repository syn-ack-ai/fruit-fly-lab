"""Faces and people (robot/faces.py, robot/people.py) without a camera or GPU,
and the face page's question and answer (robot/face_server.py)."""
import math

import numpy as np
import pytest

from robot import faces as F
from robot import people as P


def _unit(seed):
    v = np.random.default_rng(seed).normal(size=128).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(v, seed, eps=0.3):
    w = v + eps * _unit(seed)
    return (w / np.linalg.norm(w)).astype(np.float32)


# ------------------------------------------------------------------ geometry
def test_facing_from_landmarks():
    frontal = F.TEMPLATE                       # eyes, nose, mouth corners of a frontal face
    yaw, pitch, ok = F.facing(frontal)
    assert abs(yaw) < 0.02 and 0.4 < pitch < 0.6 and ok
    turned = F.TEMPLATE.copy()
    turned[2, 0] += 0.45 * 35.2                # the nose far towards one eye: the head turned
    assert not F.facing(turned)[2] and F.facing(turned)[0] > F.FACING_YAW
    up = F.TEMPLATE.copy()
    up[2, 1] = 52.0                            # the nose up at the eye line: looking up / away
    assert not F.facing(up)[2]


def test_similarity_transform_undoes_rotation_scale_and_shift():
    a = math.radians(25)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    src = (F.TEMPLATE - 56) @ R.T * 2.5 + [300, 140]       # the template, moved in a frame
    M = F.similarity_transform(src)
    back = src @ M[:, :2].T + M[:, 2]
    assert np.allclose(back, F.TEMPLATE, atol=1e-9)
    assert np.allclose(F.similarity_transform(F.TEMPLATE), [[1, 0, 0], [0, 1, 0]], atol=1e-12)


def test_yunet_decode_one_face():
    H = W = 64
    outs = {}
    for st in F.STRIDES:
        n = (H // st) * (W // st)
        outs.update({f"cls_{st}": np.zeros((1, n, 1)), f"obj_{st}": np.zeros((1, n, 1)),
                     f"bbox_{st}": np.zeros((1, n, 4)), f"kps_{st}": np.zeros((1, n, 10))})
    # stride 8, cell (col 3, row 2): centre (3 + 0.5, 2 + 0.25) * 8, size exp(ln 2) * 8 = 16
    i = 2 * 8 + 3
    outs["cls_8"][0, i], outs["obj_8"][0, i] = 0.81, 1.0
    outs["bbox_8"][0, i] = [0.5, 0.25, math.log(2), math.log(2)]
    outs["kps_8"][0, i] = [0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0]
    # a duplicate next to it (lower score) and a sub-threshold cell
    outs["cls_8"][0, i + 1], outs["obj_8"][0, i + 1] = 0.64, 1.0
    outs["bbox_8"][0, i + 1] = [-0.5, 0.25, math.log(2), math.log(2)]
    outs["cls_16"][0, 0], outs["obj_16"][0, 0] = 0.2, 0.2
    f = F.yunet_decode(outs, (H, W))
    assert f.shape == (1, 15)
    x0, y0, w, h = f[0, :4]
    assert (x0, y0, w, h) == pytest.approx((28 - 8, 18 - 8, 16, 16))
    assert f[0, 14] == pytest.approx(0.9)                  # sqrt(cls x obj)
    assert f[0, 4:6] == pytest.approx((24, 16)) and f[0, 6:8] == pytest.approx((32, 16))


# ------------------------------------------------------------------ names
@pytest.mark.parametrize("text,name", [
    ("Ben", "Ben"), ("ben", "Ben"), ("I'm Ben", "Ben"), ("my name is Mary Ann", "Mary Ann"),
    ("hi, it's sam!", "Sam"), ("call me Jo-Anne", "Jo-Anne"), ("This is O'Neil", "O'Neil"),
    ("Tim Smith", "Tim Smith"), ("Kim Lee", "Kim Lee"), ("Ibrahim Ali", "Ibrahim Ali"),
    ("José", "José"), ("zoë", "Zoë"), ("Иван", "Иван"),
    ("no", "decline"), ("No thanks", "decline"), ("skip", "decline"), ("not telling", "decline"),
    ("I'm not telling", "decline"), ("I don't know", "decline"), ("my name is none of your business", "decline"),
    ("", None), ("12345", None), ("hello hello hello what is this robot doing here", None),
    ("<script>alert(1)</script>", None), ("What?", None), ("Sorry?", None), ("who are you", None),
    ("hello there", None), ("hi Milo", None), ("Ben?", None),
])
def test_parse_name(text, name):
    assert P.parse_name(text) == name


def test_name_ok():
    assert P.name_ok("Ben") and P.name_ok("Mary Ann") and P.name_ok("O'Neil")
    assert not P.name_ok("") and not P.name_ok("a b c") and not P.name_ok("Ben1") and not P.name_ok("x" * 30)


# ------------------------------------------------------------------ the book
def test_people_book_round_trip(tmp_path):
    ben, ann = _unit(1), _unit(2)
    book = P.PeopleBook(str(tmp_path))
    assert book.identify(ben) == (None, -1.0)
    book.add("Ben", [ben, _near(ben, 10)])
    book.add("Ann", [ann])
    again = P.PeopleBook(str(tmp_path))                    # from disk
    assert again.names() == ["Ann", "Ben"]
    assert again.identify(_near(ben, 11))[0] == "Ben"
    assert again.identify(_near(ann, 12))[0] == "Ann"
    assert again.identify(_unit(3))[0] is None             # a stranger
    assert again.forget("Ben") and P.PeopleBook(str(tmp_path)).names() == ["Ann"]
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_people_book_names_never_share_a_file(tmp_path):
    book = P.PeopleBook(str(tmp_path))
    for k, n in enumerate(("Иван", "Мария", "Ann Marie", "Ann-Marie")):
        book.add(n, [_unit(20 + k)])
    assert P.PeopleBook(str(tmp_path)).names() == sorted(["Иван", "Мария", "Ann Marie", "Ann-Marie"])
    assert book.forget("Мария") and P.PeopleBook(str(tmp_path)).names() == sorted(["Иван", "Ann Marie", "Ann-Marie"])


def test_people_book_keeps_at_most_max_embeddings(tmp_path):
    book = P.PeopleBook(str(tmp_path))
    v = _unit(4)
    n = book.add("Ben", [_near(v, k, 0.5) for k in range(P.MAX_EMB + 15)])
    assert n == P.MAX_EMB and book.people["Ben"]["emb"].shape == (P.MAX_EMB, 128)


# ------------------------------------------------------------------ tracks
def _face(box, emb, facing=True):
    return {"box": box, "score": 0.9, "facing": facing, "embedding": emb}


def test_tracker_knows_ben_and_not_a_stranger(tmp_path):
    ben = _unit(5)
    book = P.PeopleBook(str(tmp_path))
    book.add("Ben", [ben])
    tr = P.FaceTracker(book)
    t = 0.0
    for k in range(5):
        seen = tr.update([_face((100 + k, 50, 160 + k, 120), _near(ben, 20 + k)),
                          _face((10, 10, 50, 60), _near(_unit(6), 30 + k, 0.1))], t)
        t += 0.2
    who = {s["box"][0] < 90: s for s in seen}
    assert who[False]["who"] == "Ben" and who[True]["who"] == "?"
    assert who[False]["facing_s"] == pytest.approx(0.8) and who[False]["id"] != who[True]["id"]
    # the stranger turns away: facing time resets; then leaves (a new track if they return)
    seen = tr.update([_face((10, 10, 50, 60), None, facing=False)], t)
    assert {s["id"]: s for s in seen}[who[True]["id"]]["facing_s"] == 0.0
    seen = tr.update([_face((12, 10, 52, 60), _unit(7))], t + 5.0)
    assert seen[0]["id"] not in (who[False]["id"], who[True]["id"])


def test_tracker_recovers_from_poor_first_views_and_splits_on_a_new_face(tmp_path):
    ben = _unit(9)
    book = P.PeopleBook(str(tmp_path))
    book.add("Ben", [ben])
    tr = P.FaceTracker(book)
    box = (100, 50, 160, 120)
    blur = _unit(59)                                       # far away: a consistent distortion
    for k in range(5):                                     # ...that matches no one
        f = ben + 3.0 * blur + 0.2 * _unit(60 + k)
        seen = tr.update([_face(box, (f / np.linalg.norm(f)).astype(np.float32))], 0.2 * k)
    assert seen[0]["who"] == "?"
    for k in range(6):                                     # closer: clearly Ben
        seen = tr.update([_face(box, _near(ben, 70 + k, 0.2))], 1.0 + 0.2 * k)
    assert seen[0]["who"] == "Ben"                         # a recent window, not a lifetime count
    first = seen[0]["id"]
    seen = tr.update([_face(box, _unit(10))], 2.4)         # someone else steps into the same spot
    assert len(seen) == 2 and {s["id"] for s in seen} - {first}


def test_tracker_enrolls_a_new_face(tmp_path):
    book = P.PeopleBook(str(tmp_path))
    tr = P.FaceTracker(book)
    sam = _unit(8)
    for k in range(P.UNKNOWN_VOTES):
        seen = tr.update([_face((100, 50, 160, 120), _near(sam, 40 + k, 0.1))], 0.2 * k)
    assert seen[0]["who"] == "?"
    assert tr.enroll(999, "Sam")["why"] == "face gone"
    assert tr.command({"cmd": "enroll", "track": seen[0]["id"], "name": "<b>"}, 1.0)["why"] == "bad name"
    ev = tr.command({"cmd": "enroll", "track": seen[0]["id"], "name": "Sam"}, 1.0)
    assert ev["event"] == "enrolled" and ev["kept"] == P.UNKNOWN_VOTES
    assert P.PeopleBook(str(tmp_path)).identify(_near(sam, 99, 0.1))[0] == "Sam"
    seen = tr.update([_face((101, 50, 161, 120), _near(sam, 50, 0.1))], 1.0)
    assert seen[0]["who"] == "Sam"


def test_tracker_will_not_give_a_known_name_to_another_face(tmp_path):
    book = P.PeopleBook(str(tmp_path))
    book.add("Ben", [_unit(11)])
    tr = P.FaceTracker(book)
    stranger = _unit(12)
    for k in range(P.UNKNOWN_VOTES):
        seen = tr.update([_face((100, 50, 160, 120), _near(stranger, 80 + k, 0.1))], 0.2 * k)
    assert tr.enroll(seen[0]["id"], "Ben")["why"] == "name taken"
    assert len(book.people["Ben"]["emb"]) == 1


def test_tracker_snooze_and_forget(tmp_path):
    book = P.PeopleBook(str(tmp_path))
    tr = P.FaceTracker(book)
    sam = _unit(13)
    for k in range(P.UNKNOWN_VOTES):
        seen = tr.update([_face((100, 50, 160, 120), _near(sam, 90 + k, 0.1))], 0.2 * k)
    assert tr.command({"cmd": "snooze", "track": seen[0]["id"], "s": 600}, 1.0)["ok"]
    seen = tr.update([_face((10, 10, 50, 60), _near(sam, 99, 0.1))], 5.0)     # back later: a new track
    assert seen[0]["snoozed"] and seen[0]["id"] != 1
    tr.update([_face((10, 10, 50, 60), _near(sam, 98, 0.1))], 700.0)          # snooze over
    seen = tr.update([_face((10, 10, 50, 60), _near(sam, 97, 0.1))], 703.0)
    assert not seen[0]["snoozed"]
    # learned, then forgotten from the command line while running
    tid = seen[0]["id"]
    for k in range(3):
        tr.update([_face((10, 10, 50, 60), _near(sam, 100 + k, 0.1))], 703.2 + 0.2 * k)
    assert tr.enroll(tid, "Sam")["event"] == "enrolled"
    P.PeopleBook(str(tmp_path)).forget("Sam")              # another process (--forget)
    import os
    os.utime(tmp_path, ns=(1, 1))                          # the folder changed
    tr.book._checked = -1e9
    seen = tr.update([_face((10, 10, 50, 60), _near(sam, 120, 0.1))], 704.0)
    assert "Sam" not in tr.book.people and seen[0]["who"] != "Sam"


# ------------------------------------------------------------------ social
def _track(i, who, facing_s, snoozed=False):
    return {"id": i, "who": who, "facing_s": facing_s, "facing": facing_s > 0, "snoozed": snoozed}


def test_social_greets_known_people_once():
    s = P.Social()
    assert s.step(0.0, [_track(1, "Ben", 0.2)])["say"] is None          # not long enough
    out = s.step(0.2, [_track(1, "Ben", 0.8)])
    assert out["say"] == "Hi Ben!" and out["events"] == ["Ben is looking at you"]
    assert s.step(5.0, [_track(1, "Ben", 5.6)])["say"] is None          # once per GREET_GAP_S
    assert s.step(5.0 + P.GREET_GAP_S, [_track(1, "Ben", 1.0)])["say"] == "Hi Ben!"


def test_social_asks_a_stranger_and_learns_the_name():
    s = P.Social()
    assert s.step(0.0, [_track(4, "?", 1.0)])["ask"] is None
    assert s.step(1.2, [_track(4, "?", 2.2), _track(5, "Ben", 2.0)])["say"] == "Hi Ben!"
    assert s.step(1.3, [_track(4, "?", 2.3), _track(5, "Ben", 2.1)])["ask"] is None   # two facing: whom?
    out = s.step(1.4, [_track(4, "?", 2.4)])
    assert out["say"] == P.Social.Q
    q = out["ask"]["id"]
    assert s.step(1.6, [_track(4, "?", 2.6)])["ask"]["id"] == q        # the question stays open
    out = s.step(3.0, [], [{"id": q + 1, "text": "Ben"}])               # an answer to another question
    assert out["commands"] == [] and out["ask"]["id"] == q
    out = s.step(3.2, [], [{"id": q, "text": "uh"}])
    assert out["say"].startswith("Sorry") and out["ask"]["id"] != q     # a new box on the page
    q2 = out["ask"]["id"]
    out = s.step(3.4, [], [{"id": q2, "text": "I'm Sam"}])
    assert out["commands"] == [{"cmd": "enroll", "track": 4, "name": "Sam"}]
    assert out["say"] == "Nice to meet you, Sam!" and out["ask"] is None
    out = s.step(3.6, [], camera_events=[{"event": "enrolled", "name": "Sam"}])
    assert s.met == {"Sam"} and out["events"] == ["you just met Sam and learned their face"]
    assert s.step(4.0, [_track(4, "Sam", 3.0)])["say"] is None          # just met


def test_social_respects_no_and_does_not_nag():
    s = P.Social()
    q = s.step(0.0, [_track(4, "?", 2.5)])["ask"]["id"]
    out = s.step(1.0, [_track(4, "?", 3.5)], [{"id": q, "text": "no thanks"}])
    assert out["say"] == "Okay, no problem."
    assert out["commands"] == [{"cmd": "snooze", "track": 4, "s": P.SNOOZE_DECLINED_S}]
    assert s.step(2.0 + P.ASK_GAP_S, [_track(9, "?", 9.0, snoozed=True)])["ask"] is None   # that face, back
    out = s.step(3.0 + P.ASK_GAP_S, [_track(5, "?", 2.5)])                               # someone else
    assert out["ask"] is not None
    out = s.step(4.0 + P.ASK_GAP_S + P.ANSWER_S, [])                                       # no answer
    assert out["ask"] is None and out["commands"] == [{"cmd": "snooze", "track": 5, "s": P.SNOOZE_UNANSWERED_S}]
    assert s.step(5.0 + P.ASK_GAP_S + P.ANSWER_S, [_track(6, "?", 3.0)])["ask"] is None   # not again so soon


def test_social_gives_up_after_retries_and_handles_failures():
    s = P.Social()
    q = s.step(0.0, [_track(4, "?", 2.5)])["ask"]["id"]
    for k in range(P.RETRIES):
        q = s.step(1.0 + k, [], [{"id": q, "text": "?!"}])["ask"]["id"]
    out = s.step(9.0, [], [{"id": q, "text": "12"}])
    assert out["ask"] is None and out["commands"][0]["cmd"] == "snooze"
    out = s.step(10.0, [], camera_events=[{"event": "enroll_failed", "name": "Sam", "track": 4, "why": "too few views"}])
    assert out["say"].startswith("Sorry")
    assert s.step(11.0, [_track(7, "?", 3.0)])["ask"] is None            # not straight away
    assert s.step(45.0, [_track(7, "?", 3.0)])["ask"] is not None        # but later
    s2 = P.Social()
    out = s2.step(0.0, [], camera_events=[{"event": "enroll_failed", "name": "Ben", "track": 4, "why": "name taken"}])
    assert "full name" in out["say"] and out["ask"] is not None


def test_one_line_per_face_update():
    from sim.habitat_bridge.brain_client import _say

    class M:
        pass
    m = M()
    assert _say(m, "Hi Ben!", "beep boop") == "Hi Ben!"
    assert _say(m, None, None) == "beep boop"                            # waited one step
    assert _say(m, None, None) is None


# ------------------------------------------------------------------ the face page
def test_face_page_question_and_answer():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import robot.face_server as fs
    fs.app.state.key = "k"
    fs.REPLIES.clear()
    fs.REPLY_TIMES.clear()
    c = TestClient(fs.app)
    h = {"X-Face-Token": "k"}
    assert c.post("/state", json={"ask": {"id": 1234567, "q": "What's your name?"}}, headers=h).status_code == 200
    assert fs.STATE["state"]["ask"] == {"id": 1234567, "q": "What's your name?"}
    assert c.post("/reply", json={"id": 2, "text": "Ben"}).status_code == 409          # not the open question
    assert c.post("/reply", json={"id": 1234567, "text": ""}).status_code == 409
    assert c.post("/reply", json={"id": 1234567}).status_code == 400
    assert c.post("/reply", content='{"id": 1234567, "text": "Ben"}',
                  headers={"Content-Type": "text/plain"}).status_code == 415          # a cross-site form
    assert c.post("/reply", json={"id": 1234567, "text": "Ben"},
                  headers={"Origin": "http://evil.example"}).status_code == 403       # another site's page
    assert c.post("/reply", content='{"id": 1e999, "text": "x"}',
                  headers={"Content-Type": "application/json"}).status_code == 400
    assert c.post("/reply", json={"id": 1234567, "text": "x" * 600}).status_code == 413
    assert c.post("/reply", json={"id": 1234567, "text": "Ben"},
                  headers={"Origin": "http://testserver"}).status_code == 200
    r = c.post("/state", json={"ask": None}, headers=h).json()
    assert r["replies"] == [{"id": 1234567, "text": "Ben"}]
    assert c.post("/state", json={}, headers=h).json()["replies"] == []
    assert c.post("/reply", json={"id": 1234567, "text": "Ben"}).status_code == 429    # every attempt counted
    with pytest.raises(ValueError):
        fs.clean_state({"ask": "what"})


# ------------------------------------------------------------------ the rover
def test_rover_world_faces_in_habitats_frame_and_camera_commands():
    from robot.rover_world import CAM_PITCH_DEG, FakeBase, RoverWorld

    class Cam:
        def __init__(self):
            self.sent = []

        def state(self):
            return {"ready": True, "stale": False, "moving": 0.0, "fps": 15.0, "pan_deg": 10.0,
                    "tilt_deg": CAM_PITCH_DEG, "person_active": 0.0, "person_stale": True}

        def command(self, *a, **k):
            pass

        def faces(self):
            return [{"id": 2, "who": "Ben", "facing_s": 1.0, "az": 5.0, "el": 3.0, "half": 4.0}]

        def events(self):
            return [{"event": "enrolled", "name": "Ben"}]

        def send(self, c):
            self.sent.append(c)
            return True

        def close(self):
            pass

    cam = Cam()
    w = RoverWorld(FakeBase(), None, None, clock=lambda: 0.0, sleep=lambda s: None, camera=cam)
    o = w.handle({"cmd": "reset"})
    assert o["faces"][0]["az"] == pytest.approx(15.0) and o["faces"][0]["el"] == pytest.approx(3.0)
    assert o["camera_events"] == [{"event": "enrolled", "name": "Ben"}]
    assert w.handle({"cmd": "camera", "send": {"cmd": "enroll", "track": 2, "name": "Ben"}}) == {"ok": True}
    assert cam.sent == [{"cmd": "enroll", "track": 2, "name": "Ben"}]


def test_head_face_angles():
    from robot.head import HFOV_DEG, _face_angles
    a = _face_angles((150, 110, 170, 130), 320, 240)       # centred, 20 px wide
    assert a["az"] == 0.0 and a["el"] == 0.0 and a["half"] == pytest.approx(round(20 / 320 * HFOV_DEG / 2, 2))
    assert _face_angles((300, 0, 320, 20), 320, 240)["az"] > 0     # right of centre: + azimuth
