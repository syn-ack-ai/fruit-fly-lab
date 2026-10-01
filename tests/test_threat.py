"""robot/threat.py (fear from meaning drives the looming circuit) and
robot/appraise.py (the vision-language model's appraisal of the camera)."""
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import shared_memory

import numpy as np

import robot.appraise as A
import robot.head as H
from cortex.personality import parse as parse_personality
from robot.threat import LEVELS, Fear, llm_level


def test_fear_lingers_and_fades():
    f = Fear()
    assert not f.update(0.0)["active"]
    assert not f.state(0.0)["active"]
    f.appraise("llm", LEVELS[1], 1.0, fade_s=4.0, what="warned")
    f.appraise("vision", LEVELS[2], 1.0, azimuth_deg=-20.0, fade_s=1.0, what="knife")
    L = f.update(1.0)
    assert L["source"] == "vision" and L["level"] == 1.0 and L["azimuth_deg"] == -20.0 and L["what"] == "knife"
    st = f.state(1000.0)
    assert st["active"] and st["half_angle_deg"] > 20 and st["expansion_rate_deg_s"] > 400
    # "no danger" and a weaker look do not end it: it fades
    f.appraise("vision", 0.0, 1.5)
    f.appraise("vision", LEVELS[1], 1.5, fade_s=1.0)
    L = f.update(1.5)
    assert L["source"] == "vision" and abs(L["level"] - math.exp(-0.5)) < 1e-3
    assert f.state(1500.0)["expansion_rate_deg_s"] < st["expansion_rate_deg_s"]
    # a stronger look raises it again
    f.appraise("vision", 1.0, 1.6, azimuth_deg=10.0, fade_s=1.0)
    assert f.update(1.6)["level"] == 1.0 and f.last["azimuth_deg"] == 10.0
    # the sight fades below the warning, then the warning fades out too
    L = f.update(4.0)
    assert L["source"] == "llm" and L["azimuth_deg"] == 0.0
    assert not f.update(30.0)["active"]
    f.appraise("vision", 1.0, 30.0)
    assert not f.update(0.5)["active"]                 # the clock went back: a new day
    assert f.count == 5


def test_fear_levels_monotone_and_llm_mapping():
    lv = []
    for x in (0.1, 0.25, 0.5, 1.0):
        f = Fear()
        f.appraise("test", x, 0.0)
        f.update(0.0)
        lv.append((f.state(0)["half_angle_deg"], f.state(0)["expansion_rate_deg_s"]))
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(lv, lv[1:]))
    assert llm_level(0) == 0.0 and llm_level("2") == 1.0 and 0 < llm_level(1) < 0.35
    assert llm_level("x") == 0.0 and llm_level(None) == 0.0 and llm_level(7) == 0.0


def test_parsers():
    assert A.parse('ok {"fear": 2, "what": "dog charging", "where": "left"}') == \
        {"fear": 2, "what": "dog charging", "where": "left"}
    assert A.parse('{"fear": 5, "where": "up"}') == {"fear": 0, "where": None, "what": ""}
    assert A.parse("no json") is None
    p = parse_personality('{"intent": "rest", "mood": "startled", "fear": 2}')
    assert p["fear"] == 2
    assert parse_personality('{"intent": "rest", "fear": "lots"}')["fear"] == 0
    assert parse_personality('{"intent": "rest"}')["fear"] == 0


class _Model(BaseHTTPRequestHandler):
    reply = {"fear": 2, "what": "a falling box", "where": "right"}
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Model.seen.append(body)
        out = json.dumps({"choices": [{"message": {"content": json.dumps(_Model.reply)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class _Pers:
    def __init__(self):
        self.events = []

    def event(self, t, text, urge=False):
        self.events.append((t, text, urge))


def test_vision_appraiser_end_to_end():
    srv = HTTPServer(("127.0.0.1", 0), _Model)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        frames = {"f": (2, b"\xff\xd8jpeg", 10.0, 0.0, False, time.time())}
        ap = A.VisionAppraiser(f"http://127.0.0.1:{srv.server_port}/v1/chat/completions", "m",
                               lambda: frames["f"], hfov_deg=60.0, period_s=0.0)
        fear, pers = Fear(), _Pers()
        new = None
        for _ in range(200):
            new = ap.step(1.0, fear, pers) or new
            if new:
                break
            time.sleep(0.01)
        assert new and new["fear"] == 2
        body = _Model.seen[-1]
        assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        L = fear.update(1.0)
        assert L["source"] == "vision" and L["level"] == 1.0
        assert abs(L["azimuth_deg"] - (10.0 + 20.0)) < 1e-9        # pan + right third of 60 deg
        assert pers.events and pers.events[-1][2] is True and "falling box" in pers.events[-1][1]
        # the same frame is not looked at twice; a moving head's frame not at all
        n = len(_Model.seen)
        frames["f"] = (2, b"x", 0.0, 0.0, False, time.time())
        ap.step(1.1, fear); time.sleep(0.05)
        frames["f"] = (4, b"x", 0.0, 0.0, True, time.time())
        ap.step(1.2, fear); time.sleep(0.05)
        assert len(_Model.seen) == n
        # "no danger" leaves the fear fading, not gone
        _Model.reply = {"fear": 0, "what": "a person", "where": None}
        frames["f"] = (6, b"x", 0.0, 0.0, False, time.time())
        got = None
        for _ in range(200):
            got = ap.step(1.3, fear) or got
            if got:
                break
            time.sleep(0.01)
        assert got["fear"] == 0 and fear.update(1.3)["source"] == "vision"
        assert not fear.update(30.0)["active"]
        assert ap.summary()["looks"] == 2 and ap.summary()["fear2"] == 1
    finally:
        srv.shutdown()


def test_head_clean_frame_roundtrip():
    shm = shared_memory.SharedMemory(create=True, size=H.SHM_SIZE)
    try:
        hdr = np.ndarray((len(H._F),), dtype=np.float64, buffer=shm.buf)
        hdr[:] = 0
        assert H.read_clean(shm, hdr) is None
        jpg = b"\xff\xd8" + bytes(range(200)) + b"\xff\xd9"

        def wc():
            shm.buf[H._CLEAN_AT:H._CLEAN_AT + len(jpg)] = jpg
            hdr[H._I["clean_len"]] = len(jpg)
            hdr[H._I["clean_pan"]] = 12.5
            hdr[H._I["clean_t"]] = 99.0
        H._seq_write(hdr, H._I["clean_seq"], wc)
        seq, data, pan, tilt, moving, t = H.read_clean(shm, hdr)
        assert data == jpg and pan == 12.5 and not moving and t == 99.0 and seq == 2
        del hdr
    finally:
        shm.close()
        shm.unlink()
