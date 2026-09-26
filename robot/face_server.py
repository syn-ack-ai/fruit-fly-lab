"""
Serves the pet's face to a browser (an iPad in Safari: open http://<host>:8010/,
then Share -> Add to Home Screen for a full-screen face; tap once to turn sound on).

    python -m robot.face_server --port 8010          # waits for a robot / simulation
    python -m robot.face_server --port 8010 --demo   # cycles through expressions

The robot (or sim/habitat_bridge/brain_client.py --face) POSTs the face state
(robot/face.FaceModel) to /state; every open page gets it over a WebSocket.

Security (review 2026-09-26): the page and WebSocket are open to the home
network (the iPad must reach them). Changing the face needs the shared token
(header X-Face-Token; $FLY_FACE_KEY or ~/.fly_face_key, mode 600, created on
first use, the same on the robot side) and a JSON body, and every field is
checked (types, ranges, allowed words, short phrases), so a web page on the
network cannot make the pet say things or break the page. Run it on the home
network only.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import math
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

HERE = Path(__file__).resolve().parent
app = FastAPI(title="Pet face")
STATE = {"state": {"gaze": [0, 0.2], "open": 0.85, "pupil": 0.4, "mouth": "cat", "mood": "calm",
                   "event": {"id": 0}}, "version": 0}
MAX_BYTES = 8192
KEY_PATH = os.path.expanduser("~/.fly_face_key")
MOUTHS = {"cat", "smile", "frown", "o", "chew"}
SOUNDS = {"mrrp", "meow", "purr", "trill", "chirp", "hiss"}


def face_key() -> str:
    """The shared token ($FLY_FACE_KEY, else ~/.fly_face_key, created mode 600)."""
    env = os.environ.get("FLY_FACE_KEY")
    if env:
        return env
    try:
        with open(KEY_PATH) as fh:
            k = fh.read().strip()
        if k:
            return k
    except FileNotFoundError:
        pass
    k = secrets.token_hex(24)
    fd = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(k)
    return k


def _num(x, lo, hi):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(x)
    return max(lo, min(hi, float(x)))


def _word(x, n=24):
    if not isinstance(x, str):
        raise ValueError(x)
    return x[:n]


def clean_state(st) -> dict:
    """A face state with only known fields, of the right types (else ValueError)."""
    if not isinstance(st, dict):
        raise ValueError("not an object")
    g = st.get("gaze", [0, 0.2])
    if not isinstance(g, list) or len(g) != 2:
        raise ValueError("gaze")
    out = {"gaze": [_num(g[0], -1, 1), _num(g[1], -1, 1)],
           "open": _num(st.get("open", 0.85), 0, 1), "pupil": _num(st.get("pupil", 0.4), 0, 1),
           "slow_blink": bool(st.get("slow_blink", False)), "blush": bool(st.get("blush", False)),
           "grooming": bool(st.get("grooming", False)),
           "mouth": st.get("mouth") if st.get("mouth") in MOUTHS else "cat",
           "mood": _word(st.get("mood", "calm")), "behaviour": _word(st.get("behaviour", ""), 60)}
    for k in ("hunger", "social"):
        if k in st:
            out[k] = _num(st[k], 0, 1)
    ev = st.get("event") or {"id": 0}
    if not isinstance(ev, dict):
        raise ValueError("event")
    say = ev.get("say")
    out["event"] = {"id": int(_num(ev.get("id", 0), 0, 2 ** 31)),
                    "say": _word(say, 60) if say is not None else None,
                    "sound": ev.get("sound") if ev.get("sound") in SOUNDS else None}
    return out


@app.get("/")
def page():
    return FileResponse(HERE / "face_page" / "face.html")


@app.post("/state")
async def post_state(req: Request):
    from fastapi.responses import JSONResponse
    if not hmac.compare_digest(req.headers.get("x-face-token", ""), app.state.key):
        return JSONResponse({"ok": False}, status_code=403)
    if not req.headers.get("content-type", "").startswith("application/json"):
        return JSONResponse({"ok": False}, status_code=415)
    body = await req.body()
    if len(body) > MAX_BYTES:
        return JSONResponse({"ok": False}, status_code=413)
    try:
        st = clean_state(json.loads(body))
    except (ValueError, TypeError, json.JSONDecodeError, OverflowError):
        return JSONResponse({"ok": False}, status_code=400)
    STATE["state"] = st
    STATE["version"] += 1
    return {"ok": True}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    seen = -1
    try:
        while True:
            if STATE["version"] != seen:
                seen = STATE["version"]
                await sock.send_text(json.dumps(STATE["state"]))
            await asyncio.sleep(0.05)
    except (WebSocketDisconnect, RuntimeError):
        pass


DEMO = [  # (seconds, FaceModel inputs)
    (4, {"person_visible": True, "person_az": -30, "person_el": 5, "person_d": 2.0, "mood": "curious", "speed": 0.1}),
    (4, {"person_visible": True, "person_az": 35, "person_el": 10, "person_d": 1.5, "mood": "eager", "speed": 0.3,
         "say": "hi!", "sound": "trill"}),
    (5, {"person_visible": True, "person_az": 0, "person_el": 25, "person_d": 0.6, "mood": "content", "petting": True,
         "sound": "purr"}),
    (3, {"startle": 1.0, "mood": "startled", "sound": "hiss"}),
    (5, {"eating": True, "mood": "happy", "goal_az": 0}),
    (6, {"mood": "sleepy", "speed": 0.0}),
    (4, {"goal_az": -60, "mood": "playful", "speed": 0.35, "sound": "chirp"}),
    (4, {"person_visible": True, "person_az": 10, "person_d": 1.0, "mood": "grumpy", "say": "not now", "sound": "mrrp"}),
]


async def demo():
    from robot.face import FaceModel
    fm = FaceModel()
    while True:
        for secs, inp in DEMO:
            for k in range(int(secs * 10)):
                x = dict(inp) if k == 0 else {kk: v for kk, v in inp.items() if kk not in ("say", "sound")}
                x.setdefault("behaviour", "demo")
                STATE["state"] = dict(fm.update(0.1, x))
                STATE["version"] += 1
                await asyncio.sleep(0.1)


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--demo", action="store_true", help="cycle through expressions")
    a = ap.parse_args()
    app.state.key = face_key()
    if a.demo:
        app.router.on_startup.append(lambda: asyncio.get_event_loop().create_task(demo()))
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
