"""
Serves the pet's face to a browser (an iPad in Safari: open http://<host>:8010/,
then Share -> Add to Home Screen for a full-screen face; tap once to turn sound on).

    python -m robot.face_server --port 8010          # waits for a robot / simulation
    python -m robot.face_server --port 8010 --demo   # cycles through expressions

The robot (or sim/habitat_bridge/brain_client.py --face) POSTs the face state
(robot/face.FaceModel) to /state; every open page gets it over a WebSocket.
No authentication: it shows a face and accepts face states, nothing else. Run
it on the home network only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

HERE = Path(__file__).resolve().parent
app = FastAPI(title="Pet face")
STATE = {"state": {"gaze": [0, 0.2], "open": 0.85, "pupil": 0.4, "mouth": "cat", "mood": "calm",
                   "event": {"id": 0}}, "version": 0}
MAX_BYTES = 8192


@app.get("/")
def page():
    return FileResponse(HERE / "face_page" / "face.html")


@app.post("/state")
async def post_state(req: Request):
    body = await req.body()
    if len(body) > MAX_BYTES:
        return {"ok": False}
    st = json.loads(body)
    if not isinstance(st, dict):
        return {"ok": False}
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
    if a.demo:
        app.router.on_startup.append(lambda: asyncio.get_event_loop().create_task(demo()))
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
