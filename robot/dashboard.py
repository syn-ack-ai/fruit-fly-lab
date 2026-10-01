"""
Milo's dashboard: what the robot sees and does, in a browser (an iPad on the
home network, or anywhere through Tailscale). Read-only: it shows, it does
not control.

    brain client --(telemetry ~5 Hz, camera JPEG ~5 Hz; localhost)--> this server
    browser <--(page, WebSocket, camera frames; needs the key)-- this server

Panels: the camera (with the person detector's box), the lidar scan around
the robot, the fly brain's descending channels, the neocortex's drives and
goal, the people in view, what Milo heard and said, learning, and the
control loop's timing.

The page shows the camera, so it needs the key: open
    http://<robot>:8080/?key=<key>
(the key: $FLY_DASHBOARD_KEY, else ~/.milo_dashboard_key, created mode 600 on
first use; the brain client prints the address). The brain client posts from
127.0.0.1 only, with the same key.

    python -m robot.dashboard --port 8080          # run by the brain client (--dashboard)
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import queue
import secrets
import threading
import time
import urllib.request
from pathlib import Path

try:            # module level: FastAPI resolves the handlers' (postponed) annotations here
    from fastapi import Request, WebSocket, WebSocketDisconnect
except ImportError:                     # the brain client's publisher needs none of it
    Request = WebSocket = WebSocketDisconnect = None

HERE = Path(__file__).resolve().parent
KEY_PATH = os.path.expanduser("~/.milo_dashboard_key")
MAX_TELEMETRY = 64 * 1024
MAX_JPEG = 512 * 1024
MAX_BLOB = 4 * 1024 * 1024                   # the brain's geometry (~1 MB), activity (~70 KB), flow
MAX_CIRCUIT = 128 * 1024 * 1024              # every neuron's strongest inputs and outputs (once)


def dashboard_key() -> str:
    env = os.environ.get("FLY_DASHBOARD_KEY")
    if env:
        return env
    try:
        with open(KEY_PATH) as fh:
            k = fh.read().strip()
        if k:
            return k
    except FileNotFoundError:
        pass
    k = secrets.token_urlsafe(18)
    fd = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # (an empty file: replaced)
    with os.fdopen(fd, "w") as fh:
        fh.write(k)
    os.chmod(KEY_PATH, 0o600)
    return k


def make_app(key: str):
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse, Response

    app = FastAPI(title="Milo dashboard", docs_url=None, redoc_url=None, openapi_url=None)
    boot = secrets.token_hex(8)                  # changes when the server restarts (DashboardPublisher)
    # three.js for the 3D brain (MIT; vendored for the lab's views): public library code, no key
    from fastapi.staticfiles import StaticFiles
    app.mount("/vendor/three", StaticFiles(directory=HERE.parent / "visualization" / "static" / "vendor" / "three"),
              name="three")
    state = {"telemetry": None, "tv": 0, "jpeg": None, "jv": 0,
             "geometry": None, "gv": 0, "activity": None, "av": 0, "flow": None, "fv": 0, "circuit": None}

    def ok_key(k) -> bool:
        return isinstance(k, str) and hmac.compare_digest(k.encode(), key.encode())

    def local(req) -> bool:
        return req.client is not None and req.client.host in ("127.0.0.1", "::1")

    @app.get("/")
    def page(key: str = ""):
        if not ok_key(key):
            return Response("Milo's dashboard needs its key: /?key=...", status_code=403, media_type="text/plain")
        return FileResponse(HERE / "dashboard_page" / "index.html",
                            headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})

    @app.get("/camera.jpg")
    def camera(key: str = ""):
        if not ok_key(key):
            return Response(status_code=403)
        if state["jpeg"] is None:
            return Response(status_code=204)
        return Response(state["jpeg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/telemetry")
    async def post_telemetry(req: Request):
        if not local(req) or not ok_key(req.headers.get("x-key", "")):
            return JSONResponse({"ok": False}, status_code=403)
        body = await req.body()
        if len(body) > MAX_TELEMETRY:
            return JSONResponse({"ok": False}, status_code=413)
        try:
            state["telemetry"] = json.loads(body)
        except ValueError:
            return JSONResponse({"ok": False}, status_code=400)
        state["tv"] += 1
        return {"ok": True, "boot": boot, "circuit": state["circuit"] is not None, "geometry": state["geometry"] is not None}

    @app.post("/camera")
    async def post_camera(req: Request):
        if not local(req) or not ok_key(req.headers.get("x-key", "")):
            return JSONResponse({"ok": False}, status_code=403)
        body = await req.body()
        if len(body) > MAX_JPEG or not body.startswith(b"\xff\xd8"):
            return JSONResponse({"ok": False}, status_code=400)
        state["jpeg"] = body
        state["jv"] += 1
        return {"ok": True}

    @app.post("/geometry")
    async def post_geometry(req: Request):
        return await _blob(req, "geometry", "gv", b"FLYG")

    @app.post("/activity")
    async def post_activity(req: Request):
        return await _blob(req, "activity", "av", None)

    @app.post("/flow")
    async def post_flow(req: Request):
        return await _blob(req, "flow", "fv", None)

    @app.post("/circuit")
    async def post_circuit(req: Request):
        """robot/brainmap.BrainActivity.circuit(): kept to answer /neuron."""
        import io
        import numpy as np
        if not local(req) or not ok_key(req.headers.get("x-key", "")):
            return JSONResponse({"ok": False}, status_code=403)
        body = await req.body()
        if len(body) > MAX_CIRCUIT:
            return JSONResponse({"ok": False}, status_code=413)
        def load():
            z = np.load(io.BytesIO(body), allow_pickle=False)
            return {k: z[k] for k in z.files}
        try:
            state["circuit"] = await asyncio.to_thread(load)      # not on the event loop
        except Exception:
            return JSONResponse({"ok": False}, status_code=400)
        return {"ok": True}

    @app.get("/neuron")
    def neuron(key: str = "", i: int = -1):
        """A drawn neuron (its position in the 3D view's order): its cell type,
        system, and strongest inputs and outputs (synapse counts; - = inhibitory)."""
        if not ok_key(key):
            return Response(status_code=403)
        c = state["circuit"]
        if c is None:
            return JSONResponse({"ok": False, "why": "not ready"}, status_code=503)
        if not 0 <= i < len(c["idx"]):
            return JSONResponse({"ok": False}, status_code=404)
        n = int(c["idx"][i])

        def partners(idx, w):
            out = []
            for t, s in zip(idx[n], w[n]):
                if t < 0:
                    break
                out.append({"i": int(c["pos_of"][t]), "type": str(c["types"][c["type_id"][t]]) or "untyped",
                            "system": str(c["systems"][c["system"][t]]), "synapses": int(s)})
            return out
        return {"ok": True, "i": i, "type": str(c["types"][c["type_id"][n]]) or "untyped",
                "system": str(c["systems"][c["system"][n]]),
                "outputs": partners(c["out_idx"], c["out_w"]), "inputs": partners(c["in_idx"], c["in_w"])}

    async def _blob(req, name, ver, magic):
        if not local(req) or not ok_key(req.headers.get("x-key", "")):
            return JSONResponse({"ok": False}, status_code=403)
        body = await req.body()
        if len(body) > MAX_BLOB or (magic and not body.startswith(magic)):
            return JSONResponse({"ok": False}, status_code=400)
        state[name] = body
        state[ver] += 1
        return {"ok": True}

    @app.get("/geometry.bin")
    def geometry(key: str = ""):
        if not ok_key(key):
            return Response(status_code=403)
        if state["geometry"] is None:
            return Response(status_code=204)
        return Response(state["geometry"], media_type="application/octet-stream",
                        headers={"Cache-Control": "no-store"})

    @app.get("/flow.bin")
    def flow(key: str = ""):
        if not ok_key(key):
            return Response(status_code=403)
        return Response(state["flow"] or b"", media_type="application/octet-stream", headers={"Cache-Control": "no-store"})

    @app.get("/activity.bin")
    def activity(key: str = ""):
        if not ok_key(key):
            return Response(status_code=403)
        if state["activity"] is None:
            return Response(status_code=204)
        return Response(state["activity"], media_type="application/octet-stream",
                        headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        if not ok_key(sock.query_params.get("key", "")):
            await sock.close(code=1008)
            return
        await sock.accept()
        seen = None
        try:
            while True:
                now = (state["tv"], state["jv"], state["gv"], state["av"], state["fv"])
                if now != seen:
                    seen = now
                    await sock.send_text(json.dumps({"t": state["telemetry"], "camera": state["jv"],
                                                     "geometry": state["gv"], "activity": state["av"],
                                                     "flow": state["fv"], "circuit": state["circuit"] is not None}))
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


def _clean(x):
    """JSON without NaN or infinity (the browser's JSON.parse rejects them): None."""
    if isinstance(x, float):
        return x if x == x and abs(x) != float("inf") else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):                          # numpy scalars
        return _clean(x.item())
    return x


class DashboardPublisher:
    """Posts telemetry and camera frames to the dashboard server from a
    background thread; the latest wins, the brain loop never waits."""

    def __init__(self, port: int = 8080, key: str | None = None, start_server: bool = True):
        import subprocess
        import sys
        self.url = f"http://127.0.0.1:{port}"
        self.key = key or dashboard_key()
        self.proc = None
        if start_server:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log = open(os.path.expanduser("~/.milo_dashboard.log"), "ab")
            self.proc = subprocess.Popen([sys.executable, "-m", "robot.dashboard", "--port", str(port),
                                          "--parent", str(os.getpid())], cwd=root, stdout=log, stderr=log)
        self.q = queue.Queue(maxsize=1)
        self.fails = 0
        self.sent = 0
        self.static = {}                          # path -> [bytes, acknowledged]: sent until the server has it
        self.boot = None                          # the server's boot id: a new one means it restarted
        threading.Thread(target=self._run, daemon=True).start()

    def put_static(self, path: str, data: bytes) -> None:
        """Data the server keeps (the brain's geometry): posted until it is
        acknowledged, and again if the server restarts."""
        self.static[path] = [data, False]

    def publish(self, telemetry: dict, jpeg_fn=None, extra_fn=None) -> None:
        """telemetry: a JSON-able dict; jpeg_fn: returns the camera's latest
        JPEG, extra_fn: (more telemetry, {path: bytes}), both called in the
        background thread."""
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.q.put_nowait((telemetry, jpeg_fn, extra_fn))
        except queue.Full:
            pass

    def _post(self, path: str, body: bytes, ctype: str) -> bytes:
        req = urllib.request.Request(self.url + path, body, {"Content-Type": ctype, "X-Key": self.key})
        return urllib.request.urlopen(req, timeout=max(1.0, len(body) / 2e6)).read()   # the circuit: tens of MB

    def _run(self) -> None:
        while True:
            tel, jpeg_fn, extra_fn = self.q.get()
            try:
                for path, item in list(self.static.items()):
                    if not item[1]:
                        self._post(path, item[0], "application/octet-stream")
                        item[1] = True
                if extra_fn is not None:
                    more, blobs = extra_fn()
                    tel = dict(tel, **(more or {}))
                    for path, data in (blobs or {}).items():
                        if data:
                            self._post(path, data, "application/octet-stream")
                r = json.loads(self._post("/telemetry", json.dumps(_clean(tel), allow_nan=False).encode(),
                                          "application/json") or b"{}")
                if r.get("boot") != self.boot:    # a new server (or the first contact): send the static data again
                    self.boot = r.get("boot")
                    for item in self.static.values():
                        item[1] = False
                jpg = jpeg_fn() if jpeg_fn is not None else None
                if jpg:
                    self._post("/camera", jpg, "image/jpeg")
                self.sent += 1
            except Exception:
                self.fails += 1                   # (the static data is re-sent only to a new server)
                time.sleep(0.5)

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--parent", type=int, default=0, help="exit when this process is gone (the brain client)")
    a = ap.parse_args()
    if a.parent:
        def watch():
            while True:
                time.sleep(2.0)
                try:
                    os.kill(a.parent, 0)
                except OSError:
                    os._exit(0)                       # the robot stopped: stop showing its camera
        threading.Thread(target=watch, daemon=True).start()
    uvicorn.run(make_app(dashboard_key()), host=a.host, port=a.port, log_level="warning",
                timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
