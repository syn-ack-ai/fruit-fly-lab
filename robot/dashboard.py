"""
Milo's dashboard: what the robot sees and does, in a browser (an iPad on the
home network, or anywhere through Tailscale). Read-only: it shows, it does
not control.

    brain client --(shared memory: telemetry and spike counts, ~5 Hz)--> this server
    head camera  --(shared memory: its preview JPEG)--------------------> this server
    browser <--(page, WebSocket, camera frames; needs the key)-- this server

The brain's process only copies (a JSON dump and its spike counts, ~0.5 ms a
copy, 5 times a second; no thread, no HTTP): the activity, levels and wiring
(robot/brainmap.ActivityMap) and everything served are this process's work,
on another core, without the brain's Python lock.

Panels: the camera (with the person detector's box), the lidar scan around
the robot, the fly brain's descending channels, the neocortex's drives and
goal, the people in view, what Milo heard and said, learning, and the
control loop's timing.

The page shows the camera, so it needs the key: open
    http://<robot>:8080/?key=<key>
(the key: $FLY_DASHBOARD_KEY, else ~/.milo_dashboard_key, created mode 600 on
first use; the brain client prints the address).

    python -m robot.dashboard --port 8080 --shm NAME   # run by the brain client (--dashboard)
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import secrets
import sys
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

try:            # module level: FastAPI resolves the handlers' (postponed) annotations here
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError:                     # the brain client's side needs none of it
    WebSocket = WebSocketDisconnect = None

HERE = Path(__file__).resolve().parent
KEY_PATH = os.path.expanduser("~/.milo_dashboard_key")
STATIC_DIR = os.path.expanduser("~/.milo_dashboard")    # the brain's geometry and circuit (once a run)
MAX_TELEMETRY = 64 * 1024
POLL_S = 0.1
# shared memory: a float64 header, the telemetry JSON, the head camera's
# shared-memory name, the spike counts (int32)
_F = ["seq", "t", "tel_len", "n", "static_v", "head_len", "head_v"]
_I = {k: i for i, k in enumerate(_F)}
_HEADER = 256
_TEL_AT = _HEADER
_HEAD_AT = _TEL_AT + MAX_TELEMETRY
_COUNTS_AT = _HEAD_AT + 256


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


def make_app(key: str, shm_name: str | None = None):
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse, Response

    app = FastAPI(title="Milo dashboard", docs_url=None, redoc_url=None, openapi_url=None)
    # three.js for the 3D brain (MIT; vendored for the lab's views): public library code, no key
    from fastapi.staticfiles import StaticFiles
    app.mount("/vendor/three", StaticFiles(directory=HERE.parent / "visualization" / "static" / "vendor" / "three"),
              name="three")
    state = {"telemetry": None, "tv": 0, "jpeg": None, "jv": 0,
             "geometry": None, "gv": 0, "activity": None, "av": 0, "flow": None, "fv": 0, "circuit": None}
    if shm_name:
        reader = BrainReader(shm_name, state)

        @app.on_event("startup")
        async def _start():
            asyncio.get_running_loop().create_task(reader.run())

    def ok_key(k) -> bool:
        return isinstance(k, str) and hmac.compare_digest(k.encode(), key.encode())

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


def _attach(name):
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    from multiprocessing import resource_tracker
    shm = shared_memory.SharedMemory(name=name)
    resource_tracker.unregister(shm._name, "shared_memory")     # the brain client owns unlink()
    return shm


class BrainReader:
    """In the server: picks up what the brain client shares (DashboardShare)
    and the head camera's preview, and computes the brain's activity
    (robot/brainmap.ActivityMap) here, off the brain's process."""

    def __init__(self, shm_name: str, state: dict):
        self.name, self.state = shm_name, state
        self.shm = self.h = None
        self.seq = self.static_v = self.head_v = None
        self.head = self.head_h = None
        self.preview_seq = None
        self.amap = None

    def _snapshot(self):
        """(t, telemetry bytes, counts) under the seqlock, or None."""
        h = self.h
        for _ in range(100):
            s1 = h[_I["seq"]]
            if int(s1) % 2:
                time.sleep(0.0005)
                continue
            tl, n = int(h[_I["tel_len"]]), int(h[_I["n"]])
            tel = bytes(self.shm.buf[_TEL_AT:_TEL_AT + tl])
            counts = np.frombuffer(self.shm.buf, np.int32, n, _COUNTS_AT).copy() if n else None
            t = float(h[_I["t"]])
            if h[_I["seq"]] == s1:
                return int(s1), t, tel, counts
        return None

    def _load_static(self):
        st = self.state
        with open(os.path.join(STATIC_DIR, "geometry.bin"), "rb") as fh:
            g = fh.read()
        z = np.load(os.path.join(STATIC_DIR, "circuit.npz"), allow_pickle=False)
        c = {k: z[k] for k in z.files}
        from robot.brainmap import ActivityMap
        self.amap = ActivityMap.from_circuit(c)
        st["geometry"], st["circuit"] = g, c
        st["gv"] += 1

    def _brain(self):
        """One poll: static data, telemetry and activity (in a worker thread)."""
        st = self.state
        if self.static_v != self.h[_I["static_v"]] and self.h[_I["static_v"]] > 0:
            self.static_v = self.h[_I["static_v"]]
            self._load_static()
        snap = self._snapshot()
        if snap is None or snap[0] == self.seq or not snap[2]:
            return
        self.seq, t, tel, counts = snap
        tel = _clean(json.loads(tel))
        if self.amap is not None and counts is not None and len(counts) == len(self.amap.system):
            systems, levels = self.amap.update(counts, now=t)
            if systems is not None:
                tel["activity"] = systems
                st["activity"], st["flow"] = levels, self.amap.flow or b"\0" * 8
                st["av"] += 1
                st["fv"] += 1
        st["telemetry"] = tel
        st["tv"] += 1

    def _camera(self):
        from robot.head import _F as HF, read_preview
        h = self.h
        if self.head_v != h[_I["head_v"]]:
            self.head_v = h[_I["head_v"]]
            n = int(h[_I["head_len"]])
            name = bytes(self.shm.buf[_HEAD_AT:_HEAD_AT + n]).decode() if n else None
            if self.head is not None:
                del self.head_h
                self.head.close()
                self.head = self.head_h = None
            if name:
                self.head = _attach(name)
                self.head_h = np.ndarray((len(HF),), np.float64, buffer=self.head.buf)
        if self.head is None:
            return
        from robot.head import _I as HI
        ps = self.head_h[HI["preview_seq"]]
        if ps == self.preview_seq:
            return
        jpg = read_preview(self.head, self.head_h)
        if jpg and jpg.startswith(b"\xff\xd8"):
            self.preview_seq = ps
            self.state["jpeg"] = jpg
            self.state["jv"] += 1

    def _poll(self):
        if self.shm is None:
            try:
                self.shm = _attach(self.name)
            except FileNotFoundError:
                return
            self.h = np.ndarray((len(_F),), np.float64, buffer=self.shm.buf)
        self._brain()
        self._camera()

    async def run(self):
        while True:
            try:
                await asyncio.to_thread(self._poll)
            except Exception as ex:                   # display only: keep serving
                print("dashboard: reading the brain failed:", repr(ex)[:200], file=sys.stderr, flush=True)
                await asyncio.sleep(1.0)
            await asyncio.sleep(POLL_S)


class DashboardShare:
    """The brain client's side: starts the server and shares with it, through
    memory: telemetry (a JSON dump) and the engine's spike counts, ~5 times
    a second; the brain's geometry and circuit once (files); the head
    camera's shared-memory name (the server reads the preview itself). No
    thread and no HTTP in the brain's process."""

    def __init__(self, port: int = 8080, n_counts: int = 0, key: str | None = None, start_server: bool = True):
        import subprocess
        self.key = key or dashboard_key()
        self.n = int(n_counts)
        self.shm = shared_memory.SharedMemory(create=True, size=_COUNTS_AT + 4 * max(self.n, 1))
        self.shm.buf[:_COUNTS_AT] = bytes(_COUNTS_AT)
        self._h = np.ndarray((len(_F),), np.float64, buffer=self.shm.buf)
        self._counts = np.ndarray((max(self.n, 1),), np.int32, buffer=self.shm.buf, offset=_COUNTS_AT)
        self._h[_I["n"]] = self.n
        self.proc = None
        self.dropped = 0
        if start_server:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.log = open(os.path.expanduser("~/.milo_dashboard.log"), "ab")
            self.proc = subprocess.Popen([sys.executable, "-m", "robot.dashboard", "--port", str(port),
                                          "--shm", self.shm.name, "--parent", str(os.getpid())],
                                         cwd=root, stdout=self.log, stderr=self.log)

    def put_static(self, geometry: bytes, circuit: bytes) -> None:
        """The brain's geometry and circuit (robot/brainmap.BrainActivity),
        once: written as files the server loads (atomically)."""
        os.makedirs(STATIC_DIR, mode=0o700, exist_ok=True)
        for name, data in (("geometry.bin", geometry), ("circuit.npz", circuit)):
            tmp = os.path.join(STATIC_DIR, name + ".tmp")
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, os.path.join(STATIC_DIR, name))
        self._h[_I["static_v"]] += 1

    def set_head(self, shm_name: str | None) -> None:
        b = (shm_name or "").encode()[:255]
        self.shm.buf[_HEAD_AT:_HEAD_AT + len(b)] = b
        self._h[_I["head_len"]] = len(b)
        self._h[_I["head_v"]] += 1

    def publish(self, telemetry: dict, spike_counts=None) -> None:
        """telemetry: JSON-able (numpy scalars and NaN allowed: the server
        cleans); spike_counts: the engine's cumulative counts."""
        body = json.dumps(telemetry, default=_jsonable).encode()
        if len(body) > MAX_TELEMETRY:
            self.dropped += 1
            return
        h = self._h
        h[_I["seq"]] += 1                                   # odd: writing
        self.shm.buf[_TEL_AT:_TEL_AT + len(body)] = body
        h[_I["tel_len"]] = len(body)
        if spike_counts is not None and self.n:
            np.copyto(self._counts, spike_counts, casting="unsafe")
        h[_I["t"]] = time.monotonic()                       # system-wide clock: the server's too
        h[_I["seq"]] += 1                                   # even: ready

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        self.proc = None
        if self.shm is not None:
            del self._h, self._counts
            self.shm.close()
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
            self.shm = None


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--parent", type=int, default=0, help="exit when this process is gone (the brain client)")
    ap.add_argument("--shm", default=None, help="the brain client's shared memory (DashboardShare)")
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
    uvicorn.run(make_app(dashboard_key(), a.shm), host=a.host, port=a.port, log_level="warning",
                timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
