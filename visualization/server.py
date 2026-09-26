"""
Fruit Fly Laboratory -- web server.

Runs the whole-brain simulation in a background thread and streams telemetry to
the browser over a WebSocket. The browser is a display and a control panel; it
contains no biology.

Run:  python -m visualization.server
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

import config
from brain.neurons.registry import load_connectome
from brain.sensory.modalities import BY_KEY, census
from simulation.engine.session import Session

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Fruit Fly Laboratory")

# --------------------------------------------------------------------------- #
# Simulation runner
# --------------------------------------------------------------------------- #


class Runner:
    """
    Owns the Session and advances it on a background thread, paced to the
    wall clock: at pace 1.0 one simulated second takes one real second.
    """

    PACE_CHUNK_MS = 5.0      # simulate in chunks this size (keeps the native
                             # engine's compute/readout pipeline busy)
    MAX_CHUNK_MS = 25.0
    MAX_BEHIND_MS = 100.0    # further behind than this: drop the backlog

    def __init__(self):
        self.connectome = load_connectome()
        self.session = Session(self.connectome, seed=0)
        self.lock = threading.Lock()
        self.running = False
        self.sim_ms_per_tick = 20.0    # chunk size when unpaced (pace 0)
        self.pace = float(os.environ.get("FLY_PACE", "1.0"))   # 0 = as fast as possible
        self.latest = None
        self.thread = None
        self._stop = threading.Event()
        self._anchor = None            # (wall s, sim ms) the pacing clock counts from
        self._rt_window = deque()      # (wall s, sim ms), last ~2 s, for the speed readout
        self.dropped_ms = 0.0          # sim time skipped because we fell behind
        self._replay = []              # recorded frames of the current experiment
        self.camera = None             # CameraFeed when the live camera is on
        self._camera_stim = None
        self.head = None               # robot.head.HeadFeed when the pan/tilt head is on
        self._head_parts = []          # (encoder, stimulus) pairs it adds
        self.head_ctrl = None

    # ------------------------------------------------------------------ loop
    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _chunk(self):
        """Sim ms to advance now, or None to wait (sleeps as needed)."""
        if self.pace <= 0:
            return self.sim_ms_per_tick
        now = time.perf_counter()
        t = self.session.engine.t_ms
        if self._anchor is None:
            self._anchor = (now, t)
        due = self._anchor[1] + (now - self._anchor[0]) * 1000.0 * self.pace
        behind = due - t
        if behind > self.MAX_BEHIND_MS:           # can't keep up: don't chase it
            self.dropped_ms += behind - self.PACE_CHUNK_MS
            self._anchor = (now, t + self.PACE_CHUNK_MS)
            behind = self.PACE_CHUNK_MS
        if behind < self.PACE_CHUNK_MS:
            time.sleep((self.PACE_CHUNK_MS - behind) / 1000.0 / self.pace)
            return None
        return float(min(int(behind), self.MAX_CHUNK_MS))

    def _loop(self):
        while not self._stop.is_set():
            if not self.running:
                self._anchor = None
                self._rt_window.clear()
                time.sleep(0.02)
                continue
            chunk = self._chunk()
            if chunk is None:
                continue
            with self.lock:
                frames = self.session.advance(chunk)
            if frames and self.head_ctrl is not None:
                try:
                    self.head_ctrl.update(frames[-1]["channels"])
                except Exception as ex:          # never let the head stop the brain
                    print("[head] controller error:", ex)
            if frames:
                f = frames[-1]
                now = time.perf_counter()
                w = self._rt_window
                w.append((now, f["t_ms"]))
                while len(w) > 2 and now - w[0][0] > 2.0:
                    w.popleft()
                span = w[-1][0] - w[0][0]
                f["realtime_factor"] = (round((w[-1][1] - w[0][1]) / 1000.0 / span, 3)
                                        if span > 0.2 else None)
                f["pace"] = self.pace
                f["lag_ms"] = (round(self._anchor[1] + (now - self._anchor[0]) * 1000.0
                                     * self.pace - f["t_ms"], 1)
                               if self._anchor and self.pace > 0 else None)
                f["dropped_ms"] = round(self.dropped_ms, 1)
                self.latest = f
                self._replay.append({
                    "t_ms": f["t_ms"], "channels": f["channels"],
                    "body": {k: f["body"][k] for k in
                             ("x_mm", "y_mm", "z_mm", "heading_deg",
                              "behaviour", "proboscis_extension",
                              "wing_angle_deg", "airborne")},
                    "active_neurons": f["active_neurons"],
                    "dn_rates": f["dn_rates"],
                })
                if len(self._replay) > 20000:
                    del self._replay[:5000]

    # --------------------------------------------------------------- controls
    def play(self):
        with self.lock:
            self.session.paused = False
        self.running = True

    def set_pace(self, pace: float):
        self.pace = max(0.0, min(4.0, float(pace)))
        self._anchor = None            # re-anchor the clock at the new pace
        self._rt_window.clear()

    def pause(self):
        self.running = False

    def reset(self, seed: int = 0):
        self.running = False
        time.sleep(0.05)
        with self.lock:
            self.session.reset(seed=seed)
            self.session.body.reset()
            self._attach_camera()
        self.latest = None
        self._replay = []
        self._anchor = None
        self._rt_window.clear()
        self.dropped_ms = 0.0

    # Composite: what a fly actually encounters when food is placed in front of
    # it. Each component drives a real, separately cited FlyWire population;
    # bundling them is an ENVIRONMENT description (category C/D), not a
    # behavioural rule. The response still comes entirely from the connectome.
    FOOD_COMPONENTS = ("odor_vinegar", "touch_leg_taste", "taste_sugar")

    def apply_stimulus(self, kind: str, params: dict) -> dict:
        if kind == "food":
            out = []
            for key in self.FOOD_COMPONENTS:
                r = self.apply_stimulus(key, params)
                out.append({"component": key, **r})
            n = sum(x.get("n_neurons", 0) for x in out)
            return {"ok": True, "composite": "food", "components": out,
                    "n_neurons": n,
                    "note": ("food odour + tarsal contact chemosensation + "
                             "proboscis sugar, driven simultaneously")}
        with self.lock:
            if kind == "looming":
                s = self.session.add_looming(
                    azimuth_deg=float(params.get("azimuth_deg", 45.0)),
                    elevation_deg=float(params.get("elevation_deg", 0.0)),
                    half_size_mm=float(params.get("half_size_mm", 5.0)),
                    speed_mm_s=float(params.get("speed_mm_s", 250.0)),
                    start_distance_mm=float(params.get("start_distance_mm", 50.0)),
                )
                return {"ok": True, "stimulus": s.state(self.session.engine.t_ms)}
            m = BY_KEY.get(kind)
            if m is None:
                return {"ok": False, "error": "unknown stimulus %r" % kind}
            if not m.supported:
                return {"ok": False, "not_modeled": True,
                        "label": m.label, "reason": m.unsupported_reason}
            s = self.session.add_modality(
                kind, intensity=float(params.get("intensity", 1.0)),
                duration_ms=float(params.get("duration_ms", 300.0)))
            from brain.sensory.modalities import resolve_neurons
            return {"ok": True, "n_neurons": int(len(resolve_neurons(m, self.connectome))),
                    "label": m.label, "citation": m.citation,
                    "stimulus": s.state(self.session.engine.t_ms)}

    def clear_stimuli(self):
        with self.lock:
            self.session.clear_stimuli()
            self._attach_camera()

    # ---------------------------------------------------------------- camera
    # The camera is a sensor, not a one-shot stimulus: while it is on it stays
    # attached across Reset and Clear, driving LC4/LPLC2 through the same
    # LoomingEncoder as the thrown rock (see brain/sensory/camera.py).
    def _attach_head(self):
        if self.head is None:
            return
        for enc, st in self._head_parts:
            if not any(x is st for _, x in self.session.encoders):
                self.session.add_stimulus(enc, st)

    def head_on(self) -> dict:
        if self.head is not None and self.head.alive:
            return {"ok": True, "head": self.head.state()}
        from brain.sensory.encoders import LoomingEncoder
        from brain.sensory.retinotopy import load_retinotopy
        from robot.head import (HeadController, HeadFeed, HeadLoomingStimulus,
                                ObjectEncoder, RestingOlfaction)
        try:
            feed = HeadFeed()
        except Exception as ex:
            return {"ok": False, "error": str(ex)}
        t0 = time.time()
        while time.time() - t0 < 25.0:             # camera reset + detector start-up
            if not feed.alive:
                err = feed.error(); feed.close()
                return {"ok": False, "error": "head process exited: " + err[-400:]}
            if feed.state().get("ready"):
                break
            time.sleep(0.1)
        else:
            feed.close()
            return {"ok": False, "error": "no frames from the head camera within 25 s"}
        loom = LoomingEncoder(self.connectome, load_retinotopy(self.connectome))
        obj = ObjectEncoder(self.connectome, feed)
        rest = RestingOlfaction(self.connectome)
        parts = [(loom, HeadLoomingStimulus(feed)), (obj, obj), (rest, rest)]
        self.hearing = None
        try:                                       # the camera's microphone -> JO-A/JO-B
            from robot.hearing import HearingEncoder, HearingFeed
            self.hearing = HearingFeed()
            ear = HearingEncoder(self.connectome, self.hearing)
            parts.append((ear, ear))
        except Exception as ex:
            print("[head] no hearing:", ex)
        with self.lock:
            self.head = feed
            self._head_parts = parts
            self._attach_head()
            self.head_ctrl = HeadController(feed, target_fn=lambda: obj.last.get("target") is not None)
        self.play()
        return {"ok": True, "head": feed.state()}

    def head_off(self) -> dict:
        with self.lock:
            feed, self.head = self.head, None
            parts = {id(st) for _, st in self._head_parts}
            self.session.encoders = [(e, st) for e, st in self.session.encoders if id(st) not in parts]
            if not self.session.encoders:
                self.session.engine.clear_poisson()
            self._head_parts = []
            self.head_ctrl = None
        if feed is not None:
            feed.close()
        if getattr(self, "hearing", None) is not None:
            self.hearing.close()
            self.hearing = None
        return {"ok": True}

    def _attach_camera(self):
        self._attach_head()
        if self.camera is None:
            return
        from brain.sensory.camera import CameraLoomingStimulus
        from brain.sensory.encoders import LoomingEncoder
        from brain.sensory.retinotopy import load_retinotopy
        if self._camera_stim is None:
            self._camera_enc = LoomingEncoder(self.connectome, load_retinotopy(self.connectome))
            self._camera_stim = CameraLoomingStimulus(self.camera)
        if not any(st is self._camera_stim for _, st in self.session.encoders):
            self.session.add_stimulus(self._camera_enc, self._camera_stim)

    def camera_on(self) -> dict:
        if self.camera is not None and self.camera.alive:
            return {"ok": True, "camera": self.camera.state()}
        from brain.sensory.camera import CameraFeed
        feed = CameraFeed(yaw_deg=float(os.environ.get("FLY_CAMERA_YAW", 0)),
                          pitch_deg=float(os.environ.get("FLY_CAMERA_PITCH", 0)),
                          rotate=int(os.environ.get("FLY_CAMERA_ROTATE", 0)))
        t0 = time.time()
        while time.time() - t0 < 8.0:              # wait for the first frame
            if not feed.alive:
                err = feed.error()
                feed.close()
                return {"ok": False, "error": "camera process exited: " + err[-400:]}
            if feed.state()["frames"] > 0:
                break
            time.sleep(0.05)
        else:
            feed.close()
            return {"ok": False, "error": "no frames from the camera within 8 s"}
        with self.lock:
            self.camera = feed
            self._camera_stim = None
            self._attach_camera()
        self.play()
        return {"ok": True, "camera": feed.state()}

    def camera_off(self) -> dict:
        with self.lock:
            feed, self.camera = self.camera, None
            if self._camera_stim is not None:
                self.session.encoders = [(e, st) for e, st in self.session.encoders
                                         if st is not self._camera_stim]
                if not self.session.encoders:
                    self.session.engine.clear_poisson()
            self._camera_stim = None
        if feed is not None:
            feed.close()
        return {"ok": True}

    # ----------------------------------------------------------------- world
    def world_start(self, p: dict) -> dict:
        """Put the fly in the closed-loop world (fly/world/)."""
        from fly.world.world import WorldConfig
        cfg = WorldConfig(
            size_mm=float(p.get("size_mm", 600.0)),
            n_fruit=int(p.get("fruits", 4)),
            wind_speed_mm_s=float(p.get("wind_mm_s", 150.0)),
            wind_from_deg=float(p.get("wind_from_deg", 180.0)),
            turbulent=bool(p.get("turbulent", True)),
            predators=bool(p.get("predators", False)),
            predator_interval_s=(float(p.get("predator_min_s", 10.0)),
                                 float(p.get("predator_max_s", 25.0))))
        senses = {k: bool(p.get(k, True)) for k in ("smell", "taste", "wind", "vision")}
        self.running = False
        time.sleep(0.05)
        with self.lock:
            self.session.set_world(cfg, neural=bool(p.get("neural", True)), senses=senses,
                                   seed=int(p.get("seed", 0)),
                                   learning=bool(p.get("learning", True)))
            tol = float(p.get("quiesce_tol_mV", 1e-3))
            if hasattr(self.session.engine, "set_quiesce_tolerance"):
                self.session.engine.set_quiesce_tolerance(tol)
            self.session.reset(seed=int(p.get("seed", 0)))
        self.latest = None
        self._replay = []
        self._anchor = None
        self.play()
        return {"ok": True, "world": self.session.world.state(),
                "senses": self.session.world_senses.provenance,
                "body": self.session.body.provenance}

    def world_stop(self) -> dict:
        self.running = False
        time.sleep(0.05)
        with self.lock:
            self.session.set_world(None)
            if hasattr(self.session.engine, "set_quiesce_tolerance"):
                self.session.engine.set_quiesce_tolerance(0.0)
            self.session.reset(seed=0)
            self._attach_camera()
        self.latest = None
        return {"ok": True}

    def silence(self, cell_type: str, on: bool) -> dict:
        with self.lock:
            c = self.connectome
            cells = c.by_cell_type(cell_type)
            if cells.empty:
                return {"ok": False, "error": "no neurons of type %r" % cell_type}
            idx = cells["idx"].to_numpy()
            eng = self.session.engine
            if on:
                eng.silence(idx)
            elif hasattr(eng, "unsilence"):          # native engine
                eng.unsilence(idx)
            else:
                eng._silenced[idx] = False
            return {"ok": True, "cell_type": cell_type,
                    "n": int(len(idx)), "silenced": on}


RUNNER: Runner = None


@app.on_event("startup")
def _startup():
    global RUNNER
    RUNNER = Runner()
    RUNNER.start()
    print("[server] connectome loaded: %s" % RUNNER.connectome)
    print("[server] engine: %s" % RUNNER.session.engine.provenance.get(
        "backend", "python (simulation/engine/lif_engine.py)"))


@app.on_event("shutdown")
def _shutdown():
    if RUNNER is None:
        return
    RUNNER.running = False
    RUNNER._stop.set()
    if RUNNER.camera is not None:
        RUNNER.camera_off()
    if RUNNER.head is not None:
        RUNNER.head_off()


# --------------------------------------------------------------------------- #
# REST API
# --------------------------------------------------------------------------- #


@app.post("/api/head/{on}")
def api_head(on: int):
    return RUNNER.head_on() if on else RUNNER.head_off()


@app.get("/api/head/state")
def api_head_state():
    h = RUNNER.head
    if h is None:
        return {"on": False}
    st = h.state()
    ctrl = RUNNER.head_ctrl
    obj = next((e for e, _ in RUNNER._head_parts if hasattr(e, "last")), None)
    f = RUNNER.latest or {}
    return {"on": True, "state": st, "moves": ctrl.log[-8:] if ctrl else [],
            "lc10a_target": obj.last if obj else None,
            "hearing": (RUNNER.hearing.state if getattr(RUNNER, "hearing", None) else None),
            "turn_bias": (f.get("channels") or {}).get("turn_bias"),
            "steer_baseline": round(ctrl.baseline, 2) if ctrl else None,
            "dn": {k: v for k, v in (f.get("dn_rates") or {}).items()
                   if k.startswith(("DNa01", "DNa02", "DNp09"))}}


@app.get("/api/head/preview.jpg")
def api_head_preview():
    h = RUNNER.head
    jpg = h.preview_jpeg() if h is not None else None
    if not jpg:
        return Response(status_code=204)
    return Response(content=jpg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/world/start")
async def api_world_start(payload: dict = None):
    return RUNNER.world_start(payload or {})


@app.post("/api/world/stop")
def api_world_stop():
    return RUNNER.world_stop()


@app.get("/api/world/state")
def api_world_state():
    s = RUNNER.session
    return {"on": s.world is not None,
            "world": None if s.world is None else s.world.state(),
            "neural": None if s.world is None else s.body.neural}


@app.get("/api/provenance")
def api_provenance():
    c = RUNNER.connectome
    return {
        "dataset": c.dataset,
        "manifest": c.manifest,
        "checksums_file": str(config.CHECKSUM_FILE),
        "session": RUNNER.session.provenance,
        "body": RUNNER.session.body.provenance,
    }


@app.get("/api/modalities")
def api_modalities():
    return {"modalities": census(RUNNER.connectome)}


@app.get("/api/state")
def api_state():
    return {
        "running": RUNNER.running,
        "t_ms": RUNNER.session.engine.t_ms,
        "pace": RUNNER.pace,
        "sim_ms_per_tick": RUNNER.sim_ms_per_tick,
        "latest": RUNNER.latest,
    }


@app.post("/api/play")
def api_play():
    RUNNER.play()
    return {"running": True}


@app.post("/api/pause")
def api_pause():
    RUNNER.pause()
    return {"running": False}


@app.post("/api/reset")
def api_reset():
    RUNNER.reset()
    return {"ok": True, "t_ms": 0.0}


@app.post("/api/pace/{value}")
def api_pace(value: float):
    """Simulated seconds per wall-clock second (1 = real time, 0 = unpaced)."""
    RUNNER.set_pace(value)
    return {"pace": RUNNER.pace}


@app.post("/api/speed/{value}")
def api_speed(value: float):
    RUNNER.sim_ms_per_tick = max(0.1, min(20.0, float(value)))
    return {"sim_ms_per_tick": RUNNER.sim_ms_per_tick}


@app.post("/api/stimulus/{kind}")
async def api_stimulus(kind: str, payload: dict = None):
    res = RUNNER.apply_stimulus(kind, payload or {})
    if res.get("ok"):
        RUNNER.play()
    return res


@app.post("/api/camera/{on}")
def api_camera(on: int):
    return RUNNER.camera_on() if int(on) else RUNNER.camera_off()


@app.get("/api/camera/state")
def api_camera_state():
    cam = RUNNER.camera
    if cam is None:
        return {"on": False}
    if not cam.alive:
        return {"on": False, "error": cam.error()[-400:]}
    return {"on": True, **cam.state()}


@app.get("/api/camera/preview.jpg")
def api_camera_preview():
    cam = RUNNER.camera
    jpg = cam.preview_jpeg() if cam is not None else None
    if not jpg:
        return Response(status_code=204)
    return Response(content=jpg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/stimulus/clear")
def api_clear():
    RUNNER.clear_stimuli()
    return {"ok": True}


@app.post("/api/silence/{cell_type}/{on}")
def api_silence(cell_type: str, on: int):
    return RUNNER.silence(cell_type, bool(int(on)))


@app.get("/api/replay")
def api_replay():
    return {"n": len(RUNNER._replay), "frames": RUNNER._replay[-4000:]}


@app.get("/api/raster")
def api_raster(last_ms: float = 300.0):
    with RUNNER.lock:
        pts = RUNNER.session.raster(last_ms)
    c = RUNNER.connectome
    sc = c.neurons["super_class"].astype(str).to_numpy()
    return {"points": [[t, i, sc[i]] for t, i in pts[-4000:]]}


@app.get("/api/circuit/{cell_type}")
def api_circuit(cell_type: str, top: int = 25):
    """Real connectivity of a cell type, for the circuit inspector."""
    c = RUNNER.connectome
    cells = c.by_cell_type(cell_type)
    if cells.empty:
        return {"error": "no neurons of type %r in %s" % (cell_type, c.dataset)}
    idx = cells["idx"].to_numpy()
    ptypes = c.neurons["primary_type"].astype(str).to_numpy()

    def agg(mat, axis_idx):
        import pandas as pd
        co = mat.tocoo()
        d = pd.DataFrame({"t": ptypes[axis_idx(co)], "s": np.abs(co.data),
                          "w": co.data})
        g = d.groupby("t").agg(syn=("s", "sum"), n=("s", "size"),
                               signed=("w", "sum")).reset_index()
        g = g.sort_values("syn", ascending=False).head(top)
        return [{"type": r.t, "synapses": int(r.syn), "connections": int(r.n),
                 "sign": "excitatory" if r.signed > 0 else "inhibitory"}
                for r in g.itertuples()]

    return {
        "cell_type": cell_type,
        "n_cells": int(len(cells)),
        "root_ids": [int(x) for x in cells["root_id"].head(20)],
        "sides": cells["side"].astype(str).value_counts().to_dict(),
        "inputs": agg(c.w[:, idx], lambda co: co.row),
        "outputs": agg(c.w[idx, :], lambda co: co.col),
        "dataset": c.dataset,
    }


@app.get("/api/neurons/positions")
def api_positions():
    """Binary Float32 xyz positions + Uint8 super-class codes for the 3D view."""
    c = RUNNER.connectome
    n = c.neurons
    xyz = n[["pos_x_nm", "pos_y_nm", "pos_z_nm"]].to_numpy(dtype=np.float64)
    xyz = np.nan_to_num(xyz) / 1000.0                      # nm -> um
    centre = np.nanmean(xyz, axis=0)
    xyz = (xyz - centre).astype(np.float32)

    classes = sorted(n["super_class"].astype(str).unique())
    code = {k: i for i, k in enumerate(classes)}
    codes = n["super_class"].astype(str).map(code).to_numpy().astype(np.uint8)

    header = json.dumps({"n": int(len(n)), "classes": classes}).encode()
    # Pad so the Float32 block starts on a 4-byte boundary (typed-array rule).
    header += b" " * ((-(len(header) + 4)) % 4)
    body = struct.pack("<I", len(header)) + header + xyz.tobytes() + codes.tobytes()
    return Response(content=body, media_type="application/octet-stream")


@app.get("/api/neurons/lookup")
def api_lookup(root_id: int):
    c = RUNNER.connectome
    try:
        i = c.idx(root_id)
    except KeyError:
        return {"error": "root_id %d is not in %s" % (root_id, c.dataset)}
    r = c.neurons.iloc[i]
    return {
        "root_id": int(r["root_id"]), "idx": int(i),
        "cell_type": str(r["primary_type"]), "super_class": str(r["super_class"]),
        "side": str(r["side"]), "nt": str(r["nt_resolved"]),
        "neuropil": str(r["primary_neuropil"]),
        "spikes": int(RUNNER.session.engine.spike_counts[i]),
        "codex_url": "https://codex.flywire.ai/app/cell_details?root_id=%d" % r["root_id"],
    }


# --------------------------------------------------------------------------- #
# WebSocket telemetry
# --------------------------------------------------------------------------- #


def _track_since(track, t_after: float) -> list:
    """Body-track samples newer than t_after (newest last), rounded for JSON."""
    out = []
    for e in reversed(track):
        if e[0] <= t_after:
            break
        out.append([round(e[0], 1), round(e[1], 3), round(e[2], 3), round(e[3], 3),
                    round(e[4], 2), round(e[5], 2), round(e[6], 1), round(e[7], 1),
                    round(e[8], 3), round(e[9], 3), int(e[10]), e[11]])
        if len(out) >= 2000:
            break
    out.reverse()
    return out


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    last_t = -1.0
    last_track = -1.0
    try:
        while True:
            f = RUNNER.latest
            if f is not None and f["t_ms"] != last_t:
                last_t = f["t_ms"]
                with RUNNER.lock:
                    track = RUNNER.session.body_track
                    if track and track[-1][0] < last_track:     # session was reset
                        last_track = -1.0
                    body_track = _track_since(track, last_track)
                    if body_track:
                        last_track = body_track[-1][0]
                    raster = RUNNER.session.raster(250.0)
                    active = np.flatnonzero(
                        RUNNER.session.recorder.window_sum).astype(np.int32)
                payload = dict(f)
                payload["track"] = body_track
                payload["raster"] = raster[-1500:]
                payload["active_idx"] = active[
                    np.linspace(0, len(active) - 1, min(len(active), 3000)).astype(int)
                ].tolist() if len(active) else []
                await sock.send_text(json.dumps(payload, default=float))
            await asyncio.sleep(0.05)
    except (WebSocketDisconnect, RuntimeError):
        return


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))


def main():
    import uvicorn
    host = os.environ.get("FLY_HOST", "127.0.0.1")
    port = int(os.environ.get("FLY_PORT", "8000"))
    print("Fruit Fly Laboratory -> http://%s:%d" % (host, port))
    # Open WebSockets never close on their own, so bound the graceful shutdown;
    # otherwise a stopped server lingers, holding the camera and CPU cores.
    uvicorn.run(app, host=host, port=port, log_level="warning",
                timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
