"""
The real rover as the brain client's world: the same commands and replies as
sim/habitat_bridge/habitat_server.py (reset / step / lidar / body / path /
close), so sim/habitat_bridge/brain_client.py drives the robot unchanged
(--rover hw), in real time: step() sends the wheel command at once, waits
until the control period has passed on the wall clock, then returns the
latest sensors.

    brain_client  --(v, w)-->  RoverWorld.step  --> UGVBase.drive (robot/ugv.py)
                  <--(obs)---   odometry pose, D500 scan (robot/d500.py),
                                person from the head camera (robot/head.py,
                                robot/detector.py), when there is one

--rover fake runs the same loop with a simulated base, lidar and person in a
small room (FakeBase / FakeLidar / FakePerson below): the brain and the timing
are real, the world is not. It tests that everything keeps up with real time
on the Jetson before the robot arrives. With --camera the person comes from
the real camera instead (the detector's GPU and CPU load included).

The camera (robot.head.HeadFeed: its own process, the person detector at
~10 Hz) is aimed CAM_PITCH_DEG up, as Habitat's camera, so the person's
azimuth, elevation and angular size mean what they mean in Habitat. A person
counts as seen only while a detection is fresh (< 0.5 s) and the head is
still; a stalled camera is counted (camera_stale_s), not a stop: the camera
is not a safety layer (the lidar is).

What the robot cannot know is filled in, marked in the reply: dist (the
person's distance) is estimated from their apparent size when they are seen
(robot.safety.person_distance), else nan; human (their position) is nan;
collisions stay 0 (no bumper); the episode is never over. path returns no
route (the cortex falls back to a straight line, as when Habitat finds none).

The command reaching the wheels is the brain's latest: step k's command is
computed from the sensors of step k-1's end, one control period of latency
(100 ms), comparable to a fly's visuomotor delay.

Below every layer of the brain client, RoverWorld sends zero speed while the
base's feedback or the lidar's revolutions are older than 0.5 s (STALE_*),
and the ESP32 stops on its own if commands stop (robot/ugv.py). A lidar scan
is handed on once, with the odometry pose at the time it was taken
(lidar_pose), not the pose at the reply.
"""
from __future__ import annotations

import collections
import math
import time

import numpy as np

CTRL_HZ = 120.0                  # Habitat's control rate: step n = n / 120 s
CAM_HEIGHT_M, CAM_PITCH_DEG = 0.35, 20.0
TARGET_HEIGHT_M, TARGET_HALF_WIDTH_M = 1.45, 0.25     # as habitat_server
LIDAR_MAX_M = 8.0


class LocalConn:
    """multiprocessing.connection's send / recv, answered in-process."""

    def __init__(self, world):
        self.world = world
        self._reply = None

    def send(self, msg):
        self._reply = self.world.handle(msg)

    def recv(self):
        r, self._reply = self._reply, None
        return r

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.world.close()


def person_geometry(pose, person_xz, hfov_deg: float) -> dict:
    """The person's head and shoulders through the head camera, as
    habitat_server.Server.summary computes it (no occlusion test)."""
    x, z, yaw_deg = pose
    yaw = math.radians(yaw_deg)
    du, dw = person_xz[0] - x, -(person_xz[1] - z)
    ahead_h = du * math.cos(yaw) + dw * math.sin(yaw)
    right = du * math.sin(yaw) - dw * math.cos(yaw)
    dh = TARGET_HEIGHT_M - CAM_HEIGHT_M
    az = math.degrees(math.atan2(right, ahead_h))
    el = math.degrees(math.atan2(dh, math.hypot(du, dw))) - CAM_PITCH_DEG
    rng = math.sqrt(du * du + dw * dw + dh * dh)
    half = math.degrees(math.atan2(TARGET_HALF_WIDTH_M, max(rng, 0.05)))
    in_fov = ahead_h > 0 and abs(az) <= hfov_deg / 2
    return {"az": az, "el": el, "half": half, "visible": bool(in_fov), "in_fov": bool(in_fov),
            "dist": math.hypot(du, dw)}


# ------------------------------------------------------------------ fakes
class FakeBase:
    """UGVBase's interface: the commanded velocity is what the wheels do."""

    def __init__(self, start=(0.0, 0.0, 0.0), room=None, radius=0.13):
        self.pose0 = start
        self.room, self.radius = room, radius
        self.reset_pose()
        self.cmd = (0.0, 0.0)
        self.t = None
        self.blocked_s = 0.0

    def reset_pose(self):
        self.x, self.z, self.yaw = self.pose0

    def drive(self, v, w):
        if not (math.isfinite(v) and math.isfinite(w)):
            v, w = 0.0, 0.0
        self.cmd = (float(v), float(w))

    def stop(self):
        self.cmd = (0.0, 0.0)

    def advance(self, dt):
        v, w = self.cmd
        yaw = self.yaw + 0.5 * math.degrees(w) * dt
        nx = self.x + v * dt * math.cos(math.radians(yaw))
        nz = self.z - v * dt * math.sin(math.radians(yaw))
        if self.room is None or self.room.free(nx, nz, self.radius):
            self.x, self.z = nx, nz
        elif v != 0.0:
            self.blocked_s += dt                 # pressed against a wall: no translation
        self.yaw = (self.yaw + math.degrees(w) * dt) % 360.0

    def state(self):
        return {"pose": (self.x, self.z, self.yaw), "battery_v": 11.5, "wheels": self.cmd,
                "fb_age_s": 0.0, "feedbacks": 0}

    def pose_at(self, t_s):
        return (self.x, self.z, self.yaw)

    def close(self):
        pass


class FakeRoom:
    """A rectangular room with a box in it (x, z in m)."""

    def __init__(self, half_x=2.0, half_z=1.5, box=((0.8, 0.4), (1.2, 0.9))):
        self.hx, self.hz = half_x, half_z
        self.box = box

    def free(self, x, z, r):
        if abs(x) > self.hx - r or abs(z) > self.hz - r:
            return False
        (x0, z0), (x1, z1) = self.box
        return not (x0 - r < x < x1 + r and z0 - r < z < z1 + r)

    def ray(self, x, z, dx, dz, extra=()):
        """Distance along (dx, dz) to the nearest wall, box or extra circle
        ((cx, cz, r), ...)."""
        best = LIDAR_MAX_M
        for (lim, p, d) in ((self.hx, x, dx), (self.hz, z, dz)):
            if d > 1e-9:
                best = min(best, (lim - p) / d)
            elif d < -1e-9:
                best = min(best, (-lim - p) / d)
        (x0, z0), (x1, z1) = self.box
        tmin, tmax = 0.0, LIDAR_MAX_M
        hit = True
        for p, d, lo, hi in ((x, dx, x0, x1), (z, dz, z0, z1)):
            if abs(d) < 1e-9:
                if not (lo <= p <= hi):
                    hit = False
            else:
                t0, t1 = (lo - p) / d, (hi - p) / d
                tmin, tmax = max(tmin, min(t0, t1)), min(tmax, max(t0, t1))
        if hit and tmin <= tmax and tmin > 0:
            best = min(best, tmin)
        for cx, cz, r in extra:
            fx, fz = x - cx, z - cz
            b = fx * dx + fz * dz
            c = fx * fx + fz * fz - r * r
            disc = b * b - c
            if disc >= 0:
                t = -b - math.sqrt(disc)
                if t > 0:
                    best = min(best, t)
        return best


class FakePerson:
    """Someone walking slowly between two points of the room."""

    def __init__(self, a=(0.4, -1.2), b=(0.4, 1.3), speed=0.3):
        self.a, self.b, self.speed = np.array(a, float), np.array(b, float), speed
        self.t = 0.0

    def xz(self):
        L = float(np.linalg.norm(self.b - self.a))
        s = (self.speed * self.t) % (2 * L)
        f = s / L if s <= L else 2 - s / L
        return tuple(self.a + f * (self.b - self.a))

    def advance(self, dt):
        self.t += dt


class FakeLidar:
    """D500's interface (latest()) from the fake room, scanned at 10 Hz."""

    def __init__(self, room, base, person=None, beams=90):
        self.room, self.base, self.person, self.beams = room, base, person, beams
        self.scan = None

    def update(self, t_s):
        x, z, yaw = self.base.x, self.base.z, self.base.yaw
        legs = ()
        if self.person is not None:
            px, pz = self.person.xz()
            legs = ((px, pz, 0.12),)
        ang = -180.0 + 360.0 * np.arange(self.beams) / self.beams
        out = []
        for a in ang:
            b = math.radians(yaw - a)
            out.append(self.room.ray(x, z, math.cos(b), -math.sin(b), legs))
        self.scan = (t_s, np.asarray(out))

    def latest(self):
        return self.scan

    def close(self):
        pass


# ------------------------------------------------------------------ world
STALE_BASE_S = 0.5               # no base feedback for this long: the wheels are stopped
STALE_LIDAR_S = 0.5              # no new lidar revolution for this long: the same


class StepTiming:
    """The brain's work per step (s between a reply and the next step):
    running mean and maximum, and the 95th percentile of the last RECENT."""
    RECENT = 600

    def __init__(self):
        self.n, self.sum, self.max = 0, 0.0, 0.0
        self.recent = collections.deque(maxlen=self.RECENT)
        self._p95 = None

    def add(self, x: float) -> None:
        self.n += 1
        self.sum += x
        self.max = max(self.max, x)
        self.recent.append(x)
        if self.n % 50 == 1:
            self._p95 = None

    def stats(self) -> dict:
        if not self.n:
            return {"compute_ms_mean": None, "compute_ms_p95": None, "compute_ms_max": None}
        if self._p95 is None:
            self._p95 = float(np.percentile(self.recent, 95))
        return {"compute_ms_mean": round(1e3 * self.sum / self.n, 1),
                "compute_ms_p95": round(1e3 * self._p95, 1), "compute_ms_max": round(1e3 * self.max, 1)}


class RoverWorld:
    def __init__(self, base, lidar=None, person=None, hfov_deg: float = 53.0, fake: bool = False,
                 clock=time.monotonic, sleep=time.sleep, camera=None):
        self.base, self.lidar, self.person, self.camera = base, lidar, person, camera
        self._cam_aimed = False
        self._cam_state = None
        self._cam_dead = False
        self.hfov = hfov_deg
        self.fake = fake
        self.clock, self.sleep = clock, sleep
        self.lidar_beams = 0
        self.reset_counters()

    def reset_counters(self) -> None:
        self.t = 0.0
        self.deadline = None                      # set by the first step
        self.overruns, self.late_s = 0, 0.0
        self.timing = StepTiming()
        self._replied = None
        self._scan_t = None                       # the last scan handed to the brain
        self.stale, self.stale_s, self.stale_events = [], 0.0, 0
        self.camera_stale_s, self.camera_fps, self.seen_s = 0.0, 0.0, 0.0

    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        if cmd == "reset":
            return self.reset()
        if cmd == "step":
            return self.step(msg["v"], msg["w"], msg.get("n", 12))
        if cmd == "lidar":
            beams = int(msg.get("beams", 90))
            if self.lidar is not None and hasattr(self.lidar, "beams") and self.lidar.beams != beams:
                return {"error": f"the lidar was opened with {self.lidar.beams} beams, not {beams}"}
            self.lidar_beams = beams
            return {"beams": beams, "angles": [(-180.0 + 360.0 * k / beams) for k in range(beams)]}
        if cmd == "body":
            from sim.habitat_bridge.bodies import BODIES
            return {"name": "rover", **BODIES["rover"]}
        if cmd == "path":
            return {"ok": False, "geodesic": None, "waypoint": [float(msg["goal"][0]), float(msg["goal"][1])]}
        if cmd in ("topdown", "video"):
            return {"error": f"{cmd}: not on the real rover"}
        if cmd == "close":
            self.base.stop()
            return {"ok": True}
        return {"error": "unknown command %r" % cmd}

    def reset(self) -> dict:
        self.base.stop()
        if hasattr(self.base, "reset_pose"):
            self.base.reset_pose()
        self.reset_counters()
        if self.fake:
            self._fake_advance(0.0)
        return self.summary()

    def _stale_sensors(self, now: float) -> list:
        """Which sensors have gone quiet (a dead serial port, a stalled lidar
        motor, a rebooted base): the layers above would act on a frozen world."""
        if self.fake:
            return []
        out = []
        age = self.base.state().get("fb_age_s")
        if age is None or age > STALE_BASE_S:
            out.append("base")
        if self.lidar_beams and self.lidar is not None:
            s = self.lidar.latest()
            if s is None or now - s[0] > STALE_LIDAR_S:
                out.append("lidar")
        return out

    def step(self, v: float, w: float, n: int) -> dict:
        period = n / CTRL_HZ
        now = self.clock()
        if self._replied is not None:
            self.timing.add(now - self._replied)
        stale = self._stale_sensors(now)
        if stale:
            # a hard layer below the brain and every other layer: no motion
            # on sensors that stopped (review 2026-09-30)
            self.base.drive(0.0, 0.0)
            if not self.stale:
                self.stale_events += 1
            self.stale_s += period
        else:
            self.base.drive(v, w)
        self.stale = stale
        if self.deadline is None:
            self.deadline = now
        self.deadline += period
        if now > self.deadline:
            # the brain took longer than the period: count it, do not try to catch up
            self.overruns += 1
            self.late_s += now - self.deadline
            self.deadline = now
        else:
            self.sleep(self.deadline - now)
        if self.fake:
            self._fake_advance(period)
        self.t += period
        out = self.summary()
        if self.camera is not None:
            if self._cam_state is None:
                if self._cam_aimed or self._cam_dead:  # ready once (the first ~8 s start it up), or died
                    self.camera_stale_s += period
            elif self._cam_state.get("stale"):
                self.camera_stale_s += period
            if out["visible"]:
                self.seen_s += period
            out["stats"].update(camera_stale_s=round(self.camera_stale_s, 1), person_seen_s=round(self.seen_s, 1))
        self._replied = self.clock()
        return out

    def _fake_advance(self, dt: float) -> None:
        steps = max(1, int(round(dt / 0.01))) if dt > 0 else 0
        for _ in range(steps):
            self.base.advance(dt / steps)
            if self.person is not None:
                self.person.advance(dt / steps)
        if self.lidar is not None:
            self.lidar.update(self.t + dt)

    def _camera_person(self) -> dict | None:
        """The person through the head camera, in Habitat's camera frame
        (azimuth + = right of the body's heading, elevation from an axis
        pitched CAM_PITCH_DEG up), or None."""
        s = self.camera.state()
        self._cam_state = s if s.get("ready") else None
        if not getattr(self.camera, "alive", True) and not self._cam_dead:
            # the camera process ended: say why once (the run goes on blind)
            self._cam_dead = True
            err = self.camera.error() if hasattr(self.camera, "error") else ""
            print("head camera stopped:", err.strip()[-500:] or "no message", flush=True)
        if not s.get("ready"):
            return None
        self.camera_fps = s.get("fps", 0.0)
        if not self._cam_aimed:
            self.camera.command(-s["pan_deg"], CAM_PITCH_DEG - s["tilt_deg"])
            self._cam_aimed = True
        if s.get("stale") or s["moving"] or not s["person_active"] or s["person_stale"]:
            return None
        from robot.safety import person_distance
        half = float(s["person_half_deg"])
        d = person_distance(half)
        if d is None:                             # too small to tell: not a person to act on
            return None
        return {"az": s["person_az"] + s["pan_deg"], "el": s["person_el"] + s["tilt_deg"] - CAM_PITCH_DEG,
                "half": half, "visible": True, "in_fov": True, "dist": float(d)}

    def summary(self) -> dict:
        st = self.base.state()
        pose = tuple(float(x) for x in st["pose"])
        seen = {"az": 0.0, "el": 0.0, "half": 0.0, "visible": False, "in_fov": False, "dist": float("nan")}
        if self.camera is not None:
            seen.update(self._camera_person() or {})
        elif self.person is not None:
            g = person_geometry(pose, self.person.xz(), self.hfov)
            seen.update({k: g[k] for k in ("az", "el", "half", "visible", "in_fov")})
            if g["visible"]:
                from robot.safety import person_distance
                d = person_distance(g["half"])
                seen["dist"] = float("nan") if d is None else float(d)
        lid = lid_pose = None
        if self.lidar_beams and self.lidar is not None:
            s = self.lidar.latest()
            if s is not None and s[0] != self._scan_t:
                # a new revolution only (the lidar and the control loop both
                # run at ~10 Hz), with the pose when it was taken
                self._scan_t = s[0]
                lid = [round(float(r), 3) for r in np.minimum(s[1], LIDAR_MAX_M)]
                lp = self.base.pose_at(s[0]) if hasattr(self.base, "pose_at") else pose
                lid_pose = [float(x) for x in lp]
        return {"t": round(self.t, 4), "dist": seen["dist"], "lidar": lid, "lidar_pose": lid_pose,
                "az": seen["az"], "el": seen["el"], "half": seen["half"], "visible": seen["visible"],
                "in_fov": seen["in_fov"], "robot": [pose[0], pose[1], pose[2]],
                "human": [float("nan"), float("nan")], "over": False,
                "collided": False, "collisions": 0, "scene_contacts": None, "scene_bumps": None,
                "blocked_s": round(getattr(self.base, "blocked_s", 0.0), 2) if self.fake else None,
                "stats": {"overruns": float(self.overruns), "late_s": round(self.late_s, 3),
                          "stale_s": round(self.stale_s, 1), "stale_events": float(self.stale_events),
                          **({"camera_fps": round(self.camera_fps, 1), "camera_stale_s": round(self.camera_stale_s, 1),
                              "person_seen_s": round(self.seen_s, 1)} if self.camera is not None else {}),
                          **self.timing.stats()},
                "stale": list(self.stale), "battery_v": st.get("battery_v"), "rover": True}

    def close(self) -> None:
        try:
            self.base.stop()
        finally:
            try:
                self.base.close()
            finally:
                try:
                    if self.lidar is not None:
                        self.lidar.close()
                finally:
                    if self.camera is not None:
                        self.camera.close()


def open_camera(camera: str | None):
    """robot.head.HeadFeed on a device ("auto": the Orbit if plugged in), or
    None. "auto" without a camera is not an error (the robot runs blind)."""
    if not camera or camera == "none":
        return None
    from robot.head import HeadFeed, find_orbit
    dev = find_orbit() if camera == "auto" else camera
    if dev is None:
        print("no head camera found: the person is never seen", flush=True)
        return None
    try:
        return HeadFeed(dev, person=True, looming=False)
    except Exception as ex:                       # never stop the robot over its camera
        print(f"head camera {dev} failed to start ({ex}): the person is never seen", flush=True)
        return None


def make_world(kind: str, ugv_port: str = "/dev/ttyTHS1", lidar_port: str | None = None,
               beams: int = 90, hfov_deg: float = 53.0, camera: str | None = None) -> RoverWorld:
    """kind "fake": simulated room, base, lidar and person (the person from
    the real camera when there is one); "hw": the robot."""
    cam = open_camera(camera)
    try:
        if kind == "fake":
            room = FakeRoom()
            base = FakeBase(start=(-1.0, 0.5, 0.0), room=room)
            person = None if cam is not None else FakePerson()
            return RoverWorld(base, FakeLidar(room, base, person, beams), person, hfov_deg, fake=True,
                              camera=cam)
        from robot.ugv import UGVBase
        base = UGVBase(ugv_port)
        lidar = None
        if lidar_port:
            from robot.d500 import D500
            try:
                lidar = D500(lidar_port, beams)
            except Exception:
                base.close()                      # stops the wheels and frees the port
                raise
        return RoverWorld(base, lidar, None, hfov_deg, camera=cam)
    except Exception:
        if cam is not None:
            cam.close()
        raise
