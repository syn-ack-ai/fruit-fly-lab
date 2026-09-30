"""
The Waveshare UGV Rover's base: its ESP32 sub-controller, driven over a serial
port with newline-terminated JSON (Waveshare wiki, "Sub-controller JSON
Command Set"; the ugv_jetson repository's base_ctrl.py).

    {"T":13,"X":v,"Z":w}     drive: X m/s forward, Z rad/s (+ = counter-
                             clockwise, left), the ESP32's closed speed loop
    {"T":136,"cmd":ms}       heartbeat: the base stops if no command arrives
                             within ms (a crashed brain stops the wheels)
    {"T":131,"cmd":1}        continuous feedback on (~10 Hz), messages T=1001
    {"T":143,"cmd":0}        no echo of our commands
    {"T":133,"X":pan,"Y":tilt,"SPD":0,"ACC":0}   the pan-tilt head, degrees

Feedback T=1001 carries (field names as sent): L, R wheel speeds (m/s), odl,
odr cumulative wheel distance (cm), v the battery voltage, and IMU fields
(ax.., gx.., mx.., r/p/y). The voltage's unit is not documented: one bench
log shows "v":1089 (centivolts, a 3S pack); values above 100 are read as
centivolts. Odometry comes from the wheel distances (odl/odr) where present,
else from the wheel speeds; the pose is Habitat's convention (x, z, yaw deg,
yaw counter-clockwise, forward = (cos yaw, -sin yaw)), so the lidar's world
points (robot/lidar.py) work unchanged.

C. APPROXIMATIONS: TRACK_M (the effective wheel track of a six-wheel skid
steer, which slips when turning) is a guess until measured on the robot
(turn in place 10 turns, compare odometry with the real angle). Skid-steer
odometry drifts; the IMU's gyro (gz) would be better for yaw once its units
are confirmed on the hardware.

    python -m robot.ugv --port /dev/ttyTHS1 --test     # feedback, then a 1 s nudge
"""
from __future__ import annotations

import collections
import json
import math
import threading
import time

TRACK_M = 0.175          # effective wheel track (m); calibrate on the robot
V_MAX = 0.5              # m/s, as the brain client's limit
W_MAX = math.radians(180.0)
HEARTBEAT_MS = 500
RESEND_S = 2.0           # feedback quiet this long: send the settings again (the ESP32 may have reset)


def voltage(raw) -> float | None:
    """The feedback's v in volts (centivolts above 100)."""
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return None
    return x / 100.0 if x > 100.0 else x


class Odometry:
    """Pose from wheel distances (preferred) or wheel speeds."""

    def __init__(self, track_m: float = TRACK_M):
        self.track = track_m
        self.reset()

    def reset(self) -> None:
        self.x = self.z = self.yaw = 0.0          # m, m, rad
        self._od = None                           # last (odl, odr) in m
        self._t = None

    def _move(self, dl: float, dr: float) -> None:
        ds = 0.5 * (dl + dr)
        dth = (dr - dl) / self.track              # + = counter-clockwise
        mid = self.yaw + 0.5 * dth
        self.x += ds * math.cos(mid)
        self.z -= ds * math.sin(mid)
        self.yaw += dth

    def update(self, fb: dict, t_s: float) -> None:
        if "odl" in fb and "odr" in fb:
            od = (float(fb["odl"]) / 100.0, float(fb["odr"]) / 100.0)
            if self._od is not None:
                dl, dr = od[0] - self._od[0], od[1] - self._od[1]
                if abs(dl) < 1.0 and abs(dr) < 1.0:   # a counter reset or glitch is not a 1 m jump
                    self._move(dl, dr)
            self._od = od
        elif "L" in fb and "R" in fb and self._t is not None:
            dt = min(max(t_s - self._t, 0.0), 0.5)
            self._move(float(fb["L"]) * dt, float(fb["R"]) * dt)
        self._t = t_s

    def pose(self) -> tuple:
        return (self.x, self.z, math.degrees(self.yaw))


class UGVBase:
    """The base on a serial port (or any transport with write(bytes) and
    read(n) -> bytes, for tests). A reader thread keeps the latest feedback."""

    def __init__(self, port: str = "/dev/ttyTHS1", baud: int = 115200, transport=None,
                 heartbeat_ms: int = HEARTBEAT_MS, track_m: float = TRACK_M):
        if transport is None:
            import serial                         # pyserial
            transport = serial.Serial(port, baud, timeout=0.05)
        self.io = transport
        self.heartbeat_ms = int(heartbeat_ms)
        self.odo = Odometry(track_m)
        self.poses = collections.deque(maxlen=64)   # (t, pose) at each feedback: pose_at()
        self.fb = {}
        self.fb_t = None
        self.n_fb = 0
        self.bad_lines = 0
        self._buf = b""
        self._wlock = threading.Lock()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.setup_errors = 0
        self._configure()
        self._th = threading.Thread(target=self._reader, daemon=True)
        self._th.start()

    # ------------------------------------------------------------ commands
    def _configure(self) -> None:
        """No echo, continuous feedback, the heartbeat stop."""
        self._configured_t = time.monotonic()
        self.send({"T": 143, "cmd": 0})
        self.send({"T": 131, "cmd": 1})
        self.send({"T": 136, "cmd": self.heartbeat_ms})

    def send(self, msg: dict) -> None:
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        with self._wlock:
            self.io.write(line)

    def drive(self, v: float, w: float) -> None:
        """v m/s forward, w rad/s counter-clockwise; clipped; NaN stops."""
        if not (math.isfinite(v) and math.isfinite(w)):
            v, w = 0.0, 0.0
        v = max(-V_MAX, min(V_MAX, float(v)))
        w = max(-W_MAX, min(W_MAX, float(w)))
        now = time.monotonic()
        with self._lock:
            quiet = self.fb_t is None or now - self.fb_t > RESEND_S
        if quiet and now - self._configured_t > RESEND_S:
            self._configure()                     # a base that reset (brownout) or was still booting
        self.send({"T": 13, "X": round(v, 3), "Z": round(w, 3)})

    def stop(self) -> None:
        self.send({"T": 13, "X": 0, "Z": 0})

    def head(self, pan_deg: float, tilt_deg: float) -> None:
        pan = max(-180.0, min(180.0, float(pan_deg)))
        tilt = max(-30.0, min(90.0, float(tilt_deg)))
        self.send({"T": 133, "X": round(pan, 1), "Y": round(tilt, 1), "SPD": 0, "ACC": 0})

    # ------------------------------------------------------------ feedback
    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self.io.read(256)
            except Exception:                     # a closed port ends the thread
                return
            if not chunk:
                continue
            try:
                self.feed(chunk, time.monotonic())
            except Exception:                     # never let one bad message end the reader
                self.bad_lines += 1
                self._buf = b""

    def feed(self, chunk: bytes, t_s: float) -> None:
        """Parse bytes from the base (the reader thread; tests call it directly)."""
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8", "replace"))
            except ValueError:
                self.bad_lines += 1
                continue
            if isinstance(msg, dict) and msg.get("T") == 1001:
                with self._lock:
                    self.fb = msg
                    self.fb_t = t_s
                    self.n_fb += 1
                    self.odo.update(msg, t_s)
                    self.poses.append((t_s, self.odo.pose()))
        if len(self._buf) > 4096:                 # no newline in 4 KB: noise
            self._buf = b""
            self.bad_lines += 1

    def state(self) -> dict:
        with self._lock:
            return {"pose": self.odo.pose(), "battery_v": voltage(self.fb.get("v")),
                    "wheels": (self.fb.get("L"), self.fb.get("R")), "fb_age_s":
                    None if self.fb_t is None else time.monotonic() - self.fb_t, "feedbacks": self.n_fb}

    def pose_at(self, t_s: float) -> tuple:
        """The odometry pose at the feedback nearest t_s (e.g. a lidar scan's time)."""
        with self._lock:
            if not self.poses:
                return self.odo.pose()
            return min(self.poses, key=lambda tp: abs(tp[0] - t_s))[1]

    def reset_pose(self) -> None:
        with self._lock:
            self.odo.reset()
            self.poses.clear()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self._stop.set()
            try:
                self.io.close()
            except Exception:
                pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyTHS1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--test", action="store_true", help="print feedback, then drive 0.1 m/s for 1 s")
    a = ap.parse_args()
    b = UGVBase(a.port, a.baud)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            time.sleep(0.5)
            print(b.state(), b.fb, flush=True)
        if a.test:
            for _ in range(10):
                b.drive(0.1, 0.0)
                time.sleep(0.1)
            b.stop()
            time.sleep(0.5)
            print("after 1 s at 0.1 m/s:", b.state(), flush=True)
    finally:
        b.close()


if __name__ == "__main__":
    main()
