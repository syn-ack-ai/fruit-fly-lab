"""
Milo's compound eyes: what the camera and the lidar see goes through the fly's
optic lobes (option B of experiments/vision_ab: robot/flyvis_eye.py).

    the head camera (a small grey frame, robot/head.py, where the head points)
      + the lidar's 1-degree scan (obstacles drawn as dark bands, nearer = darker
        and taller, all around: the parts of the fly's field the camera misses)
      -> a panorama (azimuth -180..180, + = right; elevation 90..-90)
      -> flyvis (the published fly visual system model) on the GPU, in real
         time: one 10 ms step every 10 ms of wall time, in its own process
      -> the rates of the connectome's ~61,000 optic-lobe neurons, which the
         whole-brain model carries on (lobula plate, lobula, central brain)

The eye process (run_worker) shares memory with the brain client (EyeFeed):
the brain client writes the lidar scan, the eye writes the rates; the camera
frame comes from the head process's shared memory directly. EyeRates is the
Session encoder: the eye's latest rates, picked up at most every 10 ms of
simulated time (the same read-only array in between).

The connectome-to-flyvis mapping is built once (it needs the connectome) and
saved; the eye process loads only flyvis and that file:

    PYTHONPATH=. FLY_DATASET=merged python -m robot.eye --build

PROVENANCE
A. REAL DATA: the camera image and the lidar ranges; the connectome's
   optic-lobe neurons and column map (robot/flyvis_eye.py).
B. PUBLISHED: flyvis (Lappalainen et al. 2024, Nature 634:1132; MIT).
C. APPROXIMATIONS: the lidar sees one plane: an obstacle is drawn from the
   floor to WALL_UP_M above the lidar, with a brightness from its distance
   (no texture: motion shows at its edges); the camera image is pasted at its
   field of view where the head points (the camera's 53 degrees is ~9 of the
   fly's columns); the rest of the field is plain.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from multiprocessing import shared_memory

import numpy as np

from robot.retina import PANO_H, PANO_W

# the robot's eye: 20 ms flyvis steps (the longest flyvis recommends; half the
# GPU turns of 10 ms, which the brain's own GPU blocks wait behind: Jetson
# 2026-10-01, 97 -> 79 ms a 100 ms step), and rates from activity above each
# neuron's 2 s adapting baseline (a still room fades to rest), at half the
# harness's gain (50 Hz at a type's 99th-percentile response: at 100 a person
# moving at the desk kept the looming escape DNs DNp02 / DNp04 at 3-6 Hz and
# Milo in flight; at 50, ~1 Hz, as without the eye, and HS still follows
# motion). Set before robot.flyvis_eye is imported; the eye process inherits them.
os.environ.setdefault("FLY_EYE_DT", "0.02")
os.environ.setdefault("FLY_EYE_BASELINE_S", "2")
os.environ.setdefault("FLY_EYE_RATE_MAX", "50")

from robot.flyvis_eye import DT as STEP_S  # noqa: E402  (after the defaults above)

MAP_PATH = os.path.expanduser(os.environ.get("FLY_EYE_MAP", "~/milo/models/flyvis/milo_map_%s.npz"))
PICKUP_MS = 1000.0 * STEP_S       # the brain picks up new rates at most this often (simulated time)
LIDAR_N = 360
LIDAR_MAX_M = 8.0
LIDAR_H_M = 0.2                   # the lidar above the floor
WALL_UP_M = 0.5                   # an obstacle's assumed height above the lidar
SKY, FLOOR = 0.6, 0.45            # the plain field above / below the horizon
LIDAR_MEDIAN = 5                  # scans per beam in the median the eye draws
LIDAR_HOLD = 10                   # a beam's dropouts keep its last range for this many scans (~1 s)
LIDAR_DEADBAND_M = 0.03           # smaller changes are not redrawn (or LIDAR_DEADBAND_REL of the range)
LIDAR_DEADBAND_REL = 0.10
LIDAR_SPREAD = 2                  # and each beam the median of its neighbours +-2 deg (the fly's
                                  # columns are 5.8 deg apart: a single beam's surface drift is noise)
LIDAR_PERSIST = 3                 # ... nor changes that do not last this many scans (an edge
                                  # beam alternating near / far, ~0.3 s)
AZ = -180.0 + (np.arange(PANO_W) + 0.5) * 360.0 / PANO_W
EL = 90.0 - (np.arange(PANO_H) + 0.5) * 180.0 / PANO_H
_F = ["seq", "t_wall", "ready", "stop", "n", "steps", "step_ms", "late", "lidar_seq", "cam_seq",
      "mean_hz", "error"]
_I = {k: i for i, k in enumerate(_F)}
_HEADER = 256
_LIDAR_AT = _HEADER


def map_path(dataset: str | None = None) -> str:
    return MAP_PATH % (dataset or os.environ.get("FLY_DATASET", "flywire"))


# ------------------------------------------------------------------ the scene
def area_resize(a, w, h):
    """a (H, W) -> (h, w): each output pixel the mean of the input area it
    covers (a downscale without aliasing; numpy only)."""
    def edges(n_in, n_out):
        e = np.linspace(0, n_in, n_out + 1)
        return np.minimum(np.floor(e[:-1]).astype(int), n_in - 1), np.maximum(np.ceil(e[1:]).astype(int), 1)
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.float64)
    ii[1:, 1:] = a.cumsum(0).cumsum(1)
    y0, y1 = edges(a.shape[0], h)
    x0, x1 = edges(a.shape[1], w)
    s = ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]
    return (s / ((y1 - y0)[:, None] * (x1 - x0)[None, :])).astype(np.float32)


def compose(gray=None, pan_deg=0.0, tilt_deg=0.0, ranges=None, hfov_deg=None) -> np.ndarray:
    """The panorama the eyes see: plain sky and floor, the lidar's obstacles
    all around, the camera's image where the head points."""
    img = np.where(EL[:, None] > 0, SKY, FLOOR).astype(np.float32) * np.ones((1, PANO_W), np.float32)
    if ranges is not None and len(ranges):
        r = np.asarray(ranges, np.float32)
        k = np.round((AZ + 180.0) * len(r) / 360.0).astype(int) % len(r)   # beam per column
        rc = r[k]
        hit = np.isfinite(rc) & (rc > 0.05) & (rc < LIDAR_MAX_M - 0.05)
        top = np.degrees(np.arctan2(WALL_UP_M, np.maximum(rc, 0.05)))
        bot = -np.degrees(np.arctan2(LIDAR_H_M, np.maximum(rc, 0.05)))
        shade = 0.1 + 0.35 * np.clip(rc / 4.0, 0.0, 1.0)                    # nearer = darker
        m = hit[None, :] & (EL[:, None] <= top[None, :]) & (EL[:, None] >= bot[None, :])
        img = np.where(m, shade[None, :], img)
    if gray is not None:
        from robot.head import HFOV_DEG
        hf = hfov_deg or HFOV_DEG
        h, w = gray.shape
        vf = hf * h / w
        px = 360.0 / PANO_W                                                 # deg a panorama pixel
        x0, x1 = (pan_deg - hf / 2 + 180.0) / px, (pan_deg + hf / 2 + 180.0) / px
        y0, y1 = (90.0 - (tilt_deg + vf / 2)) / px, (90.0 - (tilt_deg - vf / 2)) / px
        cx0, cx1, cy0, cy1 = int(round(x0)), int(round(x1)), int(round(y0)), int(round(y1))
        if cx1 > cx0 and cy1 > cy0:
            patch = area_resize(np.asarray(gray, np.float32) / 255.0, cx1 - cx0, cy1 - cy0)
            ya, yb = max(cy0, 0), min(cy1, PANO_H)
            xa, xb = max(cx0, 0), min(cx1, PANO_W)
            if yb > ya and xb > xa:
                img[ya:yb, xa:xb] = patch[ya - cy0:yb - cy0, xa - cx0:xb - cx0]
    return img


# ------------------------------------------------------------------ eye process
def _attach(name):
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    from multiprocessing import resource_tracker
    shm = shared_memory.SharedMemory(name=name)
    resource_tracker.unregister(shm._name, "shared_memory")     # the parent owns unlink()
    return shm


def run_worker(shm_name: str, head_shm: str | None, map_file: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(int(os.environ.get("FLY_EYE_THREADS", 1)))   # the brain needs the cores
    from robot.flyvis_eye import FlyvisEncoder
    shm = _attach(shm_name)
    hdr = np.ndarray((len(_F),), np.float64, buffer=shm.buf)
    try:
        enc = FlyvisEncoder(None, None, None, saved=map_file)
    except Exception as ex:
        print("eye: flyvis unavailable:", ex, file=sys.stderr, flush=True)
        hdr[_I["error"]] = 1.0
        return
    n = len(enc.indices)
    lidar = np.ndarray((LIDAR_N,), np.float32, buffer=shm.buf, offset=_LIDAR_AT)
    rates = np.ndarray((n,), np.float32, buffer=shm.buf, offset=_LIDAR_AT + 4 * LIDAR_N)
    head = head_hdr = None
    if head_shm:
        from robot.head import _F as HF
        head = _attach(head_shm)
        head_hdr = np.ndarray((len(HF),), np.float64, buffer=head.buf)
    from robot.head import read_gray
    parent = os.getppid()
    hdr[_I["n"]] = n
    hdr[_I["ready"]] = 1.0
    print(f"eye: flyvis ready, {n} optic-lobe neurons", file=sys.stderr, flush=True)
    t_next = time.monotonic()
    steps = late = 0
    step_ms = 0.0
    cam = None
    try:
        while os.getppid() == parent and not hdr[_I["stop"]]:
            now = time.monotonic()
            if now < t_next:
                time.sleep(t_next - now)
                continue
            k = min(3, 1 + int((now - t_next) / STEP_S))        # behind: catch up a little, then skip
            if now - t_next > 3 * STEP_S:
                late += 1
                t_next = now
            t_next += k * STEP_S
            if head is not None:
                g = read_gray(head, head_hdr)
                if g is not None:
                    cam = g
            lid = lidar.copy() if hdr[_I["lidar_seq"]] > 0 else None
            img = compose(cam[1] if cam else None, cam[2] if cam else 0.0, cam[3] if cam else 0.0, lid)
            t0 = time.perf_counter()
            r = enc.see(img, k)
            step_ms = 0.9 * step_ms + 0.1 * 1e3 * (time.perf_counter() - t0) / k
            steps += k

            def write():
                rates[:] = r
                hdr[_I["t_wall"]] = time.time()
                hdr[_I["steps"]], hdr[_I["late"]], hdr[_I["step_ms"]] = steps, late, step_ms
                hdr[_I["cam_seq"]] = cam[0] if cam else 0
                hdr[_I["mean_hz"]] = float(r.mean())
            hdr[_I["seq"]] += 1
            write()
            hdr[_I["seq"]] += 1
    finally:
        del hdr, lidar, rates
        if head is not None:
            del head_hdr
            head.close()
        shm.close()


# ------------------------------------------------------------------ the brain's side
class LidarSteadier:
    """The lidar as the eye draws it: each beam the median of its last few
    ranges, dropouts held for a while, small changes ignored. The raw D500
    scan jitters by centimetres and drops beams every revolution, and beams
    on an object's edge alternate between it and what is behind; drawn as
    is, obstacle edges flicker 10 times a second, which the fly's motion
    detectors take for motion all around (Jetson 2026-10-01, still room:
    raw, HSE 57 Hz and the looming escape DNs DNp02 / DNp04 10-20 Hz;
    median and hold, 25 / 3-8 Hz; camera alone ~0)."""

    def __init__(self, n):
        self.hist = np.full((LIDAR_MEDIAN, n), np.nan, np.float32)
        self.k = 0
        self.held = np.full(n, LIDAR_MAX_M, np.float32)
        self.missing = np.zeros(n, np.int32)
        self.out = np.full(n, LIDAR_MAX_M, np.float32)
        self.pending = np.zeros(n, np.int32)

    def update(self, r) -> np.ndarray:
        r = np.asarray(r, np.float32)
        ok = np.isfinite(r) & (r > 0.05) & (r < LIDAR_MAX_M - 0.05)
        self.missing = np.where(ok, 0, self.missing + 1)
        self.held = np.where(ok, r, np.where(self.missing > LIDAR_HOLD, LIDAR_MAX_M, self.held))
        self.hist[self.k % LIDAR_MEDIAN] = self.held
        self.k += 1
        med = np.nanmedian(self.hist[:min(self.k, LIDAR_MEDIAN)], axis=0)
        if LIDAR_SPREAD:
            nb = np.stack([np.roll(med, s) for s in range(-LIDAR_SPREAD, LIDAR_SPREAD + 1)])
            med = np.median(nb, axis=0)
        moved = np.abs(med - self.out) > np.maximum(LIDAR_DEADBAND_M,
                                                    LIDAR_DEADBAND_REL * np.minimum(med, self.out))
        self.pending = np.where(moved, self.pending + 1, 0)
        new = moved & ((self.pending >= LIDAR_PERSIST) | (self.k <= LIDAR_MEDIAN))   # the first scans: at once
        self.out = np.where(new, med, self.out).astype(np.float32)
        self.pending[new] = 0
        return self.out


class EyeFeed:
    """Starts the eye process; passes it the lidar; reads its rates."""

    STALE_S = 0.5

    def __init__(self, head_shm: str | None, map_file: str | None = None):
        self.map_file = map_file or map_path()
        if not os.path.exists(self.map_file):
            raise RuntimeError(f"no eye mapping {self.map_file}: run python -m robot.eye --build")
        m = np.load(self.map_file, allow_pickle=False)
        self.indices = m["indices"].astype(np.int64)
        self.n = len(self.indices)
        self.shm = shared_memory.SharedMemory(create=True, size=_LIDAR_AT + 4 * (LIDAR_N + self.n))
        self.shm.buf[:self.shm.size] = bytes(self.shm.size)
        self._h = np.ndarray((len(_F),), np.float64, buffer=self.shm.buf)
        self._lidar = np.ndarray((LIDAR_N,), np.float32, buffer=self.shm.buf, offset=_LIDAR_AT)
        self._rates = np.ndarray((self.n,), np.float32, buffer=self.shm.buf, offset=_LIDAR_AT + 4 * LIDAR_N)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args = [sys.executable, "-m", "robot.eye", "--shm", self.shm.name, "--map", self.map_file]
        if head_shm:
            args += ["--head-shm", head_shm]
        self._lid = None                                   # LidarSteadier, from the first scan
        self.log = open(os.path.expanduser("~/.milo_eye.log"), "ab")
        self.proc = subprocess.Popen(args, cwd=root, stdout=self.log, stderr=self.log)

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None and not self._h[_I["error"]]

    def put_lidar(self, ranges) -> None:
        """The lidar's 1-degree scan (m, + = right, from -180)."""
        if ranges is None or len(ranges) == 0:
            return
        r = np.asarray(ranges, np.float32)
        if len(r) != LIDAR_N:
            r = r[np.round(np.arange(LIDAR_N) * len(r) / LIDAR_N).astype(int) % len(r)]
        if self._lid is None:
            self._lid = LidarSteadier(LIDAR_N)
        r = self._lid.update(r)
        self._lidar[:] = r                                  # a torn scan for one step is harmless
        self._h[_I["lidar_seq"]] += 1

    def rates(self, since=None):
        """(seq, a copy of the rates as float64), or None (also when nothing
        is newer than `since`)."""
        for _ in range(100):
            s1 = self._h[_I["seq"]]
            if int(s1) % 2:
                continue
            if s1 == 0 or s1 == since:
                return None
            r = self._rates.astype(np.float64)
            if self._h[_I["seq"]] == s1:
                return int(s1), r
        return None

    def status(self) -> dict:
        h = self._h
        return {"ready": bool(h[_I["ready"]]), "alive": self.alive,
                "stale": bool(h[_I["ready"]] and time.time() - h[_I["t_wall"]] > self.STALE_S),
                "step_ms": round(float(h[_I["step_ms"]]), 2), "steps": int(h[_I["steps"]]),
                "late": int(h[_I["late"]]), "camera": bool(h[_I["cam_seq"]] > 0),
                "lidar": bool(h[_I["lidar_seq"]] > 0), "mean_hz": round(float(h[_I["mean_hz"]]), 2)}

    def close(self) -> None:
        if getattr(self, "proc", None) is None:
            return
        if self.proc.poll() is None:
            self._h[_I["stop"]] = 1.0
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        del self._h, self._lidar, self._rates
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
        self.log.close()


class EyeRates:
    """Session encoder: the eye's latest optic-lobe rates (0 Hz until it is
    ready, or when it has stopped)."""

    def __init__(self, feed: EyeFeed):
        self.feed = feed
        self.indices = feed.indices
        self._zero = np.zeros(len(self.indices))
        self._zero.flags.writeable = False
        self._rates, self._seq, self._t = self._zero, None, -math.inf
        self.last = {}

    def rates_hz(self, t_ms: float, stim=None) -> np.ndarray:
        if t_ms - self._t < PICKUP_MS and t_ms >= self._t:
            return self._rates
        self._t = t_ms
        st = self.feed._h
        if st[_I["stop"]] or not st[_I["ready"]] or time.time() - st[_I["t_wall"]] > EyeFeed.STALE_S:
            self._rates, self._seq = self._zero, None
            return self._rates
        got = self.feed.rates(self._seq)
        if got is not None and got[0] != self._seq:
            self._seq, r = got
            r.flags.writeable = False
            self._rates = r
        return self._rates

    def state(self, t_ms: float) -> dict:
        return {"kind": "eye", "active": self._rates is not self._zero}

    @property
    def provenance(self) -> dict:
        return {"drives": {"optic lobe (flyvis types)": int(len(self.indices))},
                "source": "camera + lidar -> flyvis (robot/eye.py, robot/flyvis_eye.py)"}


def build(path: str | None = None) -> str:
    """The connectome-to-flyvis mapping and calibration, saved for the eye."""
    from brain.neurons.registry import load_connectome
    from brain.sensory.retinotopy import load_retinotopy
    from robot.flyvis_eye import FlyvisEncoder
    c = load_connectome()
    enc = FlyvisEncoder(c, load_retinotopy(c), None)
    path = path or map_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    enc.save(path)
    print(f"{path}: {len(enc.indices)} neurons, {sum(1 for v in enc.n_by_type.values() if v)} cell types")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", help="build the connectome-to-flyvis mapping")
    ap.add_argument("--shm")
    ap.add_argument("--head-shm")
    ap.add_argument("--map")
    a = ap.parse_args()
    if a.build:
        build(a.map)
    else:
        run_worker(a.shm, a.head_shm, a.map or map_path())
