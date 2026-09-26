"""
Live camera -> the real looming detectors (LC4, LPLC2).

PROVENANCE
----------
D. OUR ENGINEERING : everything in this file. It replaces the *geometry* of
   the virtual thrown object (simulation/stimuli/looming.py) with the same
   quantities measured from a camera, and changes nothing downstream: the
   rates still come from LoomingEncoder (published tuning x FlyWire-derived
   receptive fields) and everything after that is the connectome.
C. APPROXIMATIONS  :
   - One looming object at a time: the largest coherently moving region.
   - Its angular half-size is the radius of a disc with the region's convex
     hull area; its expansion rate is the least-squares radial scaling of the
     optical flow about the region's centroid (pure approach -> v = s * r),
     so sideways motion and camera shake give ~0 expansion.
   - Pixels map linearly to visual angle over the lens field of view, with the
     camera axis at the fly's straight ahead (adjustable yaw/pitch). The lens
     covers a frontal patch only (IMX708: 66 x 41 deg; each fly eye ~175 deg).

The camera runs in its own process (`python -m brain.sensory.camera`), so image
processing never competes with the simulation loop for the GIL, and publishes
through shared memory. The simulation reads the latest estimate every 1 ms.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from multiprocessing import shared_memory

import numpy as np

# Lens fields of view (degrees, horizontal x vertical) for Raspberry Pi cameras.
LENS_FOV = {
    "imx708": (66.0, 41.0),          # Camera Module 3
    "imx708_noir": (66.0, 41.0),
    "imx708_wide": (102.0, 67.0),    # Camera Module 3 Wide
    "imx708_wide_noir": (102.0, 67.0),
}
DEFAULT_FOV = (66.0, 41.0)

WIDTH, HEIGHT = 160, 90              # processing resolution: ~0.4 deg/px, finer
                                     # than the fly's ~5 deg ommatidial spacing

# --------------------------------------------------------------------------- #
# Shared-memory layout
# --------------------------------------------------------------------------- #
# float64 header, then a JPEG preview. Both halves use a sequence lock: the
# writer makes the counter odd while writing and even when done.
_F = ["seq", "t_wall", "active", "azimuth_deg", "elevation_deg",
      "half_angle_deg", "expansion_rate_deg_s", "scale_rate_s", "area_frac",
      "fps", "frames", "hfov_deg", "vfov_deg", "preview_seq", "preview_len"]
_I = {k: i for i, k in enumerate(_F)}
_HEADER = 256                        # bytes reserved for the float64 fields
_PREVIEW_MAX = 256 * 1024
SHM_SIZE = _HEADER + _PREVIEW_MAX


# --------------------------------------------------------------------------- #
# Feature extraction (pure: frames in, looming estimate out)
# --------------------------------------------------------------------------- #
class LoomingExtractor:
    """Estimates one looming object's direction, size and expansion from video."""

    def __init__(self, hfov_deg: float = DEFAULT_FOV[0], vfov_deg: float = DEFAULT_FOV[1],
                 yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                 flow_thresh_deg_s: float = 8.0, min_area_frac: float = 0.002,
                 min_scale_rate_s: float = 0.4, smoothing: float = 0.5):
        import cv2
        self.cv2 = cv2
        self.hfov, self.vfov = hfov_deg, vfov_deg
        self.yaw, self.pitch = yaw_deg, pitch_deg
        self.deg_per_px = 0.5 * (hfov_deg / WIDTH + vfov_deg / HEIGHT)
        self.flow_thresh_deg_s = flow_thresh_deg_s
        self.min_area = max(4, int(min_area_frac * WIDTH * HEIGHT))
        self.min_scale_rate_s = min_scale_rate_s
        self.alpha = smoothing
        self._prev = None
        self._prev_t = None
        self.mask = np.zeros((HEIGHT, WIDTH), np.uint8)   # last region, for preview
        self.est = self._inactive()

    @staticmethod
    def _inactive() -> dict:
        return {"active": False, "azimuth_deg": 0.0, "elevation_deg": 0.0,
                "half_angle_deg": 0.0, "expansion_rate_deg_s": 0.0,
                "scale_rate_s": 0.0, "area_frac": 0.0}

    def update(self, gray: np.ndarray, t_s: float) -> dict:
        """Feed one (HEIGHT, WIDTH) uint8 frame taken at time t_s (seconds)."""
        cv2 = self.cv2
        cur = cv2.GaussianBlur(gray, (5, 5), 0)
        prev, prev_t = self._prev, self._prev_t
        self._prev, self._prev_t = cur, t_s
        if prev is None or t_s <= prev_t:
            return dict(self.est)
        dt = t_s - prev_t

        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 11, 3, 5, 1.1, 0)
        mag = np.hypot(flow[..., 0], flow[..., 1])            # px per frame
        thresh_px = self.flow_thresh_deg_s * dt / self.deg_per_px
        mask = (mag > thresh_px).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)

        raw = self._inactive()
        self.mask[:] = 0
        if n > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area >= self.min_area:
                ys, xs = np.nonzero(labels == k)
                self.mask[ys, xs] = 1
                cx, cy = cents[k]
                rx, ry = xs - cx, ys - cy
                v = flow[ys, xs]
                r2 = float(np.sum(rx * rx + ry * ry))
                s = float(np.sum(v[:, 0] * rx + v[:, 1] * ry)) / r2 / dt if r2 > 0 else 0.0
                hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
                radius_px = np.sqrt(max(cv2.contourArea(hull), area) / np.pi)
                theta = float(radius_px * self.deg_per_px)
                raw = {
                    "active": True,
                    "azimuth_deg": (cx / WIDTH - 0.5) * self.hfov + self.yaw,
                    "elevation_deg": (0.5 - cy / HEIGHT) * self.vfov + self.pitch,
                    "half_angle_deg": theta,
                    # d(theta)/dt = theta * (dr/dt)/r; ignore sub-threshold drift
                    "expansion_rate_deg_s": theta * s if abs(s) >= self.min_scale_rate_s else 0.0,
                    "scale_rate_s": s,
                    "area_frac": area / float(WIDTH * HEIGHT),
                }

        a = self.alpha
        if raw["active"]:
            if not self.est["active"]:
                self.est = dict(raw)
            else:
                for key in ("azimuth_deg", "elevation_deg", "half_angle_deg",
                            "expansion_rate_deg_s", "scale_rate_s", "area_frac"):
                    self.est[key] = a * raw[key] + (1 - a) * self.est[key]
                self.est["active"] = True
        else:
            self.est = self._inactive()
        return dict(self.est)


# --------------------------------------------------------------------------- #
# Sensor process
# --------------------------------------------------------------------------- #
def _seq_write(buf: np.ndarray, i: int, fn) -> None:
    buf[i] += 1                      # odd: writing
    fn()
    buf[i] += 1                      # even: stable


def _preview(view: np.ndarray, mask: np.ndarray, est: dict, fps: float) -> bytes:
    """What the camera sees (320x180 luma, shown at 640x360), with the region
    the extractor is tracking tinted and the looming estimate drawn on it."""
    import cv2
    scale = 4                                        # processing px -> preview px
    img = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (WIDTH * scale, HEIGHT * scale), interpolation=cv2.INTER_LINEAR)
    if mask.any():
        m = cv2.resize(mask, (WIDTH * scale, HEIGHT * scale), interpolation=cv2.INTER_NEAREST) > 0
        img[m] = (0.55 * img[m] + 0.45 * np.array([172, 212, 77])).astype(np.uint8)
    if est["active"]:
        hf, vf = est.get("_hfov", DEFAULT_FOV[0]), est.get("_vfov", DEFAULT_FOV[1])
        cx = int(((est["azimuth_deg"] - est.get("_yaw", 0)) / hf + 0.5) * WIDTH * scale)
        cy = int((0.5 - (est["elevation_deg"] - est.get("_pitch", 0)) / vf) * HEIGHT * scale)
        r = int(est["half_angle_deg"] / est["_dpp"] * scale)
        looming = est["expansion_rate_deg_s"] > 0
        cv2.circle(img, (cx, cy), max(r, 2), (129, 107, 255) if looming else (200, 200, 200), 2)
    txt = "%.0f fps" % fps
    if est["active"]:
        txt += "   size %.1f deg   expansion %+.0f deg/s" % (
            est["half_angle_deg"], est["expansion_rate_deg_s"])
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpg.tobytes() if ok else b""


def camera_fov() -> tuple:
    try:
        from picamera2 import Picamera2
        info = Picamera2.global_camera_info()
        model = info[0]["Model"] if info else ""
    except Exception:
        model = ""
    return LENS_FOV.get(model, DEFAULT_FOV), model


def run_worker(shm_name: str, yaw_deg: float, pitch_deg: float, fps: float,
               rotate: int = 0) -> None:
    """Capture loop. Exits when the parent process goes away."""
    import cv2
    from picamera2 import Picamera2

    shm = shared_memory.SharedMemory(name=shm_name, track=False)   # parent owns it
    hdr = np.ndarray((len(_F),), dtype=np.float64, buffer=shm.buf)
    (hfov, vfov), model = camera_fov()
    ext = LoomingExtractor(hfov, vfov, yaw_deg, pitch_deg)

    cam = Picamera2()
    # Full-sensor binned mode, so the whole lens field of view is used.
    full = max(cam.sensor_modes, key=lambda m: (m["size"][0] * m["size"][1]
                                                if m["fps"] >= fps else 0))
    from libcamera import Transform
    cfg = cam.create_video_configuration(
        main={"size": (320, 180), "format": "YUV420"},
        sensor={"output_size": full["size"]},
        transform=Transform(hflip=1, vflip=1) if rotate == 180 else Transform(),
        controls={"FrameRate": float(fps)})
    cam.configure(cfg)
    cam.start()

    parent = os.getppid()
    t_last, frames, rate = time.monotonic(), 0, 0.0
    try:
        while os.getppid() == parent:
            req = cam.capture_request()
            try:
                y = req.make_array("main")[:180, :320]
                ts = req.get_metadata().get("SensorTimestamp")
            finally:
                req.release()
            t = ts * 1e-9 if ts else time.monotonic()
            gray = cv2.resize(y, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            est = ext.update(gray, t)

            frames += 1
            now = time.monotonic()
            if now - t_last >= 1.0:
                rate, frames, t_last = frames / (now - t_last), 0, now

            def write():
                hdr[_I["t_wall"]] = time.time()
                for k in ("active", "azimuth_deg", "elevation_deg", "half_angle_deg",
                          "expansion_rate_deg_s", "scale_rate_s", "area_frac"):
                    hdr[_I[k]] = float(est[k])
                hdr[_I["fps"]] = rate
                hdr[_I["frames"]] += 1
                hdr[_I["hfov_deg"]], hdr[_I["vfov_deg"]] = hfov, vfov
            _seq_write(hdr, _I["seq"], write)

            if int(hdr[_I["frames"]]) % 3 == 0:            # ~10 fps preview
                jpg = _preview(y, ext.mask, {**est, "_dpp": ext.deg_per_px,
                                                "_hfov": hfov, "_vfov": vfov,
                                                "_yaw": yaw_deg, "_pitch": pitch_deg}, rate)
                if 0 < len(jpg) <= _PREVIEW_MAX:
                    def write_preview():
                        shm.buf[_HEADER:_HEADER + len(jpg)] = jpg
                        hdr[_I["preview_len"]] = len(jpg)
                    _seq_write(hdr, _I["preview_seq"], write_preview)
    finally:
        cam.stop()
        cam.close()
        del hdr
        shm.close()


# --------------------------------------------------------------------------- #
# Simulation side
# --------------------------------------------------------------------------- #
class CameraFeed:
    """Starts the sensor process and reads its latest estimate."""

    STALE_S = 0.5

    def __init__(self, yaw_deg: float = 0.0, pitch_deg: float = 0.0, fps: float = 30.0,
                 rotate: int = 0):
        self.yaw_deg, self.pitch_deg = yaw_deg, pitch_deg
        self.shm = shared_memory.SharedMemory(create=True, size=SHM_SIZE)
        self.shm.buf[:SHM_SIZE] = bytes(SHM_SIZE)
        self._hdr = np.ndarray((len(_F),), dtype=np.float64, buffer=self.shm.buf)
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "brain.sensory.camera", "--shm", self.shm.name,
             "--yaw", str(yaw_deg), "--pitch", str(pitch_deg), "--fps", str(fps),
             "--rotate", str(rotate)],
            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def error(self) -> str:
        if self.alive:
            return ""
        return (self.proc.stderr.read() or b"").decode(errors="replace")[-2000:]

    def _read(self, seq_i: int, fn):
        h = self._hdr
        for _ in range(1000):
            s1 = h[seq_i]
            if int(s1) % 2:
                continue
            out = fn()
            if h[seq_i] == s1:
                return out
        return None

    def state(self) -> dict:
        v = self._read(_I["seq"], lambda: self._hdr.copy())
        if v is None or v[_I["frames"]] == 0:
            return {"active": False, "fps": 0.0, "frames": 0, "stale": True,
                    "azimuth_deg": 0.0, "elevation_deg": 0.0, "half_angle_deg": 0.0,
                    "expansion_rate_deg_s": 0.0, "scale_rate_s": 0.0, "area_frac": 0.0}
        d = {k: float(v[_I[k]]) for k in _F if k not in ("seq", "preview_seq", "preview_len")}
        d["stale"] = time.time() - d["t_wall"] > self.STALE_S
        d["active"] = bool(d["active"]) and not d["stale"]
        return d

    def preview_jpeg(self) -> bytes | None:
        def grab():
            n = int(self._hdr[_I["preview_len"]])
            return bytes(self.shm.buf[_HEADER:_HEADER + n]) if n else None
        return self._read(_I["preview_seq"], grab)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        del self._hdr
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass


class CameraLoomingStimulus:
    """
    Drop-in for LoomingStimulus whose geometry comes from the camera. Feed it
    to LoomingEncoder exactly as the virtual thrown object is.
    """

    def __init__(self, feed):
        self.feed = feed

    def state(self, t_ms: float) -> dict:
        s = self.feed.state()
        return {
            "t_ms": t_ms,
            "source": "camera",
            "azimuth_deg": s["azimuth_deg"],
            "elevation_deg": s["elevation_deg"],
            "half_angle_deg": s["half_angle_deg"] if s["active"] else 0.0,
            "expansion_rate_deg_s": s["expansion_rate_deg_s"] if s["active"] else 0.0,
            "camera_fps": round(s["fps"], 1),
            "active": s["active"],
        }


def main():
    ap = argparse.ArgumentParser(description="camera sensor process")
    ap.add_argument("--shm", required=True)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--rotate", type=int, choices=(0, 180), default=0,
                    help="180 for a camera mounted upside down")
    a = ap.parse_args()
    run_worker(a.shm, a.yaw, a.pitch, a.fps, a.rotate)


if __name__ == "__main__":
    main()
