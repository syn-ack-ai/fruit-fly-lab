"""
The robot's head: a pan/tilt USB camera (Logitech QuickCam Orbit/Sphere AF)
that the simulated fly's brain looks through and turns.

    camera --> visual features --> real FlyWire visual neurons --> connectome
      ^                                                                |
      |                         descending steering neurons (DNa01/DNa02, DNp09)
      +---------------- pan motor <----- HeadController <--------------+

PROVENANCE
----------
A. REAL DATA : the neurons driven (LC4, LPLC2, LC10a) and their receptive fields
   (brain/sensory/retinotopy.py); every step from them to the descending
   neurons is the connectome.
B. PUBLISHED :
   - LC4 / LPLC2 encode looming (brain/sensory/encoders.LoomingEncoder).
   - LC10a responds to small moving objects and drives visually guided
     pursuit through AOTU019/AOTU025 -> DNa02 steering (Ribeiro et al. 2018,
     Cell 174:607; Hindmarsh Sten et al. 2021, Nature 595:549).
   - Flies steer toward the side with more DNa01/DNa02 activity (Rayshubskiy
     et al. 2024); DNp09 (P9) adds ipsilateral turning (Bidaye et al. 2020).
   - During self-generated turns the fly's visual motion responses are
     cancelled by an efference copy of the motor command (Kim, Fitzgerald &
     Maimon 2015, Nat Neurosci 18:1247). Here vision is blanked while the
     motor moves and briefly after.
C. OUR APPROXIMATIONS :
   - One moving object at a time (the largest moving region, as in
     brain/sensory/camera.py), or the largest person found by a detector,
     which takes priority: YOLOv8 on the Pi's AI HAT (Hailo-10H), or YOLOX
     on a Jetson's GPU through TensorRT (robot/detector.py). For a person the
     target is their head and shoulders (top third of the detection box):
     a robot-design choice of what the pet attends to, sized near LC10a's
     preferred ~15 deg at room distance.
   - LC10a drive = size tuning (log-Gaussian, peak ~15 deg, on the target's
     angular size) x receptive-field overlap, weaker for a still object than a
     moving one (a person counts as moving when their bearing changes > 5 deg/s).
   - Head controller (robot engineering, not biology): the fly's resting
     steering bias ("handedness": left DNa02 fires more at rest in this
     connectome) is learned while no target is in view and subtracted, so the
     head does not drift; target-evoked steering is used as is.
   - Head turn rate = gain x (right - left DN steering activity); the camera
     turns in steps every ~200 ms because the motor is slow (~0.3-0.5 s/move).
   - Startle (expressive mapping, robot engineering): when the long-mode
     escape command (DNp02/04/11) crosses 0.5 -- e.g. after a sudden sound via
     JO-B -> DNp11 (robot/hearing.py) -- the head flinches up 8 deg and returns
     after ~1 s. In the connectome the same sound also stops walking (DNg100
     -> ~0 Hz), which the wheeled body will use.
   - Orbit optics: ~53 deg horizontal field of view (measured: 10 deg of pan
     shifts the 320-px image by 60 px); 1/64 deg per pan unit.

Run standalone to test the sensor process:  python -m robot.head --test
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import struct
import subprocess
import sys
import threading
import time
from multiprocessing import shared_memory

import numpy as np

try:
    import fcntl
except ImportError:  # V4L2 is Linux-only; keep the sensor-independent code importable.
    fcntl = None

# ------------------------------------------------------------------ hardware
ORBIT_GLOB = "/dev/v4l/by-id/usb-046d_0994_*-video-index0"
VIDIOC_S_CTRL = 0xC008561C
CTRL_PAN_REL, CTRL_TILT_REL = 0x009A0904, 0x009A0905
CTRL_PAN_RESET, CTRL_TILT_RESET = 0x009A0906, 0x009A0907
CTRL_EXPOSURE_DYN_FPS = 0x009A0903          # off: keep 30 fps in dim light
UNITS_PER_DEG = 64.0
PAN_LIMIT_DEG, TILT_LIMIT_DEG = 68.0, 28.0
HFOV_DEG = 53.0
CAP_W, CAP_H = 320, 240
# processing image: the central 320x180 band (16:9, like brain/sensory/camera.py)
PROC_W, PROC_H = 160, 90
VFOV_DEG = HFOV_DEG * 180.0 / 320.0          # ~30 deg for the band
HAILO_HEF = "/usr/share/hailo-models/yolov8m_h10.hef"
PERSON_PERIOD_S = 0.1                        # run the detector at most every 0.1 s (10 Hz: the
                                             # control rate; frames come at 15-30 Hz with the light)
SETTLE_S = 0.25                              # vision blanked after a move ends
FACE_PERIOD_S = 0.2                          # faces (robot/faces.py, robot/people.py) at 5 Hz


def find_orbit() -> str | None:
    devs = sorted(glob.glob(ORBIT_GLOB))
    return devs[0] if devs else None


class PanTilt:
    """Relative pan/tilt through the UVC motor controls. The camera cannot
    report its angle, so it is tracked here from a reset (centred) position.

    The motor controls are Logitech extension controls that the kernel only
    exposes when they are mapped (the uvcdynctrl package's udev rule does it;
    the Pi had them, a fresh Jetson does not). Without them the head is a fixed
    camera: moves do nothing and the angles stay 0."""

    def __init__(self, dev: str):
        self.fd = os.open(dev, os.O_RDWR)
        self.pan_deg = 0.0
        self.tilt_deg = 0.0
        self.busy_until = 0.0
        self.motors = True

    def _ctrl(self, cid: int, val: int) -> None:
        fcntl.ioctl(self.fd, VIDIOC_S_CTRL, struct.pack("Ii", cid, int(val)))

    def fixed_framerate(self) -> None:
        try:
            self._ctrl(CTRL_EXPOSURE_DYN_FPS, 0)
        except OSError:
            pass

    def reset(self) -> None:
        try:
            self._ctrl(CTRL_PAN_RESET, 1)
        except OSError as ex:
            self.motors = False
            print(f"pan/tilt motor controls unavailable ({ex}): a fixed camera "
                  "(sudo apt install uvcdynctrl, then replug, maps them)", file=sys.stderr, flush=True)
            return
        time.sleep(2.5)
        self._ctrl(CTRL_TILT_RESET, 1)
        time.sleep(2.0)
        self.pan_deg = self.tilt_deg = 0.0
        self.busy_until = time.monotonic() + SETTLE_S

    def move(self, dpan_deg: float, dtilt_deg: float = 0.0) -> float:
        """Turn by (dpan, dtilt) degrees; + pan = right, + tilt = up. Returns
        the time (s) the move should take."""
        if not self.motors:
            return 0.0
        dp = float(np.clip(self.pan_deg + dpan_deg, -PAN_LIMIT_DEG, PAN_LIMIT_DEG) - self.pan_deg)
        dt = float(np.clip(self.tilt_deg + dtilt_deg, -TILT_LIMIT_DEG, TILT_LIMIT_DEG) - self.tilt_deg)
        dur = 0.0
        if abs(dp) >= 0.5:
            self._ctrl(CTRL_PAN_REL, -round(dp * UNITS_PER_DEG))     # + units turn left
            self.pan_deg += dp
            dur = max(dur, 0.15 + abs(dp) / 80.0)
        if abs(dt) >= 0.5:
            self._ctrl(CTRL_TILT_REL, -round(dt * UNITS_PER_DEG))    # + units tilt down
            self.tilt_deg += dt
            dur = max(dur, 0.15 + abs(dt) / 60.0)
        if dur:
            self.busy_until = time.monotonic() + dur + SETTLE_S
        return dur

    @property
    def moving(self) -> bool:
        return time.monotonic() < self.busy_until

    def close(self) -> None:
        os.close(self.fd)


class FixedHead:
    """PanTilt's interface for a camera without motors (or a recorded video)."""
    pan_deg = tilt_deg = 0.0
    motors = False
    moving = False

    def reset(self) -> None:
        pass

    def move(self, dpan_deg: float, dtilt_deg: float = 0.0) -> float:
        return 0.0

    def fixed_framerate(self) -> None:
        pass

    def close(self) -> None:
        pass


class PersonDetector:
    """Largest person in the frame, from YOLOv8m on the Hailo-10H (AI HAT+ 2)."""

    def __init__(self, hef: str = HAILO_HEF):
        from hailo_platform import FormatType, VDevice
        self.vd = VDevice()
        self.im = self.vd.create_infer_model(hef)
        self.im.input().set_format_type(FormatType.UINT8)
        self.cm = self.im.configure().__enter__()
        self.b = self.cm.create_bindings()
        self.out = np.empty(self.im.output().shape, dtype=np.float32)
        self.lb = np.zeros((640, 640, 3), np.uint8)

    def detect(self, rgb: np.ndarray) -> dict | None:
        """rgb: (240, 320, 3). Returns the largest person as fractions of the
        full frame {cx, cy, w, h, score}, or None."""
        import cv2
        big = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
        self.lb[80:560] = big
        self.b.input().set_buffer(self.lb)
        self.b.output().set_buffer(self.out)
        self.cm.run([self.b], 1000)
        res = self.b.output().get_buffer()
        persons = res[0] if isinstance(res, list) else []
        best = None
        for y0, x0, y1, x1, sc in np.asarray(persons).reshape(-1, 5):
            if sc < 0.5:
                continue
            # letterbox rows 80..560 of 640 hold the image
            fy0, fy1 = (y0 * 640 - 80) / 480, (y1 * 640 - 80) / 480
            area = (x1 - x0) * (fy1 - fy0)
            if best is None or area > best["area"]:
                best = {"cx": (x0 + x1) / 2, "cy": (fy0 + fy1) / 2, "w": x1 - x0,
                        "h": fy1 - fy0, "score": float(sc), "area": area}
        return best

    def close(self) -> None:
        try:
            self.cm.__exit__(None, None, None)
        except Exception:
            pass


def make_person_detector():
    """The Pi's AI HAT (Hailo) if present, else TensorRT on a Jetson
    (robot/detector.py; engine: $FLY_PERSON_ENGINE or ~/milo/models/
    yolox_tiny.engine). A detector's .bgr says which frame it takes."""
    errs = []
    if os.path.exists(HAILO_HEF):
        try:
            return PersonDetector()
        except Exception as ex:
            errs.append(f"Hailo: {ex}")
    from robot import detector
    eng = os.environ.get("FLY_PERSON_ENGINE") or detector.DEFAULT_ENGINE
    if os.path.exists(eng):
        try:
            return detector.YoloxPersons(eng)
        except Exception as ex:
            errs.append(f"TensorRT: {ex}")
    raise RuntimeError("; ".join(errs) or f"no Hailo HEF ({HAILO_HEF}) or TensorRT engine ({eng})")


# --------------------------------------------------------------- shared memory
_F = ["seq", "t_wall", "fps", "frames", "pan_deg", "tilt_deg", "moving",
      # moving object / looming (brain/sensory/camera.LoomingExtractor, head-centred)
      "obj_active", "obj_az", "obj_el", "obj_half_deg", "obj_exp_deg_s",
      # person (Hailo)
      "person_active", "person_az", "person_el", "person_half_deg", "person_score",
      "person_t", "person_ms",
      # commands from the simulation
      "cmd_seq", "cmd_dpan", "cmd_dtilt", "cmd_reset", "cmd_done", "cmd_stop",
      "preview_seq", "preview_len",
      # the whole frame, small and grey, with where the head pointed (robot/eye.py)
      "gray_seq", "gray_w", "gray_h", "gray_pan", "gray_tilt", "gray_moving"]
_I = {k: i for i, k in enumerate(_F)}
_HEADER = 512
_PREVIEW_MAX = 256 * 1024
GRAY_W, GRAY_H = 128, 96                     # 0.4 deg a pixel: finer than the fly's 5.8 deg columns
_GRAY_AT = _HEADER + _PREVIEW_MAX
SHM_SIZE = _GRAY_AT + GRAY_W * GRAY_H


def _seq_write(buf, i, fn):
    buf[i] += 1
    fn()
    buf[i] += 1


def _preview(rgb, mask, est, person, pan, tilt, moving, fps) -> bytes:
    import cv2
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
    top = int((240 - 180) / 2 * 2)                        # band offset in preview px
    if mask is not None and mask.any():
        m = cv2.resize(mask, (640, 360), interpolation=cv2.INTER_NEAREST) > 0
        band = img[top:top + 360]
        band[m] = (0.55 * band[m] + 0.45 * np.array([172, 212, 77])).astype(np.uint8)
    cv2.rectangle(img, (0, top), (639, top + 359), (90, 90, 90), 1)
    if est["active"]:
        cx = int((est["azimuth_deg"] / HFOV_DEG + 0.5) * 640)
        cy = top + int((0.5 - est["elevation_deg"] / VFOV_DEG) * 360)
        r = int(est["half_angle_deg"] / HFOV_DEG * 640)
        col = (129, 107, 255) if est["expansion_rate_deg_s"] > 0 else (220, 220, 220)
        cv2.circle(img, (cx, cy), max(r, 3), col, 2)
    if person:
        x0 = int((person["cx"] - person["w"] / 2) * 640); x1 = int((person["cx"] + person["w"] / 2) * 640)
        y0 = int((person["cy"] - person["h"] / 2) * 480); y1 = int((person["cy"] + person["h"] / 2) * 480)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 180, 60), 2)
        cv2.putText(img, "person %.0f%%" % (100 * person["score"]), (x0 + 4, max(14, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 60), 1, cv2.LINE_AA)
    txt = "%.0f fps  pan %+.0f  tilt %+.0f%s" % (fps, pan, tilt, "  [moving: vision blanked]" if moving else "")
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpg.tobytes() if ok else b""


def _emit(d: dict) -> None:
    """A message to the parent (HeadFeed), one JSON line on stdout."""
    sys.stdout.write("@@" + json.dumps(d) + "\n")
    sys.stdout.flush()


def _stdin_commands() -> collections.deque:
    """Commands from the parent, one JSON object per line on stdin."""
    q = collections.deque(maxlen=64)

    def run():
        for line in sys.stdin:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                q.append(d)
    threading.Thread(target=run, daemon=True).start()
    return q


def _face_angles(box, w: int, h: int) -> dict:
    """A face box -> head-centred azimuth, elevation and angular half-width (deg)."""
    x0, y0, x1, y1 = box
    return {"az": round(((x0 + x1) / 2 / w - 0.5) * HFOV_DEG, 2),
            "el": round((0.5 - (y0 + y1) / 2 / h) * HFOV_DEG * h / w, 2),
            "half": round((x1 - x0) / w * HFOV_DEG / 2, 2)}


def run_worker(shm_name: str, dev: str, use_person: bool = True, use_looming: bool = True,
               use_faces: bool = False) -> None:
    import cv2
    from brain.sensory.camera import LoomingExtractor

    if sys.version_info >= (3, 13):
        shm = shared_memory.SharedMemory(name=shm_name, track=False)
    else:
        # This subprocess has its own resource tracker; the parent owns unlink().
        from multiprocessing import resource_tracker
        shm = shared_memory.SharedMemory(name=shm_name)
        resource_tracker.unregister(shm._name, "shared_memory")
    hdr = np.ndarray((len(_F),), dtype=np.float64, buffer=shm.buf)
    live = dev.startswith("/dev/")
    pt = PanTilt(dev) if live else FixedHead()     # a video file: a recording, replayed
    pt.reset()
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2 if live else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    pt.fixed_framerate()
    ext = LoomingExtractor(HFOV_DEG, VFOV_DEG)
    det = None
    if use_person:
        try:
            det = make_person_detector()
        except Exception as ex:                          # no detector: carry on without
            print("person detector unavailable:", ex, file=sys.stderr)
    faces = tracker = cmds = None
    if use_faces:
        cmds = _stdin_commands()
        try:
            from robot.faces import Faces
            from robot.people import FaceTracker, PeopleBook
            faces = Faces()
            tracker = FaceTracker(PeopleBook())
        except Exception as ex:                          # no models: carry on without
            print("face recognition unavailable:", ex, file=sys.stderr)
            faces = None
    t_face_next = 0.0
    parent = os.getppid()
    last_cmd = 0
    person, person_t, det_ms, t_det_next = None, 0.0, 0.0, 0.0
    t_last, frames, rate = time.monotonic(), 0, 0.0
    try:
        while os.getppid() == parent and not hdr[_I["cmd_stop"]]:
            # commands from the simulation (efference: executed here, vision gated)
            cs = int(hdr[_I["cmd_seq"]])
            if cs != last_cmd and cs % 2 == 0:
                last_cmd = cs
                if hdr[_I["cmd_reset"]]:
                    pt.reset()
                else:
                    pt.move(float(hdr[_I["cmd_dpan"]]), float(hdr[_I["cmd_dtilt"]]))
                hdr[_I["cmd_done"]] = cs
            ok, frame = cap.read()
            if not ok:
                if not live:                               # a recording: play it again
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue
            if not live:
                time.sleep(1.0 / 15)                       # at a camera's pace
            t = time.monotonic()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            band = frame[30:210]                           # central 320x180
            gray = cv2.resize(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), (PROC_W, PROC_H),
                              interpolation=cv2.INTER_AREA)
            moving = pt.moving
            if moving or not use_looming:
                ext._prev = None                           # efference copy: no motion signal
                est = ext._inactive(); ext.est = est; ext.mask[:] = 0
            else:
                est = ext.update(gray, t)
            if det is not None and not moving and t >= t_det_next - 0.005:
                # on a 0.1 s schedule (frames come every 33-67 ms): 10 Hz on
                # average; after a gap (the head moved) start afresh
                t_det_next = t_det_next + PERSON_PERIOD_S if t - t_det_next < PERSON_PERIOD_S else t + PERSON_PERIOD_S
                try:
                    t_det = time.perf_counter()
                    person = det.detect(frame if getattr(det, "bgr", False) else rgb)
                    person_t = time.time()
                    det_ms = 1e3 * (time.perf_counter() - t_det)
                except Exception as ex:
                    print("detector error:", ex, file=sys.stderr)
                    try:
                        det.close()
                    except Exception:
                        pass
                    det = None
            if moving:
                person = None
            while cmds:
                # commands from the brain client (robot/people.Social); a failure
                # is reported back, never the end of the camera process
                c = cmds.popleft()
                try:
                    if faces is None or tracker is None:
                        ev = ({"event": "enroll_failed", "track": c.get("track"), "name": c.get("name"),
                               "why": "face recognition off"} if c.get("cmd") == "enroll" else None)
                    else:
                        ev = tracker.command(c, t)
                except Exception as ex:
                    print("face command error:", ex, file=sys.stderr)
                    ev = {"event": f"{c.get('cmd')}_failed", "track": c.get("track"), "name": c.get("name"),
                          "why": "error"}
                if ev is not None:
                    try:
                        _emit(ev)
                    except (OSError, ValueError):
                        pass
            if faces is not None:
                if not moving and t >= t_face_next - 0.005:
                    t_face_next = t_face_next + FACE_PERIOD_S if t - t_face_next < FACE_PERIOD_S else t + FACE_PERIOD_S
                    try:
                        t_f = time.perf_counter()
                        seen = tracker.update(faces(frame), t)
                        h_, w_ = frame.shape[:2]
                        for o in seen:
                            o.update(_face_angles(o["box"], w_, h_))
                        _emit({"faces": seen, "ms": round(1e3 * (time.perf_counter() - t_f), 2)})
                    except Exception as ex:
                        print("face recognition error:", ex, file=sys.stderr)
                        faces.close()
                        faces = None
            frames += 1
            if t - t_last >= 1.0:
                rate, frames, t_last = frames / (t - t_last), 0, t

            def write():
                hdr[_I["t_wall"]] = time.time()
                hdr[_I["fps"]] = rate
                hdr[_I["frames"]] += 1
                hdr[_I["pan_deg"]], hdr[_I["tilt_deg"]] = pt.pan_deg, pt.tilt_deg
                hdr[_I["moving"]] = float(moving)
                hdr[_I["person_ms"]] = det_ms
                hdr[_I["obj_active"]] = float(est["active"])
                hdr[_I["obj_az"]], hdr[_I["obj_el"]] = est["azimuth_deg"], est["elevation_deg"]
                hdr[_I["obj_half_deg"]] = est["half_angle_deg"]
                hdr[_I["obj_exp_deg_s"]] = est["expansion_rate_deg_s"]
                if person is not None:
                    # the target is the person's head and shoulders (the part the
                    # pet attends to; ~15 deg at room distance, LC10a's preferred
                    # size): width ~ box width, height ~ box height / 3, at the top
                    w_deg = person["w"] * HFOV_DEG
                    h_deg = person["h"] * HFOV_DEG * CAP_H / CAP_W
                    head = min(0.9 * w_deg, h_deg / 3.0)
                    top_deg = (0.5 - (person["cy"] - person["h"] / 2)) * HFOV_DEG * CAP_H / CAP_W
                    hdr[_I["person_active"]] = 1.0
                    hdr[_I["person_az"]] = (person["cx"] - 0.5) * HFOV_DEG
                    hdr[_I["person_el"]] = top_deg - head / 2
                    hdr[_I["person_half_deg"]] = head / 2
                    hdr[_I["person_score"]] = person["score"]
                    hdr[_I["person_t"]] = person_t
                else:
                    hdr[_I["person_active"]] = 0.0
            _seq_write(hdr, _I["seq"], write)
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (GRAY_W, GRAY_H),
                               interpolation=cv2.INTER_AREA)

            def wg():
                shm.buf[_GRAY_AT:_GRAY_AT + small.size] = small.tobytes()
                hdr[_I["gray_w"]], hdr[_I["gray_h"]] = GRAY_W, GRAY_H
                hdr[_I["gray_pan"]], hdr[_I["gray_tilt"]] = pt.pan_deg, pt.tilt_deg
                hdr[_I["gray_moving"]] = float(moving)
            _seq_write(hdr, _I["gray_seq"], wg)
            if int(hdr[_I["frames"]]) % 3 == 0:
                jpg = _preview(rgb, ext.mask, est, person, pt.pan_deg, pt.tilt_deg, moving, rate)
                if 0 < len(jpg) <= _PREVIEW_MAX:
                    def wp():
                        shm.buf[_HEADER:_HEADER + len(jpg)] = jpg
                        hdr[_I["preview_len"]] = len(jpg)
                    _seq_write(hdr, _I["preview_seq"], wp)
    finally:
        cap.release()
        if det is not None:
            det.close()
        if faces is not None:
            faces.close()
        try:
            pt.move(-pt.pan_deg, -pt.tilt_deg)             # leave the head centred
        except Exception:
            pass
        pt.close()
        del hdr
        shm.close()


# ------------------------------------------------------------ simulation side
def read_gray(shm, hdr):
    """The head's latest small grey frame from its shared memory (a seqlock
    read: None if it never settles or there is no frame yet)."""
    i = _I["gray_seq"]
    for _ in range(1000):
        s1 = hdr[i]
        if int(s1) % 2 or s1 == 0:
            if s1 == 0:
                return None
            continue
        w, h = int(hdr[_I["gray_w"]]), int(hdr[_I["gray_h"]])
        img = np.frombuffer(shm.buf, np.uint8, w * h, _GRAY_AT).reshape(h, w).copy()
        out = (int(s1), img, float(hdr[_I["gray_pan"]]), float(hdr[_I["gray_tilt"]]), bool(hdr[_I["gray_moving"]]))
        if hdr[i] == s1:
            return out
    return None


class HeadFeed:
    """Starts the head's sensor process; reads its state, sends motor commands."""

    STALE_S = 0.5

    def __init__(self, dev: str | None = None, person: bool = True, looming: bool = True,
                 faces: bool = False):
        dev = dev or find_orbit()
        if dev is None:
            raise RuntimeError("pan/tilt camera (Logitech Orbit) not found")
        self.shm = shared_memory.SharedMemory(create=True, size=SHM_SIZE)
        self.shm.buf[:SHM_SIZE] = bytes(SHM_SIZE)
        self._h = np.ndarray((len(_F),), dtype=np.float64, buffer=self.shm.buf)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args = [sys.executable, "-m", "robot.head", "--shm", self.shm.name, "--dev", dev]
        if not person:
            args.append("--no-person")
        if not looming:
            args.append("--no-looming")
        if faces:
            args.append("--faces")
        self.proc = subprocess.Popen(args, cwd=root, stderr=subprocess.PIPE,
                                     stdout=subprocess.PIPE if faces else subprocess.DEVNULL,
                                     stdin=subprocess.PIPE if faces else None)
        self._faces = None                       # (monotonic t, [tracks]) from the worker
        self._events = collections.deque(maxlen=64)
        if faces:
            threading.Thread(target=self._messages, daemon=True).start()
        # Drain the worker's stderr (libjpeg and V4L2 warnings, TensorRT's
        # log): a full pipe would block the worker for good on a long run.
        # The last few KB are kept for error().
        self._err = collections.deque(maxlen=64)
        threading.Thread(target=self._drain, daemon=True).start()

    def _messages(self) -> None:
        try:
            for line in self.proc.stdout:
                if not line.startswith(b"@@"):
                    continue
                try:
                    d = json.loads(line[2:])
                except ValueError:
                    continue
                if "faces" in d:
                    self._faces = (time.monotonic(), d["faces"])
                else:
                    self._events.append(d)
        except (OSError, ValueError):
            pass

    def faces(self, max_age_s: float = 1.0) -> list | None:
        """The faces in view (robot/people.FaceTracker tracks with head-centred
        az / el / half), or None if face recognition is off or stale (the head
        moving, the camera stalled)."""
        f = self._faces
        if f is None or time.monotonic() - f[0] > max_age_s:
            return None
        return f[1]

    def events(self) -> list:
        """Messages from the camera process since the last call (e.g. enrolled)."""
        out = []
        while self._events:
            out.append(self._events.popleft())
        return out

    def send(self, cmd: dict) -> bool:
        """A command for the camera process (e.g. {"cmd": "enroll", ...})."""
        if self.proc.stdin is None or not self.alive:
            return False
        try:
            self.proc.stdin.write((json.dumps(cmd) + "\n").encode())
            self.proc.stdin.flush()
            return True
        except (OSError, ValueError):
            return False

    def _drain(self) -> None:
        try:
            for line in self.proc.stderr:
                self._err.append(line.decode(errors="replace"))
        except (OSError, ValueError):
            pass

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def error(self) -> str:
        if self.alive:
            return ""
        try:
            self.proc.wait(timeout=1)             # the drain thread reads to the end
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.05)
        return "".join(self._err)[-2000:]

    def _read(self, i, fn):
        for _ in range(1000):
            s1 = self._h[i]
            if int(s1) % 2:
                continue
            out = fn()
            if self._h[i] == s1:
                return out
        return None

    def state(self) -> dict:
        v = self._read(_I["seq"], lambda: self._h.copy())
        if v is None or v[_I["frames"]] == 0:
            return {"ready": False, "frames": 0}
        d = {k: float(v[_I[k]]) for k in _F}
        d["ready"] = True
        d["stale"] = time.time() - d["t_wall"] > self.STALE_S
        d["person_stale"] = time.time() - d["person_t"] > 0.5
        return d

    def command(self, dpan_deg: float, dtilt_deg: float = 0.0, reset: bool = False) -> None:
        h = self._h
        h[_I["cmd_seq"]] += 1                           # odd: writing
        h[_I["cmd_dpan"]], h[_I["cmd_dtilt"]] = dpan_deg, dtilt_deg
        h[_I["cmd_reset"]] = 1.0 if reset else 0.0
        h[_I["cmd_seq"]] += 1                           # even: ready

    def gray(self):
        """The latest small grey frame: (seq, (GRAY_H, GRAY_W) uint8, pan_deg,
        tilt_deg, moving), or None (robot/eye.py reads the same memory)."""
        return read_gray(self.shm, self._h)

    def preview_jpeg(self) -> bytes | None:
        def grab():
            n = int(self._h[_I["preview_len"]])
            return bytes(self.shm.buf[_HEADER:_HEADER + n]) if n else None
        return self._read(_I["preview_seq"], grab)

    def close(self) -> None:
        if self.proc.poll() is None:
            self._h[_I["cmd_stop"]] = 1.0
            try:
                self.proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        del self._h
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass


class HeadLoomingStimulus:
    """The head camera's moving object as a looming stimulus (LoomingEncoder)."""

    def __init__(self, feed: HeadFeed):
        self.feed = feed

    def state(self, t_ms: float) -> dict:
        s = self.feed.state()
        on = s.get("ready") and not s.get("stale") and s.get("obj_active", 0) > 0
        return {"t_ms": t_ms, "source": "head", "active": bool(on),
                "azimuth_deg": s.get("obj_az", 0.0), "elevation_deg": s.get("obj_el", 0.0),
                "half_angle_deg": s.get("obj_half_deg", 0.0) if on else 0.0,
                "expansion_rate_deg_s": s.get("obj_exp_deg_s", 0.0) if on else 0.0,
                "camera_fps": round(s.get("fps", 0.0), 1)}


class ObjectEncoder:
    """Drives the real LC10a neurons (small-object detectors of the pursuit
    pathway) from the head camera: the detected person if any, otherwise the
    largest moving object. See the module docstring (B, C)."""

    CELL_TYPES = ("LC10a",)
    MAX_HZ = 150.0           # the model's standard activation rate (LIFParams.r_poi)
    PEAK_DEG = 15.0          # preferred angular (full) size
    STILL_FRACTION = 0.4     # a still object drives LC10a less than a moving one
    arousal = 1.0            # interest in the person (see __init__)

    def __init__(self, connectome, feed: HeadFeed):
        from brain.sensory.retinotopy import load_retinotopy, receptive_fields_2hop
        self.feed = feed
        # LC10a's inputs are mostly not column-assigned: two-synapse RF estimate
        rf = receptive_fields_2hop(load_retinotopy(connectome), "LC10a").dropna(subset=["azimuth_deg"])
        self.indices = rf["idx"].to_numpy(np.int64)
        order = np.argsort(self.indices)
        self.indices = self.indices[order]
        self._az = rf["azimuth_deg"].to_numpy(float)[order]
        self._el = rf["elevation_deg"].to_numpy(float)[order]
        self._sigma = np.clip(rf["rf_radius_deg"].to_numpy(float)[order], 8.0, 40.0)
        # Interest in the person, 0..1 (set by the neocortex; 1 = innate): the
        # fly's pursuit is gated by internal state (in courting males P1
        # arousal gates the LC10a pursuit pathway; Hindmarsh Sten et al. 2021).
        # Applies to the person only, not to other moving objects.
        self.arousal = 1.0
        self.last = {}

    def rates_hz(self, t_ms: float, stim=None) -> np.ndarray:
        from simulation.stimuli.looming import angular_distance_deg
        from brain.sensory.encoders import frozen, memo_of
        s = self.feed.state()
        rm = memo_of(self)
        rates = rm.zeros(len(self.indices))
        self.last = {"target": None}
        if not s.get("ready") or s.get("stale") or s.get("moving", 0) > 0:
            return rates
        if s.get("person_active", 0) > 0 and not s.get("person_stale"):
            az, el, half, kind = s["person_az"], s["person_el"], s["person_half_deg"], "person"
            # Is the person moving? Estimated once per new detection (the
            # feed's person_t), not per call: rates_hz runs every block while a
            # detection stays the same. A clock that jumps back (a new episode
            # in simulation) or a long gap starts the estimate afresh.
            pt = s.get("person_t", time.monotonic())
            prev = getattr(self, "_prev_person", None)
            if prev is None or pt < prev[0] or pt - prev[0] > 1.0:
                self._prev_person, self._pmove = (pt, az), 0.0
            elif pt - prev[0] >= 0.08:
                speed = abs(az - prev[1]) / (pt - prev[0])
                self._pmove = 0.8 * self._pmove + 0.2 * (1.0 if speed > 5.0 else 0.0)
                self._prev_person = (pt, az)
            moving = self._pmove
        elif s.get("obj_active", 0) > 0:
            az, el, half, moving, kind = s["obj_az"], s["obj_el"], s["obj_half_deg"], 1.0, "object"
        else:
            return rates
        # the same target as the last block (the camera updates at ~10 Hz,
        # the session asks every 1 ms): the same rates
        key = (az, el, half, moving, kind, self.arousal, self.PEAK_DEG, self.STILL_FRACTION, self.MAX_HZ)
        if rm.rates is not None and rm.key == key:
            self.last = dict(self._memo_last, azimuth_deg=round(az, 1))
            return rm.rates
        size = max(2 * half, 1.0)
        tuning = np.exp(-0.5 * (np.log(size / self.PEAK_DEG) / 0.8) ** 2)
        gain = self.STILL_FRACTION + (1 - self.STILL_FRACTION) * moving
        if kind == "person":
            gain *= self.arousal
        d = angular_distance_deg(az, el, self._az, self._el)
        edge = np.maximum(0.0, d - half)
        rates = self.MAX_HZ * tuning * gain * np.exp(-edge ** 2 / (2 * self._sigma ** 2))
        self.last = {"target": kind, "azimuth_deg": round(az, 1), "size_deg": round(size, 1),
                     "drive_hz": round(float(rates.max()), 1)}
        self._memo_last = dict(self.last)
        rm.rates, rm.key = frozen(rates), key
        return rm.rates

    def state(self, t_ms: float) -> dict:
        return {"kind": "head_object", "active": self.last.get("target") is not None, **self.last}

    @property
    def provenance(self) -> dict:
        return {"drives": {"LC10a": int(len(self.indices))},
                "tuning": {"max_hz": self.MAX_HZ, "peak_size_deg": self.PEAK_DEG,
                           "still_fraction": self.STILL_FRACTION},
                "source": "Ribeiro et al. 2018 Cell 174:607; Hindmarsh Sten et al. 2021 Nature 595:549"}


class HeadController:
    """Turns the head from the brain's steering output (see module docstring)."""

    GAIN_DEG_S = 200.0       # deg/s per unit turn_bias; measured: a 13-deg target at
                             # +-30 deg gives turn_bias ~+0.20 / -0.27 (LC10a at 150 Hz)
    P9_GAIN_DEG_S = 60.0
    PERIOD_S = 0.3
    MIN_STEP_DEG = 3.0
    SMOOTH_TAU_S = 0.5       # DN readout has few spikes per 50 ms window: smooth it
    BASELINE_TAU_S = 5.0     # learning the resting steering bias (no target in view)

    def __init__(self, feed: HeadFeed, target_fn=None):
        self.feed = feed
        self.target_fn = target_fn or (lambda: False)
        self.pending = 0.0
        self.t_last = None
        self._since = 0.0
        self.baseline = 0.0
        self.log = []

    def update(self, channels: dict, now_s: float | None = None) -> None:
        now_s = time.monotonic() if now_s is None else now_s
        if self.t_last is None:
            self.t_last = now_s
            return
        dt = now_s - self.t_last
        self.t_last = now_s
        # right-minus-left steering activity; + = turn right (Rayshubskiy et al. 2024)
        raw = (self.GAIN_DEG_S * channels.get("turn_bias", 0.0)
               + self.P9_GAIN_DEG_S * channels.get("forward_walk_lr", 0.0))
        self.smooth = getattr(self, "smooth", 0.0)
        self.smooth += (raw - self.smooth) * min(1.0, dt / self.SMOOTH_TAU_S)
        steer = self.smooth
        if not self.target_fn():
            self.baseline += (steer - self.baseline) * min(1.0, dt / self.BASELINE_TAU_S)
        rate = steer - self.baseline
        self.pending += rate * dt
        self._since += dt
        s = self.feed.state()
        esc = channels.get("escape_long_mode", 0.0)
        flinch_until = getattr(self, "flinch_until", 0.0)
        if esc >= 0.5 and now_s > flinch_until + 2.0 and not s.get("moving", 0):
            self.feed.command(0.0, 8.0)              # startle: head up
            self.flinch_until = now_s + 1.0
            self.log.append((round(now_s, 2), "startle"))
            return
        if flinch_until and now_s >= flinch_until and not s.get("moving", 0):
            self.feed.command(0.0, -8.0)             # settle back
            self.flinch_until = 0.0
            return
        if s.get("moving", 0):
            self.pending = 0.0                       # the turn is already under way
        elif self._since >= self.PERIOD_S and abs(self.pending) >= self.MIN_STEP_DEG:
            self.feed.command(self.pending, 0.0)
            self.log.append((round(now_s, 2), round(self.pending, 1)))
            self.log = self.log[-50:]
            self.pending = 0.0
            self._since = 0.0


def main():
    ap = argparse.ArgumentParser(description="robot head sensor process")
    ap.add_argument("--shm")
    ap.add_argument("--dev")
    ap.add_argument("--no-person", action="store_true")
    ap.add_argument("--no-looming", action="store_true", help="no moving-object estimate (saves CPU)")
    ap.add_argument("--faces", action="store_true", help="faces and who they are (robot/faces.py, robot/people.py)")
    ap.add_argument("--test", action="store_true", help="run the sensor for 10 s and print")
    a = ap.parse_args()
    if a.test:
        feed = HeadFeed(person=not a.no_person)
        t0 = time.time()
        while time.time() - t0 < 12:
            time.sleep(1)
            s = feed.state()
            print({k: round(v, 1) for k, v in s.items() if isinstance(v, float) and k in
                   ("fps", "pan_deg", "moving", "obj_active", "obj_az", "obj_half_deg",
                    "person_active", "person_az", "person_el", "person_half_deg", "person_score",
                    "person_ms")},
                  feed.error())
            if 5 < time.time() - t0 < 6.2:
                feed.command(15.0)
        feed.close()
        return
    run_worker(a.shm, a.dev or find_orbit(), use_person=not a.no_person, use_looming=not a.no_looming,
               use_faces=a.faces)


if __name__ == "__main__":
    main()


class RestingOlfaction:
    """Every olfactory receptor neuron at its measured spontaneous rate
    (Hallem & Carlson 2006, brain/sensory/olfaction.py): the brain's resting
    background while the head looks around the (odourless) room."""

    def __init__(self, connectome):
        from brain.sensory.olfaction import OlfactorySpace
        t = connectome.neurons["primary_type"].fillna("").astype(str).to_numpy()
        self.indices = np.flatnonzero(np.char.startswith(t.astype(str), "ORN_"))
        types = sorted(set(t[self.indices]))
        osp = OlfactorySpace([ty[4:] for ty in types])
        gi = {ty: i for i, ty in enumerate(types)}
        self._rates = np.array([osp.spont[gi[t[i]]] for i in self.indices])
        self._rates.flags.writeable = False       # constant (Session._rates_at: unchanged by identity)

    def rates_hz(self, t_ms: float, stim=None) -> np.ndarray:
        return self._rates

    def state(self, t_ms: float) -> dict:
        return {"kind": "resting_olfaction", "active": True}

    @property
    def provenance(self) -> dict:
        return {"drives": {"ORN (all)": int(len(self.indices))},
                "source": "Hallem & Carlson 2006, Cell 125:143 (spontaneous rates)"}
