"""
2D lidar as fly senses (the Waveshare UGV Rover's lidar; in Habitat, a
simulated scan: sim/habitat_bridge/habitat_server.py --lidar).

A 2D lidar gives distances in one horizontal plane all around the robot. It is
not vision; it feeds two of the fly's senses:

  looming   something MOVING toward the pet -> the looming detectors LC4 /
            LPLC2 (the same neurons the camera drives, robot/head.py). Self-
            motion is discounted, as flies do with an efference copy: the last
            scan is projected into the robot's new pose, and a beam looms only
            where the world came closer than a still world predicts, faster
            than MOVING_MS, within LOOM_MAX_M (LidarLooming, a stimulus for
            brain.sensory.encoders.LoomingEncoder). (Without this, the first
            test drove LC4/LPLC2 in 99.8% of steps: turning makes walls
            "approach" in some beams.)
  touch     an obstacle within TOUCH_M of the body -> the antennal and
            vibrissal head bristles (BM_Ant, BM_Vib) on that side, as a fly
            feels a wall with its antennae (LidarTouch)

Beam angles are relative to the heading, + = right (as camera azimuth).

C. APPROXIMATIONS: not a fly sense. A fly avoids obstacles while walking
mostly with vision and antennal touch; the lidar stands in for both where the
camera cannot see (behind, beside, in the dark). The brain-only connectome has
no leg mechanosensation, so head bristles carry "touch".
"""
from __future__ import annotations

import math
import os

import numpy as np

OBJECT_W_M = 0.3            # nominal obstacle width for the looming angle
LOOM_MAX_M = 2.0            # beyond this, nothing looms
TOUCH_M = 0.15              # "antenna" reach beyond the body surface
TOUCH_HZ = 80.0             # as petting (sim/habitat_bridge/home.py)
# Bristle mechanoreceptors are rapidly adapting: they signal a deflection and
# fall mostly silent under steady contact (NompC channels; Walker, Willingham &
# Zuker 2000, Science 287:2229). A steady wall at 80 Hz kept the brain in a
# touch response (backing, grooming, even proboscis extension) and the robot
# sat against walls for ~30% of the time (2026-09-28). The drive is the contact
# level minus its adapted level (time constant TOUCH_ADAPT_S) plus a small tonic
# part; FLY_TOUCH_ADAPT=0 restores the steady drive.
TOUCH_ADAPT_S = 0.5
TOUCH_TONIC = 0.15
FRONT_DEG = 30.0            # beams within this of straight ahead touch both sides
MOVING_MS = 0.3             # closing speed beyond self-motion that counts as "coming at me"


def beam_angles(beams: int) -> list:
    """Beam angles (deg, + = right), as habitat_server.set_lidar."""
    return [(-180.0 + 360.0 * k / beams) for k in range(beams)]


def ellipse_body(angles_deg, half_length_m: float, half_width_m: float) -> np.ndarray:
    """Distance from the lidar (body centre) to the body surface, per beam."""
    a = np.radians(np.asarray(angles_deg, float))
    return 1.0 / np.sqrt((np.cos(a) / half_length_m) ** 2 + (np.sin(a) / half_width_m) ** 2)


def rect_body(angles_deg, half_length_m: float, half_width_m: float) -> np.ndarray:
    """Distance from the lidar (body centre) to a rectangular body's edge, per beam."""
    a = np.radians(np.asarray(angles_deg, float))
    with np.errstate(divide="ignore"):
        return np.minimum(half_length_m / np.abs(np.cos(a)), half_width_m / np.abs(np.sin(a)))


def body_profile(angles_deg, body: dict) -> np.ndarray:
    """Per-beam body surface for a sim/habitat_bridge/bodies.py entry."""
    f = rect_body if body.get("shape") == "rect" else ellipse_body
    return f(angles_deg, body["half_len"], body["half_wid"])


class LidarScan:
    """The latest scan: ranges (m) at beam angles (deg, + = right); body = the
    body surface's distance from the lidar (scalar or per beam)."""

    def __init__(self, angles_deg, body_radius_m, max_range_m: float = 8.0):
        self.angles = np.asarray(angles_deg, float)
        self.max_range = float(max_range_m)          # a beam at max range hit nothing
        self.body = np.broadcast_to(np.asarray(body_radius_m, float), self.angles.shape).copy()
        self.ranges = np.full(len(self.angles), np.inf)
        self.t = None
        self.pose = None

    def update(self, ranges, t_s: float, pose=None) -> None:
        """pose = (x, z, yaw_deg) of the robot (odometry / IMU on the robot;
        Habitat's pose in simulation), yaw CCW in the (x, -z) plane."""
        r = np.asarray(ranges, float)
        self.ranges = np.where(np.isfinite(r) & (r > 0) & (r < self.max_range - 1e-3), r, np.inf)
        self.t = t_s
        self.pose = pose

    def points_world(self) -> np.ndarray:
        """Hit points in world (x, z), finite beams only."""
        x, z, yaw = self.pose
        ok = np.isfinite(self.ranges)
        b = np.radians(yaw - self.angles[ok])
        r = self.ranges[ok]
        return np.stack([x + r * np.cos(b), z - r * np.sin(b)], axis=1)

    def points_world_all(self) -> np.ndarray:
        """Hit points in world (x, z) for every beam, nan where nothing was hit."""
        x, z, yaw = self.pose
        b = np.radians(yaw - self.angles)
        r = np.where(np.isfinite(self.ranges), self.ranges, np.nan)
        return np.stack([x + r * np.cos(b), z - r * np.sin(b)], axis=1)

    def predict_from(self, pts) -> np.ndarray:
        """Per beam of THIS pose: range at which a still world (the given world
        points) would be seen; inf where no old point falls in the beam."""
        x, z, yaw = self.pose
        dx, dz = pts[:, 0] - x, pts[:, 1] - z
        rng = np.hypot(dx, dz)
        bearing = np.degrees(np.arctan2(-dz, dx))                   # CCW, (x, -z)
        az = ((yaw - bearing + 180.0) % 360.0) - 180.0              # + = right
        step = 360.0 / len(self.angles)
        k = np.round((az - self.angles[0]) / step).astype(int) % len(self.angles)
        out = np.full(len(self.angles), np.inf)
        np.minimum.at(out, k, rng)
        return out

    def clearance(self) -> np.ndarray:
        """Distance from the body surface to the nearest thing, per beam."""
        return np.maximum(self.ranges - self.body, 0.0)


class LidarLooming:
    """The most threatening MOVING thing as a looming stimulus (LoomingEncoder).

    Self-motion is discounted like a fly's efference copy, by matching points:
    each current hit point (world frame, from the robot's pose) is compared
    with the nearest point of the scan BASE_S earlier. A still world matches
    within the scan's own resolution (MATCH_M + range x beam spacing); a point
    that matches nothing and came closer faster than MOVING_MS is something
    coming at the pet. Its looming drive is the angular expansion rate of an
    object of width OBJECT_W_M closing at that speed. (Review 2026-09-26: the
    first version compared ranges within 4-degree bins and fired on slanted
    still walls in half the steps while walking.)"""

    BASE_S = 0.5              # compare with the scan this long ago
    FAST_BASE_S = 0.15        # .fast: a short baseline for quick rushes (a kick lasts ~0.15 s;
                              # robot/threat.FastDanger), as noisy but over a 1.6 m/s threshold
    MATCH_M = 0.08
    RADIAL_DEG = 25.0
    CONFIRM_DEG = 15.0
    EDGE_M = 0.15

    def __init__(self, scan: LidarScan):
        self.scan = scan
        self.reset()

    def reset(self) -> None:
        """A new episode / day: no history."""
        from collections import deque
        self._hist = deque()
        self.last = {"active": False}
        self.fast = {"active": False}
        self._fast_prev = {"active": False}

    def update(self) -> None:
        s = self.scan
        self.last = {"active": False}
        self.fast = {"active": False}
        if s.t is None or s.pose is None:
            self._fast_prev = self.fast
            return
        while self._hist and (s.t < self._hist[-1][0] or s.t - self._hist[0][0] > 2 * self.BASE_S):
            if s.t < self._hist[-1][0]:
                self._hist.clear()               # the clock went back: a new episode
            else:
                self._hist.popleft()
        refs = {}
        for t_old, pts_old in self._hist:
            for name, base in (("last", self.BASE_S), ("fast", self.FAST_BASE_S)):
                if s.t - t_old >= base - 1e-6:
                    refs[name] = (t_old, pts_old)  # the newest scan at least `base` old
        ok = np.isfinite(s.ranges)
        self._hist.append((s.t, s.points_world_all()))
        if not ok.any():
            return
        pts = s.points_world()                   # finite beams only, in beam order
        prev_fast = self._fast_prev
        for name, ref in refs.items():
            setattr(self, name, self._approach(s, ok, pts, ref, radial=name == "fast"))
        # a rush builds over scans: the one before saw it approaching too, on
        # about the same bearing and farther out. Something stepping out from
        # behind a doorframe appears at once (Habitat: "rushes" at 8-10 m/s)
        # The closing speed is then the lesser of the two estimates: since the
        # reference scan, and since the scan before (one step out from cover
        # still reaches back to the wall behind)
        f = self.fast
        if f["active"]:
            f["t"] = s.t
            f["confirmed"] = bool(prev_fast.get("active") and abs(((f["azimuth_deg"] - prev_fast["azimuth_deg"]
                                                                       + 180.0) % 360.0) - 180.0) <= self.CONFIRM_DEG
                                  and prev_fast["range_m"] > f["range_m"] and s.t > prev_fast["t"])
            if f["confirmed"]:
                step = (prev_fast["range_m"] - f["range_m"]) / (s.t - prev_fast["t"])
                f["closing_ms"] = min(f["closing_ms"], step)
        self._fast_prev = f

    def _approach(self, s, ok, pts, ref, radial: bool = False) -> dict:
        """The fastest-expanding beam that moved toward the robot since `ref`.
        radial: only a point whose earlier position lies farther out along its
        own line of sight (within RADIAL_DEG) and as far as its closing speed
        says: something coming AT the robot. Without this, wall coming into
        view past an edge as the robot drives reads as a rush (Habitat
        2026-10-01: 843 false rushes in 30 days, median 3.5 m/s)."""
        t_old, old = ref
        old = old[np.isfinite(old[:, 0])]
        if not len(old):
            return {"active": False}
        dt = s.t - t_old
        x, z, _ = s.pose
        idx = np.flatnonzero(np.isfinite(s.ranges))
        keep = ok[idx]
        P, r = pts[keep], s.ranges[idx][keep]
        beams = idx[keep]
        d2 = ((P[:, None, :] - old[None, :, :]) ** 2).sum(-1)
        j = np.argmin(d2, axis=1)
        dmin = np.sqrt(d2[np.arange(len(P)), j])
        r_old = np.hypot(old[j, 0] - x, old[j, 1] - z)
        closing = (r_old - r) / dt
        spacing = np.radians(360.0 / len(s.angles))
        tol = self.MATCH_M + r * spacing * 1.5
        clear = np.maximum(r - s.body[beams], 0.05)
        moving = (dmin > tol) & (closing > MOVING_MS) & (clear < LOOM_MAX_M)
        if radial and moving.any():
            to_old = old[j] - P                                # where the point was, seen from now
            ray = (P - np.array([x, z])) / np.maximum(r[:, None], 1e-6)
            cos = (to_old * ray).sum(1) / np.maximum(dmin, 1e-6)
            moving &= (cos >= np.cos(np.radians(self.RADIAL_DEG))) & (dmin <= 1.5 * closing * dt + tol)
            # ... and nearer on that line of sight than the scan before showed it
            # (projected into this pose): a wall seen at a grazing angle while
            # turning passes the test above, but its range grows
            if moving.any() and len(self._hist) >= 2:
                prev = self._hist[-2][1]
                prev = prev[np.isfinite(prev[:, 0])]
                pred = s.predict_from(prev) if len(prev) else np.full(len(s.angles), np.inf)
                moving &= np.isfinite(pred[beams]) & (r < pred[beams] - self.MATCH_M)
            # ... and not behind a much nearer neighbouring beam: a surface seen
            # edge-on as the robot turns (a box's side) slides along its edge;
            # a foot, a ball, a leg is as near as its neighbours or nearer
            rr = s.ranges
            nb = np.minimum(rr[(beams - 1) % len(rr)], rr[(beams + 1) % len(rr)])
            moving &= r <= nb + self.EDGE_M
        if not moving.any():
            return {"active": False}
        w2 = OBJECT_W_M / 2
        half = np.degrees(np.arctan(w2 / clear))
        exp = np.degrees(w2 * closing / (clear ** 2 + w2 ** 2))
        k = int(np.argmax(np.where(moving, exp, -1.0)))
        return {"active": True, "azimuth_deg": float(s.angles[beams[k]]),
                "half_angle_deg": float(half[k]), "expansion_rate_deg_s": float(exp[k]),
                "range_m": float(r[k]), "clear_m": float(clear[k]), "closing_ms": float(closing[k])}

    def state(self, t_ms: float) -> dict:
        L = self.last
        return {"t_ms": t_ms, "source": "lidar", "active": L["active"],
                "azimuth_deg": L.get("azimuth_deg", 0.0), "elevation_deg": 0.0,
                "half_angle_deg": L.get("half_angle_deg", 0.0),
                "expansion_rate_deg_s": L.get("expansion_rate_deg_s", 0.0)}


class LidarTouch:
    """Session encoder: near obstacles -> antennal / vibrissal bristles by side."""

    # FlyWire: antennal + vibrissal bristles. The MaleCNS types its 69 head
    # bristles only as "BM" (a mix of seven FlyWire bristle types); each is
    # assigned one by its wiring (brain/sensory/bm_subtypes.py) and touch
    # drives the antennal, frontal and inter-ocular ones with BM_Vib. Not the
    # fronto-orbital / orbital ones: they back the male fly up (MDN) and
    # extend its proboscis (MN9), which kept the robot backing and freezing at
    # walls (2026-09-28). FLY_TOUCH_BM=all drives all 69 (the old mapping).
    CELL_TYPES = ("BM_Ant", "BM_Vib", "BM")

    def __init__(self, connectome, scan: LidarScan):
        n = connectome.neurons
        t = n["primary_type"].fillna("").astype(str).to_numpy()
        side = n["side"].fillna("").astype(str).to_numpy()
        use = np.isin(t, self.CELL_TYPES)
        if (t == "BM").any() and os.environ.get("FLY_TOUCH_BM", "subtypes").strip().lower() != "all":
            from brain.sensory.bm_subtypes import TOUCH_SUBTYPES, load
            a = load()
            if a is None:
                import warnings
                warnings.warn("no data/metadata/bm_subtypes_malecns.csv: lidar touch drives all 69 male BM "
                              "cells (python -m brain.sensory.bm_subtypes)")
            else:
                ok = set(a["root_id"][a["subtype"].isin(TOUCH_SUBTYPES)].astype(np.int64))
                rid = n["root_id"].to_numpy().astype(np.int64)
                use &= (t != "BM") | np.isin(rid, list(ok))
        idx = np.flatnonzero(use & np.isin(side, ["left", "right"]))
        self.indices = np.sort(idx)
        self._right = side[self.indices] == "right"
        self.scan = scan
        self.last = {"left": 0.0, "right": 0.0}
        import config
        self.adapt = config.env_flag("FLY_TOUCH_ADAPT")
        self.reset()

    def reset(self) -> None:
        self._a = [0.0, 0.0]                  # adapted contact level, left / right
        self._t = None

    def levels(self) -> tuple:
        s = self.scan
        # once per scan (~10 Hz; the session asks every 1 ms block): update()
        # replaces the arrays, and holding them here keeps their ids unique
        m = getattr(self, "_lv", None)
        if m is not None and m[0] is s.ranges and m[1] is s.body and m[2] is s.angles:
            return m[3]
        near = np.clip(1.0 - s.clearance() / TOUCH_M, 0.0, 1.0)
        near[~np.isfinite(s.ranges)] = 0.0
        a = s.angles
        rear = np.abs(a) >= 180.0 - 1e-6                             # straight behind: both sides
        left = float(np.max(near[(a < FRONT_DEG) | rear], initial=0.0))    # left side, front, rear
        right = float(np.max(near[(a > -FRONT_DEG) | rear], initial=0.0))  # right side, front, rear
        self._lv = (s.ranges, s.body, s.angles, (left, right))
        return left, right

    def rates_hz(self, t_ms: float, stim=None) -> np.ndarray:
        left, right = self.levels()
        self.last = {"left": round(left, 2), "right": round(right, 2)}
        if self.adapt:
            dt = 0.0 if self._t is None or t_ms < self._t else (t_ms - self._t) / 1000.0
            self._t = t_ms
            k = 1.0 - math.exp(-dt / TOUCH_ADAPT_S) if dt > 0 else 0.0
            drive = []
            for i, lv in enumerate((left, right)):
                self._a[i] += (lv - self._a[i]) * k
                drive.append(TOUCH_TONIC * lv + (1.0 - TOUCH_TONIC) * max(lv - self._a[i], 0.0))
            left, right = drive
            self.last.update(drive_left=round(left, 2), drive_right=round(right, 2))
        from brain.sensory.encoders import memo_of
        # the same contact as the last block: the same (read-only) rates
        return memo_of(self).get((left, right), lambda: np.where(self._right, TOUCH_HZ * right, TOUCH_HZ * left))

    def state(self, t_ms: float) -> dict:
        return {"kind": "lidar_touch", "active": max(self.last.values()) > 0, **self.last}
