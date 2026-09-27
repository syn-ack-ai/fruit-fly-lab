"""
2D lidar as fly senses (the Waveshare UGV Rover's lidar; in Habitat, a
simulated scan: sim/habitat_bridge/habitat_server.py --lidar).

A 2D lidar gives distances in one horizontal plane all around the robot. It is
not vision; it feeds two of the fly's senses:

  looming   a gap closing fast in any direction -> the looming detectors
            LC4 / LPLC2 (the same neurons the camera drives, robot/head.py):
            the beam with the fastest growing apparent size of an obstacle of
            nominal width OBJECT_W_M, if it is within LOOM_MAX_M (LidarLooming,
            a stimulus for brain.sensory.encoders.LoomingEncoder)
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

import numpy as np

OBJECT_W_M = 0.3            # nominal obstacle width for the looming angle
LOOM_MAX_M = 2.0            # beyond this, nothing looms
TOUCH_M = 0.15              # "antenna" reach beyond the body surface
TOUCH_HZ = 80.0             # as petting (sim/habitat_bridge/home.py)
FRONT_DEG = 30.0            # beams within this of straight ahead touch both sides


def beam_angles(beams: int) -> list:
    """Beam angles (deg, + = right), as habitat_server.set_lidar."""
    return [(-180.0 + 360.0 * k / beams) for k in range(beams)]


def ellipse_body(angles_deg, half_length_m: float, half_width_m: float) -> np.ndarray:
    """Distance from the lidar (body centre) to the body surface, per beam."""
    a = np.radians(np.asarray(angles_deg, float))
    return 1.0 / np.sqrt((np.cos(a) / half_length_m) ** 2 + (np.sin(a) / half_width_m) ** 2)


class LidarScan:
    """The latest scan: ranges (m) at beam angles (deg, + = right); body = the
    body surface's distance from the lidar (scalar or per beam)."""

    def __init__(self, angles_deg, body_radius_m):
        self.angles = np.asarray(angles_deg, float)
        self.body = np.broadcast_to(np.asarray(body_radius_m, float), self.angles.shape).copy()
        self.ranges = np.full(len(self.angles), np.inf)
        self.t = None

    def update(self, ranges, t_s: float) -> None:
        r = np.asarray(ranges, float)
        self.ranges = np.where(np.isfinite(r) & (r > 0), r, np.inf)
        self.t = t_s

    def clearance(self) -> np.ndarray:
        """Distance from the body surface to the nearest thing, per beam."""
        return np.maximum(self.ranges - self.body, 0.0)


class LidarLooming:
    """The most threatening beam as a looming stimulus (LoomingEncoder)."""

    def __init__(self, scan: LidarScan):
        self.scan = scan
        self._half = None
        self._t = None
        self._exp = np.zeros(len(scan.angles))
        self.last = {"active": False}

    def update(self) -> None:
        s = self.scan
        clear = np.maximum(s.clearance(), 0.05)
        half = np.degrees(np.arctan((OBJECT_W_M / 2) / clear))
        half[~np.isfinite(s.ranges)] = 0.0
        if self._half is not None and s.t is not None and s.t > self._t:
            exp = (half - self._half) / (s.t - self._t)
            self._exp = 0.5 * self._exp + 0.5 * exp
        self._half, self._t = half, s.t
        cand = np.where((clear < LOOM_MAX_M) & (self._exp > 0), self._exp, 0.0)
        i = int(np.argmax(cand))
        on = cand[i] > 0
        self.last = {"active": bool(on), "azimuth_deg": float(s.angles[i]),
                     "half_angle_deg": float(half[i]) if on else 0.0,
                     "expansion_rate_deg_s": float(self._exp[i]) if on else 0.0,
                     "range_m": float(s.ranges[i])}

    def state(self, t_ms: float) -> dict:
        L = self.last
        return {"t_ms": t_ms, "source": "lidar", "active": L["active"],
                "azimuth_deg": L.get("azimuth_deg", 0.0), "elevation_deg": 0.0,
                "half_angle_deg": L.get("half_angle_deg", 0.0),
                "expansion_rate_deg_s": L.get("expansion_rate_deg_s", 0.0)}


class LidarTouch:
    """Session encoder: near obstacles -> antennal / vibrissal bristles by side."""

    CELL_TYPES = ("BM_Ant", "BM_Vib")

    def __init__(self, connectome, scan: LidarScan):
        n = connectome.neurons
        t = n["primary_type"].fillna("").astype(str).to_numpy()
        side = n["side"].fillna("").astype(str).to_numpy()
        idx = np.flatnonzero(np.isin(t, self.CELL_TYPES) & np.isin(side, ["left", "right"]))
        self.indices = np.sort(idx)
        self._right = side[self.indices] == "right"
        self.scan = scan
        self.last = {"left": 0.0, "right": 0.0}

    def levels(self) -> tuple:
        s = self.scan
        near = np.clip(1.0 - s.clearance() / TOUCH_M, 0.0, 1.0)
        near[~np.isfinite(s.ranges)] = 0.0
        a = s.angles
        left = float(np.max(near[(a < FRONT_DEG)], initial=0.0))    # left side and front
        right = float(np.max(near[(a > -FRONT_DEG)], initial=0.0))  # right side and front
        return left, right

    def rates_hz(self, t_ms: float, stim=None) -> np.ndarray:
        left, right = self.levels()
        self.last = {"left": round(left, 2), "right": round(right, 2)}
        return np.where(self._right, TOUCH_HZ * right, TOUCH_HZ * left)

    def state(self, t_ms: float) -> dict:
        return {"kind": "lidar_touch", "active": max(self.last.values()) > 0, **self.last}
