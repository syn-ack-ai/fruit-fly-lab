"""
The robot's speed limit near people: a body-level safety layer, below the fly
brain and the neocortex, as on any robot that shares a floor with people.

The fly brain chooses where to walk and how fast; this layer only lowers the
forward speed as the person gets close, so that when the pet reaches someone it
arrives at a crawl (a rub against the legs, not a collision). Backing away and
turning are never limited. The person's distance comes from the head camera
(apparent size of head and shoulders), the same estimate the neocortex uses;
the last estimate is held briefly when the person drops out of view.

C. APPROXIMATIONS: not a fly structure. Numbers are engineering choices for a
small indoor robot: 0.08 m/s inside 0.5 m (person's head-and-shoulders distance
from the camera), full speed from 2 m. Contact speed depends on the body: in
Habitat (Spot-sized robot) contact happens at 0.6-0.8 m between centres, where
the limit is 0.1-0.16 m/s; set CONTACT_M for the real robot's size.
"""
from __future__ import annotations

import math

import numpy as np

TARGET_HALF_WIDTH_M = 0.25      # head and shoulders (robot/head.py target)
HEAD_ABOVE_CAM_M = 1.10         # a standing person's head above the pet's camera
CONTACT_M = 0.5                 # at or inside this: crawl
FREE_M = 2.0                    # from here: no limit
V_CONTACT = 0.08                # m/s
V_FREE = 0.5                    # m/s (brain_client.V_MAX)
HOLD_S = 2.0                    # keep the last distance this long out of view


def person_distance(half_deg: float) -> float | None:
    """Floor distance to a person from the apparent half-width of their head
    and shoulders (deg), or None if too small to tell."""
    if half_deg <= 0.1:
        return None
    rng = TARGET_HALF_WIDTH_M / math.tan(math.radians(half_deg))
    return math.sqrt(max(rng ** 2 - HEAD_ABOVE_CAM_M ** 2, 0.0))


def speed_limit(d_m: float | None) -> float:
    if d_m is None:
        return V_FREE
    f = (d_m - CONTACT_M) / (FREE_M - CONTACT_M)
    return V_CONTACT + (V_FREE - V_CONTACT) * min(1.0, max(0.0, f))


class ProximityGovernor:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.d, self.t_seen = None, -1e9
        self.limited_s = 0.0

    def observe(self, visible: bool, half_deg: float, t_s: float) -> None:
        if visible:
            d = person_distance(half_deg)
            if d is not None:
                self.d, self.t_seen = d, t_s

    def limit(self, v: float, t_s: float, dt: float = 0.0) -> float:
        d = self.d if t_s - self.t_seen <= HOLD_S else None
        vmax = speed_limit(d)
        if v > vmax:
            self.limited_s += dt
            return vmax
        return v


# ------------------------------------------------------------ obstacles (lidar)
OBST_STOP_M = 0.08              # clearance ahead at which forward motion stops
OBST_FREE_M = 0.60              # from here: no limit
OBST_CONE_DEG = 35.0            # "ahead" = beams within this of the heading
OBST_V_MIN = 0.0


def obstacle_limit(clearance_m, angles_deg, v: float, v_free: float = V_FREE) -> float:
    """Forward speed allowed by the lidar's clearance ahead (m from the body
    surface, per beam). Slows linearly from OBST_FREE_M and stops at
    OBST_STOP_M; backing up and turning are never limited. The robot's own
    obstacle reflex (on the rover: its safety layer / ESP32), not the brain's:
    in this connectome antennal touch slows walking but does not steer around
    things (experiments/antenna_touch_test.py)."""
    if v <= 0:
        return v
    a = ((np.asarray(angles_deg, float) + 180.0) % 360.0) - 180.0
    ahead = np.abs(a) <= OBST_CONE_DEG
    c = np.asarray(clearance_m, float)[ahead]
    if c.size == 0:
        return v
    near = float(c.min())
    f = (near - OBST_STOP_M) / (OBST_FREE_M - OBST_STOP_M)
    return min(v, OBST_V_MIN + (v_free - OBST_V_MIN) * min(1.0, max(0.0, f)))


# ------------------------------------------- footprint time-to-collision (lidar)
# The Nav2 Collision Monitor "approach" idea (the rover will run the real Nav2
# node): project the commanded (v, w) forward TTC_S seconds and sweep the
# robot's footprint along that arc against the lidar points. Rule: never get
# closer than CLEAR_M to anything; if already closer (e.g. brushing a wall),
# never get closer than now -- so backing or turning away is always allowed.
# (The first version stopped whenever anything was inside the footprint, and a
# long body in a cluttered house then froze in place.)
TTC_S = 1.0
TTC_STEPS = 10
CLEAR_M = 0.05


def footprint_clearance(points_robot, half_len: float, half_wid: float) -> float:
    """Smallest distance from a rectangular footprint (robot frame: x forward,
    y left) to any point; negative inside."""
    P = np.asarray(points_robot, float).reshape(-1, 2)
    if len(P) == 0:
        return np.inf
    dx, dy = np.abs(P[:, 0]) - half_len, np.abs(P[:, 1]) - half_wid
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    return float(np.min(np.where((dx > 0) | (dy > 0), outside, inside)))


def _future(P, v, w, t):
    th = w * t
    if abs(w) > 1e-6:
        dx, dy = v / w * np.sin(th), v / w * (1 - np.cos(th))
    else:
        dx, dy = v * t, 0.0
    c, s = np.cos(th), np.sin(th)
    qx, qy = P[:, 0] - dx, P[:, 1] - dy
    return np.stack([c * qx + s * qy, -s * qx + c * qy], axis=1)


def motion_safe(points_robot, v: float, w: float, half_len: float, half_wid: float,
                ttc_s: float = TTC_S) -> bool:
    P = np.asarray(points_robot, float).reshape(-1, 2)
    if len(P) == 0 or (v == 0 and w == 0):
        return True
    floor = min(footprint_clearance(P, half_len, half_wid), CLEAR_M) - 1e-3
    for t in np.linspace(ttc_s / TTC_STEPS, ttc_s, TTC_STEPS):
        if footprint_clearance(_future(P, v, w, t), half_len, half_wid) < floor:
            return False
    return True


def ttc_filter(points_robot, v: float, w: float, half_len: float, half_wid: float) -> tuple:
    """The brain's command, slowed as little as needed to be safe; else a turn
    in place if that is safe; else stop."""
    for s in (1.0, 0.75, 0.5, 0.25):
        if motion_safe(points_robot, s * v, s * w, half_len, half_wid):
            return s * v, s * w
    for s in (1.0, 0.5):
        if motion_safe(points_robot, 0.0, s * w, half_len, half_wid):
            return 0.0, s * w
    return 0.0, 0.0


def ttc_scale(points_robot, v: float, w: float, half_len: float, half_wid: float,
              ttc_s: float = TTC_S) -> float:
    """Largest of 1, .75, .5, .25, 0 such that (s*v, s*w) is safe (tests / reporting)."""
    for s in (1.0, 0.75, 0.5, 0.25):
        if motion_safe(points_robot, s * v, s * w, half_len, half_wid, ttc_s):
            return s
    return 0.0


def scan_points_robot(ranges, angles_deg) -> np.ndarray:
    """Lidar hits in the robot frame (x forward, y left); angles + = right."""
    r = np.asarray(ranges, float)
    a = np.radians(np.asarray(angles_deg, float))
    ok = np.isfinite(r)
    return np.stack([r[ok] * np.cos(a[ok]), -r[ok] * np.sin(a[ok])], axis=1)


class CollisionMonitor:
    """ttc_filter with a recovery reflex (as Nav2's recovery behaviours / a
    robot vacuum): if the safety layer has held the robot still for STUCK_S
    while the brain wants to move, it backs up a little or turns the other way,
    whichever is safe, for RECOVER_S. A long body next to a wall cannot turn in
    place (its corners would swing into the wall) and the fly brain rarely
    walks backwards, so without this it stays stuck."""

    STUCK_S, RECOVER_S = 1.5, 1.2
    RECOVERIES = ((-0.12, 0.0), (0.0, 0.8), (0.0, -0.8), (-0.1, 0.6), (-0.1, -0.6))

    def __init__(self, half_len: float, half_wid: float):
        self.L, self.W = half_len, half_wid
        self.stuck_s, self.recover_until, self.t = 0.0, -1.0, 0.0
        self.recoveries = 0
        self._rec = (0.0, 0.0)

    def reset(self) -> None:
        """A new day: no recovery in progress, and the count restarts (it is
        reported per day; review 2026-09-27: it used to run on across days)."""
        self.stuck_s, self.recover_until, self.t = 0.0, -1.0, 0.0
        self.recoveries = 0

    def __call__(self, points_robot, v: float, w: float, dt: float) -> tuple:
        self.t += dt
        if self.t < self.recover_until and motion_safe(points_robot, *self._rec, self.L, self.W):
            return self._rec
        fv, fw = ttc_filter(points_robot, v, w, self.L, self.W)
        wants = abs(v) > 0.02 or abs(w) > 0.1
        held = abs(fv) < 0.01 and abs(fw) < 0.05
        self.stuck_s = self.stuck_s + dt if (wants and held) else 0.0
        if self.stuck_s >= self.STUCK_S:
            self.stuck_s = 0.0
            for rv, rw in self.RECOVERIES:
                if motion_safe(points_robot, rv, rw, self.L, self.W):
                    self._rec, self.recover_until = (rv, rw), self.t + self.RECOVER_S
                    self.recoveries += 1
                    return self._rec
        return fv, fw
