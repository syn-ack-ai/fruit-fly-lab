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
