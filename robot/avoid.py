"""
Steering around obstacles before reaching them, as animals do (the robot's
safety brakes, robot/safety.py, then become a seatbelt that rarely fires).

Animals do not walk up to a wall and stop: they steer toward open space while
moving. Insects balance optic flow on both eyes and so keep to the middle of
gaps (the "centering response", Srinivasan et al. 1991 Biol Cybern 65:33);
cats plan steps around obstacles with posterior parietal cortex (Drew &
Marigold 2015 Curr Opin Neurobiol 33:25). Here the 2D lidar stands in for that
vision: every control step,

  corridor   for a candidate heading, the free distance along a strip as
             wide as the body (+ MARGIN_M) in that direction
  desired    where the pet wants to go (the neocortex's goal, else straight ahead)
  chosen     the heading nearest the desired one whose corridor is free for
             LOOKAHEAD (longer at speed); if none is, the most open one.
             Keeping to the side chosen last step (SIDE_KEEP_DEG) stops it
             dithering left/right in front of an obstacle.
  urgency    0..1, how blocked the desired corridor is -- or the corridor
             straight ahead of the body, if that is worse: a pet facing a wall
             while its goal lies behind it must turn firmly, not wiggle (the
             first smoke test: the brakes held it 8 s like that)

and the result goes into the fly brain through the same steering channels the
neocortex uses (cortex/topdown.py): the goal direction (FC2 -> PFL3 -> DNa02)
and the pursuit pathway's phantom object (LC10a). With a goal, the goal is
bent around the obstacle; with none, the fly walks where it likes until
something is ahead, and is then pulled toward the open side with the urgency
as gain. Nothing here sets a motor command.

Pivot (body reflex): pinned at an obstacle (the safety brake stopped it) while
still walking forward, the fly brain's turn toward open space is weak and
flips side to side (smoke test 2026-09-27: 4 s of wiggling at a wall with the
phantom object at 90 deg). In a real fly the nerve cord's local leg reflexes
take part in such turns. The pet's body is driven from the brain's descending
steering neurons; the default brain (the complete male CNS) has a nerve cord,
but its leg-level turning readout is not yet usable (see
results/complete_brain_2026-09-27), so the body turns in place toward the
side chosen here (pivot()), at PIVOT_W, until the way ahead opens. Counted, so it can be reported.

C. APPROXIMATIONS: not a fly circuit. The brain-only connectome's own
obstacle responses (looming escape, antennal touch slowing) do not steer
around things (experiments/antenna_touch_test.py), so this plays the part of
the missing visual obstacle avoidance. The corridor is a straight strip
(vector-field-histogram-like, Borenstein & Koren 1991), not the swept arc.
"""
from __future__ import annotations

import math

import numpy as np

MARGIN_M = 0.06               # beyond the body's half width, each side
LOOKAHEAD_M = 0.45            # corridor length at rest ...
LOOKAHEAD_S = 1.2             # ... plus this many seconds of travel
CANDIDATES_DEG = np.arange(-150.0, 150.1, 5.0)
SIDE_KEEP_DEG = 20.0          # extra cost for switching sides
TARGET_STOP_M = 0.35          # a target (the person) this close ends its corridor
PIVOT_W = 0.9                 # rad/s: the body's turn in place when pinned
PINNED_V = 0.02               # m/s: "stopped" by the brake


def corridor_free(points_robot, az_deg, half_len: float, half_wid: float, margin: float = MARGIN_M):
    """Free distance (m, from the body's front edge) along heading az (deg,
    + = right) for each az; inf if nothing is in the strip. points: robot
    frame, x forward, y left (robot.safety.scan_points_robot)."""
    P = np.asarray(points_robot, float).reshape(-1, 2)
    az = np.radians(np.atleast_1d(np.asarray(az_deg, float)))
    if len(P) == 0:
        return np.full(len(az), np.inf)
    dx, dy = np.cos(az), -np.sin(az)                       # unit heading, robot frame
    along = P[:, 0, None] * dx[None] + P[:, 1, None] * dy[None]
    lat = np.abs(P[:, 0, None] * dy[None] - P[:, 1, None] * dx[None])
    blocked = (along > 0) & (lat < half_wid + margin)
    d = np.where(blocked, along - half_len, np.inf)
    return np.maximum(d.min(axis=0), 0.0)


class Avoid:
    def __init__(self, half_len: float, half_wid: float):
        self.L, self.W = half_len, half_wid
        self.reset()

    def reset(self) -> None:
        self.side = 0.0               # -1 left, +1 right, 0 none yet
        self.last = {"active": False}
        self.steps = self.active_steps = 0
        self._attend = None           # the attention check's own side memory (per day too)

    def choose(self, points_robot, desired_az: float, v: float = 0.0,
               target_dist: float | None = None) -> tuple:
        """(chosen az deg, urgency 0..1) for a desired heading (deg, + = right)."""
        look = LOOKAHEAD_M + LOOKAHEAD_S * max(v, 0.0)
        look_des = look if target_dist is None else min(look, max(target_dist - TARGET_STOP_M, 0.0))
        self.steps += 1
        fwd = corridor_free(points_robot, [desired_az, 0.0], self.L, self.W)
        free_des = float(fwd[0])
        u_des = float(np.clip(1.0 - free_des / look_des, 0.0, 1.0)) if look_des > 0 else 0.0
        # straight ahead of the body (only when heading away from the target)
        u_fwd = float(np.clip(1.0 - float(fwd[1]) / look, 0.0, 1.0)) if abs(desired_az) > 10.0 else 0.0
        urgency = max(u_des, u_fwd)
        if urgency <= 0.0:
            self.side = 0.0
            self.last = {"active": False, "desired": round(desired_az, 1), "chosen": round(desired_az, 1), "urgency": 0.0}
            return desired_az, 0.0
        free = corridor_free(points_robot, CANDIDATES_DEG, self.L, self.W)
        cost = np.abs(CANDIDATES_DEG - desired_az)
        if self.side:
            cost = cost + SIDE_KEEP_DEG * (np.sign(CANDIDATES_DEG - desired_az) == -self.side)
        ok = free >= (look_des if u_des > 0 else min(free_des, look))
        if ok.any():
            k = int(np.argmin(np.where(ok, cost, np.inf)))
        else:                                                  # nothing free: the most open way
            k = int(np.argmax(free - 1e-3 * cost))
        chosen = float(CANDIDATES_DEG[k])
        if chosen != desired_az:
            self.side = float(np.sign(chosen - desired_az))
        self.active_steps += 1
        self.last = {"active": True, "desired": round(desired_az, 1), "chosen": chosen,
                     "urgency": round(urgency, 2), "free_m": round(min(free_des, 9.0), 2),
                     "ahead_m": round(min(float(fwd[1]), 9.0), 2)}
        return chosen, urgency

    def adjust(self, cmd: dict, heading_deg: float, points_robot, v: float,
               target_dist: float | None = None, gain: float = 1.0,
               attend_dist: float | None = None) -> dict:
        """The neocortex's top-down command, bent around obstacles (world
        goal_deg is CCW; azimuths + = right, cortex.topdown.goal_azimuth).
        target_dist: the goal is a person this far away (not an obstacle);
        attend_dist: the same for an explicit attention direction."""
        from cortex.topdown import goal_azimuth
        if cmd.get("attend_explicit") and cmd.get("attend_az") is not None and cmd.get("attend_gain", 0.0) > 0:
            # an explicit attention direction (the neocortex's orienting reflex)
            # pulls the body toward the person: the goal keeps its own corridor
            # check (below) and the pull gets its own, with its own chooser
            # state (2026-09-27: checking only the goal / straight ahead tripled
            # furniture bumps with the reflex on; a shared check then rewrote
            # the goal -- second review)
            base = {k: v_ for k, v_ in cmd.items() if k not in ("attend_az", "attend_gain", "attend_explicit")}
            out = self.adjust(base, heading_deg, points_robot, v, target_dist, gain)
            if getattr(self, "_attend", None) is None:
                self._attend = Avoid(self.L, self.W)
            az, u = self._attend.choose(points_robot, float(cmd["attend_az"]), v, attend_dist)
            out["attend_az"] = az
            out["attend_gain"] = max(float(cmd["attend_gain"]), gain * u) if u > 0 else float(cmd["attend_gain"])
            out["attend_explicit"] = True
            return out
        out = {k: v_ for k, v_ in cmd.items() if k != "attend_explicit"}   # (a bend below is derived)
        g = cmd.get("goal_deg")
        if g is not None and cmd.get("goal_gain", 0.0) > 0:
            az, u = self.choose(points_robot, goal_azimuth(g, heading_deg), v, target_dist)
            if u > 0:                                  # bend the goal; turn at least as firmly as it is urgent
                gg = max(float(cmd["goal_gain"]), gain * u)
                out.update(goal_deg=(heading_deg - az) % 360.0, goal_gain=gg, attend_az=az,
                           attend_gain=max(float(cmd.get("attend_gain", cmd["goal_gain"])), gain * u))
        else:
            az, u = self.choose(points_robot, 0.0, v)
            if u > 0:                                  # wandering: a pull to the open side only when needed
                out.update(goal_deg=(heading_deg - az) % 360.0, goal_gain=gain * u,
                           attend_az=az, attend_gain=gain * u)
        return out


def pivot(avoid_last: dict, v_brain: float, v: float, w: float) -> tuple:
    """(w, pivoted): the body reflex above. v_brain = the forward speed asked
    for before the lidar brake, v = after it; w + = left (CCW), chosen
    azimuth + = right."""
    if not avoid_last.get("active") or v_brain <= 0.05 or v > PINNED_V:
        return w, False
    side = -np.sign(avoid_last.get("chosen", 0.0))        # chosen to the right -> turn clockwise (w < 0)
    if side == 0:
        return w, False
    if np.sign(w) == side and abs(w) >= PIVOT_W:
        return w, True                                     # already turning that way, fast enough
    return float(side * PIVOT_W), True


# the unstick reflex (2026-09-28): the robot sat within touch distance of a wall
# or furniture, barely moving, for 26% of all simulated time on the held-out
# runs (19% without the motor lag): at a wall the fly brain alternates backing
# away from the touch (head bristles -> MDN) and walking back toward its goal,
# and the motor lag averages that into nothing; pivot() never engaged, as the
# brain was not asking to go forward. When stuck, the robot layer points the
# goal and attention to the most open direction for UNSTICK_S -- the fly brain
# then turns itself out (as the orienting reflex does, it biases, never drives).
STUCK_NEAR_M = 0.15           # an obstacle this close to the body (lidar "touch")
STUCK_WINDOW_S = 3.0          # ... for this long ...
STUCK_MOVE_M = 0.10           # ... while the body moved less than this
UNSTICK_S = 2.5               # then pull toward the most open direction this long
UNSTICK_COOLDOWN_S = 2.0      # before it may trigger again
OPEN_DEG = np.arange(-180.0, 180.0, 10.0)


class Unstick:
    """Detects being stuck near an obstacle and, while active, returns the
    WORLD direction (goal_deg convention: CCW, as the neocortex's goals) of the
    most open corridor found when it triggered. (Review 2026-09-28: a target
    kept relative to the body turned with it, so the robot turned on past the
    open way -- up to 130 deg.)"""

    def __init__(self, half_len: float, half_wid: float):
        self.L, self.W = half_len, half_wid
        self.reset()

    def reset(self) -> None:
        self.hist = []                 # (t, x, z, near)
        self.until = self.cool = -1e9
        self.goal = None
        self.events = 0
        self.active_s = 0.0
        self.pulling = False           # set by the caller each step (logging)

    def cancel(self) -> None:
        """Stop a running pull (the robot rests, docks or yields) and forget
        the window."""
        self.until, self.hist = -1e9, []

    def step(self, t: float, xz, near_m: float, points_robot, dt: float, heading_deg: float,
             may_trigger: bool = True) -> float | None:
        """t (s), body position (m), nearest obstacle distance from the body
        surface (m), lidar points in the robot frame, heading (deg, the
        habitat yaw, CCW). may_trigger: False while the robot means to stand
        still (charging, mid-meal, yielding, beside its person). Returns the
        world goal direction to pull toward while unsticking, else None."""
        if not may_trigger:
            # standing still on purpose: stop any pull and forget the window,
            # so it does not fire the moment the stillness ends (review 2026-09-28)
            self.cancel()
            return None
        self.hist.append((t, float(xz[0]), float(xz[1]), near_m < STUCK_NEAR_M))
        while self.hist and self.hist[0][0] < t - STUCK_WINDOW_S:
            self.hist.pop(0)
        if t < self.until:
            self.active_s += dt
            return self.goal
        if t < self.cool or len(self.hist) < 2 or self.hist[-1][0] - self.hist[0][0] < STUCK_WINDOW_S - 0.15:
            return None
        if not all(h[3] for h in self.hist):
            return None
        xs = np.array([h[1] for h in self.hist]); zs = np.array([h[2] for h in self.hist])
        if math.hypot(xs.max() - xs.min(), zs.max() - zs.min()) >= STUCK_MOVE_M:
            return None
        free = corridor_free(points_robot, OPEN_DEG, self.L, self.W)
        free = np.where(np.isinf(free), 99.0, free)
        # the most open way; ties prefer smaller turns
        k = int(np.argmax(free - 1e-3 * np.abs(OPEN_DEG)))
        self.goal = (float(heading_deg) - float(OPEN_DEG[k])) % 360.0     # az + = right -> world CCW
        self.until, self.cool = t + UNSTICK_S, t + UNSTICK_S + UNSTICK_COOLDOWN_S
        self.events += 1
        self.active_s += dt                # the trigger step pulls too
        self.hist = []
        return self.goal
