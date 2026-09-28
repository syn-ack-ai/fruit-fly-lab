"""Steering toward open space (robot/avoid.py)."""
import math

import numpy as np

from robot.avoid import Avoid, corridor_free
from robot.safety import scan_points_robot

HL, HW = 0.127, 0.116          # the rover (sim/habitat_bridge/bodies.py)


def _box_points(x0, x1, y0, y1, step=0.02):
    """Outline of an axis-aligned box in the robot frame (x forward, y left)."""
    xs, ys = np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step)
    return np.concatenate([np.stack([xs, np.full_like(xs, y0)], 1), np.stack([xs, np.full_like(xs, y1)], 1),
                           np.stack([np.full_like(ys, x0), ys], 1), np.stack([np.full_like(ys, x1), ys], 1)])


def test_corridor_free_distance():
    P = np.array([[1.0, 0.0], [0.0, 0.6], [-0.5, 0.0], [0.5, 0.5]])  # ahead, left, behind, ahead-left
    f = corridor_free(P, [0.0, 90.0, -90.0, 180.0, -45.0], HL, HW)
    assert abs(f[0] - (1.0 - HL)) < 1e-9                         # the point straight ahead
    assert np.isinf(f[1])                                        # to the right: nothing
    assert abs(f[2] - (0.6 - HL)) < 1e-9                         # to the left: the point at y = 0.6
    assert abs(f[3] - (0.5 - HL)) < 1e-9                         # behind
    assert abs(f[4] - (math.hypot(0.5, 0.5) - HL)) < 1e-9        # 45 deg left: the diagonal point


def test_clear_path_is_left_alone():
    a = Avoid(HL, HW)
    P = _box_points(2.0, 2.5, -1.0, 1.0)                         # far ahead
    az, u = a.choose(P, 0.0, v=0.3)
    assert az == 0.0 and u == 0.0 and not a.last["active"]


def test_obstacle_ahead_bends_toward_the_open_side():
    a = Avoid(HL, HW)
    P = _box_points(0.5, 0.8, -0.6, 0.15)                        # box ahead, reaching further right
    az, u = a.choose(P, 0.0, v=0.3)
    assert u > 0 and az < 0                                      # steer left (az + = right)
    assert np.isinf(corridor_free(P, [az], HL, HW)[0]) or corridor_free(P, [az], HL, HW)[0] > 0.8


def test_keeps_its_side_rather_than_dithering():
    a = Avoid(HL, HW)
    P = _box_points(0.5, 0.8, -0.3, 0.3)                         # symmetric box
    az1, _ = a.choose(P, 0.0, v=0.3)
    P2 = _box_points(0.5, 0.8, -0.32, 0.28)                      # nudged so the other side is marginally better
    az2, _ = a.choose(P2, 0.0, v=0.3)
    assert np.sign(az1) == np.sign(az2)


def test_goal_is_bent_but_target_person_is_not_an_obstacle():
    a = Avoid(HL, HW)
    person = np.array([[0.9, y] for y in np.linspace(-0.12, 0.12, 7)])   # legs 0.9 m ahead
    cmd = {"goal_deg": 90.0, "goal_gain": 1.0}
    out = a.adjust(cmd, heading_deg=90.0, points_robot=person, v=0.3, target_dist=0.9)
    assert out["goal_deg"] == 90.0                               # walking up to the person is allowed
    out = a.adjust(cmd, heading_deg=90.0, points_robot=person, v=0.3)
    assert out["goal_deg"] != 90.0                               # an unknown thing there: go around


def test_wandering_gets_a_pull_only_when_needed():
    a = Avoid(HL, HW)
    assert a.adjust({}, 0.0, _box_points(3.0, 3.5, -1, 1), v=0.3) == {}
    out = a.adjust({}, 0.0, _box_points(0.4, 0.7, -0.5, 0.1), v=0.3)
    assert out["goal_gain"] > 0 and out["attend_az"] < 0         # pulled to the left, the open side
    assert abs(((out["goal_deg"] - 0.0 + 180) % 360) - 180 + out["attend_az"]) < 1e-9


def test_room_walk_with_steering_does_not_touch():
    """A kinematic robot in a room with a table, always wanting to go straight,
    steered only by the chosen azimuth: it must not touch anything."""
    from robot.lidar import beam_angles
    segs = [((-2, -2), (2, -2)), ((2, -2), (2, 2)), ((2, 2), (-2, 2)), ((-2, 2), (-2, -2)),
            ((0.3, -0.3), (0.9, -0.3)), ((0.9, -0.3), (0.9, 0.3)), ((0.9, 0.3), (0.3, 0.3)), ((0.3, 0.3), (0.3, -0.3))]
    ang = beam_angles(90)

    def scan(x, y, th):
        out = []
        for a_ in ang:
            b = th - math.radians(a_)
            bx, by, best = math.cos(b), math.sin(b), 8.0
            for (ax, ay), (cx, cy) in segs:
                ex, ey = cx - ax, cy - ay
                den = bx * ey - by * ex
                if abs(den) < 1e-12:
                    continue
                t = ((ax - x) * ey - (ay - y) * ex) / den
                u = ((ax - x) * by - (ay - y) * bx) / den
                if t > 1e-6 and 0 <= u <= 1:
                    best = min(best, t)
            out.append(best)
        return np.array(out)

    def min_clear(x, y, th):
        c, s = math.cos(th), math.sin(th)
        best = np.inf
        for (ax, ay), (cx, cy) in segs:
            for k in np.linspace(0, 1, 60):
                px, py = ax + k * (cx - ax) - x, ay + k * (cy - ay) - y
                lx, ly = c * px + s * py, -s * px + c * py
                dx, dy = abs(lx) - HL, abs(ly) - HW
                best = min(best, math.hypot(max(dx, 0), max(dy, 0)) if (dx > 0 or dy > 0) else max(dx, dy))
        return best

    a = Avoid(HL, HW)
    x, y, th, v, dt = -1.2, 0.05, 0.0, 0.3, 0.1
    worst = np.inf
    for _ in range(300):
        pts = scan_points_robot(scan(x, y, th), ang)
        az, u = a.choose(pts, 0.0, v)
        w = max(-2.0, min(2.0, -math.radians(az) * 2.0))           # a simple steering response
        vv = v * max(0.2, math.cos(math.radians(az)))
        th += w * dt
        x, y = x + vv * math.cos(th) * dt, y + vv * math.sin(th) * dt
        worst = min(worst, min_clear(x, y, th))
    assert worst > 0.0, worst


def test_facing_a_wall_with_the_goal_behind_turns_firmly():
    # smoke test 2026-09-27: goal behind (explore), wall ahead; the goal corridor
    # was free so the reflex did nothing and the brakes held it for 8 s
    a = Avoid(HL, HW)
    wall = np.array([[0.2, y] for y in np.linspace(-1.0, 1.0, 41)])
    cmd = {"goal_deg": 180.0, "goal_gain": 0.39}                 # heading 0: the goal is straight behind
    out = a.adjust(cmd, heading_deg=0.0, points_robot=wall, v=0.3)
    assert a.last["active"] and a.last["urgency"] > 0.5
    assert out["goal_gain"] > 0.5 and out["attend_gain"] > 0.5 and abs(out["attend_az"]) >= 90
    # a goal straight ahead with a free way: untouched
    assert a.adjust({"goal_deg": 0.0, "goal_gain": 0.4}, 0.0, wall + [3.0, 0.0], v=0.3) == {"goal_deg": 0.0, "goal_gain": 0.4}


def test_pivot_only_when_pinned():
    from robot.avoid import PIVOT_W, pivot
    last = {"active": True, "chosen": -90.0}                     # open side: left
    assert pivot(last, v_brain=0.3, v=0.0, w=0.2) == (PIVOT_W, True)      # pinned: turn left (CCW)
    assert pivot(last, v_brain=0.3, v=0.2, w=0.2) == (0.2, False)         # still moving: the brain steers
    assert pivot(last, v_brain=0.0, v=0.0, w=0.2) == (0.2, False)         # not trying to walk: leave it
    assert pivot({"active": False}, 0.3, 0.0, 0.2) == (0.2, False)
    assert pivot({"active": True, "chosen": 60.0}, 0.3, 0.0, 0.0) == (-PIVOT_W, True)  # open side right


def test_orienting_pull_is_checked_in_its_own_direction():
    """The neocortex's orienting reflex pulls toward a person (attend_az, + =
    right): avoidance must check THAT corridor, not straight ahead."""
    a = Avoid(HL, HW)
    right_box = _box_points(0.25, 0.6, -0.6, -0.15)          # an obstacle ahead-right, the way to the person
    out = a.adjust({"attend_az": 45.0, "attend_gain": 1.0}, 0.0, right_box, v=0.3)
    assert out["attend_az"] != 45.0                           # bent ...
    f = corridor_free(right_box, [45.0, out["attend_az"]], HL, HW)     # (+ = right, as attend_az)
    assert f[1] > f[0]                                        # ... to a clearer corridor
    a = Avoid(HL, HW)
    clear = a.adjust({"attend_az": 45.0, "attend_gain": 1.0}, 0.0, _box_points(3.0, 3.5, 2.0, 2.5), v=0.3)
    assert clear == {"attend_az": 45.0, "attend_gain": 1.0}   # nothing in the way: left alone



def test_orienting_pull_leaves_the_goal_to_its_own_check():
    """Second review 2026-09-27: with a goal AND the orienting reflex, the goal
    must not be rewritten to the person's corridor."""
    a = Avoid(HL, HW)
    right_box = _box_points(0.25, 0.6, -0.6, -0.15)          # blocks the way to the person (right)
    cmd = {"goal_deg": 180.0, "goal_gain": 0.8, "attend_az": 45.0, "attend_gain": 1.0}   # goal: to the left
    out = a.adjust(cmd, heading_deg=90.0, points_robot=right_box, v=0.3)
    assert out["goal_deg"] == 180.0                           # the left goal's corridor is clear: untouched
    assert out["attend_az"] != 45.0                           # the pull toward the person is bent
