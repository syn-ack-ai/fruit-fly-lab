"""The neocortex's obstacle map and routes (cortex/obstacle_map.py)."""
import math

import numpy as np

from cortex.obstacle_map import ObstacleMap
from robot.lidar import beam_angles

# a 6 x 6 m room with a 1.2 x 1.2 m table in the middle (world x, z)
SEGS = [((-3, -3), (3, -3)), ((3, -3), (3, 3)), ((3, 3), (-3, 3)), ((-3, 3), (-3, -3)),
        ((-0.6, -0.6), (0.6, -0.6)), ((0.6, -0.6), (0.6, 0.6)), ((0.6, 0.6), (-0.6, 0.6)), ((-0.6, 0.6), (-0.6, -0.6))]
ANG = beam_angles(90)


def scan(x, z, yaw_deg):
    out = []
    for a in ANG:
        b = math.radians(yaw_deg - a)
        bx, bz, best = math.cos(b), -math.sin(b), 8.0
        for (ax, az), (cx, cz) in SEGS:
            ex, ez = cx - ax, cz - az
            den = bx * ez - bz * ex
            if abs(den) < 1e-12:
                continue
            t = ((ax - x) * ez - (az - z) * ex) / den
            u = ((ax - x) * bz - (az - z) * bx) / den
            if t > 1e-6 and 0 <= u <= 1:
                best = min(best, t)
        out.append(best)
    return out


def _mapped():
    m = ObstacleMap(0.116)
    for pose in [(-2.0, 0.0, 0.0), (-2.0, 1.5, 0.0), (-2.0, -1.5, 0.0), (2.0, 0.0, 180.0)]:
        m.update(pose, scan(*pose), ANG)
    return m


def test_map_marks_table_and_free_floor():
    m = _mapped()
    i, j = m.cell(-0.6, 0.0)                     # the table's near edge
    assert m.occupied()[int(i), int(j)]
    i, j = m.cell(-1.5, 0.0)                     # open floor between pet and table
    assert m.seen[int(i), int(j)] and not m.occupied()[int(i), int(j)]


def test_route_goes_around_the_table():
    m = _mapped()
    path = m.plan((-2.0, 0.0), (2.0, 0.0))
    assert path is not None
    occ = m.occupied()
    for x, z in path:                            # never through the table (or within the body's half width)
        assert not (abs(x) < 0.6 + 0.116 and abs(z) < 0.6 + 0.116), (x, z)
    assert max(abs(z) for _, z in path) > 0.6 + 0.116          # it goes round the side
    rp = m.route_point((-2.0, 0.0), (2.0, 0.0))
    assert rp is not None and 0.5 < math.hypot(rp[0] + 2.0, rp[1]) < 0.8   # the next point, ~0.6 m along


def test_straight_line_when_nothing_is_in_the_way():
    m = _mapped()
    rp = m.route_point((-2.0, -2.0), (-2.0, 2.0))
    assert rp is not None and abs(rp[0] + 2.0) < 0.15


def test_moved_furniture_is_forgotten():
    m = _mapped()
    i, j = (int(v) for v in m.cell(-0.6, 0.0))
    assert m.occupied()[i, j]
    global SEGS
    saved, SEGS = SEGS, SEGS[:4]                 # the table is gone
    try:
        for _ in range(8):
            m.update((-2.0, 0.0, 0.0), scan(-2.0, 0.0, 0.0), ANG)
    finally:
        SEGS = saved
    assert not m.occupied()[i, j]


def test_cortex_routes_its_goal_around_the_table():
    import pytest
    pytest.importorskip("torch")
    from cortex.v0 import CortexV0
    cx = CortexV0(seed=0)
    cx.enable_route(0.116)
    for pose in [(-2.0, 0.0, 0.0), (-2.0, 1.5, 0.0), (-2.0, -1.5, 0.0), (2.0, 0.0, 180.0)]:
        cx.observe_scan(pose, scan(*pose), ANG)
    cx.day_log = {}
    rp = cx._route((-0.9, 0.0), (2.0, 0.0), 0.0)                # right in front of the table
    assert rp is not None and abs(rp[1]) > 0.2                   # it heads round, not into it
