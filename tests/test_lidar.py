"""2D lidar as fly senses (robot/lidar.py)."""
import numpy as np

from robot.lidar import LidarLooming, LidarScan, LidarTouch, beam_angles, body_profile, ellipse_body, rect_body


def _scan():
    ang = beam_angles(8)                  # -180, -135, ..., 135
    return LidarScan(ang, 0.2), ang


def test_an_approaching_obstacle_looms_from_its_direction():
    ang = beam_angles(90)                  # a realistic scan (4-degree beams)
    scan = LidarScan(ang, 0.2)
    loom = LidarLooming(scan)
    i = ang.index(44.0)                    # front right
    for k, r in enumerate([1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.7]):   # 2 m/s, pet standing still
        rr = np.full(90, 8.0)
        rr[i] = r
        scan.update(rr, 0.1 * k, pose=(0.0, 0.0, 0.0))
        loom.update()
    st = loom.state(0.0)
    assert st["active"] and st["azimuth_deg"] == 44.0 and st["expansion_rate_deg_s"] > 0


def test_a_still_wall_does_not_loom():
    scan, _ = _scan()
    loom = LidarLooming(scan)
    for k in range(8):
        scan.update(np.full(8, 0.6), 0.1 * k, pose=(0.0, 0.0, 0.0))
        loom.update()
    assert not loom.state(0.0)["active"]


class _C:
    def __init__(self):
        import pandas as pd
        self.neurons = pd.DataFrame({"primary_type": ["BM_Ant", "BM_Ant", "BM_Vib", "LC4"],
                                     "side": ["left", "right", "right", "left"]})


def test_touch_is_felt_on_the_side_of_the_obstacle():
    scan, ang = _scan()
    touch = LidarTouch(_C(), scan)
    rr = np.full(8, 8.0)
    rr[ang.index(-90.0)] = 0.25            # 5 cm from the left side
    scan.update(rr, 0.0)
    r = touch.rates_hz(0.0)
    left = r[~touch._right]
    assert left.max() > 50 and r[touch._right].max() == 0


def test_rect_body_rover():
    import math
    # the rover (sim/habitat_bridge/bodies.py): front 0.127, side 0.116, corner 0.172
    from sim.habitat_bridge.bodies import BODIES
    b = rect_body([0.0, 90.0, 180.0, -90.0], 0.127, 0.116)
    assert np.allclose(b, [0.127, 0.116, 0.127, 0.116])
    corner = math.degrees(math.atan2(0.116, 0.127))
    assert abs(rect_body([corner], 0.127, 0.116)[0] - math.hypot(0.127, 0.116)) < 1e-9
    assert np.allclose(body_profile([0.0, 90.0], BODIES["rover"]), [0.127, 0.116])
    assert np.allclose(body_profile([0.0, 90.0], BODIES["spot"]), [0.55, 0.25])


def test_ellipse_body():
    b = ellipse_body([0.0, 90.0, 180.0], 0.55, 0.25)
    assert np.allclose(b, [0.55, 0.25, 0.55])


def _wall_scan(pose, beams=90):
    """Ranges to a straight wall at x = 1 (world), from pose (x, z, yaw)."""
    import math
    ang = beam_angles(beams)
    x, z, yaw = pose
    rr = []
    for a in ang:
        b = math.radians(yaw - a)
        c = math.cos(b)
        rr.append((1.0 - x) / c if c > 1e-6 else np.inf)
    return np.array(rr), ang


def test_turning_beside_a_still_wall_does_not_loom():
    rr, ang = _wall_scan((0.0, 0.0, 0.0))
    scan = LidarScan(ang, 0.2)
    loom = LidarLooming(scan)
    for k in range(8):                                  # spin 10 deg per step
        pose = (0.0, 0.0, 10.0 * k)
        rr, _ = _wall_scan(pose)
        scan.update(rr, 0.1 * k, pose=pose)
        loom.update()
        assert not loom.state(0.0)["active"]


def test_something_coming_at_a_still_pet_looms():
    ang = beam_angles(90)
    scan = LidarScan(ang, 0.2)
    loom = LidarLooming(scan)
    i = ang.index(0.0)
    for k, r in enumerate([1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6]):   # 2 m/s toward the pet
        rr = np.full(90, 8.0)
        rr[i] = r
        scan.update(rr, 0.1 * k, pose=(0.0, 0.0, 0.0))
        loom.update()
    assert loom.state(0.0)["active"] and loom.state(0.0)["azimuth_deg"] == 0.0


def test_obstacle_limit():
    from robot.safety import obstacle_limit
    assert obstacle_limit([0.05, 1.0], [0.0, 90.0], 0.4) == 0.0     # wall just ahead: stop
    assert obstacle_limit([0.05, 1.0], [120.0, 0.0], 0.4) == 0.4    # behind: free
    assert obstacle_limit([0.05], [0.0], -0.2) == -0.2              # backing up: free
    assert 0.0 < obstacle_limit([0.34], [0.0], 0.4) < 0.4


def test_server_lidar_turns_with_the_robot(monkeypatch):
    """Regression (review 2026-09-26): the server converted yaw to radians twice,
    so the simulated beams stayed fixed to the world. Mock Habitat's ray cast:
    a wall at world x = 2; a robot facing -z (yaw 90 deg CCW) must see it on its
    RIGHT (+90) at 2.0 m and nothing straight ahead."""
    import math
    import sys
    import types
    mn = types.ModuleType("magnum")

    class V3:
        def __init__(s, x, y, z):
            s.x, s.y, s.z = x, y, z
    mn.Vector3 = V3
    hs = types.ModuleType("habitat_sim")
    hs.geo = types.SimpleNamespace(Ray=lambda o, d: types.SimpleNamespace(o=o, d=d))
    monkeypatch.setitem(sys.modules, "magnum", mn)
    monkeypatch.setitem(sys.modules, "habitat_sim", hs)

    class Sim:
        def cast_ray(self, ray, max_distance):
            hits = []
            if ray.d.x > 1e-9:
                t = (2.0 - ray.o.x) / ray.d.x
                if t <= max_distance:
                    hits = [types.SimpleNamespace(object_id=99, ray_distance=t)]
            return types.SimpleNamespace(hits=hits, has_hits=lambda: bool(hits))
    from sim.habitat_bridge.habitat_server import Server
    srv = Server.__new__(Server)
    srv.set_lidar(8)
    robot = types.SimpleNamespace(sim_obj=types.SimpleNamespace(object_id=1, link_object_ids={}))
    r = dict(zip(srv.lidar_angles, srv._lidar(Sim(), robot, np.zeros(3), math.atan2(1.0, 0.0))))
    assert abs(r[90.0] - 2.0) < 1e-6
    assert r[0.0] == srv.LIDAR_MAX_M


def _room_segments():
    return [((-3, -3), (3, -3)), ((3, -3), (3, 3)), ((3, 3), (-3, 3)), ((-3, 3), (-3, -3)),
            ((0.5, 0.5), (1.5, 0.5)), ((1.5, 0.5), (1.5, 1.5)), ((1.5, 1.5), (0.5, 1.5)), ((0.5, 1.5), (0.5, 0.5))]


def _cast(segs, x, z, bx, bz, rmax=8.0):
    best = rmax
    for (ax, az), (cx, cz) in segs:
        ex, ez = cx - ax, cz - az
        den = bx * ez - bz * ex
        if abs(den) < 1e-12:
            continue
        t = ((ax - x) * ez - (az - z) * ex) / den
        u = ((ax - x) * bz - (az - z) * bx) / den
        if t > 1e-6 and 0 <= u <= 1:
            best = min(best, t)
    return best


def _room_scan(segs, pose, ang):
    import math
    x, z, yaw = pose
    out = []
    for a in ang:
        b = math.radians(yaw - a)
        out.append(_cast(segs, x, z, math.cos(b), -math.sin(b)))
    return np.array(out)


def test_walking_and_turning_in_a_still_room_does_not_loom():
    """Regression (review 2026-09-26): walking made slanted still walls loom."""
    import math
    ang = beam_angles(90)
    segs = _room_segments()
    for v, wdeg in ((0.3, 0.0), (0.3, 20.0), (0.0, 40.0)):
        scan = LidarScan(ang, ellipse_body(ang, 0.55, 0.25))
        loom = LidarLooming(scan)
        x, z, yaw = -1.5, -1.5, 30.0
        active = 0
        for k in range(300):
            pose = (x, z, yaw)
            scan.update(_room_scan(segs, pose, ang), 0.1 * k, pose=pose)
            loom.update()
            active += loom.last["active"]
            nx, nz = x + v * 0.1 * math.cos(math.radians(yaw)), z - v * 0.1 * math.sin(math.radians(yaw))
            if abs(nx) < 2.3 and abs(nz) < 2.3 and not (0.0 < nx < 2.0 and 0.0 < nz < 2.0):
                x, z = nx, nz
            else:
                yaw += 90
            yaw = (yaw + wdeg * 0.1) % 360
        assert active <= 3, (v, wdeg, active)


def test_a_person_walking_at_a_walking_pet_looms():
    import math
    ang = beam_angles(90)
    segs0 = _room_segments()
    scan = LidarScan(ang, ellipse_body(ang, 0.55, 0.25))
    loom = LidarLooming(scan)
    hits = 0
    for k in range(20):
        t = 0.1 * k
        px = 2.4 - 0.8 * t                                   # a person (0.3 m wide) walking -x at 0.8 m/s
        person = [((px, -1.65), (px, -1.35))]
        pose = (-1.5 + 0.2 * t, -1.5, 0.0)                   # the pet walks +x at 0.2 m/s, facing +x
        scan.update(_room_scan(segs0 + person, pose, ang), t, pose=pose)
        loom.update()
        if loom.last["active"]:
            hits += 1
            assert abs(loom.last["azimuth_deg"]) <= 20.0
    assert hits >= 5


def test_touch_from_straight_behind_is_felt_on_both_sides():
    scan, ang = _scan()
    touch = LidarTouch(_C(), scan)
    rr = np.full(8, 8.0)
    rr[ang.index(-180.0)] = 0.25
    scan.update(rr, 0.0)
    assert touch.levels()[0] > 0 and touch.levels()[1] > 0


def test_ttc_checks_turning_and_backing_up():
    from robot.safety import ttc_scale
    L, W = 0.55, 0.25
    wall_ahead = np.array([[1.0, y] for y in np.linspace(-0.5, 0.5, 11)])
    assert ttc_scale(wall_ahead, 0.5, 0.0, L, W) < 1.0          # driving at it: slowed
    assert ttc_scale(wall_ahead, -0.3, 0.0, L, W) == 1.0        # backing away: free
    behind = np.array([[-0.7, 0.0]])
    assert ttc_scale(behind, -0.3, 0.0, L, W) < 1.0             # backing into it: slowed
    corner = np.array([[0.45, -0.40]])                           # right of the nose
    assert ttc_scale(corner, 0.0, -1.0, L, W) < 1.0             # turning right swings the nose into it
    assert ttc_scale(corner, 0.0, 1.0, L, W) == 1.0             # turning left: free


def test_ttc_never_freezes_a_pet_that_is_touching_a_wall():
    """Regression: the first version stopped whenever anything was inside the
    footprint, and the pet froze against furniture."""
    from robot.safety import ttc_filter
    L, W = 0.55, 0.25
    touching = np.array([[0.56, y] for y in np.linspace(-0.3, 0.3, 7)])   # nose on a wall
    v, w = ttc_filter(touching, 0.3, 0.0, L, W)
    assert v == 0.0                                              # not further into it
    v, w = ttc_filter(touching, -0.2, 0.0, L, W)
    assert v == -0.2                                             # backing away: allowed
    # turning while touching can be unsafe for this long body (the filter alone
    # may then stop it); the CollisionMonitor's recovery gets it out within
    # STUCK_S (review 2026-09-27: the old assertion here could never fail)
    from robot.safety import CollisionMonitor
    cm = CollisionMonitor(L, W)
    outs = [cm(touching, 0.3, 1.0, 0.1) for _ in range(int(CollisionMonitor.STUCK_S / 0.1) + 2)]
    assert any(o != (0.0, 0.0) for o in outs)                    # some way out, not frozen


def test_collision_monitor_recovers_from_a_corner():
    """A long body in a corner: forward and turning in place are unsafe; after
    STUCK_S the monitor backs out."""
    from robot.safety import CollisionMonitor, motion_safe
    L, W = 0.55, 0.25
    wall_front = [[0.60, y] for y in np.linspace(-0.6, 0.6, 13)]
    wall_right = [[x, -0.30] for x in np.linspace(-0.6, 0.6, 13)]
    pts = np.array(wall_front + wall_right)
    assert not motion_safe(pts, 0.2, 0.0, L, W) and not motion_safe(pts, 0.0, -0.8, L, W)
    cm = CollisionMonitor(L, W)
    out = [cm(pts, 0.3, -0.5, 0.1) for _ in range(30)]
    assert any(v < 0 or abs(w) > 0 for v, w in out[15:])        # it moves again
    assert cm.recoveries >= 1
