"""2D lidar as fly senses (robot/lidar.py)."""
import numpy as np

from robot.lidar import LidarLooming, LidarScan, LidarTouch, beam_angles, ellipse_body


def _scan():
    ang = beam_angles(8)                  # -180, -135, ..., 135
    return LidarScan(ang, 0.2), ang


def test_an_approaching_obstacle_looms_from_its_direction():
    scan, ang = _scan()
    loom = LidarLooming(scan)
    i = ang.index(45.0)                    # front right
    for k, r in enumerate([1.8, 1.4, 1.0, 0.7]):
        rr = np.full(8, 8.0)
        rr[i] = r
        scan.update(rr, 0.1 * k)
        loom.update()
    st = loom.state(0.0)
    assert st["active"] and st["azimuth_deg"] == 45.0 and st["expansion_rate_deg_s"] > 0


def test_a_still_wall_does_not_loom():
    scan, _ = _scan()
    loom = LidarLooming(scan)
    for k in range(4):
        scan.update(np.full(8, 0.6), 0.1 * k)
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


def test_ellipse_body():
    b = ellipse_body([0.0, 90.0, 180.0], 0.55, 0.25)
    assert np.allclose(b, [0.55, 0.25, 0.55])
