"""The real rover's drivers and world, without hardware: robot/ugv.py,
robot/d500.py, robot/rover_world.py."""
import json
import math

import numpy as np
import pytest

from robot import d500
from robot.rover_world import (FakeBase, FakeLidar, FakePerson, FakeRoom, LocalConn, RoverWorld,
                               person_geometry)
from robot.ugv import Odometry, UGVBase, voltage


# ------------------------------------------------------------------ D500
def test_crc_table_is_ldrobots():
    # the first entries of the LD19 / STL-19P manual's table (poly 0x4D)
    assert list(d500.CRC_TABLE[:8]) == [0x00, 0x4D, 0x9A, 0xD7, 0x79, 0x34, 0xE3, 0xAE]


def _revolution(dist_mm_at, n_packets=30):
    """Packets covering one turn; dist_mm_at(angle_deg) -> mm."""
    out = b""
    span = 360.0 / n_packets
    for p in range(n_packets):
        a0 = p * span
        a1 = a0 + span * 11 / 12
        angs = [a0 + (a1 - a0) * k / 11 for k in range(12)]
        out += d500.packet(a0, a1, [dist_mm_at(a) for a in angs])
    return out


def test_parse_packet_and_resync():
    pkt = d500.packet(10.0, 20.0, [1000 + k for k in range(12)])
    bad = bytearray(pkt)
    bad[10] ^= 0xFF                              # corrupt: CRC fails
    buf = bytearray(b"\x00\x54junk" + bytes(bad) + pkt)
    pts, nbad = d500.parse(buf)
    assert nbad == 1
    assert len(pts) == 12
    assert pts[0][0] == pytest.approx(10.0) and pts[-1][0] == pytest.approx(20.0)
    assert pts[3][1] == pytest.approx(1.003)
    assert len(buf) < d500.PACKET_LEN            # consumed


def test_angle_wraps_inside_a_packet():
    pts, _ = d500.parse(bytearray(d500.packet(355.0, 5.0, [500] * 12)))
    assert pts[0][0] == pytest.approx(355.0)
    assert pts[-1][0] == pytest.approx(5.0)


def test_revolution_to_beams_right_is_positive():
    # a wall 1 m away to the right (sensor angle 90, clockwise), 3 m elsewhere
    near = lambda a: 1000 if abs(((a - 90.0 + 180.0) % 360.0) - 180.0) < 10 else 3000
    lid = d500.D500(transport=_NullIO(), beams=36)
    lid.feed(d500.packet(300.0, 311.0, [500] * 12), 0.0)   # started mid-turn: this partial turn is dropped
    lid.feed(_revolution(near), 0.05)
    assert lid.latest() is None
    lid.feed(d500.packet(0.0, 11.0, [3000] * 12), 0.1)   # the next turn starts: the first full one is complete
    t, r = lid.latest()
    ang = d500.beam_angles(36)
    assert r[np.argmin(np.abs(ang - 90.0))] == pytest.approx(1.0)
    assert r[np.argmin(np.abs(ang + 90.0))] == pytest.approx(3.0)
    lid.close()


def test_bin_scan_mounting_and_direction():
    pts = [(30.0, 2.0, 100)]
    ang = d500.beam_angles(36)
    r = d500.bin_scan(pts, 36, zero_deg=0.0, clockwise=True)
    assert r[np.argmin(np.abs(ang - 30.0))] == 2.0
    r = d500.bin_scan(pts, 36, zero_deg=0.0, clockwise=False)
    assert r[np.argmin(np.abs(ang + 30.0))] == 2.0
    r = d500.bin_scan(pts, 36, zero_deg=30.0)               # the sensor's 30 deg points ahead
    assert r[np.argmin(np.abs(ang))] == 2.0
    # no return (distance 0) and nothing at all: the maximum range
    assert (d500.bin_scan([(0.0, 0.0, 100)], 36) == d500.MAX_RANGE_M).all()


class _NullIO:
    def read(self, n):
        import time
        time.sleep(0.01)
        return b""

    def close(self):
        pass


# ------------------------------------------------------------------ UGV
class _FakeSerial:
    def __init__(self):
        self.written = []

    def write(self, b):
        self.written.append(json.loads(b.decode()))

    def read(self, n):
        import time
        time.sleep(0.01)
        return b""

    def close(self):
        pass


def test_ugv_setup_drive_and_limits():
    io = _FakeSerial()
    b = UGVBase(transport=io, heartbeat_ms=400)
    assert io.written[:3] == [{"T": 143, "cmd": 0}, {"T": 131, "cmd": 1}, {"T": 136, "cmd": 400}]
    b.drive(0.2, -0.5)
    assert io.written[-1] == {"T": 13, "X": 0.2, "Z": -0.5}
    b.drive(5.0, 100.0)                          # clipped
    assert io.written[-1]["X"] == 0.5 and io.written[-1]["Z"] == pytest.approx(math.pi, abs=1e-3)
    b.drive(float("nan"), 0.3)                   # never a NaN on the wire: stop
    assert io.written[-1] == {"T": 13, "X": 0.0, "Z": 0.0}
    b.head(200, -40)
    assert io.written[-1] == {"T": 133, "X": 180.0, "Y": -30.0, "SPD": 0, "ACC": 0}
    b.close()
    assert io.written[-1] == {"T": 13, "X": 0, "Z": 0}


def test_ugv_feedback_and_odometry():
    b = UGVBase(transport=_FakeSerial())
    t = 0.0
    for od in range(0, 101, 10):                 # 1 m straight ahead, in pieces, split across reads
        line = json.dumps({"T": 1001, "L": 0.1, "R": 0.1, "odl": od, "odr": od, "v": 1150}).encode() + b"\n"
        b.feed(line[:7], t)
        b.feed(line[7:], t)
        t += 0.1
    b.feed(b"garbage\n{\"T\":1002}\n", t)
    st = b.state()
    x, z, yaw = st["pose"]
    assert x == pytest.approx(1.0) and z == pytest.approx(0.0) and yaw == pytest.approx(0.0)
    assert st["battery_v"] == pytest.approx(11.5)
    assert b.bad_lines == 1 and b.n_fb == 11
    b.close()


def test_odometry_turn_and_conventions():
    o = Odometry(track_m=0.2)
    o.update({"odl": 0, "odr": 0}, 0.0)
    # right wheel forward, left back: counter-clockwise (left) turn in place
    quarter = (math.pi / 2) * 0.2 / 2             # each wheel's arc for 90 deg: (track / 2) x angle
    o.update({"odl": -100 * quarter, "odr": 100 * quarter}, 1.0)
    assert o.pose()[2] == pytest.approx(90.0)
    o2 = Odometry()
    o2.yaw = math.radians(90.0)                  # facing +90 (CCW from +x): forward is -z
    o2._move(1.0, 1.0)
    assert o2.x == pytest.approx(0.0, abs=1e-12) and o2.z == pytest.approx(-1.0)
    assert voltage(1089) == pytest.approx(10.89) and voltage(11.2) == 11.2 and voltage(None) is None


# ------------------------------------------------------------------ world
def test_person_geometry_matches_habitat_convention():
    ahead = person_geometry((0.0, 0.0, 0.0), (2.0, 0.0), 53.0)
    assert ahead["az"] == pytest.approx(0.0) and ahead["visible"]
    right = person_geometry((0.0, 0.0, 0.0), (2.0, 1.0), 53.0)      # +z is to the right when facing +x
    assert right["az"] > 0
    behind = person_geometry((0.0, 0.0, 0.0), (-2.0, 0.0), 53.0)
    assert not behind["visible"]


def test_fake_lidar_sees_walls():
    room = FakeRoom(half_x=2.0, half_z=1.5, box=((10, 10), (11, 11)))
    base = FakeBase(start=(0.0, 0.0, 0.0), room=room)
    lid = FakeLidar(room, base, None, beams=4)                      # beams at -180, -90, 0, 90
    lid.update(0.0)
    r = lid.latest()[1]
    assert r[2] == pytest.approx(2.0)                                # ahead: +x wall
    assert r[3] == pytest.approx(1.5)                                # right: +z wall
    assert r[0] == pytest.approx(2.0) and r[1] == pytest.approx(1.5)


def test_rover_world_real_time_steps():
    clock = {"t": 100.0}
    slept = []
    world = RoverWorld(FakeBase(), None, None, clock=lambda: clock["t"],
                       sleep=lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s)))
    conn = LocalConn(world)
    conn.send({"cmd": "reset"})
    conn.recv()
    for k in range(3):
        conn.send({"cmd": "step", "v": 0.2, "w": 0.0, "n": 12})
        obs = conn.recv()
        clock["t"] += 0.03                       # the brain's work: 30 ms
    assert slept[0] == pytest.approx(0.1) and slept[1] == pytest.approx(0.07)
    assert obs["t"] == pytest.approx(0.3) and obs["stats"]["overruns"] == 0
    clock["t"] += 0.2                            # a slow step: over the period
    conn.send({"cmd": "step", "v": 0.0, "w": 0.0, "n": 12})
    obs = conn.recv()
    assert obs["stats"]["overruns"] == 1
    assert world.base.cmd == (0.0, 0.0)
    conn.send({"cmd": "lidar", "beams": 90})
    assert len(conn.recv()["angles"]) == 90
    conn.send({"cmd": "path", "goal": [1.0, 2.0]})
    assert conn.recv()["ok"] is False
    conn.send({"cmd": "nonsense"})
    assert "error" in conn.recv()


def test_fake_world_moves_and_scans():
    room = FakeRoom()
    base = FakeBase(start=(-1.0, 0.0, 0.0), room=room)
    person = FakePerson()
    world = RoverWorld(base, FakeLidar(room, base, person, 90), person, 150.0, fake=True,
                       clock=lambda: 0.0, sleep=lambda s: None)
    world.handle({"cmd": "lidar", "beams": 90})
    world.handle({"cmd": "reset"})
    for _ in range(10):
        obs = world.handle({"cmd": "step", "v": 0.3, "w": 0.0, "n": 12})
    assert obs["robot"][0] == pytest.approx(-0.7, abs=1e-6)          # 1 s at 0.3 m/s
    assert len(obs["lidar"]) == 90 and max(obs["lidar"]) <= 8.0
    assert obs["human"][0] != obs["human"][0]                        # unknown on the robot: nan


def test_lost_packet_keeps_previous_range():
    lid = d500.D500(transport=_NullIO(), beams=36)
    lid.feed(d500.packet(300.0, 311.0, [500] * 12), 0.0)
    lid.feed(_revolution(lambda a: 1500), 0.05)
    lid.feed(d500.packet(0.0, 11.0, [1500] * 12), 0.1)
    first = lid.latest()[1].copy()
    # the next turn loses the packets covering 90..110 deg (CRC errors): those
    # beams keep 1.5 m instead of reading as open space
    pk = b"".join(d500.packet(a0, a0 + 11.0, [2000] * 12) for a0 in np.arange(12.0, 360.0, 12.0)
                  if not (84.0 <= a0 <= 108.0))
    lid.feed(pk, 0.15)
    lid.feed(d500.packet(0.0, 11.0, [2000] * 12), 0.2)
    r = lid.latest()[1]
    ang = d500.beam_angles(36)
    assert r[np.argmin(np.abs(ang - 100.0))] == pytest.approx(1.5)
    assert r[np.argmin(np.abs(ang - 40.0))] == pytest.approx(2.0)
    assert first == pytest.approx(np.full(36, 1.5))
    lid.close()


def test_ugv_pose_at_and_settings_resent():
    io = _FakeSerial()
    b = UGVBase(transport=io)
    for k, od in enumerate((0, 10, 20)):
        b.feed(json.dumps({"T": 1001, "odl": od, "odr": od}).encode() + b"\n", 10.0 + k)
    assert b.pose_at(11.1)[0] == pytest.approx(0.1)
    assert b.pose_at(50.0)[0] == pytest.approx(0.2)
    # feedback goes quiet (e.g. the ESP32 reset): the settings are sent again
    b.fb_t = -1e9
    b._configured_t = -1e9
    n = len(io.written)
    b.drive(0.1, 0.0)
    assert {"T": 136, "cmd": 500} in io.written[n:]
    b.close()


class _Stale:
    """A base whose feedback stopped, for RoverWorld."""
    def __init__(self):
        self.cmds = []

    def drive(self, v, w):
        self.cmds.append((v, w))

    def stop(self):
        self.cmds.append((0.0, 0.0))

    def state(self):
        return {"pose": (0.0, 0.0, 0.0), "battery_v": 11.0, "fb_age_s": None}

    def close(self):
        pass


class _OldScan:
    beams = 90

    def latest(self):
        return (0.0, np.full(90, 2.0))            # taken at t=0 and never again

    def close(self):
        pass


def test_stale_sensors_stop_the_wheels():
    clock = {"t": 10.0}
    base = _Stale()
    w = RoverWorld(base, _OldScan(), None, clock=lambda: clock["t"], sleep=lambda s: None)
    w.handle({"cmd": "lidar", "beams": 90})
    w.handle({"cmd": "reset"})
    obs = w.handle({"cmd": "step", "v": 0.3, "w": 0.2, "n": 12})
    assert base.cmds[-1] == (0.0, 0.0)
    assert set(obs["stale"]) == {"base", "lidar"}
    assert obs["stats"]["stale_events"] == 1


def test_world_hands_each_scan_once_with_its_pose():
    room = FakeRoom()
    base = FakeBase(start=(0.0, 0.0, 0.0), room=room)
    world = RoverWorld(base, FakeLidar(room, base, None, 90), None, fake=True,
                       clock=lambda: 0.0, sleep=lambda s: None)
    world.handle({"cmd": "lidar", "beams": 90})
    o1 = world.handle({"cmd": "reset"})
    assert o1["lidar"] is not None and o1["lidar_pose"] == [0.0, 0.0, 0.0]
    o2 = world.summary()                           # no new revolution since: nothing new
    assert o2["lidar"] is None
