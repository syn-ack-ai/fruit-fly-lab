"""
The LDRobot D500 lidar kit (STL-19P sensor; the LD19 protocol) on a serial
port, as the scans robot/lidar.py and the Habitat server use: ranges (m) at
fixed beam angles (deg, + = right of the heading, from -180), 8 m where
nothing was hit.

Protocol (LDRobot STL-19P datasheet; LD19 development manual): 230400 baud,
47-byte packets
    0x54, 0x2C (type 1, 12 points), speed (u16, deg/s), start angle (u16,
    0.01 deg), 12 x {distance (u16, mm), intensity (u8)}, end angle (u16,
    0.01 deg), timestamp (u16, ms, wraps at 30000), CRC-8 (poly 0x4D) of the
    46 bytes before it
all little-endian. A point's angle is interpolated between the start and end
angles. The sensor spins clockwise seen from above (angles grow to the
right), ~10 revolutions a second; a revolution ends when the angle wraps.

C. APPROXIMATIONS / TO CALIBRATE ON THE ROBOT: ZERO_DEG (the sensor angle
that points straight ahead, from its mounting) and CLOCKWISE; a beam's range
is the nearest point within its angular bin; a distance of 0 (no return) or
an intensity below MIN_INTENSITY is ignored.

    python -m robot.d500 --port /dev/ttyUSB0          # print scans
"""
from __future__ import annotations

import math
import struct
import threading
import time

import numpy as np

PACKET_LEN = 47
HEADER, VERLEN = 0x54, 0x2C
MAX_RANGE_M = 8.0          # as the Habitat server's lidar (habitat_server.Server.LIDAR_MAX_M)
MIN_INTENSITY = 0
FINE_BEAMS = 360               # the display's scan (1 degree)
ZERO_DEG = 0.0
CLOCKWISE = True


def _crc_table():
    t = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ 0x4D) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        t.append(c)
    return bytes(t)


CRC_TABLE = _crc_table()


def crc8(data: bytes) -> int:
    c = 0
    for b in data:
        c = CRC_TABLE[(c ^ b) & 0xFF]
    return c


def parse(buf: bytearray) -> tuple:
    """Consume whole packets from buf. Returns (points, bad): points is a list
    of (angle_deg, distance_m, intensity) in sensor angles; bad counts packets
    that failed the CRC (resynchronised on the next header)."""
    pts, bad = [], 0
    i = 0
    n = len(buf)
    while i + PACKET_LEN <= n:
        if buf[i] != HEADER or buf[i + 1] != VERLEN:
            i += 1
            continue
        pkt = bytes(buf[i:i + PACKET_LEN])
        if crc8(pkt[:-1]) != pkt[-1]:
            bad += 1
            i += 1
            continue
        start = struct.unpack_from("<H", pkt, 4)[0] / 100.0
        end = struct.unpack_from("<H", pkt, 42)[0] / 100.0
        if end < start:
            end += 360.0
        step = (end - start) / 11.0
        for k in range(12):
            d, inten = struct.unpack_from("<HB", pkt, 6 + 3 * k)
            pts.append(((start + step * k) % 360.0, d / 1000.0, inten))
        i += PACKET_LEN
    del buf[:i]
    return pts, bad


def beam_angles(beams: int) -> np.ndarray:
    """As robot.lidar.beam_angles: -180 .. 180 - 360 / beams (deg, + = right)."""
    return -180.0 + 360.0 * np.arange(beams) / beams


def bin_scan(points, beams: int, zero_deg: float = ZERO_DEG, clockwise: bool = CLOCKWISE,
             max_range: float = MAX_RANGE_M, min_intensity: int = MIN_INTENSITY,
             empty: float | None = None) -> np.ndarray:
    """One revolution's points -> ranges at beam_angles(beams): the nearest
    valid point in each beam's bin; a bin without points gets `empty`
    (default max_range: nothing there)."""
    out = np.full(beams, max_range if empty is None else empty)
    if not points:
        return out
    p = np.asarray(points, float)
    ok = (p[:, 1] > 0.0) & (p[:, 2] >= min_intensity)
    p = p[ok]
    if not len(p):
        return out
    az = p[:, 0] - zero_deg
    if not clockwise:
        az = -az
    az = (az + 180.0) % 360.0 - 180.0                     # + = right
    step = 360.0 / beams
    k = np.round((az + 180.0) / step).astype(int) % beams
    hit = np.minimum(p[:, 1], max_range)
    if empty is not None and np.isnan(empty):
        out[k] = max_range                        # bins with points start from max_range
    np.minimum.at(out, k, hit)
    return out


class D500:
    """The lidar on a serial port (or a transport with read(n) -> bytes, for
    tests). A reader thread assembles revolutions; latest() is the newest scan."""

    def __init__(self, port: str = "/dev/ttyUSB0", beams: int = 90, transport=None,
                 zero_deg: float = ZERO_DEG, clockwise: bool = CLOCKWISE, baud: int = 230400):
        if transport is None:
            import serial                         # pyserial
            transport = serial.Serial(port, baud, timeout=0.05)
        self.io = transport
        self.beams, self.zero, self.cw = int(beams), float(zero_deg), bool(clockwise)
        self._buf = bytearray()
        self._rev = []
        self._last_angle = None
        self._lock = threading.Lock()
        self.scan = None                          # (t_s, ranges)
        self.fine = None                          # (t_s, ranges at FINE_BEAMS), for displays
        self._first = True                        # the first revolution is partial: dropped
        self.revolutions = 0
        self.errors = 0
        self.bad_packets = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._reader, daemon=True)
        self._th.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self.io.read(512)
            except Exception:
                return
            if chunk:
                try:
                    self.feed(chunk, time.monotonic())
                except Exception:                 # never let one bad chunk end the reader
                    self.errors += 1
                    self._buf.clear()

    def feed(self, chunk: bytes, t_s: float) -> None:
        self._buf += chunk
        pts, bad = parse(self._buf)
        self.bad_packets += bad
        for p in pts:
            if self._last_angle is not None and p[0] < self._last_angle - 180.0:
                # the angle wrapped: a revolution is complete. Bins no point
                # fell into (a packet lost to a CRC error) keep the previous
                # revolution's range rather than reading as open space.
                if self._first:
                    self._first = False           # started mid-turn: partial
                else:
                    ranges = bin_scan(self._rev, self.beams, self.zero, self.cw, empty=np.nan)
                    prev = self.scan[1] if self.scan is not None else None
                    gap = np.isnan(ranges)
                    ranges[gap] = prev[gap] if prev is not None else MAX_RANGE_M
                    # a finer scan of the same turn for displays (robot/dashboard.py):
                    # the D500 gives ~450 points a turn; the brain uses `beams`
                    fine = bin_scan(self._rev, FINE_BEAMS, self.zero, self.cw)
                    with self._lock:
                        self.scan = (t_s, ranges)
                        self.fine = (t_s, fine)
                        self.revolutions += 1
                self._rev = []
            self._rev.append(p)
            self._last_angle = p[0]

    def latest(self):
        with self._lock:
            return self.scan

    def close(self) -> None:
        self._stop.set()
        try:
            self.io.close()
        except Exception:
            pass


def packet(start_deg: float, end_deg: float, dists_mm, intens=None, speed=3600, ts=0) -> bytes:
    """A valid packet (tests, and a fake lidar)."""
    intens = [200] * 12 if intens is None else intens
    b = struct.pack("<BBHH", HEADER, VERLEN, speed, int(round(start_deg * 100)) % 36000)
    for d, i in zip(dists_mm, intens):
        b += struct.pack("<HB", int(d), int(i))
    b += struct.pack("<HH", int(round(end_deg * 100)) % 36000, ts)
    return b + bytes([crc8(b)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--beams", type=int, default=36)
    a = ap.parse_args()
    lid = D500(a.port, a.beams)
    try:
        while True:
            time.sleep(1.0)
            s = lid.latest()
            print(f"revolutions {lid.revolutions} bad packets {lid.bad_packets}", flush=True)
            if s is not None:
                print(" ".join(f"{az:+.0f}:{r:.2f}" for az, r in zip(beam_angles(a.beams), s[1])), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        lid.close()


if __name__ == "__main__":
    main()
