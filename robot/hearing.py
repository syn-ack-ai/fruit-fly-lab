"""
The robot's hearing: a USB microphone -> the fly's real auditory neurons.

    microphone -> band envelopes -> adaptation -> JO-A / JO-B rates -> connectome

PROVENANCE
----------
A. REAL DATA : the Johnston's organ neurons driven (JO-A, JO-B; FlyWire v783,
   both antennae) and everything downstream of them.
B. PUBLISHED :
   - Johnston's organ subgroups A and B detect sound (antennal vibration);
     A is tuned to higher and B to lower frequencies, together covering the
     ~100-800 Hz range of courtship song (Kamikouchi et al. 2009, Nature
     458:165; Yorozu et al. 2009, Nature 458:201).
C. OUR APPROXIMATIONS :
   - A fly hears near-field particle velocity; a microphone measures pressure.
     We use the pressure envelope in two bands: 100-300 Hz -> JO-B,
     300-800 Hz -> JO-A.
   - Both antennae get the same drive (one mono microphone: no direction).
   - The response adapts to steady sound (as the wind channel in
     fly/world/senses.py does): the band power is smoothed over ~40 ms, the
     background is its slow (~4 s) average in dB, and
     rate = R * s / (s + K) with s = dB above background minus a 6 dB
     deadband, so room hum, fans and ordinary fluctuations are ignored and
     new sounds register.

Standalone check:  python -m robot.hearing --test   (prints levels for 8 s)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time

import numpy as np

RATE = 16000
FRAME = 320                      # 20 ms
BANDS = {"B": (100.0, 300.0), "A": (300.0, 800.0)}
MAX_HZ = 150.0                   # the model's standard activation rate
HALF_DB = 10.0                   # dB above background (past the deadband) for half drive
DEADBAND_DB = 6.0
SMOOTH_S = 0.04                  # band power smoothing
BACKGROUND_S = 4.0               # background (adaptation) time constant


def find_mic(name_hint: str = "0x46d0x994") -> str | None:
    """ALSA device of the USB microphone (default: the Logitech Orbit)."""
    out = subprocess.run(["arecord", "-l"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.match(r"card (\d+): (\S+) .*device (\d+):", line)
        if m and name_hint in line:
            return "plughw:%s,%s" % (m.group(1), m.group(3))
    m = re.search(r"card (\d+):.*device (\d+):", out)
    return "plughw:%s,%s" % (m.group(1), m.group(2)) if m else None


class BandEnvelope:
    """Per-frame band levels (dB) and their adapted drive (dB above floor)."""

    def __init__(self, rate: int = RATE, frame: int = FRAME):
        self.rate, self.frame = rate, frame
        f = np.fft.rfftfreq(frame, 1.0 / rate)
        self.masks = {k: (f >= lo) & (f < hi) for k, (lo, hi) in BANDS.items()}
        self.win = np.hanning(frame)
        self.floor = {k: None for k in BANDS}
        self.power = {k: None for k in BANDS}
        self.dt = frame / rate

    def update(self, x: np.ndarray) -> dict:
        X = np.abs(np.fft.rfft(x * self.win)) ** 2
        out = {}
        for k, m in self.masks.items():
            pw = X[m].sum() / self.frame + 1e-12
            ps = self.power[k]
            ps = pw if ps is None else ps + (pw - ps) * min(1.0, self.dt / SMOOTH_S)
            self.power[k] = ps
            db = 10.0 * np.log10(ps)
            fl = self.floor[k]
            fl = db if fl is None else fl + (db - fl) * min(1.0, self.dt / BACKGROUND_S)
            self.floor[k] = fl
            out[k] = {"db": db, "above_db": max(0.0, db - fl - DEADBAND_DB)}
        return out

    @staticmethod
    def rate_hz(above_db: float) -> float:
        return MAX_HZ * above_db / (above_db + HALF_DB)


class HearingFeed:
    """Captures the microphone in a thread (arecord pipe) and keeps the latest
    JO-A / JO-B drive. Cheap enough for the simulation process."""

    def __init__(self, device: str | None = None):
        self.device = device or find_mic()
        if self.device is None:
            raise RuntimeError("no capture device found (arecord -l)")
        self.env = BandEnvelope()
        self.state = {"A": 0.0, "B": 0.0, "db_A": -120.0, "db_B": -120.0, "t": 0.0}
        self.proc = subprocess.Popen(
            ["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._stop = False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        nbytes = FRAME * 2
        while not self._stop:
            buf = self.proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            x = np.frombuffer(buf, np.int16).astype(np.float64) / 32768.0
            o = self.env.update(x)
            self.state = {"A": BandEnvelope.rate_hz(o["A"]["above_db"]),
                          "B": BandEnvelope.rate_hz(o["B"]["above_db"]),
                          "db_A": o["A"]["db"], "db_B": o["B"]["db"], "t": time.time()}

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self):
        self._stop = True
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class HearingEncoder:
    """Drives the real JO-A and JO-B neurons (both antennae) from HearingFeed."""

    def __init__(self, connectome, feed: HearingFeed):
        t = connectome.neurons["primary_type"].fillna("").astype(str).to_numpy()
        a = np.flatnonzero(np.char.startswith(t.astype(str), "JO-A"))
        b = np.flatnonzero(np.char.startswith(t.astype(str), "JO-B"))
        self.indices = np.concatenate([a, b])
        self._is_a = np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)])
        order = np.argsort(self.indices)
        self.indices, self._is_a = self.indices[order], self._is_a[order]
        self.feed = feed

    def rates_hz(self, t_ms: float, stim=None) -> np.ndarray:
        s = self.feed.state
        if time.time() - s["t"] > 0.3:                   # no fresh audio
            return np.zeros(len(self.indices))
        return np.where(self._is_a, s["A"], s["B"])

    def state(self, t_ms: float) -> dict:
        s = self.feed.state
        return {"kind": "hearing", "active": s["A"] > 1 or s["B"] > 1,
                "jo_a_hz": round(s["A"], 1), "jo_b_hz": round(s["B"], 1)}

    @property
    def provenance(self) -> dict:
        return {"drives": {"JO-A": int(self._is_a.sum()), "JO-B": int((~self._is_a).sum())},
                "bands_hz": BANDS, "max_hz": MAX_HZ, "half_db": HALF_DB,
                "source": "Kamikouchi et al. 2009 Nature 458:165; Yorozu et al. 2009 Nature 458:201"}


def main():
    ap = argparse.ArgumentParser(description="microphone -> JO-A/JO-B drive")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--secs", type=float, default=8.0)
    a = ap.parse_args()
    feed = HearingFeed()
    print("device", feed.device)
    t0 = time.time()
    while time.time() - t0 < a.secs:
        time.sleep(0.25)
        s = feed.state
        print("JO-B %5.1f Hz (%.0f dB)   JO-A %5.1f Hz (%.0f dB)" % (s["B"], s["db_B"], s["A"], s["db_A"]))
    feed.close()


if __name__ == "__main__":
    main()
