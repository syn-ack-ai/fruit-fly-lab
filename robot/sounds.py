"""
Milo's robot sounds: a small synthesised "voice" without words, servo whirrs
and clunks -- expressive robot noises in the spirit of film robots (WALL-E,
R2-D2), made from scratch here (no samples).

    voice   a pulse source (harmonics, pitch contour with vibrato and a little
            jitter) shaped by vowel formants that glide (oo -> ah -> ee ...):
            the harmonics are weighted by the formants' resonance at each
            moment (additive synthesis), so pitch and vowel move smoothly; a
            touch of ring modulation and saturation makes it electronic
    whistle sine glides with vibrato (chirps, trills)
    servo   a motor spinning up or down: gear-whine harmonics and filtered
            noise, amplitude-modulated at the gear rate
    clunk   a short metallic knock (inharmonic damped partials)

Each sound is a phrase of these; make(name) returns float32 mono in -1..1.

    python -m robot.sounds --play             # all of them (macOS afplay / Linux aplay)
    python -m robot.sounds --out DIR          # write WAV files
"""
from __future__ import annotations

import numpy as np

SR = 24000
# vowel formants (F1, F2, F3) of a small voice: an adult's x 1.3
_VOWELS = {k: tuple(1.3 * np.array(v)) for k, v in {
    "oo": (300, 870, 2240), "oh": (450, 800, 2830), "ah": (730, 1090, 2440),
    "uh": (600, 1170, 2390), "eh": (530, 1840, 2480), "ee": (280, 2250, 2900),
    "mm": (250, 1000, 2200)}.items()}
_BW = (90.0, 130.0, 200.0)                   # formant bandwidths, Hz
_GAIN = (1.0, 0.6, 0.25)


def _ease(a, b, n, shape="cos"):
    x = np.linspace(0.0, 1.0, n)
    if shape == "cos":
        x = 0.5 - 0.5 * np.cos(np.pi * x)
    elif shape == "fast":
        x = 1.0 - (1.0 - x) ** 3
    elif shape == "slow":
        x = x ** 2
    return a + (b - a) * x


def _env(n, attack=0.02, release=0.06, sr=SR):
    t = np.arange(n) / sr
    dur = n / sr
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.minimum(1.0, (dur - t) / max(release, 1e-4)).clip(0, 1)


def voice(syllables, sr=SR, rng=None, vibrato=(5.5, 0.025), breath=0.03, buzz=0.12):
    """syllables: [(dur_s, f0_start, f0_end, vowel_start, vowel_end, amp[, shape])].
    Returns one phrase (syllables joined smoothly)."""
    rng = rng or np.random.default_rng(0)
    f0, F, amp = [], [], []
    for syl in syllables:
        dur, fa, fb, va, vb, a = syl[:6]
        shape = syl[6] if len(syl) > 6 else "cos"
        n = int(sr * dur)
        f0.append(_ease(np.log(fa), np.log(fb), n, shape))
        F.append(np.stack([_ease(_VOWELS[va][i], _VOWELS[vb][i], n) for i in range(3)], 1))
        amp.append(a * _env(n, 0.015, min(0.05, dur / 3), sr))
    f0 = np.exp(np.concatenate(f0))
    F = np.concatenate(F)
    amp = np.concatenate(amp)
    n = len(f0)
    t = np.arange(n) / sr
    f0 = f0 * (1.0 + vibrato[1] * np.sin(2 * np.pi * vibrato[0] * t)
               + 0.006 * np.convolve(rng.standard_normal(n), np.ones(200) / 200, "same") * 14)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    out = np.zeros(n)
    kmax = int((sr / 2 - 500) / f0.min())
    for k in range(1, min(kmax, 40) + 1):
        fk = k * f0
        ok = fk < sr / 2 - 500
        res = sum(g / (1.0 + ((fk - F[:, i]) / _BW[i]) ** 2) for i, g in enumerate(_GAIN))
        out += ok * (res + 0.004) * np.sin(k * phase) / k ** 0.6
    out /= max(np.abs(out).max(), 1e-9)
    out += breath * np.convolve(rng.standard_normal(n), [0.5, 0.5], "same")
    # electronic colour: ring-modulated copy and a soft square edge
    out = (1 - buzz) * out + buzz * out * np.sin(2 * np.pi * 2.0 * f0.mean() * t)
    return np.tanh(1.6 * out * amp) / np.tanh(1.6)


def whistle(dur, fa, fb, sr=SR, vib=(9.0, 0.03), amp=0.5, shape="cos"):
    n = int(sr * dur)
    t = np.arange(n) / sr
    f = np.exp(_ease(np.log(fa), np.log(fb), n, shape)) * (1 + vib[1] * np.sin(2 * np.pi * vib[0] * t))
    ph = 2 * np.pi * np.cumsum(f) / sr
    return amp * (np.sin(ph) + 0.15 * np.sin(2 * ph)) * _env(n, 0.008, min(0.04, dur / 3), sr)


def servo(dur, fa, fb, sr=SR, amp=0.35, rng=None, gear=0.18):
    rng = rng or np.random.default_rng(1)
    n = int(sr * dur)
    t = np.arange(n) / sr
    f = _ease(fa, fb, n, "cos")
    ph = 2 * np.pi * np.cumsum(f) / sr
    whine = sum(np.sin(k * ph) / k for k in (1, 2, 3, 5))
    noise = np.convolve(rng.standard_normal(n), np.ones(6) / 6, "same")
    am = 1.0 + gear * np.sin(ph / 6.0)                 # the gear teeth
    x = (0.6 * whine + 0.5 * noise * (0.5 + 0.5 * np.sin(ph))) * am
    return amp * x / max(np.abs(x).max(), 1e-9) * _env(n, 0.03, 0.08, sr)


def clunk(sr=SR, amp=0.5, f=420.0, dur=0.12):
    n = int(sr * dur)
    t = np.arange(n) / sr
    x = sum(a * np.exp(-t * d) * np.sin(2 * np.pi * f * r * t)
            for r, a, d in ((1.0, 1.0, 45), (2.76, 0.5, 70), (5.4, 0.25, 110), (8.9, 0.12, 160)))
    return amp * x / max(np.abs(x).max(), 1e-9)


def gap(d, sr=SR):
    return np.zeros(int(sr * d))


def _mix(*parts):
    """Overlapping parts: (offset_s, signal)."""
    n = max(int(SR * o) + len(x) for o, x in parts)
    out = np.zeros(n)
    for o, x in parts:
        i = int(SR * o)
        out[i:i + len(x)] += x
    return out


def _seq(*xs):
    return np.concatenate(xs)


def _phrases():
    return {
        # the personality's set (cortex/personality.SOUNDS)
        "beep": lambda: _seq(whistle(0.07, 1500, 1650, amp=0.4), gap(0.03), voice([(0.12, 620, 700, "eh", "ee", 0.8)])),
        "boop": lambda: voice([(0.22, 420, 300, "oo", "oo", 0.9, "fast")], vibrato=(4, 0.01)),
        "chirp": lambda: _seq(voice([(0.10, 500, 900, "uh", "ee", 0.8, "slow")]), whistle(0.08, 1800, 2600, amp=0.3)),
        "trill": lambda: voice([(0.45, 700, 820, "ee", "eh", 0.8)], vibrato=(17, 0.09)),
        "whirr": lambda: _seq(servo(0.35, 180, 520), servo(0.25, 520, 260, amp=0.25)),
        "buzz": lambda: voice([(0.12, 260, 300, "mm", "mm", 0.6), (0.30, 300, 190, "mm", "uh", 0.9, "slow")],
                              buzz=0.45, vibrato=(25, 0.03)),
        "song": lambda: voice([(0.16, 520, 600, "ah", "ah", 0.8), (0.12, 700, 780, "ee", "ee", 0.8),
                               (0.16, 600, 660, "ah", "eh", 0.8), (0.30, 820, 620, "oo", "oo", 0.9)],
                              vibrato=(6.5, 0.035)),
        # moods and moments
        "startle": lambda: _mix((0.0, servo(0.12, 300, 900, amp=0.35, gear=0.4)), (0.0, clunk(amp=0.35)),
                                (0.06, voice([(0.07, 420, 1100, "oh", "ah", 1.0, "fast"),
                                              (0.20, 1100, 520, "ah", "oh", 0.8, "slow")], vibrato=(9, 0.04)))),
        "curious": lambda: _seq(voice([(0.20, 330, 360, "mm", "mm", 0.6)]), gap(0.06),
                                voice([(0.12, 420, 460, "oo", "oo", 0.8), (0.22, 460, 860, "oo", "ee", 0.9, "slow")])),
        "happy": lambda: _seq(voice([(0.09, 600, 760, "ee", "ee", 0.8)]), gap(0.03),
                              voice([(0.09, 680, 860, "ee", "ee", 0.8)]), gap(0.03),
                              voice([(0.22, 780, 1050, "ee", "eh", 0.9, "fast")], vibrato=(8, 0.05)),
                              whistle(0.08, 2000, 2700, amp=0.2)),
        "sad": lambda: voice([(0.25, 520, 560, "ah", "ah", 0.8), (0.55, 560, 300, "ah", "oo", 0.8, "slow")],
                             vibrato=(5, 0.05)),
        "greet": lambda: _seq(voice([(0.13, 560, 720, "ah", "ee", 0.9)]), gap(0.04),
                              voice([(0.10, 900, 820, "ee", "ee", 0.8), (0.22, 820, 1000, "ee", "ee", 0.9, "fast")])),
        "sleepy": lambda: _mix((0.0, voice([(0.35, 380, 480, "ah", "ah", 0.7, "slow"),
                                            (0.6, 480, 230, "ah", "oo", 0.7, "slow")], vibrato=(4, 0.02))),
                               (0.75, servo(0.5, 400, 90, amp=0.2))),
        "wow": lambda: voice([(0.12, 380, 420, "oo", "oo", 0.8), (0.40, 420, 900, "oo", "ah", 0.9, "slow"),
                              (0.25, 900, 600, "ah", "oo", 0.8)], vibrato=(6, 0.04)),
        "uhoh": lambda: _seq(voice([(0.16, 620, 640, "uh", "uh", 0.9)]), gap(0.05),
                             voice([(0.28, 470, 380, "oh", "oh", 0.9, "slow")])),
    }


NAMES = tuple(_phrases())


def make(name: str, sr: int = SR) -> np.ndarray:
    """A sound as float32 mono, peak 0.8, at `sr` (resampled from SR)."""
    f = _phrases().get(name)
    if f is None:
        return np.zeros(0, np.float32)
    x = f()
    # a small speaker: nothing below ~150 Hz (first-order high-pass)
    a = np.exp(-2 * np.pi * 150 / SR)
    y = np.empty_like(x)
    prev_x = prev_y = 0.0
    for i, v in enumerate(x):
        prev_y = a * (prev_y + v - prev_x)
        prev_x = v
        y[i] = prev_y
    y = 0.8 * y / max(np.abs(y).max(), 1e-9)
    if sr != SR:
        n = int(len(y) * sr / SR)
        y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
    return y.astype(np.float32)


def write_wav(path: str, x: np.ndarray, sr: int = SR) -> None:
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    import argparse
    import os
    import shutil
    import subprocess
    import tempfile
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(NAMES))
    ap.add_argument("--out", help="write NAME.wav files here")
    ap.add_argument("--play", action="store_true")
    a = ap.parse_args()
    d = a.out or tempfile.mkdtemp(prefix="milo_sounds_")
    os.makedirs(d, exist_ok=True)
    player = shutil.which("afplay") or shutil.which("aplay")
    for name in a.names:
        p = os.path.join(d, f"{name}.wav")
        write_wav(p, make(name))
        if a.play and player:
            print(name, flush=True)
            subprocess.run([player, p], check=False)
            time.sleep(0.4)
    print("wrote", d)


if __name__ == "__main__":
    main()
