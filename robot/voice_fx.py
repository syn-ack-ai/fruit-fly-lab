"""
Make a voice sound like a little robot: higher (pitch shifted, same speed) and
a little metallic (ring modulation, a short comb resonance, light bit-crush).
numpy + scipy only; a sentence takes a few milliseconds.

    y = robotize(x, sr, **PRESETS["little"])
"""
from __future__ import annotations

import numpy as np

PRESETS = {
    # higher and slightly metallic: a small friendly robot
    "little": {"semitones": 5.0, "ring_hz": 0.0, "ring_mix": 0.0, "comb_ms": 3.0, "comb_fb": 0.35, "crush_bits": 0},
    # higher, with a gentle ring-modulated buzz
    "buzzy": {"semitones": 5.0, "ring_hz": 70.0, "ring_mix": 0.25, "comb_ms": 2.5, "comb_fb": 0.3, "crush_bits": 0},
    # tin-can: a stronger resonance and a little crunch
    "tin": {"semitones": 4.0, "ring_hz": 0.0, "ring_mix": 0.0, "comb_ms": 1.6, "comb_fb": 0.55, "crush_bits": 10},
    # very small robot
    "tiny": {"semitones": 8.0, "ring_hz": 45.0, "ring_mix": 0.15, "comb_ms": 2.0, "comb_fb": 0.3, "crush_bits": 0},
}


def time_stretch(x: np.ndarray, factor: float, sr: int, frame_ms: float = 40.0, hop_ms: float = 10.0,
                 search_ms: float = 8.0) -> np.ndarray:
    """WSOLA: x played `factor` times as long, pitch unchanged."""
    if abs(factor - 1.0) < 1e-6:
        return x.copy()
    n, hop_out = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    hop_in, search = hop_out / factor, int(sr * search_ms / 1000)
    win = np.hanning(n).astype(np.float32)
    out_len = int(len(x) * factor) + n
    y = np.zeros(out_len, np.float32)
    norm = np.zeros(out_len, np.float32)
    xp = np.concatenate([x, np.zeros(n + 2 * search + 1, np.float32)]).astype(np.float32)
    prev = None
    k = 0
    while True:
        pos_out = k * hop_out
        centre = int(round(k * hop_in))
        if centre >= len(x) or pos_out + n > out_len:
            break
        best = centre
        if prev is not None:
            # the shift within +-search that best continues the previous frame
            lo, hi = max(0, centre - search), centre + search
            target = prev
            seg = xp[lo:hi + n]
            if len(seg) >= n:
                c = np.correlate(seg, target, mode="valid")
                best = lo + int(np.argmax(c))
        frame = xp[best:best + n]
        y[pos_out:pos_out + n] += frame * win
        norm[pos_out:pos_out + n] += win
        prev = xp[best + hop_out:best + hop_out + n] if best + hop_out + n <= len(xp) else xp[best:best + n]
        k += 1
    y = y / np.maximum(norm, 1e-3)
    return y[:int(len(x) * factor)]


def pitch_shift(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Higher (or lower) by `semitones`, same duration."""
    if not semitones:
        return x.copy()
    from scipy.signal import resample_poly
    f = 2.0 ** (semitones / 12.0)
    stretched = time_stretch(x, f, sr)                    # longer by f ...
    up, down = 1000, int(round(1000 * f))
    return resample_poly(stretched, up, down).astype(np.float32)   # ... then played f times faster


def robotize(x: np.ndarray, sr: int, semitones: float = 5.0, ring_hz: float = 0.0, ring_mix: float = 0.0,
             comb_ms: float = 3.0, comb_fb: float = 0.35, crush_bits: int = 0) -> np.ndarray:
    y = pitch_shift(np.asarray(x, np.float32), sr, semitones)
    if ring_hz and ring_mix:
        t = np.arange(len(y)) / sr
        y = (1 - ring_mix) * y + ring_mix * y * np.sin(2 * np.pi * ring_hz * t).astype(np.float32)
    if comb_ms and comb_fb:
        from scipy.signal import lfilter
        d = max(1, int(sr * comb_ms / 1000))
        a = np.zeros(d + 1)
        a[0], a[d] = 1.0, -comb_fb                         # y[n] = x[n] + fb * y[n - d]
        y = lfilter([1.0 - comb_fb], a, y).astype(np.float32)
    if crush_bits:
        q = 2.0 ** (crush_bits - 1)
        y = np.round(y * q) / q
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y * (0.9 / peak) if peak > 0.9 else y).astype(np.float32)
