"""robot/voice.py and robot/speech.py without models or audio devices."""
import numpy as np

from robot import voice as V
from robot import voice_fx as FX


def test_sentences_split_for_streaming():
    assert V.sentences("Hi! I don't think we've met. What's your name?") == \
        ["Hi!", "I don't think we've met.", "What's your name?"]
    long = "I am a little robot, " * 12
    parts = V.sentences(long)
    assert len(parts) > 1 and all(len(p) <= 140 for p in parts)
    assert V.sentences("  ") == [] and len(" ".join(V.sentences("x" * 999))) <= V.MAX_CHARS


def test_robot_voice_effect_keeps_length_and_level():
    sr = 24000
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)          # a voiced "ah" at 180 Hz
    for name, p in FX.PRESETS.items():
        y = FX.robotize(x, sr, **p)
        assert abs(len(y) - len(x)) < sr * 0.02, name                    # same duration
        assert np.isfinite(y).all() and np.abs(y).max() <= 0.91, name
    y = FX.pitch_shift(x, sr, 12.0)                                      # an octave up: ~360 Hz
    spec = np.abs(np.fft.rfft(y[2000:-2000] * np.hanning(len(y) - 4000)))
    f = np.argmax(spec) * sr / (len(y) - 4000)
    assert 330 < f < 390


def test_find_speaker_prefers_env(monkeypatch):
    monkeypatch.setenv("FLY_SPEAKER", "plughw:9,0")
    assert V.find_speaker() == "plughw:9,0"
