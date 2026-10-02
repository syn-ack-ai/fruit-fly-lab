"""
Milo's voice: words and robot sounds from the rover's own speaker, in its own
process.

    text -> sentences -> Pocket TTS (Kyutai; int8, sherpa-onnx, CPU) in the
    voice of a reference recording -> robot_fx (higher, a little metallic;
    robot/voice_fx.py) -> aplay -> the speaker

Pocket TTS (100M parameters, CC-BY-4.0) copies the voice of a short reference
recording: Stuart Bell from Kyutai's voice-zero set (CC0). On the Orin's CPU,
2 threads, it speaks faster than real time (a 6 s sentence in ~3.5 s), so the
first sentence plays while the next is made. Sounds (beep, chirp, curious,
startle ... and the fly brain's own song) are synthesised by robot/sounds.py. The parent mutes the ears while
Milo speaks (the "speaking" messages say for how long), so it does not hear
itself.

Models (~/milo/models/tts; not in the repository):
    curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
    tar xjf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
    mkdir -p voices && curl -L -o voices/stuart_bell.wav \\
        https://huggingface.co/kyutai/tts-voices/resolve/main/voice-zero/stuart_bell.wav

The speaker: $FLY_SPEAKER (an ALSA device, e.g. plughw:3,0), else the first
USB playback device, else "null" (no sound; for tests).

    python -m robot.voice --say "Hello, I'm Milo."        # speak once
    python -m robot.voice --say "Hello" --wav out.wav     # write a file instead
"""
from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

TTS_DIR = os.path.expanduser(os.environ.get("FLY_TTS_DIR", "~/milo/models/tts"))
POCKET = "sherpa-onnx-pocket-tts-int8-2026-01-26"
VOICE = os.environ.get("FLY_VOICE_REF", "stuart_bell")
STYLE = os.environ.get("FLY_VOICE_STYLE", "tin")           # robot/voice_fx.PRESETS (chosen 2026-10-01)
THREADS = 2
MAX_CHARS = 300
from robot.sounds import NAMES as SOUNDS          # robot/sounds.py: beep ... startle, curious, happy ...


def find_speaker() -> str:
    """$FLY_SPEAKER, else the first USB playback device, else "null"."""
    env = os.environ.get("FLY_SPEAKER")
    if env:
        return env
    try:
        out = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "null"
    for line in out.splitlines():
        m = re.match(r"card (\d+): .*?\[(.*?)\], device (\d+):", line)
        if m and ("USB" in line or "usb" in m.group(2).lower()):
            return f"plughw:{m.group(1)},{m.group(3)}"
    return "null"


def sentences(text: str) -> list:
    """Speakable pieces: sentences (not after "Dr." or "e.g."), and long ones
    split at commas."""
    text = " ".join(str(text or "").split())[:MAX_CHARS]
    out = []
    for s in re.split(r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.!?])\s+", text):
        while len(s) > 120 and "," in s[60:]:
            k = s.index(",", 60) + 1
            out.append(s[:k].strip())
            s = s[k:].strip()
        if s.strip():
            out.append(s.strip())
    return out


def synth_sound(name: str, sr: int = 24000) -> np.ndarray:
    """A robot sound (robot/sounds.py: a small wordless voice, servo whirrs;
    the personality's set, moods, the startle and the fly brain's song)."""
    from robot.sounds import make
    return make(name, sr)


def _emit(d: dict) -> None:
    sys.stdout.write("@@" + json.dumps(d) + "\n")
    sys.stdout.flush()


class Speaker:
    """A long-lived aplay fed raw 16-bit mono; a writer thread plays queued
    pieces in order. Each piece is reported as it is written with how long
    the queued audio will still play (a playback clock: aplay buffers ahead),
    so the ears stay muted until Milo has finished. A failed aplay (speaker
    unplugged) is restarted, at most every few seconds."""

    def __init__(self, device: str, sr: int, on_start=None, wav_path: str | None = None):
        import queue
        self.sr, self.on_start, self.device = sr, on_start, device
        self.q = queue.Queue()
        self.pending = 0                          # queued, not yet written
        self.end = 0.0                            # monotonic time the written audio ends
        self.errors = 0
        self.wav = None
        self.proc = None
        if wav_path:
            import wave
            self.wav = wave.open(wav_path, "wb")
            self.wav.setnchannels(1); self.wav.setsampwidth(2); self.wav.setframerate(sr)
        else:
            self._open()
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self) -> None:
        self.proc = subprocess.Popen(["aplay", "-q", "-D", self.device, "-f", "S16_LE", "-r", str(self.sr), "-c", "1",
                                      "-t", "raw"], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def play(self, x: np.ndarray, label: str = "") -> None:
        self.pending += 1
        self.q.put((np.clip(np.asarray(x, np.float32), -1, 1), label))

    def _run(self) -> None:
        last_fail = -1e9
        while True:
            item = self.q.get()
            if item is None:
                return
            x, label = item
            dur = len(x) / self.sr
            now = time.monotonic()
            self.end = max(self.end, now) + dur
            if self.on_start:
                self.on_start(self.end - now, label)
            pcm = (x * 32767).astype(np.int16).tobytes()
            try:
                if self.wav is not None:
                    self.wav.writeframes(pcm)
                else:
                    if self.proc is None or self.proc.poll() is not None:
                        if now - last_fail < 5.0:
                            raise OSError("speaker unavailable")
                        self._open()
                    self.proc.stdin.write(pcm)
                    self.proc.stdin.flush()
            except (OSError, ValueError) as ex:
                self.errors += 1
                last_fail = now
                self.end = now                    # nothing is playing
                print(f"speaker error ({self.device}): {ex}", file=sys.stderr, flush=True)
                try:
                    if self.proc is not None:
                        self.proc.kill()
                except OSError:
                    pass
                self.proc = None
            finally:
                self.pending -= 1

    def idle(self) -> bool:
        return self.pending <= 0

    def close(self, wait_s: float = 30.0) -> None:
        deadline = time.monotonic() + wait_s
        while not self.idle() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.q.put(None)
        if self.wav is not None:
            self.wav.close()
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=max(0.5, min(10.0, self.end - time.monotonic() + 0.5)))
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()


class Mouth:
    """Pocket TTS in the reference voice, made robotic."""

    def __init__(self, voice: str = VOICE, style: str = STYLE, threads: int = THREADS):
        import sherpa_onnx as s
        import wave
        d = os.path.join(TTS_DIR, POCKET)
        self.tts = s.OfflineTts(s.OfflineTtsConfig(model=s.OfflineTtsModelConfig(pocket=s.OfflineTtsPocketModelConfig(
            decoder=f"{d}/decoder.int8.onnx", encoder=f"{d}/encoder.onnx", lm_flow=f"{d}/lm_flow.int8.onnx",
            lm_main=f"{d}/lm_main.int8.onnx", text_conditioner=f"{d}/text_conditioner.onnx",
            token_scores_json=f"{d}/token_scores.json", vocab_json=f"{d}/vocab.json"), num_threads=threads)))
        ref = voice if voice.endswith(".wav") else os.path.join(TTS_DIR, "voices", voice + ".wav")
        with wave.open(ref) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
            ch, self.ref_sr = w.getnchannels(), w.getframerate()
        self.ref = x[::ch][: self.ref_sr * 10].tolist()
        from robot.voice_fx import PRESETS
        self.fx = PRESETS.get(style) if style != "none" else None
        self.sr = 24000

    def speak(self, text: str) -> np.ndarray:
        import sherpa_onnx as s
        from robot.voice_fx import robotize
        g = s.GenerationConfig()
        g.reference_audio, g.reference_sample_rate = self.ref, self.ref_sr
        a = self.tts.generate(text, g)
        self.sr = a.sample_rate
        y = np.nan_to_num(np.asarray(a.samples, np.float32))
        return robotize(y, self.sr, **self.fx) if self.fx else y


def run_worker(device: str | None = None, wav_path: str | None = None) -> None:
    """The voice's process: {"say": text} / {"sound": name} on stdin; "speaking"
    (seconds, as each piece starts) and "done" on stdout."""
    try:
        os.nice(5)                                  # the brain loop first
    except OSError:
        pass
    mouth = Mouth()
    spk = Speaker(device or find_speaker(), mouth.tts.sample_rate,
                  on_start=lambda dur, label: _emit({"speaking": round(dur, 2), "what": label}), wav_path=wav_path)
    cmds = collections.deque()
    ev = threading.Event()

    def read():
        for line in sys.stdin:
            try:
                c = json.loads(line)
            except ValueError:
                continue
            if isinstance(c, dict):
                cmds.append(c)
                ev.set()
        cmds.append({"quit": True})
        ev.set()
    threading.Thread(target=read, daemon=True).start()
    _emit({"ready": True, "device": device or find_speaker()})
    parent = os.getppid()
    while os.getppid() == parent:
        ev.wait(1.0)
        ev.clear()
        while cmds:
            c = cmds.popleft()
            if c.get("quit"):
                spk.close(wait_s=30.0)               # finish what is being said, within a limit
                return
            if c.get("sound") in SOUNDS:
                spk.play(synth_sound(c["sound"], mouth.sr), "sound:" + c["sound"])
            if c.get("say"):
                for s in sentences(c["say"]):
                    t0 = time.perf_counter()
                    try:
                        y = mouth.speak(s)
                    except Exception as ex:            # one bad sentence is not the end of the voice
                        print("voice error:", ex, file=sys.stderr, flush=True)
                        continue
                    _emit({"made": round(len(y) / mouth.sr, 2), "ms": round(1e3 * (time.perf_counter() - t0))})
                    spk.play(y, s[:40])
    spk.close()


class Voice:
    """Starts the voice's process; say() and sound() never block the caller."""

    def __init__(self, device: str | None = None, wav_path: str | None = None):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args = [sys.executable, "-m", "robot.voice", "--worker"]
        if device:
            args += ["--device", device]
        if wav_path:
            args += ["--wav", wav_path]
        self.proc = subprocess.Popen(args, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        os.set_blocking(self.proc.stdin.fileno(), False)
        self._speaking = collections.deque(maxlen=64)
        self._err = collections.deque(maxlen=64)
        self.ready = False
        self.device = None
        self.stats = collections.Counter()
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()

    def _read(self) -> None:
        try:
            for line in self.proc.stdout:
                if not line.startswith(b"@@"):
                    continue
                try:
                    d = json.loads(line[2:])
                except ValueError:
                    continue
                if d.get("ready"):
                    self.ready, self.device = True, d.get("device")
                elif "speaking" in d:
                    self._speaking.append(float(d["speaking"]))
                    self.stats["pieces"] += 1
        except (OSError, ValueError):
            pass

    def _drain(self) -> None:
        try:
            for line in self.proc.stderr:
                self._err.append(line.decode(errors="replace"))
        except (OSError, ValueError):
            pass

    def _send(self, d: dict) -> None:
        """Never blocks the brain loop: a worker that stopped reading loses the message."""
        if self.proc.stdin is None or self.proc.poll() is not None:
            return
        try:
            os.write(self.proc.stdin.fileno(), (json.dumps(d) + "\n").encode())
        except (BlockingIOError, OSError, ValueError):
            self.stats["dropped"] += 1

    def say(self, text: str) -> None:
        if text:
            self.stats["said"] += 1
            self._send({"say": str(text)[:MAX_CHARS]})

    def sound(self, name: str) -> None:
        if name in SOUNDS:
            self._send({"sound": name})

    def speaking(self) -> list:
        """Durations (s) of pieces that started playing since the last call."""
        out = []
        while self._speaking:
            out.append(self._speaking.popleft())
        return out

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def error(self) -> str:
        return "" if self.alive else "".join(self._err)[-2000:]

    def close(self, wait_s: float = 5.0) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()               # the worker finishes what it is saying
                self.proc.wait(timeout=wait_s)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--wav", help="write to a WAV file instead of the speaker")
    ap.add_argument("--say")
    ap.add_argument("--sound", choices=SOUNDS)
    a = ap.parse_args()
    if a.worker:
        run_worker(a.device, a.wav)
        return
    v = Voice(a.device, a.wav)
    if a.sound:
        v.sound(a.sound)
    if a.say:
        v.say(a.say)
    v.close(wait_s=60.0)
    print("voice:", v.device, dict(v.stats), v.error())


if __name__ == "__main__":
    main()
