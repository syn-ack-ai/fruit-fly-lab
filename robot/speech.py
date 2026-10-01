"""
Milo hears words: speech recognition on the robot, in its own process.

    microphone (arecord, 16 kHz mono) -> Silero VAD (one segment per utterance)
        -> Parakeet TDT 0.6B v2 (NVIDIA, CC-BY-4.0; int8, sherpa-onnx, CPU) -> text

Parakeet TDT 0.6B v2: 6.05% mean WER on the Hugging Face Open ASR Leaderboard
(English; the best of the models small enough to run beside the fly brain on
an Orin Nano: the ~2-3B leaders need the GPU and memory the brain uses). On
the Orin's CPU, 2 threads, 7.4 s of speech is transcribed in 1.0 s, with
punctuation and capitals; it runs only while someone speaks. The worker is
niced so the brain loop keeps its core.

Audio is never stored; only the recognised text leaves the process. While
Milo itself speaks (robot/voice.py), the ears are muted (Ears.mute).

Models (~/milo/models/asr; not in the repository):
    curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
    tar xjf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
    curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

    python -m robot.speech --file some.wav     # transcribe a file
    python -m robot.speech --test              # print what the microphone hears for 20 s

C. APPROXIMATIONS: one utterance = speech between pauses of >= 0.5 s (at most
MAX_SPEECH_S); words are not streamed while someone is still speaking.
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

RATE = 16000
ASR_DIR = os.path.expanduser(os.environ.get("FLY_ASR_DIR", "~/milo/models/asr"))
ASR_MODEL = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
THREADS = 2
MIN_SILENCE_S = 0.5
MIN_SPEECH_S = 0.25
MAX_SPEECH_S = 15.0
CHUNK = 512                     # Silero VAD's window at 16 kHz


class Transcriber:
    """Parakeet TDT through sherpa-onnx: 16 kHz float samples -> text."""

    def __init__(self, model_dir: str | None = None, threads: int = THREADS):
        import sherpa_onnx
        d = model_dir or os.path.join(ASR_DIR, ASR_MODEL)
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=os.path.join(d, "encoder.int8.onnx"), decoder=os.path.join(d, "decoder.int8.onnx"),
            joiner=os.path.join(d, "joiner.int8.onnx"), tokens=os.path.join(d, "tokens.txt"),
            num_threads=int(threads), model_type="nemo_transducer")
        self.ms = 0.0

    def __call__(self, samples: np.ndarray, rate: int = RATE) -> str:
        t0 = time.perf_counter()
        s = self.rec.create_stream()
        s.accept_waveform(rate, np.asarray(samples, np.float32))
        self.rec.decode_stream(s)
        self.ms = 1e3 * (time.perf_counter() - t0)
        return s.result.text.strip()


def make_vad(path: str | None = None):
    import sherpa_onnx
    c = sherpa_onnx.VadModelConfig()
    c.silero_vad.model = path or os.path.join(ASR_DIR, "silero_vad.onnx")
    c.silero_vad.min_silence_duration = MIN_SILENCE_S
    c.silero_vad.min_speech_duration = MIN_SPEECH_S
    if hasattr(c.silero_vad, "max_speech_duration"):
        c.silero_vad.max_speech_duration = MAX_SPEECH_S
    c.sample_rate = RATE
    return sherpa_onnx.VoiceActivityDetector(c, buffer_size_in_seconds=MAX_SPEECH_S + 15)


def _source(device: str | None, wav: str | None):
    """Chunks of float32 samples at 16 kHz: the microphone, or a WAV file
    (played at its own pace, then silence; for tests)."""
    if wav:
        import wave
        with wave.open(wav) as w:
            if w.getframerate() != RATE or w.getnchannels() != 1:
                raise ValueError(f"{wav}: need 16 kHz mono")
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        x = np.concatenate([x, np.zeros(RATE * 2, np.float32)])
        for i in range(0, len(x), CHUNK):
            time.sleep(CHUNK / RATE)
            yield x[i:i + CHUNK]
        while True:
            time.sleep(CHUNK / RATE)
            yield np.zeros(CHUNK, np.float32)
    from robot.hearing import find_mic
    device = device or find_mic()
    if device is None:
        raise RuntimeError("no microphone (arecord -l)")
    p = subprocess.Popen(["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            buf = p.stdout.read(CHUNK * 2)
            if not buf or len(buf) < CHUNK * 2:
                raise RuntimeError("the microphone stopped (arecord ended)")
            yield np.frombuffer(buf, np.int16).astype(np.float32) / 32768.0
    finally:
        p.terminate()


def _emit(d: dict) -> None:
    sys.stdout.write("@@" + json.dumps(d) + "\n")
    sys.stdout.flush()


def run_worker(device: str | None = None, wav: str | None = None) -> None:
    """The ears' process: utterances in, text out (JSON lines on stdout);
    {"cmd": "mute", "s": seconds} on stdin."""
    try:
        os.nice(5)                                  # the brain loop first
    except OSError:
        pass
    asr, vad = Transcriber(), make_vad()
    mute_until = [0.0]

    def commands():
        for line in sys.stdin:
            try:
                c = json.loads(line)
            except ValueError:
                continue
            if isinstance(c, dict) and c.get("cmd") == "mute":
                mute_until[0] = max(mute_until[0], time.monotonic() + min(float(c.get("s", 0.0)), 30.0))
    threading.Thread(target=commands, daemon=True).start()
    _emit({"ready": True})
    parent = os.getppid()
    muted = False
    for chunk in _source(device, wav):
        if os.getppid() != parent:
            return
        if time.monotonic() < mute_until[0]:
            if not muted:
                vad.reset()                         # drop a half-heard utterance (Milo's own voice)
                muted = True
            continue
        muted = False
        vad.accept_waveform(chunk)
        while not vad.empty():
            seg = np.array(vad.front.samples, np.float32)
            vad.pop()
            text = asr(seg)
            if text:
                _emit({"heard": text, "dur_s": round(len(seg) / RATE, 2), "ms": round(asr.ms, 1),
                       "t": time.time()})


class Ears:
    """Starts the ears' process; collects what it heard; mutes it while Milo speaks."""

    def __init__(self, device: str | None = None, wav: str | None = None):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args = [sys.executable, "-m", "robot.speech", "--worker"]
        if device:
            args += ["--device", device]
        if wav:
            args += ["--wav", wav]
        self.proc = subprocess.Popen(args, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        os.set_blocking(self.proc.stdin.fileno(), False)
        self._heard = collections.deque(maxlen=32)
        self._err = collections.deque(maxlen=64)
        self.ready = False
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
                    self.ready = True
                elif d.get("heard"):
                    self._heard.append(d)
                    self.stats["utterances"] += 1
        except (OSError, ValueError):
            pass

    def _drain(self) -> None:
        try:
            for line in self.proc.stderr:
                self._err.append(line.decode(errors="replace"))
        except (OSError, ValueError):
            pass

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def error(self) -> str:
        return "" if self.alive else "".join(self._err)[-2000:]

    def heard(self) -> list:
        """Utterances since the last call: [{"heard": text, "dur_s", "ms", "t"}]."""
        out = []
        while self._heard:
            out.append(self._heard.popleft())
        return out

    def mute(self, seconds: float) -> None:
        """Do not listen for this long (Milo is speaking). Never blocks."""
        if self.proc.stdin is None or not self.alive:
            return
        try:
            os.write(self.proc.stdin.fileno(), (json.dumps({"cmd": "mute", "s": float(seconds)}) + "\n").encode())
        except (BlockingIOError, OSError, ValueError):
            pass

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--wav", help="(worker) a 16 kHz mono WAV instead of the microphone")
    ap.add_argument("--file", help="transcribe a 16 kHz mono WAV file and exit")
    ap.add_argument("--test", action="store_true", help="print what the microphone hears for 20 s")
    a = ap.parse_args()
    if a.worker:
        run_worker(a.device, a.wav)
        return
    if a.file:
        import wave
        with wave.open(a.file) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
            rate = w.getframerate()
        asr = Transcriber()
        print(repr(asr(x, rate)), f"({len(x) / rate:.1f} s in {asr.ms:.0f} ms)")
        return
    if a.test:
        ears = Ears(a.device)
        t0 = time.time()
        while time.time() - t0 < 20 and ears.alive:
            time.sleep(0.2)
            for h in ears.heard():
                print(f"heard ({h['dur_s']} s, {h['ms']:.0f} ms): {h['heard']}", flush=True)
        print("ears:", "ready" if ears.ready else "not ready", ears.error())
        ears.close()


if __name__ == "__main__":
    main()
