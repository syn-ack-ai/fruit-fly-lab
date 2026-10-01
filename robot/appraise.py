"""
Is there real danger? The vision-language model looks at the camera.

The fly's looming circuit is driven on the robot by meaning, not geometry
(robot/threat.py). This is the meaning: about once a second (APPRAISE_S) the
latest clean camera frame (robot/head.py) goes to the language model the
personality uses (Gemma 4 E4B through PAIR; it reads images, ~0.3 s a frame
on the Mac Studio, 2026-10-01), with a narrow question: is something here a
real danger to a small wheeled robot? People and things nearby or coming
closer are not. The answer {"fear": 0 / 1 / 2, "what", "where"} becomes a
Fear appraisal (level robot/threat.LEVELS) aimed at that side of the camera's
view; the personality is told what was seen so its mood and words follow.

Calls run in a background thread; a frame taken while the head moves is not
sent (blurred). If the model is unreachable nothing is appraised (no fear),
and errors are counted.

C. APPROXIMATIONS: robot engineering. ~1 s per look plus the reply's latency:
this is context fear (a dog in the room, something falling over, smoke), not
a reflex to a kick. Like the personality, the camera can be shown anything:
a false "danger" makes Milo startle (stop, back away), which the safety layers
below still limit.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import urllib.request

from robot.threat import LEVELS

APPRAISE_S = 1.0               # one look this often (wall clock)
FRAME_MAX_AGE_S = 0.5          # an older frame (the camera stalled) is not sent
WHERE_FRAC = {"left": -1.0 / 3.0, "ahead": 0.0, "right": 1.0 / 3.0}

PROMPT = ("You are the eyes of {name}, a small wheeled home robot about 25 cm tall, looking through its "
          "camera. Judge only REAL danger to the robot right now: being kicked, stepped on or hit, "
          "something thrown or falling toward it, an animal attacking it, water, fire or smoke, an edge "
          "it could fall from. People, pets and things that are simply there, moving, or coming closer "
          "are NOT danger; neither is being looked at, picked up gently or played with.\n"
          "Reply with ONLY one JSON object:\n"
          '{{"fear": 0 (no danger) or 1 (something to be wary of) or 2 (danger now), '
          '"what": a few words naming what you see that matters, '
          '"where": "left" or "ahead" or "right" or null}}')


def parse(text: str) -> dict | None:
    a = text.find("{")
    if a < 0:
        return None
    try:
        r, _ = json.JSONDecoder().raw_decode(text[a:])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(r, dict):
        return None
    try:
        fear = int(r.get("fear", 0))
    except (TypeError, ValueError):
        fear = 0
    where = r.get("where") if r.get("where") in WHERE_FRAC else None
    what = r.get("what")
    return {"fear": fear if fear in LEVELS else 0, "where": where,
            "what": str(what)[:60] if what not in (None, "", "null") else ""}


class VisionAppraiser:
    """frame_fn() -> (seq, jpeg bytes, pan_deg, tilt_deg, moving, wall time) or None
    (robot.head.HeadFeed.clean_jpeg)."""

    def __init__(self, url: str, model: str, frame_fn, hfov_deg: float, name: str = "Milo",
                 timeout_s: float = 10.0, period_s: float = APPRAISE_S):
        self.url, self.model, self.frame_fn, self.hfov = url, model, frame_fn, float(hfov_deg)
        self.system = PROMPT.format(name=name)
        self.timeout, self.period = timeout_s, period_s
        self._lock = threading.Lock()
        self._pending = None
        self._reply = None
        self._last_seq = None
        self._next = 0.0
        self.last = {}
        self.stats = {"looks": 0, "failures": 0, "latency_s": 0.0, "fear1": 0, "fear2": 0}

    def step(self, t_s: float, fear, personality=None) -> dict | None:
        """Once per control step: hand a finished appraisal to `fear` (and the
        personality), and start the next look when due. Returns the new
        appraisal, if one arrived."""
        new = None
        with self._lock:
            rep, self._reply = self._reply, None
        if rep is not None:
            r, pan, lat = rep
            self.stats["latency_s"] += lat
            self.last = dict(r, latency_s=round(lat, 2))
            new = self.last
            if r["fear"]:
                self.stats[f"fear{r['fear']}"] += 1
                frac = WHERE_FRAC.get(r["where"], 0.0)
                fear.appraise("vision", LEVELS[r["fear"]], t_s, azimuth_deg=pan + frac * self.hfov,
                              what=r["what"])
                if personality is not None:
                    kind = "DANGER" if r["fear"] == 2 else "something to be wary of"
                    personality.event(t_s, f"your camera sees {kind}: {r['what'] or 'something'}"
                                           f"{' on the ' + r['where'] if r['where'] in ('left', 'right') else ''}",
                                      urge=r["fear"] == 2)
        now = time.monotonic()
        if self._pending is None or not self._pending.is_alive():
            if now >= self._next:
                f = self.frame_fn()
                if (f is not None and f[0] != self._last_seq and not f[4]
                        and time.time() - f[5] <= FRAME_MAX_AGE_S):
                    self._last_seq = f[0]
                    self._next = now + self.period
                    self._pending = threading.Thread(target=self._ask, args=(f[1], f[2]), daemon=True)
                    self._pending.start()
        return new

    def _ask(self, jpeg: bytes, pan: float) -> None:
        body = {"model": self.model, "temperature": 0, "max_tokens": 60,
                "messages": [{"role": "system", "content": self.system},
                             {"role": "user", "content": [
                                 {"type": "text", "text": "What the camera sees now."},
                                 {"type": "image_url",
                                  "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}]}]}
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as fh:
                text = json.load(fh)["choices"][0]["message"]["content"]
            r = parse(text)
        except Exception:
            r = None
        lat = time.monotonic() - t0
        self.stats["looks"] += 1
        if r is None:
            self.stats["failures"] += 1
            return
        with self._lock:
            self._reply = (r, pan, lat)

    def summary(self) -> dict:
        n = max(self.stats["looks"] - self.stats["failures"], 1)
        return {**{k: v for k, v in self.stats.items() if k != "latency_s"},
                "mean_latency_s": round(self.stats["latency_s"] / n, 2), "last": self.last}


def main():
    """Watch the appraisals live: python -m robot.appraise --camera /dev/video0"""
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default=None, help="V4L2 device (default: find the Orbit)")
    ap.add_argument("--url", default=os.environ.get("FLY_LLM_URL", "http://127.0.0.1:1234/v1/chat/completions"))
    ap.add_argument("--model", default="gemma-4-e4b-it-mlx")
    ap.add_argument("--hfov", type=float, default=53.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--save", default=None, help="keep each frame Gemma judged afraid here")
    a = ap.parse_args()
    from robot.head import HeadFeed
    from robot.threat import Fear
    feed = HeadFeed(a.camera, person=False, looming=False)
    seen = {}

    def frame():
        f = feed.clean_jpeg()
        if f is not None:
            seen[f[0]] = f[1]
            if len(seen) > 8:
                seen.pop(next(iter(seen)))
        return f
    vis = VisionAppraiser(a.url, a.model, frame, a.hfov)
    fear = Fear()
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < a.seconds:
            t = time.monotonic() - t0
            r = vis.step(t, fear)
            if r is not None:
                L = fear.update(t)
                print(f"{t:6.1f} s  fear {r['fear']}  {r['what']!r:40} where {r['where']}  "
                      f"({r['latency_s']:.2f} s)  -> looming drive {L.get('level', 0.0)}", flush=True)
                if a.save and r["fear"] and vis._last_seq in seen:
                    os.makedirs(a.save, exist_ok=True)
                    with open(os.path.join(a.save, f"fear{r['fear']}_{t:06.1f}.jpg"), "wb") as fh:
                        fh.write(seen[vis._last_seq])
            time.sleep(0.05)
    finally:
        feed.close()
        print(vis.summary())


if __name__ == "__main__":
    main()
