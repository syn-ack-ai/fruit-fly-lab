"""
The pet's face (shown on an iPad or any browser: robot/face_server.py).

Most of the expression comes straight from the fly brain and body, so the face
shows what the connectome is doing; the personality layer (cortex/personality.py)
adds a mood and the occasional word:

  eyes look at     the person (camera bearing), else where the pet is heading
  eyes open wide   startle: escape command neurons (GF DNp01, DNp02/04/11)
  pupils           arousal: a lens aperture, wide when startled, playing or chasing; small when calm
  heavy lids       resting a long time (sleepy), or a slow blink while being
                   petted or content near its person (a contented "I trust you")
  mouth            chewing while the proboscis is out (eating); otherwise the mood
  blush            being petted
  speech bubble    the personality's words; sounds are synthesised in the browser

FacePublisher sends the state to the face server without ever blocking the
brain loop (latest state wins; a background thread posts it).
"""
from __future__ import annotations

import json
import math
import queue
import threading
import urllib.request


class FaceModel:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.rest_s = 0.0
        self.startle = 0.0
        self.state = {}
        self.event = {"id": 0, "say": None, "sound": None}

    def update(self, dt: float, s: dict) -> dict:
        """s: person_visible, person_az (deg, + right), person_el, person_d (m),
        goal_az, startle (0..1), speed (m/s), behaviour, eating, grooming,
        petting, mood, say, sound, hunger, social."""
        self.startle = max(float(s.get("startle", 0.0)), self.startle * math.exp(-dt / 0.8))
        speed = abs(float(s.get("speed", 0.0)))
        self.rest_s = self.rest_s + dt if speed < 0.02 and not s.get("eating") else 0.0
        if s.get("person_visible"):
            gx = max(-1.0, min(1.0, float(s.get("person_az", 0.0)) / 60.0))
            gy = max(-1.0, min(1.0, -float(s.get("person_el", 0.0)) / 40.0))
        elif s.get("goal_az") is not None:
            gx, gy = max(-1.0, min(1.0, float(s["goal_az"]) / 90.0)), 0.1
        else:
            gx, gy = 0.0, 0.2
        mood = s.get("mood") or "calm"
        near = s.get("person_visible") and (s.get("person_d") or 9.0) < 1.2
        openness = 0.85
        if self.rest_s > 8.0:
            openness = max(0.3, 0.85 - 0.04 * (self.rest_s - 8.0))
        if mood == "sleepy":
            openness = min(openness, 0.45)
        slow_blink = bool(s.get("petting")) or (near and mood in ("content", "happy", "calm"))
        if self.startle > 0.2:
            openness = 1.0
        arousal = max(self.startle, min(1.0, speed / 0.4), 0.8 if mood in ("playful", "eager") else 0.0)
        mouth = {"happy": "smile", "eager": "smile", "playful": "smile", "grumpy": "frown",
                 "startled": "o", "hungry": "o"}.get(mood, "neutral")
        if s.get("eating"):
            mouth = "chew"
        elif self.startle > 0.2:
            mouth = "o"
        self.state = {"gaze": [round(gx, 3), round(gy, 3)], "open": round(openness, 3),
                      "pupil": round(0.3 + 0.7 * arousal, 3), "slow_blink": slow_blink,
                      "mouth": mouth, "blush": bool(s.get("petting")), "mood": mood,
                      "grooming": bool(s.get("grooming")), "behaviour": s.get("behaviour", ""),
                      "hunger": round(float(s.get("hunger", 0.0)), 2),
                      "social": round(float(s.get("social", 0.0)), 2)}
        # words and sounds are events: numbered, so a page that misses a state
        # update still plays each one once
        snd = s.get("sound") if s.get("sound") not in (None, "none") else None
        if s.get("say") or snd:
            self.event = {"id": self.event["id"] + 1, "say": s.get("say"), "sound": snd}
        self.state["event"] = self.event
        return self.state


class FacePublisher:
    """POST the face state to robot/face_server.py; never blocks the caller."""

    def __init__(self, url: str = "http://127.0.0.1:8010/state", key: str | None = None):
        from robot.face_server import face_key
        self.url = url
        self.key = key or face_key()
        self.q = queue.Queue(maxsize=1)
        self.fails = 0
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, state: dict) -> None:
        try:
            self.q.get_nowait()                      # drop a stale state
        except queue.Empty:
            pass
        try:
            self.q.put_nowait(state)
        except queue.Full:
            pass

    def _run(self):
        while True:
            st = self.q.get()
            try:
                req = urllib.request.Request(self.url, json.dumps(st).encode(),
                                             {"Content-Type": "application/json", "X-Face-Token": self.key})
                urllib.request.urlopen(req, timeout=1.0).read()
            except Exception:
                self.fails += 1
