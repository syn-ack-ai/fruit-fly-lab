"""
A scripted person who talks to the pet in the Habitat home (speech arrives as
text: the stand-in for speech recognition on the robot's microphone).

  call    now and then, "come here, <name>!" (anywhere in the house)
  praise  "good <name>!" when the pet arrives within ARRIVE_M after a call
  scold   "ouch, careful!" when the pet runs into them (the pet was moving
          toward the person faster than they were moving toward it: ground
          truth, as a person would judge it)

Records whether each call was answered and how fast (evaluation only). The
same rules and random seed apply in every condition, but WHEN calls happen
depends on the pet (no call while it is within 2 m, and the random draws follow
from that), so conditions hear different call times. Without a personality
layer nothing listens.
"""
from __future__ import annotations

import math

import numpy as np

CALL_PER_S = 1 / 45.0        # about one call every 45 s
ANSWER_S = 25.0              # a call is answered if the pet arrives within this
ARRIVE_M = 1.2
SCOLD_GAP_S = 5.0            # one "ouch" per collision, not one per contact frame
CALLS = ("come here, {name}!", "{name}, come!", "where are you, {name}?")


class ScriptedPerson:
    def __init__(self, name: str = "Mote", seed: int = 0):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.call_t = None
        self.calls = []              # (t, answered_after_s or None)
        self.heard = []              # (t, kind, text)
        self.scolds = 0
        self.last_scold = -1e9

    def step(self, t: float, dt: float, obs: dict, prev: dict, bumped: bool, v: float) -> str | None:
        say = None
        if self.call_t is not None:
            if obs["dist"] < ARRIVE_M:
                self.calls.append((self.call_t, round(t - self.call_t, 1)))
                self.call_t = None
                say = ("praise", f"good {self.name}!")
            elif t - self.call_t > ANSWER_S:
                self.calls.append((self.call_t, None))
                self.call_t = None
        if (say is None and bumped and t - self.last_scold >= SCOLD_GAP_S
                and self._pet_ran_into_me(obs, prev, v, dt)):
            self.scolds += 1
            self.last_scold = t
            say = ("scold", "ouch, careful!")
        if (say is None and self.call_t is None and obs["dist"] > 2.0
                and self.rng.random() < CALL_PER_S * dt):
            self.call_t = t
            say = ("call", CALLS[self.rng.integers(len(CALLS))].format(name=self.name))
        if say is None:
            return None
        self.heard.append((round(t, 1),) + say)
        return say[1]

    @staticmethod
    def _pet_ran_into_me(obs, prev, v, dt):
        rx, rz = prev["robot"][0] - prev["human"][0], prev["robot"][1] - prev["human"][1]
        d = math.hypot(rx, rz) or 1e-6
        ux, uz = rx / d, rz / d
        hvx = (obs["human"][0] - prev["human"][0]) / dt
        hvz = (obs["human"][1] - prev["human"][1]) / dt
        yaw = math.radians(prev["robot"][2])
        pet = -(v * math.cos(yaw) * ux - v * math.sin(yaw) * uz)
        return pet > 0.02 and pet >= hvx * ux + hvz * uz

    def summary(self) -> dict:
        answered = [a for _, a in self.calls if a is not None]
        return {"calls": len(self.calls) + (self.call_t is not None), "answered": len(answered),
                "median_answer_s": float(np.median(answered)) if answered else None,
                "scolds": self.scolds, "heard": self.heard}
