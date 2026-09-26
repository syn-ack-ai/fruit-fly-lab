"""
Neocortex agents for the Habitat home (sim/habitat_bridge/brain_client.py).

Each control step the client calls  cmd = cortex.act(obs, home, t_s, frame)
and hands cmd to cortex/topdown.TopDown, which turns it into drive to the fly
brain's goal (FC2), pursuit (LC10a) and teaching (DAN) neurons.

  oracle  knows the house (Habitat's navigation mesh) and always sets the goal
          to the next point on the shortest path to the food bowl. Not a pet:
          an UPPER BOUND that measures whether the top-down channels can steer
          the fly brain at all.
  v0      the prototype neocortex (cortex/v0.py): a cognitive map learned from
          its own odometry, drives, and a critic.
  v0_manners  v0 with cat-like manners around its person (cortex/v0.py).
"""
from __future__ import annotations

import math

from cortex.topdown import bearing_deg


class OracleCortex:
    def __init__(self, target: str = "bowl", ahead_m: float = 0.8, **_):
        self.target, self.ahead = target, ahead_m

    def reset(self, episode, home, conn) -> None:
        self.conn, self.home = conn, home
        self.goal_xz = next(p for n, p, *_ in home.sources if n == self.target)
        self._last = {}

    def act(self, obs, home, t_s, frame) -> dict:
        self.conn.send({"cmd": "path", "goal": list(self.goal_xz), "ahead": self.ahead})
        p = self.conn.recv()
        x, z, _ = obs["robot"]
        g = bearing_deg((x, z), p["waypoint"])
        at = math.hypot(x - self.goal_xz[0], z - self.goal_xz[1]) < 0.6
        self._last = {"geodesic": None if p["geodesic"] is None else round(p["geodesic"], 2)}
        # at the bowl the goal switches off (eating stops the body anyway)
        return {"goal_deg": g, "goal_gain": 0.0 if at else 1.0}

    def log_state(self) -> dict:
        return self._last

    def summary(self) -> dict:
        return {"kind": "oracle", "target": self.target}

    def save(self) -> None:
        pass


def make_cortex(kind: str, state_path: str | None = None, seed: int = 0):
    if kind == "oracle":
        return OracleCortex()
    if kind in ("v0", "v0_amnesic", "v0_manners"):
        from cortex.v0 import CortexV0
        return CortexV0(state_path=state_path, seed=seed, amnesic=kind == "v0_amnesic",
                        manners=kind == "v0_manners")
    raise ValueError(kind)
