"""
Neocortex v0: the first prototype of the pet's learned higher brain.

    senses + fly-brain readout ─┐
                                ├─> drives (hunger, company, curiosity)
    odometry ──> cognitive map ─┤   place memory (food, bitter, owner, novelty)
                                ├─> goal choice -> path on the map -> goal
                                │     direction -> FC2 goal + LC10a "attend"
    rewards ──> critic V(s) ────┴─> reward-prediction error -> PAM / PPL1 DANs

What it learns (nothing about places or people is built in):
  - a cognitive map: places (0.5 m cells) it has been, joined by the moves it
    has made between them (a topological map; Tolman 1948; O'Keefe & Nadel
    1978). Paths are planned only over moves it has made.
  - what each place is worth: where it tasted food or bitterness, where it
    met or was petted by its person, and how familiar each place is. These
    memories persist across days (state_path).
  - where its person was last seen (bearing + estimated distance), so it can
    go and look when it misses them and they are out of view.
  - a critic: the value of the current situation, learned by temporal
    differences (Sutton 1988) on the GPU. Its prediction error is the
    dopamine-like teaching signal sent to the fly's mushroom body
    (Schultz, Dayan & Montague 1997), so odours that PREDICT food can be
    learned before food is tasted.

What is innate (as in any animal): the drives and how they rise and fall, and
the rule that picks the most valuable reachable place for the current drives.

Manners (manners=True): cat-like, not polite-at-all-times. When it wants
company (social drive >= SEEK_SOCIAL) it goes to its person and contact is
welcome (rubbing against the legs; the robot's safety layer, robot/safety.py,
makes it arrive at a crawl). When it has had enough (just petted), it loses
interest in chasing the person (arousal down) and steps back if closer than
PERSONAL_M. Whatever it wants, it never cuts in front of someone walking:
within YIELD_M of a walking person and not behind them, it steps out of their
path. The person's walking is estimated from its own sightings (position from
apparent size, smoothed over ~0.5 s).
The fly brain still does all the moving: the cortex only sets a goal
direction (and how strongly), and the fly's own pursuit, walking rhythm,
feeding, grooming and startle stay in charge of the body.

C. APPROXIMATIONS (Habitat prototype): the pose is Habitat's ground truth,
standing in for wheel odometry + visual localisation on the robot; person
distance comes from their apparent size (head and shoulders 0.5 m wide).
"""
from __future__ import annotations

import heapq
import math
import os
import pickle
import time
from collections import deque

import numpy as np

from cortex.topdown import bearing_deg
from robot.safety import person_distance

CELL_M = 0.5
REPLAN_S = 1.0
WAYPOINT_AHEAD_M = 0.8
GIVE_UP_S = 15.0             # an unreachable goal is marked blocked
FOOD_GIVE_UP_S = 40.0        # time allowed to reach a remembered food spot
FINAL_APPROACH_M = 1.5       # within this, steer to the exact remembered spot
EMPTY_SPOT_M = 0.25          # at the remembered spot and no food: it may be gone
COMMIT = 1.3                 # a new goal must beat the current one by 30%
PATH_COST = 0.03             # utility per metre of path
# drives (per second of simulated time; a "day" is minutes, so they are fast)
HUNGER_RISE, HUNGER_EAT = 0.004, 0.08
SOCIAL_RISE, SOCIAL_PET, SOCIAL_NEAR = 0.003, 0.25, 0.01
CURIOSITY = 0.35
# memories
FOOD_LEARN, FOOD_EXTINCT = 0.5, 0.01
OWNER_LEARN, OWNER_DECAY = 0.3, 0.002
SEEN_TAU_S = 60.0
# manners
SEEK_SOCIAL = 0.35           # wants company at or above this social drive
CONTENT_AROUSAL = 0.3        # interest in chasing the person when content
PERSONAL_M = 0.8             # content: step back if closer than this
YIELD_M = 1.2                # a walking person: get out of their path within this
WALKING_MS = 0.25            # estimated person speed that counts as walking
# rest (cat-like naps): sleepiness builds while awake, faster when active, and
# drains while resting; a content, not-hungry, sleepy pet goes to a favourite
# spot (where it has napped before) and rests there (top-down "rest" channel)
SLEEP_RISE, SLEEP_ACTIVE, SLEEP_FALL = 0.004, 0.004, 0.03   # per second
SLEEPY, AWAKE = 0.6, 0.15
REST_LEARN = 0.02
# personality (cortex/personality.py): an intention multiplies the matching
# utility, so the drive behind it still decides how much it matters
INTENT_GAIN = 0.5            # 1.5 let a chatty whim beat hunger (talking-pet run 1, 2026-09-26)
FEEDBACK_R = 0.5             # praise / scolding: reward like a pat on the head
# critic
GAMMA = 0.98                 # per 100 ms step: ~5 s horizon
RPE_DEADBAND = 2.0           # surprises only: |error| > 2 running SDs
NEIGH = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def cell_of(x, z):
    return (int(math.floor(x / CELL_M)), int(math.floor(z / CELL_M)))


def centre(c):
    return ((c[0] + 0.5) * CELL_M, (c[1] + 0.5) * CELL_M)


class CognitiveMap:
    def __init__(self):
        self.nodes = {}          # cell -> {"visits", "food", "bitter", "owner"}
        self.edges = {}          # cell -> set(cell)
        self.blocked = set()     # frontier cells it could not reach

    def visit(self, c, prev):
        n = self.nodes.setdefault(c, {"visits": 0, "food": 0.0, "bitter": 0.0, "owner": 0.0})
        n.setdefault("rest", 0.0)
        n.setdefault("fx", 0.0); n.setdefault("fz", 0.0); n.setdefault("fn", 0.0)
        n["visits"] += 1
        self.edges.setdefault(c, set())
        if prev is not None and prev != c:
            self.edges.setdefault(prev, set()).add(c)
            self.edges[c].add(prev)
        return n

    def food_point(self, c):
        """Where in this place food was tasted (mean position), or its centre."""
        n = self.nodes.get(c)
        if n and n.get("fn", 0.0) > 0:
            return (n["fx"] / n["fn"], n["fz"] / n["fn"])
        return centre(c)

    def frontier(self, c):
        """Unvisited neighbouring cells (possible new ground)."""
        return [(c[0] + a, c[1] + b) for a, b in NEIGH
                if (c[0] + a, c[1] + b) not in self.nodes and (c[0] + a, c[1] + b) not in self.blocked]

    def distances(self, src):
        """Dijkstra over moves it has made: {cell: (metres, previous cell)}."""
        dist = {src: (0.0, None)}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u][0]:
                continue
            for v in self.edges.get(u, ()):
                nd = d + CELL_M * math.hypot(v[0] - u[0], v[1] - u[1])
                if v not in dist or nd < dist[v][0]:
                    dist[v] = (nd, u)
                    heapq.heappush(pq, (nd, v))
        return dist

    @staticmethod
    def path_to(dist, dst):
        p = [dst]
        while dist[p[-1]][1] is not None:
            p.append(dist[p[-1]][1])
        return p[::-1]


class Critic:
    """V(s): a small MLP trained online by TD(0) with a short replay buffer."""

    def __init__(self, n_in, seed=0):
        import torch
        torch.manual_seed(seed)
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, 64), torch.nn.Tanh(),
                                       torch.nn.Linear(64, 64), torch.nn.Tanh(),
                                       torch.nn.Linear(64, 1)).to(self.dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.buf = deque(maxlen=20000)
        self.rng = np.random.default_rng(seed)

    def value(self, phi):
        with self.torch.no_grad():
            return float(self.net(self.torch.as_tensor(phi, dtype=self.torch.float32, device=self.dev)))

    def learn(self, phi, r, phi2, batch=64):
        self.buf.append((phi, r, phi2))
        idx = self.rng.integers(0, len(self.buf), size=min(batch, len(self.buf)))
        idx[-1] = len(self.buf) - 1                 # always include the newest step
        P, R, P2 = (np.array([self.buf[i][k] for i in idx], np.float32) for k in range(3))
        t = self.torch
        P, R, P2 = (t.as_tensor(a, device=self.dev) for a in (P, R, P2))
        with t.no_grad():
            target = R + GAMMA * self.net(P2).squeeze(-1)
        loss = ((self.net(P).squeeze(-1) - target) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach())


class CortexV0:
    N_FOURIER = 16

    def __init__(self, state_path: str | None = None, seed: int = 0, amnesic: bool = False,
                 manners: bool = False, naps: bool = False):
        self.state_path = state_path
        self.manners = manners
        self.naps = naps                            # cat-like rest (off: as v0 / v0_manners were run)
        self.amnesic = amnesic                      # control: place memories wiped every day
        self.rng = np.random.default_rng(seed)
        # random Fourier features of position: a place code for the critic
        self.freq = self.rng.normal(0, 1.0 / 2.0, size=(self.N_FOURIER, 2))
        self.phase = self.rng.uniform(0, 2 * np.pi, self.N_FOURIER)
        self.map = CognitiveMap()
        self.critic = Critic(self._n_features(), seed)
        self.days = 0
        self.owner_prior = {}                       # cell -> learned "where my person hangs out"
        self.lifetime = {"rewards": 0.0, "rpe_abs": 0.0, "steps": 0}
        if state_path and os.path.exists(os.path.join(state_path, "cortex.pkl")):
            self._load()

    # ------------------------------------------------------------- features
    def _n_features(self):
        return 2 * self.N_FOURIER + 10

    def _features(self, pos, node, home, obs):
        f = self.freq @ np.array(pos)
        smell = float(home.conc["L"].sum() + home.conc["R"].sum()) if home else 0.0
        # what the fly smells: every source blended, left minus right antenna
        diff = float(home.conc["L"].sum() - home.conc["R"].sum()) if home else 0.0
        return np.concatenate([np.sin(f + self.phase), np.cos(f + self.phase),
                               [self.hunger, self.social, float(obs["visible"]), smell,
                                10.0 * diff, float(home.taste == "sweet") if home else 0.0,
                                float(home.taste == "bitter") if home else 0.0,
                                float(home.petting) if home else 0.0,
                                node["food"], node["owner"]]]).astype(np.float32)

    # --------------------------------------------------------------- episode
    def reset(self, episode, home, conn) -> None:
        if self.amnesic:
            # control: everything learned about places is wiped, including the
            # critic (its inputs include position, so it holds a value map)
            self.map, self.owner_prior = CognitiveMap(), {}
            self.critic = Critic(self._n_features(), int(self.rng.integers(1 << 30)))
        self.hunger, self.social = 0.6, 0.5          # a new day: hungry, wants company
        self.sleepy, self.resting, self.want_rest = 0.2, False, False
        self.prev_xz = None
        self.t = 0.0
        self.prev_cell = None
        self.goal = None                             # (cell, kind, utility, t_set)
        self.food_avoid = {}                         # timed-out food cells, until a later retry
        self.next_plan = 0.0
        self.person_seen = None                      # (x, z, t)
        self.person_vel = (0.0, 0.0)                 # estimated, m/s
        self.manner = None
        self.intent, self.feedback = "none", 0
        self.prev_phi = None
        self.prev_events = (0, 0)
        self.rpe_sd = 0.1
        self.last = {}
        self.day_log = {"goals": {}, "rpe_pos": 0, "rpe_neg": 0, "new_cells": 0, "manners_s": {}}
        self.cells_at_start = len(self.map.nodes)
        self.replay = []                             # (phi, cmd, reward) for the world model

    def know_place(self, xz, kind: str = "food", value: float = 0.5) -> None:
        """A place known without having walked there: the robot's charging dock
        at "birth" (a robot starts life on its dock and knows where it is, as a
        kitten is shown its bowl). It has no map edges yet, so it is planned to
        by straight-line distance until the pet has walked there."""
        c = cell_of(*xz)
        n = self.map.visit(c, None)
        n["visits"] -= 1                             # known, not visited
        n[kind] = max(n[kind], value)
        if kind == "food":
            n["fx"], n["fz"], n["fn"] = xz[0], xz[1], 1.0

    def set_personality(self, intent: str, feedback: int = 0) -> None:
        """From cortex/personality.py: the current intention, and praise (+1) or
        scolding (-1) heard since the last step (counted once)."""
        self.intent = intent or "none"
        self.feedback += int(feedback)

    # ------------------------------------------------------------------ step
    def act(self, obs, home, t_s, frame) -> dict:
        dt = t_s - self.t if t_s > self.t else 0.1
        self.t = t_s
        x, z, yaw = obs["robot"]
        c = cell_of(x, z)
        node = self.map.visit(c, self.prev_cell)
        self.prev_cell = c

        # --- what happened (rewards) and memories of places
        eating = bool(home and home.taste == "sweet" and frame is not None
                      and frame["body"].get("proboscis_extension", 0.0) > 0.5)
        sweet = bool(home and home.taste == "sweet")
        bitter = bool(home and home.taste == "bitter")
        pets, treats = (home.stats["pets"], home.stats["treats"]) if home else (0, 0)
        new_pet, new_treat = pets > self.prev_events[0], treats > self.prev_events[1]
        self.prev_events = (pets, treats)
        r = 0.1 * eating + 0.03 * sweet - 0.1 * bitter + 0.5 * new_pet + 0.5 * new_treat
        r += FEEDBACK_R * max(-1, min(1, self.feedback))
        self.feedback = 0
        if sweet:
            node["food"] += FOOD_LEARN * (1.0 - node["food"])
            node["fx"] += x; node["fz"] += z; node["fn"] += 1.0  # exactly where
        elif node["food"] > 0:
            fx, fz = self.map.food_point(c)
            if math.hypot(x - fx, z - fz) < EMPTY_SPOT_M:       # right there, and nothing:
                node["food"] -= FOOD_EXTINCT * node["food"]      # slowly forgets empty bowls
        if bitter:
            node["bitter"] += FOOD_LEARN * (1.0 - node["bitter"])
        if new_pet or new_treat:
            node["owner"] += OWNER_LEARN * (1.0 - node["owner"])
        # the person: where are they (estimated from apparent size)?
        person_d = person_distance(obs["half"]) if obs["visible"] else None
        if person_d is not None:
            d = person_d
            b = math.radians(yaw - obs["az"])                  # az + = right
            px, pz = x + d * math.cos(b), z - d * math.sin(b)
            prev = self.person_seen
            if prev is not None and 0.05 <= t_s - prev[2] <= 1.0:
                k = min(1.0, (t_s - prev[2]) / 0.5)
                vx, vz = (px - prev[0]) / (t_s - prev[2]), (pz - prev[1]) / (t_s - prev[2])
                self.person_vel = (self.person_vel[0] + k * (vx - self.person_vel[0]),
                                   self.person_vel[1] + k * (vz - self.person_vel[1]))
            elif prev is None or t_s - prev[2] > 1.0:
                self.person_vel = (0.0, 0.0)
            self.person_seen = (px, pz, t_s)
            pc = cell_of(px, pz)
            self.owner_prior[pc] = self.owner_prior.get(pc, 0.0) + OWNER_LEARN * (1 - self.owner_prior.get(pc, 0.0))
        for k in list(self.owner_prior):
            self.owner_prior[k] *= (1 - OWNER_DECAY * dt)

        # --- drives
        if home is not None and getattr(home, "hunger", None) is not None:
            self.hunger = float(home.hunger)        # battery mode: hunger is the charge
        else:
            self.hunger = float(np.clip(self.hunger + HUNGER_RISE * dt - HUNGER_EAT * dt * eating, 0, 1))
        near = person_d is not None and person_d < 1.5     # from apparent size, not ground truth
        self.social = float(np.clip(self.social + SOCIAL_RISE * dt - SOCIAL_NEAR * dt * near
                                    - SOCIAL_PET * (new_pet or new_treat), 0, 1))
        speed = 0.0 if self.prev_xz is None else math.hypot(x - self.prev_xz[0], z - self.prev_xz[1]) / max(dt, 1e-3)
        self.prev_xz = (x, z)
        if self.resting:
            self.sleepy = max(0.0, self.sleepy - SLEEP_FALL * dt)
            node["rest"] += REST_LEARN * dt * (1.0 - node["rest"])      # a good place to nap
            if self.sleepy <= AWAKE or self.hunger > 0.7 or new_pet or self.intent in ("seek_person", "eat", "explore"):
                self.resting = False
        else:
            self.sleepy = min(1.0, self.sleepy + (SLEEP_RISE + SLEEP_ACTIVE * min(1.0, speed / 0.3)) * dt)
        self.want_rest = (self.naps and not self.resting and self.hunger < 0.5
                          and (self.sleepy >= SLEEPY or (self.intent == "rest" and self.sleepy >= 0.3))
                          and (self.social < SEEK_SOCIAL or self.intent == "rest"))

        # --- critic: reward-prediction error -> dopamine
        phi = self._features((x, z), node, home, obs)
        rpe_drive, delta = 0.0, 0.0
        if self.prev_phi is not None:
            v, v2 = self.critic.value(self.prev_phi), self.critic.value(phi)
            delta = r + GAMMA * v2 - v
            self.critic.learn(self.prev_phi, r, phi)
            self.rpe_sd += 0.01 * (abs(delta) - self.rpe_sd)
            z_ = delta / max(self.rpe_sd, 1e-3) / 3.0
            if abs(z_) > RPE_DEADBAND / 3.0:
                rpe_drive = float(np.clip(z_, -1, 1))
                self.day_log["rpe_pos" if rpe_drive > 0 else "rpe_neg"] += 1
        self.prev_phi = phi
        self.lifetime["rewards"] += r
        self.lifetime["rpe_abs"] += abs(delta)
        self.lifetime["steps"] += 1

        # --- choose where to go (every REPLAN_S)
        if t_s >= self.next_plan:
            self.next_plan = t_s + REPLAN_S
            self._plan(c, t_s, sweet)
        cmd = {"goal_deg": None, "goal_gain": 0.0, "rpe": rpe_drive}
        if self.goal is not None:
            gcell, kind, util, t_set = self.goal
            wp = self._waypoint(c, gcell, (x, z), kind)
            cmd["goal_deg"] = bearing_deg((x, z), wp)
            cmd["goal_gain"] = float(np.clip(util / 0.5, 0.3, 1.0))
        if self.manners:
            cmd = self._manners(cmd, x, z, t_s, dt)
            if self.manner == "yield":
                self.resting = False                  # never nap in a walking person's path
        if self.resting:
            cmd.update(goal_deg=None, goal_gain=0.0, rest=1.0, arousal=min(cmd.get("arousal", 1.0), 0.2))
            self.day_log["rest_s"] = self.day_log.get("rest_s", 0.0) + dt
        self.replay.append((phi, [cmd["goal_deg"] or 0.0, cmd["goal_gain"], rpe_drive], r))
        self.last = {"hunger": round(self.hunger, 2), "social": round(self.social, 2),
                     "goal_kind": None if self.goal is None else self.goal[1], "manner": self.manner,
                     "sleepy": round(self.sleepy, 2), "resting": self.resting,
                     "cells": len(self.map.nodes), "delta": round(delta, 3)}
        return cmd

    def _manners(self, cmd, x, z, t_s, dt):
        wants = self.social >= SEEK_SOCIAL and self.intent != "give_space"
        cmd["arousal"] = 1.0 if wants else CONTENT_AROUSAL
        mode = "want" if wants else "content"
        ps = self.person_seen
        if ps is not None and t_s - ps[2] < 1.0:
            px, pz = ps[0], ps[1]
            rx, rz = x - px, z - pz
            d = math.hypot(rx, rz)
            vx, vz = self.person_vel
            speed = math.hypot(vx, vz)
            if speed > WALKING_MS and d < YIELD_M and rx * vx + rz * vz > -0.2 * speed:
                # in (or beside) a walking person's path: step out of it, to the side it is on
                ux, uz = vx / speed, vz / speed
                sx, sz = -uz, ux
                if sx * rx + sz * rz < 0:
                    sx, sz = -sx, -sz
                cmd.update(goal_deg=bearing_deg((x, z), (x + sx, z + sz)), goal_gain=1.0, arousal=0.0)
                mode = "yield"
            elif speed > WALKING_MS:
                mode = "follow" if wants else "content"
            elif wants:
                mode = "greet"                          # contact welcome (at a crawl: robot/safety.py)
            elif d < PERSONAL_M:
                cmd.update(goal_deg=bearing_deg((x, z), (x + rx, z + rz)), goal_gain=0.6)
                mode = "give_space"
        self.manner = mode
        self.day_log["manners_s"][mode] = self.day_log["manners_s"].get(mode, 0.0) + dt
        return cmd

    def _utilities(self, here, t_s):
        dist = self.map.distances(here)
        # known places not yet joined to the map (know_place): straight-line
        # distance x 1.5 as a detour allowance
        hx, hz = centre(here)
        cand = dict(dist)
        for cell, n in self.map.nodes.items():
            if cell not in cand and n["food"] > 0:
                cx, cz = centre(cell)
                cand[cell] = (1.5 * math.hypot(cx - hx, cz - hz), None)
        U = {}
        for cell, (d, _) in cand.items():
            n = self.map.nodes[cell]
            u_food = self.hunger * n["food"]
            u_owner = self.social * max(n["owner"], self.owner_prior.get(cell, 0.0))
            if self.naps:
                # the pet (battery mode): past half hungry, food becomes urgent
                # (up to 3x) and company matters less (a hungry cat goes to the
                # bowl; in the first battery run a low pet kept visiting its person)
                starving = max(0.0, self.hunger - 0.5) / 0.5
                u_food *= 1.0 + 2.0 * starving
                u_owner *= 1.0 - 0.8 * starving
            if self.person_seen is not None:
                pc = cell_of(*self.person_seen[:2])
                if cell == pc or cell in [(pc[0] + a, pc[1] + b) for a, b in NEIGH]:
                    u_owner = max(u_owner, self.social * math.exp(-(t_s - self.person_seen[2]) / SEEN_TAU_S))
            u_new = CURIOSITY * (len(self.map.frontier(cell)) / 8.0) / math.sqrt(1 + n["visits"] / 20.0)
            if self.intent in ("seek_person", "follow"):
                u_owner *= 1 + INTENT_GAIN
            elif self.intent == "give_space":
                u_owner /= 1 + INTENT_GAIN
            elif self.intent == "eat":
                u_food *= 1 + INTENT_GAIN
            elif self.intent == "explore":
                u_new *= 1 + INTENT_GAIN
            elif self.intent == "rest":
                u_new /= 1 + INTENT_GAIN
            u_rest = self.sleepy * (0.3 + n.get("rest", 0.0)) if self.want_rest else 0.0
            u = u_food + u_owner + u_new + u_rest - n["bitter"] - PATH_COST * d
            kind = max((("food", u_food), ("owner", u_owner), ("explore", u_new), ("rest", u_rest)),
                       key=lambda kv: kv[1])[0]
            U[cell] = (u, kind)
        return U, dist

    def _plan(self, here, t_s, sweet=False):
        U, dist = self._utilities(here, t_s)
        for cell, until in list(self.food_avoid.items()):
            if t_s >= until:
                del self.food_avoid[cell]
        if self.goal is not None and self.goal[1] == "food":
            gcell, kind, util, t_set = self.goal
            if sweet:                                  # arrived: the fly eats
                self.goal = None
                self._dist = dist
                return
            best_u = max(u for u, _ in U.values())
            if t_s - t_set <= FOOD_GIVE_UP_S and gcell in U and best_u < COMMIT * U[gcell][0]:
                self.goal = (gcell, kind, U[gcell][0], t_set)   # keep going, even inside the cell
                self._dist = dist
                return
            if t_s - t_set > FOOD_GIVE_UP_S:
                self.food_avoid[gcell] = t_s + FOOD_GIVE_UP_S
            self.goal = None
        if self.goal is not None:
            gcell, kind, util, t_set = self.goal
            reached = gcell == here
            if reached and kind == "explore":
                fr = self.map.frontier(here)          # step off the map into new ground
                if fr:
                    self.goal = (fr[self.rng.integers(len(fr))], "explore_new", util, t_s)
                    return
            if (not reached and t_s - t_set > GIVE_UP_S):
                if kind == "explore_new":
                    self.map.blocked.add(gcell)
                self.goal = None
            elif not reached and gcell in U and max(u for u, _ in U.values()) < COMMIT * U[gcell][0]:
                self.goal = (gcell, kind, U[gcell][0], t_set)
                return
            elif not reached and kind == "explore_new" and gcell not in self.map.nodes:
                return
        # a food spot it just gave up on is not chosen as FOOD again for a while
        # (other reasons to go there, e.g. its person, still count)
        choices = {cell: value for cell, value in U.items()
                   if not (cell in self.food_avoid and value[1] == "food")}
        if not choices:
            self.goal = None
            self._dist = dist
            return
        best = max(choices.items(), key=lambda kv: kv[1][0])
        cell, (u, kind) = best
        if kind == "rest" and (cell == here or (self.goal is not None and self.goal[1] == "rest" and self.goal[0] == here)):
            self.resting, self.want_rest, self.goal = True, False, None     # settle here
            self._dist = dist
            return
        if u <= 0.02 or (cell == here and (kind != "food" or sweet)):
            self.goal = None                          # content: the fly brain does as it likes
        else:
            self.goal = (cell, kind, u, t_s)
            self.day_log["goals"][kind] = self.day_log["goals"].get(kind, 0) + 1
        self._dist = dist

    def _waypoint(self, here, gcell, pos=None, kind=None):
        if gcell not in self.map.nodes:               # a step into new ground
            return centre(gcell)
        if kind == "food" and pos is not None:
            fp = self.map.food_point(gcell)
            if math.hypot(pos[0] - fp[0], pos[1] - fp[1]) < FINAL_APPROACH_M or gcell == here:
                return fp                             # final approach: the exact spot
        dist = getattr(self, "_dist", None)
        if dist is None or here not in dist or gcell not in dist or dist[here][0] != 0.0:
            dist = self._dist = self.map.distances(here)
        if gcell not in dist:
            return centre(gcell)
        path = CognitiveMap.path_to(dist, gcell)
        hx, hz = centre(here)
        for p in path[1:]:
            px, pz = centre(p)
            if math.hypot(px - hx, pz - hz) >= WAYPOINT_AHEAD_M:
                return (px, pz)
        return centre(gcell)

    # ------------------------------------------------------------- reporting
    def log_state(self) -> dict:
        return self.last

    def summary(self) -> dict:
        food = sorted(((round(n["food"], 2), c) for c, n in self.map.nodes.items() if n["food"] > 0.1), reverse=True)
        return {"kind": "v0", "day": self.days, "cells": len(self.map.nodes),
                "new_cells": len(self.map.nodes) - self.cells_at_start,
                "food_places": len(food), "best_food": [centre(c) for _, c in food[:2]],
                "goals": self.day_log["goals"], "rpe_pos_steps": self.day_log["rpe_pos"],
                "rpe_neg_steps": self.day_log["rpe_neg"],
                "hunger_end": round(self.hunger, 2), "social_end": round(self.social, 2),
                "manners_s": {k: round(v, 1) for k, v in self.day_log["manners_s"].items()},
                "rest_s": round(self.day_log.get("rest_s", 0.0), 1)}

    def save(self) -> None:
        self.days += 1
        if not self.state_path:
            return
        os.makedirs(self.state_path, exist_ok=True)
        with open(os.path.join(self.state_path, "cortex.pkl"), "wb") as fh:
            pickle.dump({"map": self.map, "owner_prior": self.owner_prior, "days": self.days,
                         "critic": self.critic.net.state_dict(), "lifetime": self.lifetime,
                         "freq": self.freq, "phase": self.phase}, fh)
        if self.replay:
            P = np.array([p for p, _, _ in self.replay], np.float32)
            A = np.array([a for _, a, _ in self.replay], np.float32)
            R = np.array([r for _, _, r in self.replay], np.float32)
            np.savez_compressed(os.path.join(self.state_path, "replay_day%03d.npz" % (self.days - 1)),
                                phi=P, action=A, reward=R)

    def _load(self):
        with open(os.path.join(self.state_path, "cortex.pkl"), "rb") as fh:
            s = pickle.load(fh)
        self.map, self.owner_prior, self.days = s["map"], s["owner_prior"], s["days"]
        self.critic.net.load_state_dict(s["critic"])
        self.lifetime, self.freq, self.phase = s["lifetime"], s["freq"], s["phase"]
