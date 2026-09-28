"""
Rewards in the Habitat home: a food bowl, a bitter plant, and an owner who pets
the pet and gives it treats -- so the fly's mushroom body has something to learn.

The world only provides senses and rewards; what the pet does is decided by the
connectome (through fly/body/foraging_body.ForagingBody's readout).

A. REAL DATA : the neurons driven (every ORN by glomerulus and side, sugar and
   bitter GRNs, head-bristle mechanosensory neurons, the PAM/PPL1 dopamine
   neurons) and everything downstream.
B. PUBLISHED :
   - Odours drive ORNs by their measured spontaneous rates and responses
     (Hallem & Carlson 2006; brain/sensory/olfaction.py). The bowl smells like
     fermenting fruit; the plant like green leaves.
   - Tasting sugar activates reward dopamine neurons (PAM beta'2/gamma4 =
     PAM05-08) and bitter the punishment ones (PPL101/103) (Kirkhart & Scott
     2015; Yamagata et al. 2015) -- driven directly, as in fly/world/senses.py,
     because the model does not recover the taste -> DAN route.
   - Touch to the head bristles drives the antennal/head grooming command aDN
     (DNge078) in the connectome (Hampel et al. 2015).
C. OUR APPROXIMATIONS :
   - Odour = a still-air halo around each source, c = strength * exp(-d/L),
     L = 1.2 m (room scale), sampled at two "antennae" 0.10 m left and right
     of a point 0.45 m ahead of the robot base (Spot is ~1 m long; the rover:
     0.15 m, sim/habitat_bridge/bodies.py). No walls
     block the odour (indoor air mixes around furniture).
   - Taste = contact: the robot base within EAT_R_M (0.6 m; rover 0.35 m) of the bowl/plant.
     "Eating" is counted while the proboscis channel (MN9 motor neurons) holds
     the proboscis extended, which also stops the body (ForagingBody).
   - The owner: when the pet is within 1.5 m and sees them, each second there
     is a PET_RATE chance of a 1.5 s pat on the head (head-bristle neurons at
     80 Hz) and a TREAT_RATE chance of a 1 s treat (sugar taste + reward DANs).

Battery mode (battery=robot.battery.Battery): the bowl is the charging dock and
hunger is the battery (robot/battery.py). Charging happens while the pet is at
the dock with its proboscis out. Hunger sets how strongly the dock is sensed,
as in real flies, where starvation raises the sensitivity of sugar-taste
neurons (dopamine; Inagaki et al. 2012 Cell 148:583; Marella et al. 2012 Neuron
73:941; Inagaki, Panse & Anderson 2014 Neuron 84:806) and
of food-odour ORNs (sNPF; Root et al. 2011 Cell 145:133): the dock's taste,
its reward-DAN drive and its odour are scaled by 0.15 + 0.85 x hunger (odour:
0.3 + 0.7 x hunger), so a charged pet passes the dock without the feeding
reflex. Treats from the owner are not scaled.
"""
from __future__ import annotations

import math

import numpy as np

BOWL_XZ = (-5.29, -4.49)        # a navigable spot in the small HSSD house
PLANT_XZ = (-0.40, -6.17)
ODOUR_L_M = 1.2
ANTENNA_AHEAD_M, ANTENNA_HALF_SEP_M = 0.45, 0.10
EAT_R_M = 0.6
TASTE_HZ, REINFORCE_HZ, TOUCH_HZ = 120.0, 100.0, 80.0
OWNER_R_M = 1.5
PET_S, TREAT_S = 1.5, 1.0
REWARD_DANS = ("PAM05", "PAM06", "PAM07", "PAM08")
PUNISH_DANS = ("PPL101", "PPL103")


class HomeWorld:
    def __init__(self, pet_rate: float = 0.15, treat_rate: float = 0.05, plant: bool = True,
                 seed: int = 0, battery=None, hunger_senses: bool = True, body: str = "spot",
                 odour: bool = True):
        from sim.habitat_bridge.bodies import BODIES
        self.pet_rate, self.treat_rate = pet_rate, treat_rate
        self.antenna_ahead, self.eat_r = BODIES[body]["antenna_ahead"], BODIES[body]["eat_r"]
        # odour=False: the real rover has no nose -- the ORNs fire at their
        # spontaneous rates only (no odour halo around the dock or plant)
        self.odour = odour
        self.battery = battery
        self.docked_by_nav = False              # set by the client's emergency return
        self.hunger_senses = hunger_senses      # False: control, the dock is sensed at full strength
        self.sources = [("bowl", BOWL_XZ, "fermenting_fruit", 1.0, "sweet")]
        if plant:
            self.sources.append(("plant", PLANT_XZ, "leaves", 1.0, "bitter"))
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.pet_until = self.treat_until = -1.0
        self.t = 0.0
        self.conc = {"L": np.zeros(len(self.sources)), "R": np.zeros(len(self.sources))}
        self.taste = None
        self.events = []
        self.stats = {"first_bowl_s": None, "near_bowl_s": 0.0, "at_bowl_s": 0.0, "eating_s": 0.0,
                      "at_plant_s": 0.0, "pets": 0, "treats": 0, "near_person_s": 0.0,
                      "eating_when_full_s": 0.0}
        if self.battery is not None:
            self.battery.day_start()

    @property
    def hunger(self) -> float | None:
        """Battery mode: how hungry (0..1) the pet is; else None. During a meal
        it stays hungry until satiated (robot/battery.py sense_hunger), so the
        neocortex does not pull it away from the dock mid-meal (it did in the
        first meal run: a meal started at 48% ended at 61% when the cortex,
        no longer hungry, chose its person)."""
        return None if self.battery is None else self.battery.sense_hunger

    def sense_gain(self) -> tuple:
        """(taste, odour) gain on the dock's signals: hunger-dependent sensitivity."""
        if self.battery is None or not self.hunger_senses:
            return (1.0, 1.0)
        h = self.battery.sense_hunger             # 1 during a meal (robot/battery.py)
        return (0.15 + 0.85 * h, 0.3 + 0.7 * h)

    def step(self, obs: dict, dt: float, proboscis: float, speed: float = 0.0,
             resting: bool = False) -> None:
        x, z, yaw = obs["robot"]
        a = math.radians(yaw)
        # yaw is measured in the (x, -z) plane, CCW seen from above
        # (habitat_server.summary): forward = (cos a, sin a) and left =
        # (-sin a, cos a) in (x, -z), i.e. in (x, z):
        fx, fz = math.cos(a), -math.sin(a)
        lx, lz = -math.sin(a), -math.cos(a)
        hx, hz = x + self.antenna_ahead * fx, z + self.antenna_ahead * fz
        for side, s in (("L", 1.0), ("R", -1.0)):
            ax, az_ = hx + s * ANTENNA_HALF_SEP_M * lx, hz + s * ANTENNA_HALF_SEP_M * lz
            self.conc[side] = np.array([st * math.exp(-math.hypot(ax - p[0], az_ - p[1]) / ODOUR_L_M)
                                        for _, p, _, st, _ in self.sources])
            if not self.odour:
                self.conc[side] = np.zeros(len(self.sources))
        self.taste = None
        st = self.stats
        for name, p, _, _, taste in self.sources:
            d = math.hypot(x - p[0], z - p[1])
            if name == "bowl":
                if d < 1.0:
                    st["near_bowl_s"] += dt
                if d < self.eat_r:
                    if st["first_bowl_s"] is None and not self.docked_by_nav:   # its own arrival, not the nav rescue
                        st["first_bowl_s"] = round(self.t, 1)
                        self.events.append((round(self.t, 1), "reached bowl"))
                    st["at_bowl_s"] += dt
                    if proboscis > 0.5:
                        st["eating_s"] += dt
                        # eating although satiated: at or above FULL and not in a meal
                        if self.battery is not None and self.battery.soc >= 0.9 and not self.battery.meal:
                            st["eating_when_full_s"] += dt
            elif d < self.eat_r:
                st["at_plant_s"] += dt
            if d < self.eat_r:
                self.taste = taste
        if self.battery is not None:
            d_dock = math.hypot(x - BOWL_XZ[0], z - BOWL_XZ[1])
            self.battery.step(dt, speed, resting,
                              charging=d_dock < self.eat_r and (proboscis > 0.5 or self.docked_by_nav),
                              feeding=d_dock < self.eat_r and proboscis > 0.5)
        # the owner
        if obs["dist"] < OWNER_R_M:
            st["near_person_s"] += dt
            if obs["visible"]:
                if self.t >= self.pet_until and self.rng.random() < self.pet_rate * dt:
                    self.pet_until = self.t + PET_S
                    st["pets"] += 1
                    self.events.append((round(self.t, 1), "petted"))
                if self.t >= self.treat_until and self.rng.random() < self.treat_rate * dt:
                    self.treat_until = self.t + TREAT_S
                    st["treats"] += 1
                    self.events.append((round(self.t, 1), "treat"))
        self.t += dt

    @property
    def petting(self) -> bool:
        return self.t < self.pet_until

    @property
    def treating(self) -> bool:
        return self.t < self.treat_until


class HomeSenses:
    """Session encoder: ORNs (with spontaneous rates), taste, head touch and
    reinforcement DANs from HomeWorld."""

    def __init__(self, connectome, world: HomeWorld):
        from brain.sensory.modalities import BY_KEY, resolve_neurons
        from brain.sensory.olfaction import OlfactorySpace
        self.world = world
        n = connectome.neurons
        t = n["primary_type"].fillna("").astype(str).to_numpy()
        side = n["side"].fillna("").astype(str).to_numpy()
        orn_types = sorted({x for x in t if x.startswith("ORN_")})
        self.olf = OlfactorySpace([x[4:] for x in orn_types])
        groups, self._orn = {}, {"L": [], "R": []}
        from brain.sensory.orn_side import orn_sides
        oside = orn_sides(connectome)      # side of entry, inferred for unlabelled ORNs
        for gi, ty in enumerate(orn_types):
            for sd, k in (("left", "L"), ("right", "R")):
                idx = np.flatnonzero((t == ty) & (oside == sd))
                if len(idx):
                    groups["orn_%s_%s" % (ty, k)] = idx
                    self._orn[k].append(("orn_%s_%s" % (ty, k), gi))
        groups["sugar"] = np.asarray(resolve_neurons(BY_KEY["taste_sugar"], connectome), np.int64)
        groups["bitter"] = np.asarray(resolve_neurons(BY_KEY["taste_bitter"], connectome), np.int64)
        groups["touch"] = np.asarray(resolve_neurons(BY_KEY["touch_head"], connectome), np.int64)
        groups["reward_dan"] = np.flatnonzero(np.isin(t, REWARD_DANS))
        groups["punish_dan"] = np.flatnonzero(np.isin(t, PUNISH_DANS))
        names = list(groups)
        idx = np.concatenate([groups[k] for k in names])
        gid = np.concatenate([np.full(len(groups[k]), i) for i, k in enumerate(names)])
        if np.unique(idx).size != idx.size:
            raise ValueError("sensory groups overlap")
        order = np.argsort(idx, kind="stable")
        self.indices, self._gid = idx[order], gid[order]
        self._ascending = np.arange(len(self.indices))
        self._npos = {k: i for i, k in enumerate(names)}
        self._opos = {k: np.array([self._npos[nm] for nm, _ in v]) for k, v in self._orn.items()}
        self._ogi = {k: np.array([gi for _, gi in v]) for k, v in self._orn.items()}
        self._blend = np.array([self.olf.sources[src] for _, _, src, _, _ in world.sources])
        self.reinforcement = True
        self.group_sizes = {k: int(len(v)) for k, v in groups.items()}

    def rates_hz(self, t_ms: float, stim=None) -> np.ndarray:
        w = self.world
        G = np.zeros(len(self._npos))
        g_taste, g_odour = w.sense_gain()
        src_gain = np.array([g_odour if nm == "bowl" else 1.0 for nm, *_ in w.sources])
        for k in ("L", "R"):
            r = self.olf.rates((w.conc[k] * src_gain) @ self._blend)
            G[self._opos[k]] = r[self._ogi[k]]
        sweet = w.taste == "sweet" or w.treating
        if sweet:
            g = 1.0 if w.treating else g_taste
            G[self._npos["sugar"]] = TASTE_HZ * g
            if self.reinforcement:
                G[self._npos["reward_dan"]] = REINFORCE_HZ * g
        elif w.taste == "bitter":
            G[self._npos["bitter"]] = TASTE_HZ
            if self.reinforcement:
                G[self._npos["punish_dan"]] = REINFORCE_HZ
        if w.petting:
            G[self._npos["touch"]] = TOUCH_HZ
        return G[self._gid]

    def state(self, t_ms: float) -> dict:
        w = self.world
        return {"kind": "home", "active": True, "odour_L": round(float(w.conc["L"].sum()), 3),
                "odour_R": round(float(w.conc["R"].sum()), 3), "taste": w.taste,
                "petting": w.petting, "treat": w.treating}
