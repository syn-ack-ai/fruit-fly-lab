"""
A world for the simulated fly: fruit that smells and tastes, wind that carries
the smell, and predators that attack.

SCOPE AND HONESTY
-----------------
This is the environment, not the fly. It decides where things are, what the
fly's senses receive, and what happens physically (eating, being caught,
walls). It never decides what the fly does: that comes only from the
connectome's descending neurons through the body model.

A. REAL PHYSICS / GEOMETRY
   - Angular size and expansion of an approaching predator are exact geometry
     (theta = atan(r / d)), as in simulation/stimuli/looming.py.
   - Each antenna samples the odour at its own position (antennae ~0.4 mm
     apart), so any left/right difference is physical.
B. PUBLISHED VALUES (cited where used)
   - Predator attacks use l/|v| = 25 ms, inside the 10-80 ms range of
     Drosophila escape experiments (Card & Dickinson 2008; von Reyn et al. 2014).
   - Odour plumes are intermittent: filaments separated by clean air
     (Murlis et al. 1992, Annu Rev Entomol 37:505; van Breugel & Dickinson
     2014, Curr Biol 24:274).
C. OUR APPROXIMATIONS
   - Time-averaged plume: a Gaussian plume that widens downwind plus a
     still-air halo around each fruit. Intermittency is a two-state
     (filament / gap) Markov process whose filament probability rises with
     the mean concentration.
   - Predators fly straight at where the fly was at launch (a strike, not a
     homing pursuit).
   - Energy is bookkeeping only; hunger does not change the brain.

Units: mm, ms, degrees. Heading 0 = +x, counter-clockwise positive (as in
fly/body/fly_body.py). Head-centred azimuth: 0 = ahead, positive = right.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

HEAD_OFFSET_MM = 1.2        # head centre ahead of the body origin
ANTENNA_HALF_SEP_MM = 0.2   # antennae ~0.4 mm apart
FLY_HEAD_HEIGHT_MM = 0.8


# What each kind of fruit smells like (a source in brain/sensory/olfaction.py)
# and tastes like. Mouldy fruit is bitter and smells of geosmin.
FRUIT_KINDS = {
    "fermenting": {"source": "fermenting_fruit", "taste": "sweet", "strength": 1.0},
    "ripe": {"source": "ripe_fruit", "taste": "sweet", "strength": 0.7},
    "mouldy": {"source": "mould", "taste": "bitter", "strength": 1.0},
    "citrus": {"source": "citrus", "taste": "none", "strength": 0.8},
}
DEFAULT_MIX = ("fermenting", "ripe", "mouldy", "fermenting", "citrus", "ripe")


@dataclass
class Fruit:
    x: float
    y: float
    radius: float = 12.0
    kind: str = "fermenting"   # see FRUIT_KINDS
    amount: float = 1.0        # 0 = eaten up

    @property
    def taste(self) -> str:
        return FRUIT_KINDS[self.kind]["taste"]

    @property
    def odour(self) -> float:
        return FRUIT_KINDS[self.kind]["strength"]

    def as_dict(self):
        return {"x": round(self.x, 1), "y": round(self.y, 1), "r": self.radius,
                "kind": self.kind, "taste": self.taste, "amount": round(self.amount, 3)}


@dataclass
class Predator:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    radius: float
    t_launch: float
    t_end: float
    min_dist: float = float("inf")

    def pos(self):
        return np.array([self.x, self.y, self.z])


@dataclass
class WorldConfig:
    size_mm: float = 600.0             # square arena, centred on the origin
    n_fruit: int = 4                   # kinds cycle through DEFAULT_MIX
    leaf_background: float = 0.05      # faint green-leaf odour everywhere (a garden)
    wind_speed_mm_s: float = 150.0     # 0.15 m/s breeze
    wind_from_deg: float = 180.0       # wind blows FROM this direction (world frame)
    turbulent: bool = True             # intermittent plume filaments
    predators: bool = False            # no staged attacks: a natural setting (option kept for experiments)
    predator_interval_s: tuple = (10.0, 25.0)
    predator_speed_mm_s: float = 1000.0
    predator_radius_mm: float = 25.0   # l/|v| = 25 ms
    predator_start_mm: float = 250.0
    eat_rate_per_s: float = 0.02       # fruit amount consumed per second of feeding
    seed: int = 0


class World:
    def __init__(self, config: WorldConfig | None = None):
        self.cfg = config or WorldConfig()
        self.reset(self.cfg.seed)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int = 0) -> None:
        cfg = self.cfg
        self.rng = np.random.default_rng(seed)
        h = cfg.size_mm / 2
        self.fruits = []
        for i in range(cfg.n_fruit):
            for _ in range(100):                      # keep fruit apart and off the start
                x, y = self.rng.uniform(-0.8 * h, 0.8 * h, 2)
                if math.hypot(x, y) > 120 and all(math.hypot(x - f.x, y - f.y) > 150
                                                  for f in self.fruits):
                    break
            self.fruits.append(Fruit(float(x), float(y), kind=DEFAULT_MIX[i % len(DEFAULT_MIX)]))
        self.predator = None
        self._next_attack = self._draw_attack_time(0.0)
        self._filament = False                       # inside an odour filament
        self.stats = {"eaten_sweet": 0.0, "eaten_bitter": 0.0, "eaten_none": 0.0, "attacks": 0,
                      "escaped": 0, "caught": 0, "fruit_visits": 0,
                      "time_on_fruit_ms": 0.0, "first_food_ms": None,
                      "distance_mm": 0.0, "airborne_ms": 0.0}
        self.energy = 1.0
        self.events = []
        self._on_fruit = None
        self._last_xy = None
        self.senses = {}

    def _draw_attack_time(self, t_ms: float) -> float:
        lo, hi = self.cfg.predator_interval_s
        return t_ms + 1000.0 * self.rng.uniform(lo, hi)

    @property
    def wind_vec(self) -> np.ndarray:
        """Direction the air moves TO (unit), times speed (mm/s)."""
        return np.array(self.wind_xy)

    @property
    def wind_xy(self) -> tuple:
        a = math.radians(self.cfg.wind_from_deg + 180.0)
        return (self.cfg.wind_speed_mm_s * math.cos(a), self.cfg.wind_speed_mm_s * math.sin(a))

    def senses_rounded(self) -> dict:
        return {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)
                for k, v in self.senses.items()}

    # ------------------------------------------------------------------ odour
    def mean_odour(self, x: float, y: float) -> np.ndarray:
        """Time-averaged concentration from each fruit (1 ~ at its surface)."""
        U = self.cfg.wind_speed_mm_s
        wx, wy = self.wind_xy
        u = (wx / U, wy / U) if U > 1e-6 else (0.0, 0.0)
        out = np.zeros(len(self.fruits))
        for k, f in enumerate(self.fruits):
            if f.amount <= 0:
                continue
            dx, dy = x - f.x, y - f.y
            d = math.hypot(dx, dy)
            halo = math.exp(-max(0.0, d - f.radius) / 35.0)          # still-air halo
            plume = 0.0
            if U > 1e-6:
                along = dx * u[0] + dy * u[1]
                cross = abs(dx * u[1] - dy * u[0])
                if along > 0:
                    sig = 10.0 + 0.18 * along                        # widens downwind
                    plume = (10.0 / sig) ** 0.5 * math.exp(-cross ** 2 / (2 * sig ** 2)) \
                        * math.exp(-along / 900.0)
            out[k] = f.odour * min(1.0, f.amount * 4) * max(halo, plume)
        return out

    FILAMENT_MS = 40.0          # mean duration of an odour filament (C)

    def odour_at_antennae(self, body, dt_ms: float) -> tuple:
        """Per-fruit concentration at the left and right antenna (arrays).

        With turbulence the fly is either inside a filament (concentration
        above the mean) or in cleaner air between filaments. The fraction of
        time in filaments rises with the mean concentration, so the time
        average equals the mean plume. The antennae are 0.4 mm apart and share
        the filament state; their means differ only by position.
        """
        s = body.state
        hd = math.radians(s.heading_deg)
        hx = s.x_mm + HEAD_OFFSET_MM * math.cos(hd)
        hy = s.y_mm + HEAD_OFFSET_MM * math.sin(hd)
        lx, ly = -math.sin(hd), math.cos(hd)          # unit vector to the fly's left
        cl = self.mean_odour(hx + ANTENNA_HALF_SEP_MM * lx, hy + ANTENNA_HALF_SEP_MM * ly)
        cr = self.mean_odour(hx - ANTENNA_HALF_SEP_MM * lx, hy - ANTENNA_HALF_SEP_MM * ly)
        tot = float(cl.sum() + cr.sum()) / 2
        if not self.cfg.turbulent or tot <= 0:
            return cl, cr
        p_on = min(0.9, 0.1 + 0.8 * min(1.0, tot))
        t_off = self.FILAMENT_MS * (1 - p_on) / p_on
        if self._filament:
            if self.rng.random() < dt_ms / self.FILAMENT_MS:
                self._filament = False
        elif self.rng.random() < dt_ms / t_off:
            self._filament = True
        # filament: mean / p_on (times the gap floor), gap: 5% of the mean
        k = (1 - 0.05 * (1 - p_on)) / p_on if self._filament else 0.05
        return cl * k, cr * k

    # --------------------------------------------------------------- predator
    def _launch(self, body, t_ms: float) -> None:
        cfg = self.cfg
        s = body.state
        az = self.rng.uniform(0, 2 * math.pi)
        el = math.radians(self.rng.uniform(25, 60))
        d = cfg.predator_start_mm
        target = np.array([s.x_mm, s.y_mm, s.z_mm + FLY_HEAD_HEIGHT_MM])
        start = target + d * np.array([math.cos(el) * math.cos(az),
                                       math.cos(el) * math.sin(az), math.sin(el)])
        v = (target - start) / d * cfg.predator_speed_mm_s
        # fly past the target point, then leave
        t_end = t_ms + 1000.0 * (d + 80.0) / cfg.predator_speed_mm_s
        self.predator = Predator(*start, *v, cfg.predator_radius_mm, t_ms, t_end)
        self.stats["attacks"] += 1
        self._event(t_ms, "predator attack")

    def looming(self, body) -> dict:
        """The predator as the fly's eyes see it (head-centred), or inactive."""
        p = self.predator
        if p is None:
            return {"active": False, "half_angle_deg": 0.0, "expansion_rate_deg_s": 0.0,
                    "azimuth_deg": 0.0, "elevation_deg": 0.0}
        s = body.state
        eye = np.array([s.x_mm, s.y_mm, s.z_mm + FLY_HEAD_HEIGHT_MM])
        rel = p.pos() - eye
        d = float(np.linalg.norm(rel))
        d_eff = max(d, p.radius * 1.02)
        theta = math.degrees(math.atan(p.radius / d_eff))
        # d(theta)/dt = -r / (d^2 + r^2) * dd/dt ; dd/dt = v_rel . rel/d
        v_fly = np.array([s.speed_mm_s * math.cos(math.radians(s.heading_deg)),
                          s.speed_mm_s * math.sin(math.radians(s.heading_deg)),
                          getattr(s, "vz_mm_s", 0.0)])
        v_rel = np.array([p.vx, p.vy, p.vz]) - v_fly
        dd = float(v_rel @ rel) / max(d, 1e-6)
        dtheta = math.degrees(-p.radius / (d_eff ** 2 + p.radius ** 2) * dd)
        # direction in head-centred coordinates (azimuth + = right)
        bearing = math.degrees(math.atan2(rel[1], rel[0]))
        az = ((s.heading_deg - bearing) + 180.0) % 360.0 - 180.0
        el = math.degrees(math.atan2(rel[2], math.hypot(rel[0], rel[1])))
        return {"active": True, "half_angle_deg": theta, "expansion_rate_deg_s": dtheta,
                "azimuth_deg": az, "elevation_deg": el, "distance_mm": d}

    # ------------------------------------------------------------------- step
    def step(self, dt_ms: float, t_ms: float, body) -> None:
        """Advance the world by dt_ms after the body has moved."""
        cfg = self.cfg
        s = body.state
        dt_s = dt_ms / 1000.0
        h = cfg.size_mm / 2

        # walls (physics): the arena is a closed box; the fly slides along it
        for attr in ("x_mm", "y_mm"):
            v = getattr(s, attr)
            if abs(v) > h:
                setattr(s, attr, math.copysign(h, v))
                # reflect the heading component into the wall
                hd = math.radians(s.heading_deg)
                cx, cy = math.cos(hd), math.sin(hd)
                if attr == "x_mm":
                    cx = -cx
                else:
                    cy = -cy
                s.heading_deg = math.degrees(math.atan2(cy, cx)) % 360.0
        if s.airborne and s.z_mm > 300.0:              # ceiling
            s.z_mm = 300.0
            s.vz_mm_s = min(0.0, getattr(s, "vz_mm_s", 0.0))

        if self._last_xy is not None:
            self.stats["distance_mm"] += math.hypot(s.x_mm - self._last_xy[0],
                                                    s.y_mm - self._last_xy[1])
        self._last_xy = (s.x_mm, s.y_mm)
        if s.airborne:
            self.stats["airborne_ms"] += dt_ms

        # fruit contact: taste, and eating when the proboscis is extended
        self._on_fruit = None
        if not s.airborne:
            for f in self.fruits:
                if f.amount > 0 and math.hypot(s.x_mm - f.x, s.y_mm - f.y) <= f.radius:
                    self._on_fruit = f
                    break
        f = self._on_fruit
        if f is not None:
            if not getattr(self, "_was_on", False):
                self.stats["fruit_visits"] += 1
                self._event(t_ms, "reached %s fruit" % f.kind)
            self.stats["time_on_fruit_ms"] += dt_ms
            if s.proboscis_extension > 0.5:
                bite = min(f.amount, cfg.eat_rate_per_s * dt_s)
                f.amount -= bite
                self.stats["eaten_" + f.taste] += bite
                if f.taste == "sweet":
                    self.energy = min(1.5, self.energy + bite * 20)
                    if self.stats["first_food_ms"] is None:
                        self.stats["first_food_ms"] = t_ms
                        self._event(t_ms, "first meal")
        self._was_on = f is not None
        # metabolism: flying costs ~5x walking (bookkeeping only)
        self.energy = max(0.0, self.energy - dt_s * (0.004 if not s.airborne else 0.02))

        # predators
        if cfg.predators and self.predator is None and t_ms >= self._next_attack:
            self._launch(body, t_ms)
        p = self.predator
        if p is not None:
            p.x += p.vx * dt_s
            p.y += p.vy * dt_s
            p.z += p.vz * dt_s
            eye = np.array([s.x_mm, s.y_mm, s.z_mm + FLY_HEAD_HEIGHT_MM])
            d = float(np.linalg.norm(p.pos() - eye))
            p.min_dist = min(p.min_dist, d)
            if d <= p.radius + 1.5:
                self.stats["caught"] += 1
                self._event(t_ms, "CAUGHT by predator")
                self.predator = None
                self._next_attack = self._draw_attack_time(t_ms)
                self._respawn(body, t_ms)
            elif t_ms >= p.t_end or p.z < -p.radius:
                self.stats["escaped"] += 1
                self._event(t_ms, "survived attack (closest %.0f mm)" % (p.min_dist - p.radius))
                self.predator = None
                self._next_attack = self._draw_attack_time(t_ms)

    def _respawn(self, body, t_ms: float) -> None:
        """A caught fly is replaced by a new one somewhere else (keeps the run going)."""
        h = self.cfg.size_mm / 2
        body.reset()
        body.state.x_mm, body.state.y_mm = (float(v) for v in self.rng.uniform(-0.8 * h, 0.8 * h, 2))
        body.state.heading_deg = float(self.rng.uniform(0, 360))
        body._t_ms = t_ms
        self._last_xy = None

    def _event(self, t_ms: float, what: str) -> None:
        self.events.append({"t_ms": round(t_ms, 1), "event": what})
        if len(self.events) > 200:
            del self.events[:50]

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        p = self.predator
        return {
            "size_mm": self.cfg.size_mm,
            "wind": {"speed_mm_s": self.cfg.wind_speed_mm_s, "from_deg": self.cfg.wind_from_deg},
            "fruits": [f.as_dict() for f in self.fruits],
            "predator": (None if p is None else
                         {"x": round(p.x, 1), "y": round(p.y, 1), "z": round(p.z, 1),
                          "r": p.radius, "vx": p.vx, "vy": p.vy, "vz": p.vz}),
            "stats": {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in self.stats.items()},
            "energy": round(self.energy, 3),
            "on_fruit": None if self._on_fruit is None else self._on_fruit.kind,
            "senses": self.senses_rounded(),
            "events": self.events[-8:],
        }
