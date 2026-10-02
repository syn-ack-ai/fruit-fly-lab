"""
Fear from meaning, not from geometry: what drives the fly's looming circuit
(LC4 / LPLC2 -> giant fibre DNp01, DNp02 / DNp04 / DNp11) on the robot.

In the fly, anything that expands on the eye fast enough is a predator and
LC4 / LPLC2 fire. A pet robot lives with people: someone walking up, the dock
coming closer as it drives to it, walls sweeping past are not dangers (the
flyvis eye in Habitat: escape states ~46% of the time, ROADMAP.md section 0).
So on the robot the looming neurons are driven by an APPRAISAL of what is
seen and heard: the vision model (what it is) and the language model (what
it means: "Milo, watch out!", a hazard it is told about). Geometry -- where
the thing is (the camera's or lidar's bearing) -- only aims the stimulus; the
connectome downstream decides what the startle looks like.

    appraisals (source, level 0..1, azimuth, until)  ->  the strongest live one
        -> a virtual looming disc at that azimuth: half-angle and expansion
           rate grow with the level  -> brain.sensory.encoders.LoomingEncoder

The same idea in mammals: the amygdala's fast route (crude looming, the
superior colliculus) is gated and overridden by cortical appraisal of meaning
(LeDoux 1996; the "high road"); looming escape in mice depends on context
(Evans et al. 2018 Nature 558:590). Here only the high road is kept: a robot's
crude route (anything approaching) is what made Milo flee its own home.

C. APPROXIMATIONS: robot engineering, not a fly circuit. The level -> looming
mapping (HALF_DEG, EXP_DEG_S) was set so that level 1 makes the escape command
cross its threshold and level <= 0.3 does not (tests/test_threat.py and
experiments/fear_levels.py; the trimmed robot brain). FLY_LOOM=geometry gives
back the fly's own looming from the camera and the lidar (the fly lab).
"""
from __future__ import annotations

import math
import os

LOOM_MODE = os.environ.get("FLY_LOOM", "meaning")
if LOOM_MODE not in ("meaning", "geometry"):
    raise SystemExit(f"FLY_LOOM={LOOM_MODE!r}: 'meaning' (default) or 'geometry'")

HALF_DEG = (3.0, 30.0)        # the disc's half-angle at level 0+ and 1
EXP_DEG_S = 500.0             # its expansion rate at level 1
GAMMA = 2.5                   # drive = level ** GAMMA: the circuit is steep (a 70 deg/s disc
                              # already crosses the escape threshold), "wary" must stay below it
FADE_S = {"vision": 2.0,       # time constant of a fear's fading: a look comes every
          "llm": 4.0,          # ~1 s and its answer flickers (knife test 2026-10-01: 2, 1,
          "test": 2.0,         # 0, 1, 1, 0, 1), a reply comes once and lasts a while
          "fast": 1.5}         # the fast route (FastDanger): a rush is over quickly
FLOOR = 0.05                  # a fear faded below this is gone
LEVELS = {0: 0.0, 1: 0.25, 2: 1.0}   # the language model's fear 0 / 1 (wary) / 2 (danger)


class Fear:
    """The appraisals now in force, as a stimulus for LoomingEncoder.

    appraise() from any source. Fear lingers, as in animals: each source's
    level fades exponentially (FADE_S) from its last raise; a stronger
    appraisal replaces it, a weaker one or "no danger" (0) leaves it fading
    rather than ending it at once. The stimulus is the strongest source now.
    Times are the caller's clock (simulated time in Habitat, the rover's clock
    on the robot): a clock that goes back (a new day) clears all."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._a = {}                 # source -> (level, azimuth_deg, elevation_deg, t_set, fade_s, what)
        self._t = None
        self.last = {"active": False}
        self.count = 0               # appraisals with level > 0 (for the logs)

    @staticmethod
    def _now(a, t_s: float) -> float:
        return a[0] * math.exp(-max(t_s - a[3], 0.0) / a[4])

    def appraise(self, source: str, level: float, t_s: float, azimuth_deg: float | None = None,
                 elevation_deg: float = 0.0, fade_s: float | None = None, what: str = "") -> None:
        """level 0..1; azimuth + = right of the body's heading, None = straight
        ahead. Raises this source's fear to `level` if that is above what is
        left of it; otherwise it keeps fading."""
        self._clock(t_s)
        level = min(max(float(level), 0.0), 1.0)
        if level <= 0.0:
            return
        self.count += 1
        old = self._a.get(source)
        if old is not None and self._now(old, t_s) > level:
            return
        fade = FADE_S.get(source, 2.0) if fade_s is None else float(fade_s)
        self._a[source] = (level, 0.0 if azimuth_deg is None else float(azimuth_deg),
                           float(elevation_deg), t_s, fade, what)

    def _clock(self, t_s: float) -> None:
        if self._t is not None and t_s < self._t:
            self._a.clear()
        self._t = t_s

    def update(self, t_s: float) -> dict:
        """The strongest fear now (once per control step); faded ones are dropped."""
        self._clock(t_s)
        now = {k: self._now(a, t_s) for k, a in self._a.items()}
        for k in [k for k, v in now.items() if v < FLOOR]:
            del self._a[k], now[k]
        if not self._a:
            self.last = {"active": False}
            return self.last
        src = max(now, key=now.get)
        lv, (_, az, el, _, _, what) = now[src], self._a[src]
        g = lv ** GAMMA
        self.last = {"active": True, "source": src, "level": round(lv, 3), "azimuth_deg": az,
                     "elevation_deg": el, "what": what,
                     "half_angle_deg": HALF_DEG[0] + (HALF_DEG[1] - HALF_DEG[0]) * g,
                     "expansion_rate_deg_s": EXP_DEG_S * g}
        return self.last

    def state(self, t_ms: float) -> dict:
        """LoomingEncoder's stimulus (brain.sensory.encoders.LoomingStimulus)."""
        L = self.last
        return {"t_ms": t_ms, "source": "fear", "active": L["active"],
                "azimuth_deg": L.get("azimuth_deg", 0.0), "elevation_deg": L.get("elevation_deg", 0.0),
                "half_angle_deg": L.get("half_angle_deg", 0.0),
                "expansion_rate_deg_s": L.get("expansion_rate_deg_s", 0.0)}


def llm_level(fear) -> float:
    """The language model's fear field (0, 1, 2) as a level."""
    try:
        return LEVELS.get(int(fear), 0.0)
    except (TypeError, ValueError):
        return 0.0


# ------------------------------------------------------------------ fast route
# Something RUSHING at Milo (faster than walking pace, about to arrive) is
# seen by the lidar 10 times a second (robot/lidar.LidarLooming: points that
# came closer than a still world predicts from the robot's own motion). What
# it is comes from what the camera last saw on that side (the person or toy
# detector, robot/seeing.py's labels), and whether that is a danger when it
# rushes in was asked of the language model once per kind of thing
# (data/metadata/fast_danger_valence.json): a foot or a shoe is, a person or a
# ball is to be wary of, a teddy bear is not. Nothing waits on a slow model:
# the reflex is fast, the meaning was looked up beforehand.
FAST_MS = 1.6                 # closing speed beyond self-motion: faster than walking (~1.2 m/s)
TTC_S = 0.6                   # arriving within this
VALENCE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "metadata", "fast_danger_valence.json")
UNKNOWN = "something unknown"


def load_valence(path: str = VALENCE_FILE) -> dict:
    """label -> 0 / 1 / 2 (the language model's answer for that thing rushing in)."""
    import json
    with open(path) as fh:
        return {k: int(v["fear"]) for k, v in json.load(fh)["valence"].items()}


class FastDanger:
    """The fast route: lidar rush x what it is -> a Fear appraisal ("fast")."""

    def __init__(self, valence: dict | None = None):
        self.valence = valence if valence is not None else load_valence()
        self.last = {"rush": False}
        self.count = {0: 0, 1: 0, 2: 0}

    def what(self, az: float, scene: list) -> tuple:
        """The most dangerous description of what is at bearing az: scene =
        [(label, centre_az, half_width_deg)]; (UNKNOWN, its valence) if none."""
        best = None
        for label, c, half in scene:
            if abs(((az - c + 180.0) % 360.0) - 180.0) <= half:
                v = self.valence.get(label, self.valence.get(UNKNOWN, 1))
                if best is None or v > best[1]:
                    best = (label, v)
        return best or (UNKNOWN, self.valence.get(UNKNOWN, 1))

    def step(self, t: float, rush: dict, scene: list, fear: Fear) -> dict:
        """rush = LidarLooming.last ({"active", "azimuth_deg", "clear_m",
        "closing_ms"}); appraises `fear` when something rushes in."""
        self.last = {"rush": False}
        if not rush.get("active") or not rush.get("confirmed", True) or rush.get("closing_ms", 0.0) < FAST_MS:
            return self.last
        ttc = rush.get("clear_m", rush.get("range_m", 9.0)) / max(rush["closing_ms"], 1e-6)
        if ttc > TTC_S:
            return self.last
        az = float(rush["azimuth_deg"])
        label, v = self.what(az, scene)
        self.count[v] += 1
        self.last = {"rush": True, "what": label, "valence": v, "az": round(az), "ttc_s": round(ttc, 2),
                     "closing_ms": round(rush["closing_ms"], 2)}
        if v:
            fear.appraise("fast", LEVELS[v], t, azimuth_deg=az, what=f"{label} rushing in")
        return self.last

