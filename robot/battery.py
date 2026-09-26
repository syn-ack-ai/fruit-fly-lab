"""
The pet's battery is its stomach: "hunger" is how low the charge is, and
"eating" is charging at the dock (the connectome's feeding circuit still does
the eating: the proboscis motor neurons must hold the proboscis out at the dock).

    hunger = 0 above FULL (90%), rising linearly to 1 at HUNGRY (20%)

On the robot, charge() / drain() are replaced by the measured state of charge.
In simulation (C. APPROXIMATION) the charge drains at IDLE per second, more when
moving (MOVE per second at 0.5 m/s), less when resting, and fills at CHARGE per
second while docked and "eating". Rates are scaled to Habitat's 2-minute days:
a pet that is active all day uses ~0.2 of a full charge, so it needs the dock
every 2-3 days (the first rates, 3x faster, emptied a full battery in under
two days and left no time to find the dock; run of 2026-09-26, stopped).
"""
from __future__ import annotations

FULL, HUNGRY = 0.90, 0.20
IDLE, MOVE, REST = 0.0006, 0.0016, 0.0002    # fraction of charge per second
CHARGE = 0.02                                 # per second, docked and eating
V_REF = 0.5                                   # m/s for MOVE
RESCUE = 0.3          # a flat pet is put on its dock overnight (counted as a failure)


class Battery:
    def __init__(self, soc: float = 0.7):
        self.soc = float(soc)
        self.stats = {"start": round(self.soc, 3), "min": self.soc, "charged": 0.0,
                      "low_s": 0.0, "flat_s": 0.0, "rescued": False}

    def day_start(self) -> None:
        """A new day. A pet that ran flat cannot reach its dock, so (as an owner
        would) it is carried there overnight and starts at RESCUE; the rescue is
        recorded as a failure (review 2026-09-26: otherwise every later day of
        that lifetime was dead data)."""
        rescued = self.flat
        if rescued:
            self.soc = RESCUE
        self.stats = {"start": round(self.soc, 3), "min": self.soc, "charged": 0.0,
                      "low_s": 0.0, "flat_s": 0.0, "rescued": rescued}

    @property
    def hunger(self) -> float:
        return min(1.0, max(0.0, (FULL - self.soc) / (FULL - HUNGRY)))

    @property
    def flat(self) -> bool:
        return self.soc <= 0.0

    def step(self, dt: float, speed: float, resting: bool, charging: bool) -> None:
        use = (REST if resting else IDLE) + MOVE * min(1.0, abs(speed) / V_REF)
        gain = CHARGE if charging and self.soc < 1.0 else 0.0
        self.soc = min(1.0, max(0.0, self.soc + (gain - use) * dt))
        st = self.stats
        st["charged"] += gain * dt
        st["min"] = min(st["min"], self.soc)
        st["low_s"] += dt if self.soc < HUNGRY else 0.0
        st["flat_s"] += dt if self.flat else 0.0

    def summary(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.stats.items()} | {"end": round(self.soc, 3)}
