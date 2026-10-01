"""
The central-complex heading compass, read and driven through real neurons.

A. REAL DATA : E-PG, P-EN (PEN1, PEN2), P-EG and Delta7 neurons and their
   protocerebral-bridge (PB) glomeruli, from FlyWire v783 community labels
   (e.g. "EPG_L5"); PB side = soma side.
B. PUBLISHED : the E-PG population carries a bump of activity whose position
   in the PB (two copies, glomeruli 1-8 per side, 45 deg per glomerulus)
   encodes heading, persists in darkness, and is moved by P-EN neurons, which
   receive angular-velocity input asymmetrically on the two sides (Seelig &
   Jayaraman 2015, Nature 521:186; Green et al. 2017, Nature 546:101;
   Turner-Evans et al. 2017, eLife 6:e23496).
C. OUR APPROXIMATIONS : the heading read-out is the circular mean over
   glomeruli of E-PG spike counts per side.

PB angle convention (glom_angle): heading theta sits at left glomerulus
1 - theta/45 and right glomerulus 1 + theta/45 (mod 8). The two PB bumps move
in parallel across the bridge, so glomerulus index runs in opposite angular
directions in the two halves. Checked in FlyWire: E-PG cells of left
glomerulus g share Delta7 targets with right glomerulus (10 - g) mod 8
(L1-R1, L2-R8, L3-R7, L5-R5, L7-R3), and PFL3 neurons sharing FC2 input in
the fan-shaped body pair up consistently only under this mapping. Which of
the two directions counts as counter-clockwise is not fixed by these data
(heading is supplied from odometry, not integrated by P-EN); it is chosen so
that the PFL3 -> LAL -> DNa02 pathway turns the fly toward its goal
(brain/navigation/goal.py).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config

TYPES = {"EPG": "EPG", "PEN_a/PEN1": "PEN1", "PEN_b/PEN2": "PEN2", "PEG": "PEG",
         "PEN_a(PEN1)": "PEN1", "PEN_b(PEN2)": "PEN2"}      # the last two: MaleCNS type names


def glom_angle(side: str, glom: int) -> float:
    """Heading angle (deg) represented by PB glomerulus `glom` on `side`."""
    a = (glom - 1) * 45.0
    return (-a) % 360.0 if side == "L" else a % 360.0


class Compass:
    def __init__(self, connectome):
        n = connectome.neurons
        t = n["primary_type"].fillna("").astype(str)
        lab = pd.read_csv(config.LABELS_CSV, usecols=["root_id", "label"])
        want = set(n[t.isin(list(TYPES))].root_id)
        lab = lab[lab.root_id.isin(want)].groupby("root_id").label.apply(" ".join)
        self.cells = {}                     # (key, side 'L'/'R', glomerulus) -> idx array
        for ty, key in TYPES.items():
            sub = n[t == ty]
            for idx, rid, side in zip(sub["idx"], sub["root_id"], sub["side"].astype(str)):
                nums = re.findall(key + r"_[LR](\d)", lab.get(rid, ""))
                if not nums:
                    continue
                k = (key, side[0].upper(), int(nums[0]))
                self.cells.setdefault(k, []).append(int(idx))
        self.cells = {k: np.array(v, np.int64) for k, v in self.cells.items()}

    def group(self, key: str, side: str | None = None, glom=None) -> np.ndarray:
        out = [v for (k, s, g), v in self.cells.items()
               if k == key and (side is None or s == side)
               and (glom is None or g in np.atleast_1d(glom))]
        return np.concatenate(out) if out else np.empty(0, np.int64)

    def profile(self, spike_counts: np.ndarray, side: str, key: str = "EPG") -> np.ndarray:
        """Spike counts per PB glomerulus 1..8 on one side."""
        return np.array([spike_counts[self.group(key, side, g)].sum() for g in range(1, 9)], float)

    def phase(self, spike_counts: np.ndarray, side: str) -> tuple[float, float]:
        """Bump position (deg, glomerulus 1 = 0 deg) and strength (0-1) on one side."""
        p = self.profile(spike_counts, side)
        if p.sum() <= 0:
            return float("nan"), 0.0
        ang = np.deg2rad([glom_angle(side, g) for g in range(1, 9)])
        z = (p * np.exp(1j * ang)).sum() / p.sum()
        return float(np.rad2deg(np.angle(z)) % 360.0), float(abs(z))


# --------------------------------------------------------------------------- #
# Heading input (calibrated 2026-09-25; see data/metadata/dynamics_calibrated.json
# "central_complex")
# --------------------------------------------------------------------------- #
LANDMARK_PEAK_HZ = 120.0     # Kakaria & de Bivort 2017: sensory-like input to E-PG subsets
LANDMARK_SIGMA_DEG = 30.0


class CompassDrive:
    """
    Puts a heading into the E-PG compass as a landmark-like input and reads the
    bump back as a heading.

    Why a landmark input rather than angular velocity: in the whole-brain model
    the E-PG ring holds a bump only when its internal synapses are strengthened
    (x1.3-1.5) and its adaptation removed, and then the bump is pinned to one or
    two preferred positions; angular-velocity input to P-EN1 (graded or Poisson,
    either side) did not rotate it. A Poisson drive to the E-PGs at the current
    heading (the route visual ring neurons provide in the fly; the way Kakaria &
    de Bivort 2017 moved their spiking PB/EB bump) makes the unmodified ring
    track heading within ~5 deg at 45-180 deg/s. The heading itself comes from
    the body: integrated angular velocity here (the robot's IMU/odometry, or the
    simulated body), i.e. path integration is done outside the connectome.

    Convention: heading in degrees, counter-clockwise positive (fly/body/*);
    heading theta drives each E-PG with a Gaussian of circular distance to its
    glomerulus angle (glom_angle: mirrored between PB halves, as in the fly),
    so the read-out bump phase equals the heading (the landmark-to-bump
    offset is arbitrary in a real fly).

    Use as a Session stimulus:  session.add_stimulus(drive, drive)
    """

    def __init__(self, compass: Compass, heading_deg: float = 0.0,
                 peak_hz: float = LANDMARK_PEAK_HZ, sigma_deg: float = LANDMARK_SIGMA_DEG):
        self.compass = compass
        self.heading = float(heading_deg) % 360.0
        self.peak_hz, self.sigma = float(peak_hz), float(sigma_deg)
        idx, ang = [], []
        for side in ("L", "R"):
            for g in range(1, 9):
                cells = compass.group("EPG", side, g)
                idx.append(cells)
                ang.append(np.full(len(cells), glom_angle(side, g)))
        idx, ang = np.concatenate(idx), np.concatenate(ang)
        order = np.argsort(idx)
        self.indices, self._ang = idx[order], ang[order]
        self.enabled = True

    # ------------------------------------------------------------------ input
    def set_heading(self, heading_deg: float) -> None:
        self.heading = float(heading_deg) % 360.0

    def turn(self, omega_deg_s: float, dt_ms: float) -> None:
        """Integrate angular velocity (+ = counter-clockwise / left turn)."""
        self.heading = (self.heading + omega_deg_s * dt_ms / 1000.0) % 360.0

    def rates_hz(self, t_ms: float = 0.0, stim=None) -> np.ndarray:
        from brain.sensory.encoders import memo_of
        rm = memo_of(self)
        if not self.enabled:
            return rm.zeros(len(self.indices))
        # unchanged while not turning: the same (read-only) rates
        return rm.get((self.heading, self.peak_hz, self.sigma), lambda: self.peak_hz * np.exp(
            -0.5 * (((self.heading - self._ang + 180.0) % 360.0 - 180.0) / self.sigma) ** 2))

    def state(self, t_ms: float = 0.0) -> dict:
        return {"kind": "compass", "active": self.enabled, "heading_deg": round(self.heading, 1)}

    # --------------------------------------------------------------- read-out
    def heading_estimate(self, spike_counts: np.ndarray) -> tuple[float, float]:
        """Heading (deg) and bump strength from E-PG spikes over a window
        (both PB sides pooled)."""
        z, tot = 0j, 0.0
        for side in ("L", "R"):
            p = self.compass.profile(spike_counts, side)
            ang = np.deg2rad([glom_angle(side, g) for g in range(1, 9)])
            z += (p * np.exp(1j * ang)).sum()
            tot += p.sum()
        if tot <= 0:
            return float("nan"), 0.0
        z /= tot
        return float(np.rad2deg(np.angle(z)) % 360.0), float(abs(z))

    @property
    def provenance(self) -> dict:
        return {"drives": {"EPG": int(len(self.indices))}, "peak_hz": self.peak_hz,
                "sigma_deg": self.sigma,
                "source": ("Seelig & Jayaraman 2015; Kakaria & de Bivort 2017 "
                           "(Front Behav Neurosci 11:8); heading from body odometry")}
