"""
Goal-directed steering through the central complex: a goal in the fan-shaped
body (FC2 neurons) is compared with the heading bump (E-PG) by PFL3 neurons,
which drive the steering descending neurons (DNa02) through the lateral
accessory lobe (LAL).

A. REAL DATA (FlyWire v783):
   - PFL3 (24) and PFL2 (12) neurons: each takes its heading input in the
     protocerebral bridge from E-PG neurons of one glomerulus (glomerulus
     labels from labels.csv via brain/navigation/compass.Compass), so a
     PFL3's "PB angle" = compass.glom_angle(side, glomerulus), the same frame
     CompassDrive uses. Its output side is the LAL hemisphere receiving most of
     its output synapses (contralateral to the soma for every PFL3).
   - FC2 (A/B/C, 85) neurons have no column labels; each one's "goal angle" is
     the synapse-weighted circular mean of the PB angles of the PFL3 neurons it
     contacts in the fan-shaped body (83/85 contact PFL3).
B. PUBLISHED: FC2 neurons hold a goal direction; PFL3 neurons compare it with
   the heading bump and drive turning toward the goal through DNa02; PFL2
   neurons increase forward/turning drive when heading is far from the goal
   (Westeinde et al. 2024, Nature 626:819; Mussells Pires et al. 2024, Nature
   626:808; Matheson et al. 2022, Nat Commun 13:4613).
C. OUR APPROXIMATIONS:
   - The goal is imposed as Poisson drive to FC2 cells with a Gaussian tuning
     over the circular distance between the goal and each cell's goal angle
     (peak PEAK_HZ, width SIGMA_DEG). In a fly the goal is learned or set by
     other circuits; here the "neocortex" (or a test) sets it.
   - GOAL_OFFSET_DEG: the frame between FC2 goal angles and heading angles is
     not given by the anatomy we use (PB->FB projections are offset per
     side), so it is calibrated once so that goal = heading gives no turn.

RESULTS (Mac, 2026-09-25; whole brain, calibrated dynamics, resting ORN
input, CompassDrive heading, FC2 peak 120 Hz, sigma 45 deg):
   - Column structure: PFL3 cells sharing FC2 input pair up across output
     sides only with the mirrored PB angle map (compass.glom_angle); then
     left-output PFL3 prefer heading = goal column + ~51 deg, right-output
     ~ -76 deg, PFL2 ~ +150 deg (far from goal), as published.
   - Open loop: PFL3 R-L changes sign with goal - heading at headings 0, 90
     and 270 deg (zero crossing within ~15 deg of 0; |R-L| 3-8 Hz), partly at
     180 deg; DNa02 R-L follows (e.g. heading 270: +9..+13 Hz for goals
     clockwise, -17..-31 Hz counter-clockwise). PFL2 activity is lowest at
     goal = heading and highest near 180 deg at every heading.
   - PFL3 rates stay low (<= ~10 Hz): FB5A (4 GABAergic tangential cells,
     driven by FC2) is the dominant inhibitory input to PFL3.
   - Closed loop (ForagingBody): with the default spontaneous turning noise
     (90 deg/s SD) heading error is not reduced; at 30 deg/s it drops from
     72-77 deg (goal off) to 42-50 deg (goal on), with a steady ~+45-75 deg
     counter-clockwise offset (left-turn handedness). Go home (goal = bearing
     to home from body position, 80 mm away, 40 s): 3/6 reach home vs 1/6
     without goal (default body); 5/6 vs 1/6 at 30 deg/s noise, paths
     1.1-1.4x the straight line.

Sign convention (fly/body/*): angles in degrees, counter-clockwise positive;
steer > 0 = turn right (DescendingReadout turn_bias > 0), i.e. heading
decreases. A goal counter-clockwise of the heading (goal - heading > 0)
should give a left turn (turn_bias < 0).
"""
from __future__ import annotations

import collections

import numpy as np

PEAK_HZ = 120.0          # the validated strength (tests, exam, dynamics json)
SIGMA_DEG = 45.0
GOAL_OFFSET_DEG = 0.0        # set by calibration (see GoalDrive docstring)


def _cmean(angles, weights):
    w = np.asarray(weights, float)
    if w.sum() <= 0:
        return float("nan"), 0.0
    z = (w * np.exp(1j * np.deg2rad(np.asarray(angles, float)))).sum() / w.sum()
    return float(np.rad2deg(np.angle(z)) % 360.0), float(abs(z))


class GoalCircuit:
    """Column assignments of PFL3 / PFL2 / FC2 derived from connectivity."""

    def __init__(self, connectome, compass):
        n = connectome.neurons
        t = n["primary_type"].fillna("").astype(str).to_numpy()
        side = n["side"].fillna("").astype(str).to_numpy()
        from brain.navigation.compass import glom_angle
        epg_glom = {}
        for (k, s, g), cells in compass.cells.items():
            if k == "EPG":
                for i in cells:
                    epg_glom[int(i)] = glom_angle(s, g)
        WT = connectome.w.T.tocsr()
        W = connectome.w.tocsr()
        self.pfl = {}                          # idx -> (type, pb_angle, strength, out_side)
        for ty in ("PFL3", "PFL2"):
            for j in np.flatnonzero(t == ty):
                r = WT.getrow(j)
                A, Wt = [], []
                for i, w in zip(r.indices, r.data):
                    if int(i) in epg_glom:
                        A.append(epg_glom[int(i)])
                        Wt.append(abs(w))
                ang, st = _cmean(A, Wt)
                o = W.getrow(j)
                out_l = np.abs(o.data[side[o.indices] == "left"]).sum()
                out_r = np.abs(o.data[side[o.indices] == "right"]).sum()
                self.pfl[int(j)] = (ty, ang, st, "L" if out_l > out_r else "R")
        pfl3_ang = {j: v[1] for j, v in self.pfl.items() if v[0] == "PFL3" and v[1] == v[1]}
        self.fc2 = []                          # (idx, goal_angle, strength, synapses)
        for i in np.flatnonzero(np.isin(t, ["FC2A", "FC2B", "FC2C"])):
            r = W.getrow(i)
            A, Wt = [], []
            for j, w in zip(r.indices, r.data):
                if int(j) in pfl3_ang:
                    A.append(pfl3_ang[int(j)])
                    Wt.append(abs(w))
            ang, st = _cmean(A, Wt)
            if Wt:
                self.fc2.append((int(i), ang, st, int(sum(Wt))))
        self.dna02 = {s: np.flatnonzero((t == "DNa02") & (side == sd)) for s, sd in (("L", "left"), ("R", "right"))}

    def pfl_group(self, ty: str, out_side: str) -> np.ndarray:
        return np.array(sorted(j for j, v in self.pfl.items() if v[0] == ty and v[3] == out_side), np.int64)


class GoalDrive:
    """
    Sets a goal direction as FC2 drive and reads the steering it causes.
    Use as a Session stimulus:  session.add_stimulus(drive, drive)
    """

    def __init__(self, circuit: GoalCircuit, goal_deg: float = 0.0, peak_hz: float = PEAK_HZ,
                 sigma_deg: float = SIGMA_DEG, offset_deg: float = GOAL_OFFSET_DEG):
        self.c = circuit
        self.goal = float(goal_deg) % 360.0
        self.peak_hz, self.sigma, self.offset = float(peak_hz), float(sigma_deg), float(offset_deg)
        idx = np.array([f[0] for f in circuit.fc2], np.int64)
        ang = np.array([f[1] for f in circuit.fc2], float)
        order = np.argsort(idx)
        self.indices, self._ang = idx[order], ang[order]
        self.enabled = True
        self.gain = 1.0          # scales the FC2 drive (e.g. valence gating)

    def set_goal(self, goal_deg: float) -> None:
        self.goal = float(goal_deg) % 360.0

    def rates_hz(self, t_ms: float = 0.0, stim=None) -> np.ndarray:
        from brain.sensory.encoders import memo_of
        rm = memo_of(self)
        if not self.enabled:
            return rm.zeros(len(self.indices))
        # set per control step, asked every block: the same (read-only) rates
        return rm.get((self.goal, self.offset, self.gain, self.peak_hz, self.sigma), lambda: (
            self.gain * self.peak_hz * np.exp(
                -0.5 * (((self.goal + self.offset - self._ang + 180.0) % 360.0 - 180.0) / self.sigma) ** 2)))

    def state(self, t_ms: float = 0.0) -> dict:
        return {"kind": "goal", "active": self.enabled, "goal_deg": round(self.goal, 1)}

    def readout(self, spike_counts: np.ndarray, window_ms: float) -> dict:
        """PFL3 / PFL2 output-side rates and DNa02 rates (Hz) over a window."""
        hz = lambda idx: float(spike_counts[idx].sum() / max(len(idx), 1) / (window_ms * 1e-3))
        c = self.c
        return {"PFL3_L": hz(c.pfl_group("PFL3", "L")), "PFL3_R": hz(c.pfl_group("PFL3", "R")),
                "PFL2_L": hz(c.pfl_group("PFL2", "L")), "PFL2_R": hz(c.pfl_group("PFL2", "R")),
                "DNa02_L": hz(c.dna02["L"]), "DNa02_R": hz(c.dna02["R"])}

    @property
    def provenance(self) -> dict:
        return {"drives": {"FC2": int(len(self.indices))}, "peak_hz": self.peak_hz,
                "sigma_deg": self.sigma, "offset_deg": self.offset,
                "source": "Westeinde et al. 2024; Mussells Pires et al. 2024 (Nature 626)"}
