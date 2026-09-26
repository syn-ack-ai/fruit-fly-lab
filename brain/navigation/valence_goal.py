"""
Learned odour valence -> a central-complex goal: what the mushroom body has
learned about the current odour switches on (or reverses) a navigational goal.

PROVENANCE
----------
A. REAL DATA : Kenyon cell -> MBON synapses and their learned weights
   (brain/plasticity/mushroom_body.py); which dopamine neurons teach each
   MBON (real DAN -> MBON synapse counts); the FC2 goal cells and PFL3/PFL2 ->
   DNa02/DNa03 steering path (brain/navigation/goal.py). In FlyWire the
   anatomical route MBON12/MBON13 -> FB5AB -> hDeltaC -> FC2B exists, but in
   the model conditioning changes MBON output (avoidance MBONs 96 -> 85 Hz for
   a rewarded odour) without changing FB5AB/hDeltaC/FC2/PFL activity.
B. PUBLISHED :
   - Odour does not point the way by itself: MBON and lateral-horn outputs
     converge on the fan-shaped-body tangential neuron FB5AB, which GATES
     hDeltaC, whose bump holds a direction (upwind, from wind-tuned PFNs);
     PFL3 compares that goal with heading and steers (Matheson et al. 2022,
     Nat Commun 13:4613; addendum 2024: the hDeltaC line may label hDeltaK).
   - MBONs in compartments taught by reward (PAM) dopamine neurons promote
     avoidance, those taught by punishment (PPL1) promote approach; learning
     depresses KC -> MBON synapses, so a rewarded odour loses avoidance-MBON
     drive (Aso et al. 2014 eLife 3:e04580; Owald & Waddell 2015).
   - The goal is held by FC2 and compared with heading by PFL3/PFL2
     (Westeinde et al. 2024; Mussells Pires et al. 2024).
C. OUR APPROXIMATIONS (a model extension standing in for the FB5AB gate):
   - Learned valence of the current odour = the learning-induced loss of
     KC -> MBON drive from the currently active Kenyon cells, avoidance MBONs
     counted positive and approach MBONs negative, normalised by that drive
     (in [-1, 1]; 0 for a naive fly), scaled down when the odour-evoked drive
     is weak (DRIVE_REF). Only odour-evoked (repeated) KC firing
     counts (above a slow per-KC baseline and a one-spike floor): the model's
     KCs fire spontaneously, some tonically, and their synapses are also
     depressed during training, which otherwise gave a training-dependent
     valence even in clean air.
   - Goal direction: upwind when the fly senses airflow (allocentric, from its
     heading and the sensed wind angle), otherwise the direction in which the
     odour rose while it moved (an average of d(odour)/dt x heading vector).
     Negative valence reverses the goal (downwind / down-gradient).
   - Gate: the FC2 drive scales with |valence| (full at VALENCE_FULL).
"""
from __future__ import annotations

import math

import numpy as np

# calibrated on the model: after 4 pairings (experiment 05) a trained odour
# presented alone reads +0.03..+0.05 (rewarded) / -0.03..-0.06 (punished)
VALENCE_FULL = 0.05          # |valence| giving the full FC2 drive
VALENCE_MIN = 0.01           # below this, no goal
KC_TAU_MS = 200.0            # smoothing of Kenyon cell activity
KC_FLOOR = 1.0               # trace discounted per KC: single spontaneous spikes don't count
KC_BASE_TAU_MS = 5000.0      # slow per-KC baseline (tonic firing) subtracted
DRIVE_REF = 500.0            # evoked KC->MBON drive (trace x edges) below which valence
                             # is scaled down: clean air ~20-40, a fruit odour 500-9000
GRAD_TAU_MS = 1500.0         # memory of the experienced odour gradient
MIN_AIRSPEED = 50.0          # mm/s of sensed airflow to use the upwind goal


class ValenceGoal:
    def __init__(self, mb, goal_drive, compass_drive):
        self.mb, self.goal, self.compass = mb, goal_drive, compass_drive
        # valence sign of each KC->MBON edge: + if its MBON is taught by reward
        # (PAM: an avoidance MBON), - if by punishment (PPL1: approach MBON)
        dan_types = mb.types[mb.dan]
        pam = np.char.startswith(dan_types.astype(str), "PAM")
        ppl = np.char.startswith(dan_types.astype(str), "PPL1")
        A = mb.teacher                                   # DAN x MBON (fractions)
        pam_share = A[pam].sum(0)
        ppl_share = A[ppl].sum(0)
        mbon_sign = np.where(pam_share > 2 * ppl_share, 1.0,
                             np.where(ppl_share > 2 * pam_share, -1.0, 0.0))
        self.edge_sign = mbon_sign[mb.edge_mbon]
        self.kc_act = np.zeros(len(mb.kc))
        self.kc_base = np.zeros(len(mb.kc))
        self.valence = 0.0
        self.drive = 0.0
        self._grad = np.zeros(2)
        self._last_c = None
        self.goal.enabled = False

    def reset(self) -> None:
        """A new episode: ongoing activity estimates are cleared (the learned
        mushroom-body weights are not)."""
        self.kc_act[:] = 0.0
        self.kc_base[:] = 0.0
        self.valence = self.drive = 0.0
        self._grad[:] = 0.0
        self._last_c = None
        self.goal.enabled = False

    # ----------------------------------------------------------------- update
    def step(self, spikes: np.ndarray, dt_ms: float, heading_deg: float,
             odour: float | None = None, air_from_deg: float | None = None,
             airspeed: float = 0.0) -> None:
        mb = self.mb
        k = mb._kc_pos[spikes]
        k = k[k >= 0]
        self.kc_act *= math.exp(-dt_ms / KC_TAU_MS)
        if k.size:
            np.add.at(self.kc_act, k, 1.0)
        self.kc_base += (self.kc_act - self.kc_base) * (1.0 - math.exp(-dt_ms / KC_BASE_TAU_MS))
        a = np.maximum(self.kc_act - self.kc_base - KC_FLOOR, 0.0)[mb.edge_kc]
        drive = a.sum()
        self.drive = float(drive)
        if drive > 0:
            self.valence = float((a * (1.0 - mb.weights) * self.edge_sign).sum() / max(drive, DRIVE_REF))
        else:
            self.valence *= math.exp(-dt_ms / KC_TAU_MS)
        # direction: upwind if airflow is sensed, else the experienced gradient
        h = math.radians(heading_deg)
        if odour is not None:
            if self._last_c is not None:
                dc = (odour - self._last_c) / max(dt_ms, 1e-6)
                decay = math.exp(-dt_ms / GRAD_TAU_MS)
                self._grad = self._grad * decay + dc * np.array([math.cos(h), math.sin(h)])
            self._last_c = odour
        if air_from_deg is not None and airspeed >= MIN_AIRSPEED:
            # air comes FROM heading - air_from (air_from: + = from the right)
            direction = (heading_deg - air_from_deg) % 360.0
        elif np.hypot(*self._grad) > 0:
            direction = math.degrees(math.atan2(self._grad[1], self._grad[0])) % 360.0
        else:
            direction = None
        v = self.valence
        if direction is None or abs(v) < VALENCE_MIN:
            self.goal.enabled = False
            return
        if v < 0:
            direction = (direction + 180.0) % 360.0
        self.goal.enabled = True
        self.goal.set_goal(direction)
        self.goal.gain = min(1.0, abs(v) / VALENCE_FULL)

    def state(self) -> dict:
        return {"valence": round(self.valence, 4), "goal_on": bool(self.goal.enabled),
                "goal_deg": round(self.goal.goal, 1),
                "gain": round(getattr(self.goal, "gain", 1.0), 2)}
