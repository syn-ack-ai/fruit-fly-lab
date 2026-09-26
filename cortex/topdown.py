"""
How the neocortex talks to the fly brain: top-down drive to a few real
FlyWire neuron populations. The cortex never sets the body's velocity; it only
biases the connectome, and the connectome's descending neurons move the body.

Channels (each a Session stimulus; overlapping drives take the maximum):

  compass   E-PG heading input from the body's odometry yaw
            (brain/navigation/compass.CompassDrive; landmark-like input).
  goal      a goal direction as FC2 drive; the central complex compares it
            with the heading and steers through PFL3 -> DNa02
            (brain/navigation/goal.GoalDrive).
  attend    a "phantom" small object for the pursuit pathway: LC10a drive at an
            azimuth, as if something worth chasing were there. LC10a -> AOTU019/
            025 -> DNa02 is the fly's strongest steering route in our model
            (it tracks a person to 10-20 deg).
  dopamine  the cortex critic's reward-prediction error into the mushroom
            body's teaching neurons: positive -> reward PAM05-08, negative ->
            punishment PPL101/103 (the same DANs taste drives in the home).
  arousal   interest in the person: a gain (0..1) on the camera's LC10a drive
            for the person (robot.head.ObjectEncoder.arousal). Pursuit is
            state-gated in flies (P1 arousal gates the LC10a pathway in courting
            males; Hindmarsh Sten et al. 2021); the cortex sets the state.

PROVENANCE
----------
A. REAL DATA : the neurons driven (E-PG, FC2, LC10a, PAM05-08, PPL101/103).
B. PUBLISHED : FC2 goal -> PFL3 steering (Westeinde et al. 2024; Mussells Pires
   et al. 2024); LC10a pursuit (Ribeiro et al. 2018; Hindmarsh Sten et al.
   2021); DAN teaching signals (Aso et al. 2014).
C. OUR APPROXIMATIONS : the cortex is not a fly structure. In mammals cortex
   biases subcortical circuits (basal ganglia, superior colliculus, midbrain
   dopamine); here the same idea is applied to the fly brain's goal, pursuit
   and teaching inputs. Nothing about how to move is encoded in these drives.
"""
from __future__ import annotations

import math

import numpy as np

ATTEND_SIZE_DEG = 15.0       # ObjectEncoder.PEAK_DEG: the preferred target size
ATTEND_MAX_AZ = 90.0         # a goal behind the pet is placed at the side
DAN_MAX_HZ = 100.0           # as taste reinforcement (fly/world/senses.py)
REWARD_DANS = ("PAM05", "PAM06", "PAM07", "PAM08")
PUNISH_DANS = ("PPL101", "PPL103")


class AttendEncoder:
    """LC10a drive for a phantom object (same receptive fields and tuning as
    robot.head.ObjectEncoder, which drives LC10a from the camera)."""

    def __init__(self, obj_encoder):
        o = obj_encoder
        self.indices, self._az, self._el, self._sigma = o.indices, o._az, o._el, o._sigma
        self.max_hz = o.MAX_HZ
        self.az, self.gain = 0.0, 0.0

    def set(self, az_deg: float, gain: float) -> None:
        self.az = float(np.clip(az_deg, -ATTEND_MAX_AZ, ATTEND_MAX_AZ))
        self.gain = float(np.clip(gain, 0.0, 1.0))

    def rates_hz(self, t_ms: float = 0.0, stim=None) -> np.ndarray:
        if self.gain <= 0:
            return np.zeros(len(self.indices))
        from simulation.stimuli.looming import angular_distance_deg
        half = ATTEND_SIZE_DEG / 2
        d = angular_distance_deg(self.az, 0.0, self._az, self._el)
        edge = np.maximum(0.0, d - half)
        return self.gain * self.max_hz * np.exp(-edge ** 2 / (2 * self._sigma ** 2))

    def state(self, t_ms: float = 0.0) -> dict:
        return {"kind": "cortex_attend", "active": self.gain > 0,
                "az_deg": round(self.az, 1), "gain": round(self.gain, 2)}


class DopamineEncoder:
    """Critic reward-prediction error -> PAM (positive) / PPL1 (negative)."""

    def __init__(self, connectome):
        t = connectome.neurons["primary_type"].fillna("").astype(str).to_numpy()
        pam = np.flatnonzero(np.isin(t, REWARD_DANS))
        ppl = np.flatnonzero(np.isin(t, PUNISH_DANS))
        self.indices = np.concatenate([pam, ppl])
        self._is_pam = np.concatenate([np.ones(len(pam), bool), np.zeros(len(ppl), bool)])
        o = np.argsort(self.indices)
        self.indices, self._is_pam = self.indices[o], self._is_pam[o]
        self.rpe = 0.0

    def set(self, rpe: float) -> None:
        self.rpe = float(np.clip(rpe, -1.0, 1.0))

    def rates_hz(self, t_ms: float = 0.0, stim=None) -> np.ndarray:
        r = DAN_MAX_HZ * self.rpe
        return np.where(self._is_pam, max(r, 0.0), max(-r, 0.0))

    def state(self, t_ms: float = 0.0) -> dict:
        return {"kind": "cortex_dopamine", "active": self.rpe != 0.0, "rpe": round(self.rpe, 3)}


class TopDown:
    """All cortex -> fly channels for one Session. Call attach() after every
    Session.reset() (which clears stimuli), then apply() each control step."""

    def __init__(self, connectome, obj_encoder, channels=("goal", "attend", "dopamine")):
        from brain.navigation.compass import Compass, CompassDrive
        from brain.navigation.goal import GoalCircuit, GoalDrive
        self.channels = set(channels)
        cx = Compass(connectome)
        self.compass = CompassDrive(cx)
        self.goal = GoalDrive(GoalCircuit(connectome, cx))
        self.goal.gain = 0.0
        self.attend = AttendEncoder(obj_encoder)
        self.dopamine = DopamineEncoder(connectome)
        self.obj = obj_encoder
        self.last = {}

    def attach(self, ses) -> None:
        if "goal" in self.channels:
            ses.add_stimulus(self.compass, self.compass)
            ses.add_stimulus(self.goal, self.goal)
        if "attend" in self.channels:
            ses.add_stimulus(self.attend, self.attend)
        if "dopamine" in self.channels:
            ses.add_stimulus(self.dopamine, self.dopamine)

    def apply(self, heading_deg: float, cmd: dict) -> None:
        """cmd: goal_deg (world frame, CCW) and goal_gain; attend_az (+ = right)
        and attend_gain are derived from the goal unless given; rpe."""
        self.compass.set_heading(heading_deg)
        g = cmd.get("goal_deg")
        gain = float(cmd.get("goal_gain", 0.0)) if g is not None else 0.0
        if g is not None:
            self.goal.set_goal(g)
        self.goal.gain = gain if "goal" in self.channels else 0.0
        if "attend_az" in cmd:
            az, again = cmd["attend_az"], cmd.get("attend_gain", 0.0)
        elif g is not None:
            az, again = goal_azimuth(g, heading_deg), gain
        else:
            az, again = 0.0, 0.0
        self.attend.set(az, again if "attend" in self.channels else 0.0)
        self.dopamine.set(cmd.get("rpe", 0.0) if "dopamine" in self.channels else 0.0)
        arousal = float(np.clip(cmd.get("arousal", 1.0), 0.0, 1.0)) if "arousal" in self.channels else 1.0
        self.obj.arousal = arousal
        self.last = {"goal_deg": None if g is None else round(g % 360.0, 1), "goal_gain": round(gain, 2),
                     "attend_az": round(self.attend.az, 1), "attend_gain": round(self.attend.gain, 2),
                     "rpe": round(self.dopamine.rpe, 3), "arousal": round(arousal, 2)}


def goal_azimuth(goal_deg: float, heading_deg: float) -> float:
    """Where the goal lies in the pet's view (deg, + = right): a goal
    counter-clockwise of the heading is to the LEFT."""
    return -(((goal_deg - heading_deg) + 180.0) % 360.0 - 180.0)


def bearing_deg(frm_xz, to_xz) -> float:
    """World bearing from one floor point to another in the robot's yaw frame
    (habitat_server: yaw is CCW from +x in the (x, -z) plane)."""
    return math.degrees(math.atan2(-(to_xz[1] - frm_xz[1]), to_xz[0] - frm_xz[0])) % 360.0
