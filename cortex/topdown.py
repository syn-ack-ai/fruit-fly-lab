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
  rest      sleep pressure (0..1): drive to the ER5 ring neurons (sleep drive;
            Liu et al. 2016 Cell 165:1347) plus the body's stand-in for sleep-
            promoting nerve-cord neurons (fly/body/foraging_body.py rest_level).
            In the calibrated model (experiments/sleep_drive_test.py, 6 seeds,
            matched Poisson sets) ER5 at 80 Hz silences ExR1 and lowers the
            walking command DNg100 by 18% (paired p = 0.075); the R23E10 dFB
            types by 8% (p = 0.42), both together 11% (p = 0.026). So the brain-
            side effect is modest and most of the settling comes from the body
            gate. (A first, uncontrolled test reported -35% for ER5 and ExR1
            silencing by dFB; neither held with matched controls.)
  excite    social excitement (0..1) -> the male P1 courtship-arousal neurons
            (ExciteEncoder): the fly brain's song command pIP10 follows, and
            the robot voices it. Male brains only.
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


REST_HZ = 80.0
# excitement 1 -> P1 at 24 Hz: pIP10 ~46 Hz (sings) while the proboscis motor
# neuron MN9 stays at ~5 Hz. At 40 Hz P1 drives MN9 to ~14 Hz (~20 before
# brain-wide depression), which extended the proboscis in Habitat -- the
# courtship "licking" step (orient, tap, sing, lick) -- and froze the robot.
# Walking slows while P1 is on (DNg100 4.8 -> 2.7 Hz): Milo slows to sing. Measured with the calibrated
# dynamics the robot runs, on its resting receptor input (and about the same
# without it); experiments/song_test.py, 2026-09-28.
P1_MAX_HZ = 24.0


def p1_indices(connectome) -> np.ndarray:
    """The male P1 courtship-arousal neurons: MaleCNS pC1 types also named
    pMP4 / pMP-e (Yu et al. 2010; Cachero et al. 2010), from the labels. Empty
    on FAFB (P1 is male-specific)."""
    import config
    if not config.MALE_CNS:
        return np.empty(0, np.int64)
    import pandas as pd
    lab = pd.read_csv(config.LABELS_CSV)
    ids = set(lab.loc[lab["label"].astype(str).str.contains("pMP4|pMP-e"), "root_id"])
    return np.flatnonzero(connectome.neurons["root_id"].isin(ids).to_numpy()).astype(np.int64)


class ExciteEncoder:
    """Social excitement (0..1, the neocortex) -> P1. In male flies P1 sets a
    persistent state of courtship arousal and, with a target, drives the song
    command pIP10 (von Philipsborn et al. 2011; Hoopfer et al. 2015 eLife
    4:e11346). In the complete male brain, with the resting receptor input, P1
    at 30 Hz drives pIP10 to ~55 Hz and the nerve cord's song circuit; pIP10 is
    silent at rest and barely moved by a visual target alone
    (experiments/song_test.py). So excitement makes the fly brain SING, and
    the robot voices it."""

    def __init__(self, connectome):
        self.indices = p1_indices(connectome) if connectome is not None else np.empty(0, np.int64)
        self.level = 0.0

    @classmethod
    def empty(cls):
        return cls(None)

    def set(self, level: float) -> None:
        self.level = float(np.clip(level, 0.0, 1.0))

    def rates_hz(self, t_ms: float = 0.0, stim=None) -> np.ndarray:
        return np.full(len(self.indices), P1_MAX_HZ * self.level)

    def state(self, t_ms: float = 0.0) -> dict:
        return {"kind": "cortex_excite", "active": self.level > 0, "level": round(self.level, 2)}


class RestEncoder:
    """Sleep pressure -> ER5 ring neurons."""

    def __init__(self, connectome):
        t = connectome.neurons["primary_type"].fillna("").astype(str).to_numpy()
        self.indices = np.flatnonzero(t == "ER5")
        self.level = 0.0

    def set(self, level: float) -> None:
        self.level = float(np.clip(level, 0.0, 1.0))

    def rates_hz(self, t_ms: float = 0.0, stim=None) -> np.ndarray:
        return np.full(len(self.indices), REST_HZ * self.level)

    def state(self, t_ms: float = 0.0) -> dict:
        return {"kind": "cortex_rest", "active": self.level > 0, "level": round(self.level, 2)}


def reflex_channel(reflex: str) -> str:
    """The top-down channel a robot reflex steers with: "attend" drives the
    pursuit neurons LC10a at the target's bearing, "goal" sets a
    central-complex goal there. On male brains LC10a also walks the body
    BACKWARD (see TopDown.attend_from_goal; review 2026-09-28).

    "orient" (cortex/v0.py): "attend" on every brain. Through the goal the
    male robot turned toward a person who appeared 67% / 52% of the time
    instead of 86% / 77% (tuning seeds 11-13, lidar / lidar steering,
    2026-09-28); the reflex lasts 2 s. FLY_ORIENT_CHANNEL overrides.
    "unstick" (robot/avoid.Unstick): "goal" on male brains, where LC10a
    fights the pull away from the wall; "attend" on FAFB
    (results/touch_subtypes_2026-09-28/). FLY_UNSTICK_CHANNEL overrides."""
    import os
    import config
    default = {"orient": "attend", "unstick": "goal" if config.MALE_CNS else "attend"}[reflex]
    v = (os.environ.get(f"FLY_{reflex.upper()}_CHANNEL") or default).strip().lower()
    if v not in ("goal", "attend"):
        raise ValueError(f"FLY_{reflex.upper()}_CHANNEL must be 'goal' or 'attend', not {v!r}")
    return v


class TopDown:
    """All cortex -> fly channels for one Session. Call attach() after every
    Session.reset() (which clears stimuli), then apply() each control step."""

    def __init__(self, connectome, obj_encoder, channels=("goal", "attend", "dopamine"),
                 attend_from_goal: bool | None = None):
        from brain.navigation.compass import Compass, CompassDrive
        from brain.navigation.goal import GoalCircuit, GoalDrive
        self.channels = set(channels)
        # Should a navigation goal ALSO drive the pursuit neurons (LC10a)? On
        # FAFB it helps (forward walking rises); in the male CNS LC10a is the
        # courtship-pursuit pathway and the phantom drive makes the robot walk
        # BACKWARD (MDN via DNpe023: 3 -> 24 Hz; forward DNg100 falls), while
        # the central complex's goal channel (FC2 -> PFL3) drives it forward and
        # steers it correctly (2026-09-28). Default: on for FAFB, off for the
        # male brains; FLY_ATTEND_FROM_GOAL overrides. Explicit attention (the
        # orienting and unstick reflexes, cmd["attend_explicit"]) always applies.
        import os
        import config
        self.attend_from_goal = config.env_flag(
            "FLY_ATTEND_FROM_GOAL", (not config.MALE_CNS) if attend_from_goal is None else bool(attend_from_goal))
        cx = Compass(connectome)
        self.compass = CompassDrive(cx)
        self.goal = GoalDrive(GoalCircuit(connectome, cx))
        self.goal.gain = 0.0
        self.attend = AttendEncoder(obj_encoder)
        self.dopamine = DopamineEncoder(connectome)
        self.obj = obj_encoder
        self.rest = RestEncoder(connectome)
        # (reads the male labels file: only when the channel is used)
        self.excite = ExciteEncoder(connectome) if "excite" in self.channels else ExciteEncoder.empty()
        self.ses = None
        self.last = {}

    def attach(self, ses) -> None:
        if "goal" in self.channels:
            ses.add_stimulus(self.compass, self.compass)
            ses.add_stimulus(self.goal, self.goal)
        if "attend" in self.channels:
            ses.add_stimulus(self.attend, self.attend)
        if "dopamine" in self.channels:
            ses.add_stimulus(self.dopamine, self.dopamine)
        if "rest" in self.channels:
            ses.add_stimulus(self.rest, self.rest)
        if "excite" in self.channels and len(self.excite.indices):
            ses.add_stimulus(self.excite, self.excite)
        self.ses = ses

    def apply(self, heading_deg: float, cmd: dict) -> None:
        """cmd: goal_deg (world frame, CCW) and goal_gain; attend_az (+ = right)
        and attend_gain: applied if cmd["attend_explicit"] (the orienting and
        unstick reflexes), else only with attend_from_goal (FAFB), where they
        are also derived from the goal when not given; rpe, rest, excite."""
        self.compass.set_heading(heading_deg)
        g = cmd.get("goal_deg")
        gain = float(cmd.get("goal_gain", 0.0)) if g is not None else 0.0
        if g is not None:
            self.goal.set_goal(g)
        self.goal.gain = gain if "goal" in self.channels else 0.0
        if "attend_az" in cmd and (cmd.get("attend_explicit") or self.attend_from_goal):
            az, again = cmd["attend_az"], cmd.get("attend_gain", 0.0)
        elif g is not None and self.attend_from_goal:
            az, again = goal_azimuth(g, heading_deg), gain
        else:
            az, again = 0.0, 0.0
        self.attend.set(az, again if "attend" in self.channels else 0.0)
        self.dopamine.set(cmd.get("rpe", 0.0) if "dopamine" in self.channels else 0.0)
        arousal = float(np.clip(cmd.get("arousal", 1.0), 0.0, 1.0)) if "arousal" in self.channels else 1.0
        self.obj.arousal = arousal
        rest = float(np.clip(cmd.get("rest", 0.0), 0.0, 1.0)) if "rest" in self.channels else 0.0
        self.rest.set(rest)
        excite = float(np.clip(cmd.get("excite", 0.0), 0.0, 1.0)) if "excite" in self.channels else 0.0
        self.excite.set(excite)
        body = getattr(self.ses, "body", None)
        if body is not None and hasattr(body, "rest_level"):
            body.rest_level = rest
        self.last = {"goal_deg": None if g is None else round(g % 360.0, 1), "goal_gain": round(gain, 2),
                     "attend_az": round(self.attend.az, 1), "attend_gain": round(self.attend.gain, 2),
                     "rpe": round(self.dopamine.rpe, 3), "arousal": round(arousal, 2), "rest": round(rest, 2),
                     "excite": round(excite, 2)}


def goal_azimuth(goal_deg: float, heading_deg: float) -> float:
    """Where the goal lies in the pet's view (deg, + = right): a goal
    counter-clockwise of the heading is to the LEFT."""
    return -(((goal_deg - heading_deg) + 180.0) % 360.0 - 180.0)


def bearing_deg(frm_xz, to_xz) -> float:
    """World bearing from one floor point to another in the robot's yaw frame
    (habitat_server: yaw is CCW from +x in the (x, -z) plane)."""
    return math.degrees(math.atan2(-(to_xz[1] - frm_xz[1]), to_xz[0] - frm_xz[0])) % 360.0
