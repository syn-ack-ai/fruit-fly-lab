"""
A body that can walk, fly and land, for the closed-loop world (fly/world/).

SCOPE AND HONESTY
-----------------
Like fly_body.py this is a body model (category C/D), downstream of all neural
simulation. Its inputs are the descending-neuron motor channels and the
proboscis motor-neuron drive; it never sees the world's fruit, odour or
predators.

What is NOT in the connectome, and therefore modelled here:
  - The ventral nerve cord (leg and wing motor circuits, central pattern
    generators) is absent from FlyWire FAFB, and the published brain model is
    silent at rest. Real flies walk and fly spontaneously. So the body has a
    simple SPONTANEOUS LOCOMOTION process ("exploration"): walking bouts
    separated by stops, with random turning, occasional spontaneous takeoffs,
    and flight bouts that end in a landing. Walking at ~12 mm/s in bouts of a
    few seconds follows Martin (2004, J Neurosci Methods 133:37) and Strauss &
    Heisenberg (1990).
  - The brain's descending neurons MODULATE and OVERRIDE this:
      * DNg100 (BDN2), the command neuron for forward walking, sets walking
        speed: its activity controls stepping frequency (Bidaye et al. 2020;
        Pugliese et al., bioRxiv 2025.09.12.675944). Speed during a walking
        bout = DNG100_MM_PER_HZ x its firing rate (C: the gain is chosen so
        the model's resting DNg100 rate gives the ordinary 12 mm/s), and a
        strong DNg100 command (>= the forward threshold) starts a bout.
      * DNp09 (P9) also starts forward walking, with an ipsilateral turn.
      * Steering follows Rayshubskiy et al. 2025 (eLife 102230): turning
        velocity is linear in the right-left difference of DNa02 activity
        (up to ~800 deg/s), through a BIPHASIC filter ("a biphasic filter
        converts a sustained input into a transient output"), whereas DNa01
        steers with lower gain through a monophasic (sustained) filter. So
        DNa02's contribution is high-passed (DNA02_HP_TAU_S; C) and DNa01's
        is sustained. A steady left-right asymmetry therefore produces a
        modest sustained bias (this individual's "handedness"), while changes
        (e.g. an odour arriving on one side) produce brisk turns.
    MDN walks backward, the Giant Fibre and DNp02/04/11 trigger escape takeoff,
    DNp07/DNp10 trigger landing, DNg02 sets flight power and its left/right
    difference steers in flight, proboscis motor neurons extend the proboscis
    (feeding requires standing still).
  - In flight, escape commands produce a banked evasive turn away from the
    more active side (flies evade looming objects with rapid banked turns;
    Muijres et al. 2014, Science 344:172).

With neural=False the body runs the spontaneous process alone: the control
condition for "what does the brain add?".
"""
from __future__ import annotations

import math

import numpy as np

from brain.motor.descending import CHANNEL_HALF_MAX_HZ
from fly.body.fly_body import (BACKWARD_SPEED_MM_S, BACKWARD_THRESHOLD,
                               ESCAPE_THRESHOLD, FORWARD_THRESHOLD,
                               GF_TAKEOFF_LATENCY_MS, JUMP_SPEED_MM_S,
                               LONG_MODE_PREP_MS, LONG_MODE_THRESHOLD,
                               MAX_TURN_RATE_DEG_S, MAX_WALK_SPEED_MM_S,
                               P9_TURN_DEG_S, PROBOSCIS_THRESHOLD, BodyState)

# --- C: spontaneous locomotion -------------------------------------------------
WALK_SPEED_MM_S = 12.0
MEAN_STOP_S = 2.0
MEAN_WALK_S = 5.0
TURN_NOISE_DEG_S = 90.0          # SD of spontaneous angular velocity
TURN_TAU_S = 0.25
SPONT_TAKEOFF_PER_S = 1 / 60.0   # while walking
# --- neural readout gains (see docstring) ----------------------------------------
# model DNg100 rate under resting sensory input, FAFB (C; cognition/calibrate_body.py,
# 2026-09-29: 18.5 Hz. The first value, 14.6 Hz on 2026-09-25, had no warm-up and made
# FAFB walk 27% faster). Each brain's own rate comes from body_readout_<dataset>.json;
# FAFB's is also the reference the forward threshold was set on (tests check they agree).
FAFB_DNG100_REST_HZ = 18.5
# C: holding the proboscis out. A feeding fly keeps its proboscis extended for
# seconds while it drinks; the model's proboscis motor neurons fire in bursts
# of ~250 ms with pauses (the male brain's two MN9 cells: 0-40 Hz per 250 ms at
# a mean of 8.6 Hz). The body follows their rate over the last ~second
# (PROBOSCIS_TAU_MS, averaged before the readout's saturation), extends above
# PROBOSCIS_THRESHOLD and retracts below PROBOSCIS_RELEASE x that. Replayed
# spike trains (2026-09-29, experiments/feeding_hold_test.py): sustained sugar
# -> extended 98% (male, scaled; FAFB 99%), resting input 0%, the P1
# excitement channel at its maximum 7% (the 50 ms window without hysteresis:
# 16% / 55% / 1.5% / 9%). After sugar it retracts in a median 1.2 s on the
# resting input and 2.9 s under maximal excitement. The release point is
# above the male brain's MN9 rate under maximal excitement (3.5 Hz, 6.2
# scaled); at 0.5 x the threshold excitement right after a meal held the
# proboscis out (and the robot still) about twice as long.
PROBOSCIS_TAU_MS = 1000.0
PROBOSCIS_RELEASE = 0.65
PROBOSCIS_MAX_HZ = 500.0         # a rate the readout cannot exceed (guards the average)


def _dng100_rest_hz() -> float:
    """The dataset's own resting DNg100 rate (cognition/calibrate_body.py ->
    data/metadata/body_readout_<dataset>.json), else FAFB's (FAFB_DNG100_REST_HZ).
    The complete male brain rests at ~3 Hz: read with FAFB's rate it walked at
    a quarter of a fly's pace -- the whole "walks 3x less than FAFB" gap in
    Habitat (2026-09-28)."""
    import json
    import warnings
    import config
    p = config.METADATA_DIR / f"body_readout_{config.BRAIN_KEY}.json"
    if p.exists():
        d = json.loads(p.read_text())
        try:
            from cognition.calibrate_body import dynamics_hash
            if d.get("dynamics_sha256_16") and d["dynamics_sha256_16"] != dynamics_hash():
                warnings.warn(f"{p.name} was measured with other calibrated dynamics: re-run "
                              "python -m cognition.calibrate_body --write")
        except Exception:
            pass
        hz = float(d["dng100_rest_hz"])
        if not (np.isfinite(hz) and hz >= 0.5):         # speed = rate / hz: must be a real rate
            raise ValueError(f"{p}: dng100_rest_hz {hz!r} is not a usable resting rate")
        return hz
    if config.MALE_CNS:
        warnings.warn(f"no {p.name}: the body uses FAFB's resting DNg100 rate ({FAFB_DNG100_REST_HZ} Hz) for a male brain, "
                      "which rests near 3 Hz -- run python -m cognition.calibrate_body --write")
    return FAFB_DNG100_REST_HZ


DNG100_REST_HZ = _dng100_rest_hz()
DNG100_MM_PER_HZ = WALK_SPEED_MM_S / DNG100_REST_HZ
MAX_NEURAL_WALK_MM_S = 35.0
DNA02_TURN_DEG_S = 800.0         # per unit right-left activation difference (high gain)
DNA01_TURN_DEG_S = 200.0         # lower gain (Rayshubskiy et al. 2025)
DNA02_HP_TAU_S = 0.5             # biphasic filter: slow (negative) lobe (C)
STEER_FAST_TAU_S = 0.1           # fast lobe / monophasic smoothing of DN rates (C)
SPEED_TAU_S = 0.3                # stepping frequency follows DNg100 smoothly (C)
# C: switching walking direction follows the forward / backward command
# channels over ~WALK_CMD_TAU_S (a step or two), not one 50 ms window. The
# male brain's MDN groups are 2 cells per side at ~5 Hz: a chance burst of 3
# spikes in 50 ms read as 30 Hz and walked the robot backward 7.5% of the time
# in Habitat (FAFB, MDN ~0.3 Hz: never), 2026-09-29. FLY_WALK_CMD_TAU=0: the
# readout before (for comparisons): no smoothing and no scaled DNg100 trigger.
WALK_CMD_TAU_S = 0.2
# ... and a bout starts when DNg100, scaled to FAFB's resting rate (the
# reference the forward threshold was set on; cognition/calibrate_body.py),
# reaches FORWARD_THRESHOLD: the male brain's DNg100 fires ~6x slower, so its
# absolute rate never did, except in chance 50 ms bursts.
DNG100_REF_REST_HZ = FAFB_DNG100_REST_HZ
# C: the DNg100 readouts average the same expected number of spikes on every
# brain: the smoothing times above were set on FAFB (2 cells at 18.5 Hz); the
# male brain's 2 cells rest at 3.1 Hz, so read over 0.3 s its speed was mostly
# shot noise (SD 1.6x rest vs FAFB 0.74x) and the speed limits cut its bursts
# off: mean 1.30x rest, 0.81x after the wheels' 0.5 m/s (FAFB 1.32 -> 1.15),
# 2026-09-29. DNG100_TAU_SCALE stretches SPEED_TAU_S and WALK_CMD_TAU_S for
# DNg100 by FAFB's resting rate / this brain's (1 on FAFB).
DNG100_TAU_SCALE = max(1.0, DNG100_REF_REST_HZ / DNG100_REST_HZ)
MAX_WALK_TURN_DEG_S = 800.0
# --- C: flight -------------------------------------------------------------------
CRUISE_SPEED_MM_S = 250.0
CRUISE_ALT_MM = 50.0
MEAN_FLIGHT_S = 4.0
FLIGHT_TURN_NOISE_DEG_S = 60.0
FLIGHT_POWER_TURN_DEG_S = 600.0  # yaw from DNg02 left/right difference
EVASIVE_TURN_DEG = 90.0
EVASIVE_MS = 60.0
LAND_DESCENT_MM_S = 150.0
LANDING_THRESHOLD = 0.30
FLIGHT_ESCAPE_THRESHOLD = 0.35
GRAVITY_MM_S2 = 9810.0


def _env_seconds(name: str, default: float) -> float:
    """A time constant from the environment: a number of seconds, or
    0/false/no/off/"" (config.env_flag's off words) for none."""
    import os
    v = os.environ.get(name)
    if v is None:
        return default
    if v.strip().lower() in ("", "0", "false", "no", "off"):
        return 0.0
    x = float(v)
    if not (np.isfinite(x) and x >= 0.0):
        raise ValueError(f"{name} must be >= 0 seconds, not {v!r}")
    return x


class ForagingBody:
    def __init__(self, neural: bool = True, seed: int = 0,
                 spontaneous_takeoff_per_s: float = SPONT_TAKEOFF_PER_S):
        self.neural = neural
        self.spont_takeoff = spontaneous_takeoff_per_s
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self._seed)
        self.state = BodyState()
        self.events = []
        self._t_ms = 0.0
        self._walking = False
        self._turn_noise = 0.0
        self._escape_armed_at = None
        self._escape_mode = None
        self._prob_hz = 0.0
        self._prob_on = False
        import config
        self._prob_hold = config.env_flag("FLY_PROBOSCIS_HOLD", True)   # 0: the 50 ms readout before 2026-09-29
        self._escape_complete = False
        self._phase = "ground"          # ground | jump | cruise | landing
        self._flight_end = 0.0
        self._evade_until = -1.0
        self._evade_rate = 0.0
        self._a02_lp = None             # slow component of DNa02 L-R (high-pass state)
        self._a02_f = 0.0               # fast (smoothed) DNa02 L-R
        self._a01_f = 0.0               # smoothed DNa01 L-R (monophasic)
        self._p09_f = 0.0
        self._hz100 = None              # smoothed DNg100 rate
        self._fwd_s = self._bwd_s = 0.0  # smoothed forward / backward channels
        self._dng_s = 0.0                # smoothed DNg100, scaled to FAFB's rest (Hz)
        self._cmd_tau_s = _env_seconds("FLY_WALK_CMD_TAU", WALK_CMD_TAU_S)
        # FLY_DNG100_TAU_SCALE=0: FAFB's smoothing times on every brain (before 2026-09-29)
        self._dng_tau_scale = DNG100_TAU_SCALE if config.env_flag("FLY_DNG100_TAU_SCALE", True) else 1.0
        # Rest / sleep pressure (0..1), set top-down (cortex/topdown.py "rest").
        # The brain side drives ER5 ring neurons; this is the missing nerve
        # cord's side: sleep-promoting VNC neurons (VNC-SP; Jones et al. 2023
        # PLoS Biol) lengthen stops and shorten spontaneous walking bouts.
        # Command neurons (DNg100/DNp09 over threshold) still start a bout, so
        # a strong stimulus wakes a resting pet.
        self.rest_level = 0.0

    # ----------------------------------------------------------------- update
    def update(self, dt_ms: float, channels: dict, t_ms: float,
               escape_laterality: float = 0.0, proboscis_drive: float = 0.0) -> BodyState:
        s = self.state
        self._t_ms = t_ms
        dt_s = dt_ms / 1000.0
        ch = channels if self.neural else {}
        prob = proboscis_drive if self.neural else 0.0
        takeoff = ch.get("escape_takeoff", 0.0)
        long_mode = ch.get("escape_long_mode", 0.0)
        forward = ch.get("forward_walk", 0.0)
        backward = ch.get("backward_walk", 0.0)
        turn_bias = ch.get("turn_bias", 0.0)

        # spontaneous angular velocity (Ornstein-Uhlenbeck)
        noise = FLIGHT_TURN_NOISE_DEG_S if s.airborne else TURN_NOISE_DEG_S
        a = dt_s / TURN_TAU_S
        self._turn_noise += -a * self._turn_noise + noise * math.sqrt(2 * a) * self.rng.standard_normal()

        if s.airborne:
            self._fly(dt_ms, t_ms, ch, escape_laterality)
        else:
            self._ground(dt_ms, t_ms, ch, prob, takeoff, long_mode, forward,
                         backward, turn_bias, escape_laterality)
        s.heading_deg = (s.heading_deg + s.turn_rate_deg_s * dt_s) % 360.0
        rad = math.radians(s.heading_deg)
        s.x_mm += math.cos(rad) * s.speed_mm_s * dt_s
        s.y_mm += math.sin(rad) * s.speed_mm_s * dt_s
        return s

    # ----------------------------------------------------------------- ground
    def _ground(self, dt_ms, t_ms, ch, prob, takeoff, long_mode, forward,
                backward, turn_bias, laterality):
        s = self.state
        dt_s = dt_ms / 1000.0
        if self._escape_complete and takeoff < ESCAPE_THRESHOLD and long_mode < LONG_MODE_THRESHOLD:
            self._escape_complete = False
        if self._escape_armed_at is None and not self._escape_complete:
            if takeoff >= ESCAPE_THRESHOLD:
                self._arm(t_ms, "short", "GF (DNp01) escape command", takeoff)
            elif long_mode >= LONG_MODE_THRESHOLD:
                self._arm(t_ms, "long", "long-mode escape command (DNp02/04/11)", long_mode)
        if self._escape_armed_at is not None:
            el = t_ms - self._escape_armed_at
            s.speed_mm_s, s.turn_rate_deg_s = 0.0, 0.0
            if self._escape_mode == "short":
                s.behaviour = "escape (short mode, GF-driven)"
                s.leg_extension = min(1.0, el / GF_TAKEOFF_LATENCY_MS)
                if el >= GF_TAKEOFF_LATENCY_MS:
                    self._takeoff(t_ms, laterality, directed=False, why="escape (short mode)")
            else:
                s.behaviour = "escape (long mode, preparing)"
                s.wing_angle_deg = 90.0 * min(1.0, el / LONG_MODE_PREP_MS)
                s.leg_extension = min(1.0, el / LONG_MODE_PREP_MS)
                if el >= LONG_MODE_PREP_MS:
                    self._takeoff(t_ms, laterality, directed=True, why="escape (long mode)")
            return

        # proboscis (real motor neurons); feeding needs the fly to stand still.
        # The drive stands for a rate (readout: hz / (hz + CHANNEL_HALF_MAX_HZ));
        # the rate is averaged, then saturated again (PROBOSCIS_TAU_MS).
        half = CHANNEL_HALF_MAX_HZ
        hz = half * prob / max(1e-9, 1.0 - prob) if np.isfinite(prob) else 0.0
        hz = min(max(hz, 0.0), PROBOSCIS_MAX_HZ)
        if self._prob_hold:
            self._prob_hz += (hz - self._prob_hz) * (1.0 - math.exp(-dt_ms / PROBOSCIS_TAU_MS))
            held = self._prob_hz / (self._prob_hz + half)
            self._prob_on = held >= PROBOSCIS_THRESHOLD * (PROBOSCIS_RELEASE if self._prob_on else 1.0)
        else:
            self._prob_on = prob >= PROBOSCIS_THRESHOLD
        target = 1.0 if self._prob_on else 0.0
        s.proboscis_extension += (target - s.proboscis_extension) * min(1.0, dt_ms / 40.0)
        s.wing_angle_deg *= 0.92
        s.leg_extension *= 0.9
        feeding = s.proboscis_extension > 0.5

        # spontaneous walk/stop bouts
        rest = self.rest_level
        if self._walking:
            if self.rng.random() < dt_s * (1 + 3 * rest) / MEAN_WALK_S:
                self._walking = False
        elif self.rng.random() < dt_s / (MEAN_STOP_S * (1 + 9 * rest)):
            self._walking = True

        beh = "walking" if self._walking else "resting"
        # neural modulation (see docstring)
        if self._cmd_tau_s > 0:
            k = 1.0 - math.exp(-dt_s / self._cmd_tau_s)
            self._fwd_s += (forward - self._fwd_s) * k
            self._bwd_s += (backward - self._bwd_s) * k
            forward, backward = self._fwd_s, self._bwd_s
            if ch.get("hz_DNg100") is not None:
                kd = 1.0 - math.exp(-dt_s / (self._cmd_tau_s * self._dng_tau_scale))
                self._dng_s += (ch["hz_DNg100"] * DNG100_REF_REST_HZ / DNG100_REST_HZ - self._dng_s) * kd
                forward = max(forward, self._dng_s / (self._dng_s + CHANNEL_HALF_MAX_HZ))
        hz100 = ch.get("hz_DNg100")
        if forward >= FORWARD_THRESHOLD:
            self._walking = True                      # command neurons start a bout
            beh = "walking forward (DNg100/DNp09)"
        if not self.neural or hz100 is None:
            speed = WALK_SPEED_MM_S if self._walking else 0.0
        else:
            if self._hz100 is None:
                self._hz100 = DNG100_REST_HZ
            self._hz100 += (hz100 - self._hz100) * min(1.0, dt_s / (SPEED_TAU_S * self._dng_tau_scale))
            speed = (min(MAX_NEURAL_WALK_MM_S, DNG100_MM_PER_HZ * self._hz100)
                     if self._walking else 0.0)
        turn = self._turn_noise if self._walking else 0.0
        if self.neural:
            kf = min(1.0, dt_s / STEER_FAST_TAU_S)
            self._a02_f += (ch.get("lr_DNa02", 0.0) - self._a02_f) * kf
            self._a01_f += (ch.get("lr_DNa01", 0.0) - self._a01_f) * kf
            self._p09_f += (ch.get("lr_DNp09", 0.0) - self._p09_f) * kf
            if self._a02_lp is None:
                self._a02_lp = self._a02_f
            self._a02_lp += (self._a02_f - self._a02_lp) * min(1.0, dt_s / DNA02_HP_TAU_S)
            # DNa02: fast lobe minus slow lobe (biphasic); DNa01: sustained
            steer = -(DNA02_TURN_DEG_S * (self._a02_f - self._a02_lp)
                      + DNA01_TURN_DEG_S * self._a01_f
                      + P9_TURN_DEG_S * self._p09_f)
            turn += steer
            if abs(steer) > 30.0 and not self._walking:
                speed = 0.3 * WALK_SPEED_MM_S         # pivot toward the turn
                beh = "turning %s (DNa01/02)" % ("left" if steer > 0 else "right")
        if backward >= BACKWARD_THRESHOLD:
            speed = -BACKWARD_SPEED_MM_S * backward
            beh = "walking backward (MDN)"
        if feeding:
            speed, turn, beh = 0.0, 0.0, "feeding (proboscis extended)"
        s.speed_mm_s += (speed - s.speed_mm_s) * min(1.0, dt_ms / 30.0)
        s.turn_rate_deg_s = max(-MAX_WALK_TURN_DEG_S, min(MAX_WALK_TURN_DEG_S, turn))
        s.behaviour = beh

        if (self._walking and not feeding and self.spont_takeoff > 0
                and self.rng.random() < dt_s * self.spont_takeoff):
            self._takeoff(t_ms, 0.0, directed=False, why="spontaneous")

    def _arm(self, t_ms, mode, what, strength):
        self._escape_armed_at, self._escape_mode = t_ms, mode
        self.state.escape_mode = mode
        self._log(t_ms, what, strength)

    def _takeoff(self, t_ms, laterality, directed, why):
        s = self.state
        s.airborne = True
        s.vz_mm_s = JUMP_SPEED_MM_S * 0.6
        s.speed_mm_s = max(s.speed_mm_s, 0.5 * JUMP_SPEED_MM_S)
        s.wing_angle_deg = 90.0
        s.proboscis_extension = 0.0
        self._prob_hz, self._prob_on = 0.0, False      # an escape interrupts feeding
        self._fwd_s = self._bwd_s = self._dng_s = 0.0  # ... and a walk: it starts afresh on landing
        if directed:
            # away from the more active escape side: a threat on the right drives
            # the right escape DNs more (laterality > 0; checked with looming at
            # +-60 deg), so turn left = counter-clockwise (Card & Dickinson 2008)
            s.heading_deg = (s.heading_deg + 90.0 * laterality) % 360.0
        self._phase = "jump"
        self._flight_end = t_ms + 1000.0 * self.rng.exponential(MEAN_FLIGHT_S)
        self._escape_armed_at = None
        s.behaviour = "takeoff (%s)" % why
        self._log(t_ms, "takeoff (%s)" % why, 1.0)

    # ----------------------------------------------------------------- flight
    def _fly(self, dt_ms, t_ms, ch, laterality):
        s = self.state
        dt_s = dt_ms / 1000.0
        power = ch.get("flight_power", 0.0)
        speed_target = CRUISE_SPEED_MM_S * (0.7 + 0.6 * power)
        turn = self._turn_noise - FLIGHT_POWER_TURN_DEG_S * ch.get("flight_power_lr", 0.0) \
            - MAX_TURN_RATE_DEG_S * ch.get("turn_bias", 0.0)
        s.leg_extension *= 0.95

        # evasive banked turn on an escape command (flight-state escape)
        esc = max(ch.get("escape_takeoff", 0.0), ch.get("escape_long_mode", 0.0))
        if esc >= FLIGHT_ESCAPE_THRESHOLD and t_ms > self._evade_until + 200.0:
            away = 1.0 if laterality >= 0 else -1.0        # threat on the right -> turn left
            self._evade_rate = away * EVASIVE_TURN_DEG / (EVASIVE_MS / 1000.0)
            self._evade_until = t_ms + EVASIVE_MS
            self._log(t_ms, "evasive turn in flight", esc)
        if t_ms < self._evade_until:
            turn = self._evade_rate
            speed_target *= 1.4

        if self._phase == "jump":
            s.vz_mm_s -= GRAVITY_MM_S2 * dt_s
            if s.vz_mm_s <= 50.0:                        # wings take over
                self._phase = "cruise"
        elif self._phase == "cruise":
            s.vz_mm_s = 4.0 * (CRUISE_ALT_MM - s.z_mm)
            landing = ch.get("landing", 0.0)
            if landing >= LANDING_THRESHOLD or t_ms >= self._flight_end:
                self._phase = "landing"
                self._log(t_ms, "landing (%s)" % ("DNp07/DNp10" if landing >= LANDING_THRESHOLD
                                                  else "end of flight bout"), landing)
        else:                                            # landing
            s.vz_mm_s = -LAND_DESCENT_MM_S
            speed_target = min(speed_target, 60.0)
            s.leg_extension = 1.0
        s.z_mm += s.vz_mm_s * dt_s
        s.speed_mm_s += (speed_target - s.speed_mm_s) * min(1.0, dt_ms / 80.0)
        s.turn_rate_deg_s = max(-2000.0, min(2000.0, turn))
        s.behaviour = {"jump": "takeoff", "cruise": "flying", "landing": "landing"}[self._phase]
        if t_ms < self._evade_until:
            s.behaviour = "evasive turn (flight)"
        if s.z_mm <= 0.0 and self._phase != "jump":
            s.z_mm, s.vz_mm_s, s.airborne = 0.0, 0.0, False
            s.speed_mm_s, s.leg_extension, s.wing_angle_deg = 0.0, 0.0, 0.0
            self._phase = "ground"
            self._walking = False
            self._escape_complete = True
            s.behaviour = "landed"
            self._log(t_ms, "landed", 0.0)
        elif s.z_mm < 0.0:
            s.z_mm = 0.0

    # ---------------------------------------------------------------- helpers
    def _log(self, t_ms, what, strength):
        self.events.append({"t_ms": round(t_ms, 1), "event": what,
                            "strength": round(float(strength), 3)})
        if len(self.events) > 200:
            del self.events[:50]

    def as_dict(self) -> dict:
        from dataclasses import asdict
        d = asdict(self.state)
        d["events"] = self.events[-12:]
        d["flight_phase"] = self._phase
        d["neural"] = self.neural
        return d

    @property
    def provenance(self) -> dict:
        return {"role": "closed-loop body (walk, fly, land), downstream of the brain",
                "neural_control": self.neural,
                "spontaneous_locomotion": {
                    "walk_speed_mm_s": WALK_SPEED_MM_S, "mean_stop_s": MEAN_STOP_S,
                    "mean_walk_s": MEAN_WALK_S, "spontaneous_takeoff_per_s": self.spont_takeoff,
                    "mean_flight_s": MEAN_FLIGHT_S},
                "caveat": ("The ventral nerve cord is not in FlyWire FAFB: locomotor "
                           "rhythms, spontaneous activity and flight control are a "
                           "kinematic model the brain's descending neurons modulate.")}
