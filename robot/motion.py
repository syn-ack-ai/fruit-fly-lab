"""
Motor dynamics between the brain's command and the wheels.

The fly body's turn rate is read from descending-neuron spike counts in each
100 ms control window, so it jumps from step to step (+101, +38, -34, +16, +93
deg/s): a simulated base follows it exactly and wiggles its heading about three
times a second (score_pets.py, 2026-09-27: 135-190 heading reversals per active
minute with either brain). An animal does not move like that: muscles and legs
integrate motor-neuron spikes over hundreds of milliseconds, and a body with
mass cannot reverse its turn instantly. A rover's motors do the same. This is
the smoothing an animal's body gives for free: a first-order lag on the
commanded speed and turn rate, optionally in several stages (a cascade: the
servo-like start and stop of a robot such as WALL-E, and much stronger
rejection of the step-to-step jitter for the same lag). It sits BEFORE the
robot's safety layers, so braking for an obstacle, the dock or a flat battery
stays immediate.
"""
from __future__ import annotations

import math


class MotorLag:
    """`stages` first-order lags in series, each with time constant tau.

    With `tau_w_fast` the turn lag adapts to how CONSISTENT the brain's turn
    command is: jitter (a command that keeps flipping sign) is filtered with
    tau_w, a deliberate turn (the same sign for a few hundred ms, e.g. toward a
    person who just appeared) with tau_w_fast. Consistency k = |mean| / mean|w|
    over CONSIST_S (1 = all one way, 0 = pure jitter); tau = tau_w + k^2
    (tau_w_fast - tau_w). (2026-09-27: with a fixed 1.5 s lag the robot turned
    toward a person appearing only ~50% of the time.)"""

    CONSIST_S = 0.3

    def __init__(self, tau_v_s: float, tau_w_s: float, stages: int = 1, tau_w_fast_s: float | None = None):
        self.tau_v, self.tau_w = float(tau_v_s), float(tau_w_s)
        self.tau_w_fast = None if tau_w_fast_s is None else float(tau_w_fast_s)
        self.stages = max(1, int(stages))
        self.reset()

    def reset(self) -> None:
        self._v = [0.0] * self.stages
        self._w = [0.0] * self.stages
        self.v = self.w = 0.0
        self._m = self._a = 0.0            # running mean of the command and of its size

    def turn_tau(self, w: float, dt: float) -> float:
        if self.tau_w_fast is None:
            return self.tau_w
        al = self._alpha(dt, self.CONSIST_S)
        self._m += (w - self._m) * al
        self._a += (abs(w) - self._a) * al
        k = abs(self._m) / self._a if self._a > 1e-9 else 0.0
        return self.tau_w + k * k * (self.tau_w_fast - self.tau_w)

    @staticmethod
    def _alpha(dt: float, tau: float) -> float:
        return 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)

    def sync(self, v: float, w: float) -> None:
        """Set the state to the velocity actually applied (after the safety
        layers): no wind-up while they override the command."""
        if (v, w) != (self.v, self.w):
            self._v = [float(v)] * self.stages
            self._w = [float(w)] * self.stages
            self.v, self.w = float(v), float(w)

    def __call__(self, v: float, w: float, dt: float) -> tuple:
        av, aw = self._alpha(dt, self.tau_v), self._alpha(dt, self.turn_tau(w, dt))
        for i in range(self.stages):
            self._v[i] += (v - self._v[i]) * av
            self._w[i] += (w - self._w[i]) * aw
            v, w = self._v[i], self._w[i]
        self.v, self.w = v, w
        return v, w
