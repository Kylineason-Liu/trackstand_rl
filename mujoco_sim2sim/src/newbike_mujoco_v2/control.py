"""Scalar equivalent of Isaac v2's external, per-environment velocity integral."""

import math

import numpy as np


class VelocityIntegral:
    """Trapezoidal I with the training controller's conditional anti-windup.

    State persists across target changes. Reset clears it; the first update only
    primes the previous-error sample. No motor-side or passive friction is added.
    """

    def __init__(self, gain_nm_per_rad):
        if not math.isfinite(gain_nm_per_rad) or gain_nm_per_rad < 0:
            raise ValueError("Integral gain must be finite and nonnegative")
        self.gain_nm_per_rad = float(gain_nm_per_rad)
        self.reset()

    def reset(self):
        self.error_rad = self.torque_nm = self.previous_error = 0.0
        self.initialized = False
        self.limited_updates = 0

    def update(self, error, dt, proportional_gain, effort_limit):
        if self.gain_nm_per_rad == 0:
            return self.torque_nm
        increment = 0.5 * (error + self.previous_error) * dt if self.initialized else 0.0
        proposed = self.gain_nm_per_rad * (self.error_rad + increment)
        candidate = float(np.clip(proposed, -effort_limit, effort_limit))
        delta = candidate - self.torque_nm
        total = proportional_gain * error + candidate
        if (total > effort_limit and delta > 0) or (total < -effort_limit and delta < 0):
            candidate = self.torque_nm
        self.limited_updates += int(candidate != proposed)
        self.torque_nm = candidate
        self.error_rad = candidate / self.gain_nm_per_rad
        self.previous_error = float(error)
        self.initialized = True
        return candidate
