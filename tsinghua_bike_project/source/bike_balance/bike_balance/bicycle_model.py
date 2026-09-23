"""Three-state nominal bicycle model for roll-steer balance control.

States:  x = [steering_angle, roll_angle, roll_rate]^T   (rad, rad, rad/s)
Input:   u = steering_angular_velocity                     (rad/s)

The continuous-time model is  x_dot = A(v) x + Bu(v) u  where the three
speed-dependent coefficients a1, a2, a4 are recomputed on every speed update.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BicycleParameters:
    """Physical parameters of the simplified roll-steer bicycle model.

    Attributes
    ----------
    rear_contact_to_com:
        Horizontal distance from rear-wheel ground contact to COM (m).  *a*.
    wheelbase:
        Distance between front and rear wheel contacts (m).  *b*.
    trail:
        Steering trail / caster offset (m).  *c*.
    com_height:
        Height of COM above ground (m).  *h*.
    mass:
        Total vehicle mass (kg).  *m*.
    roll_inertia:
        Equivalent roll inertia about the ground-longitudinal axis (kg·m²).
        When *None* (default), approximated as 4/3 m h².
    steering_axis_angle:
        Steering axis inclination from vertical (rad).  *lambda*.
    gravity:
        Gravitational acceleration (m/s²).
    """

    rear_contact_to_com: float = 0.117
    wheelbase: float = 0.271
    trail: float = 0.0
    com_height: float = 0.292
    mass: float = 5.0
    roll_inertia: float | None = None
    steering_axis_angle: float = 0.0
    gravity: float = 9.81

    @property
    def effective_roll_inertia(self) -> float:
        if self.roll_inertia is not None:
            return float(self.roll_inertia)
        return 4.0 / 3.0 * self.mass * self.com_height**2


@dataclass(frozen=True)
class LinearBicycleSystem:
    """Snapshot of the system matrices at one forward speed."""

    A: np.ndarray   # (3, 3) continuous-time state matrix
    Bu: np.ndarray  # (3, 1) input matrix
    Bd: np.ndarray  # (3, 1) disturbance matrix (not used here)
    Cm: np.ndarray  # (3, 3) measurement matrix (full-state = identity)
    Co: np.ndarray  # (1, 3) controlled-output matrix (steering angle)
    dt: float       # discrete-time step (s)

    @property
    def nx(self) -> int:
        return self.A.shape[0]

    @property
    def nu(self) -> int:
        return self.Bu.shape[1]

    @property
    def no(self) -> int:
        return self.Co.shape[0]


class LinearBicycleModel:
    """Speed-scheduled 3-state roll-steer bicycle model.

    Call ``update_system_parameters(speed)`` whenever the forward speed
    changes; the controller should then call ``update_system_and_gain``
    to recompute the feedback gain for the new A, Bu matrices.
    """

    def __init__(
        self,
        dt: float,
        parameters: BicycleParameters | None = None,
        *,
        track_roll: bool = False,
        min_forward_speed: float = 1.0,
        forward_speed: float = 2.0,
    ) -> None:
        self.dt = float(dt)
        self.parameters = parameters or BicycleParameters()
        self.track_roll = bool(track_roll)
        self.update_system_parameters(forward_speed, min_forward_speed=min_forward_speed)

    # ------------------------------------------------------------------
    def update_system_parameters(
        self,
        forward_speed: float,
        *,
        gravity: float | None = None,
        min_forward_speed: float = 1.0,
    ) -> LinearBicycleSystem:
        """Recompute A, Bu from physical parameters at the given speed.

        Speeds between zero and *min_forward_speed* are clamped upward to
        avoid model degeneracy.  Negative speeds are treated as reverse
        motion (the model is for forward riding, though).
        """
        p = self.parameters
        g = p.gravity if gravity is None else float(gravity)
        speed = float(forward_speed)
        minimum = abs(float(min_forward_speed))

        direction = -1.0 if speed < 0.0 else 1.0
        scheduled_speed = direction * max(minimum, abs(speed))

        a = p.rear_contact_to_com
        b = p.wheelbase
        c = p.trail
        h = p.com_height
        m = p.mass
        inertia = p.effective_roll_inertia
        cos_lambda = math.cos(p.steering_axis_angle)

        a1 = m * a * h * scheduled_speed * cos_lambda / (b * inertia)
        a2 = (m * scheduled_speed**2 * h - m * a * c * g) * cos_lambda / (b * inertia)
        a4 = m * g * h / inertia

        self.forward_speed = speed
        self.scheduled_speed = scheduled_speed
        self.min_forward_speed = minimum

        self.A = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [a2, a4, 0.0]])
        self.Bu = np.array([[1.0], [0.0], [a1]])
        self.Bd = np.array([[0.0], [0.0], [1.0]])
        self.Cm = np.eye(3)
        self.Co = (
            np.array([[0.0, 1.0, 0.0]])
            if self.track_roll
            else np.array([[1.0, 0.0, 0.0]])
        )

        self.sys = LinearBicycleSystem(
            A=self.A.copy(),
            Bu=self.Bu.copy(),
            Bd=self.Bd.copy(),
            Cm=self.Cm.copy(),
            Co=self.Co.copy(),
            dt=self.dt,
        )
        return self.sys

    # CamelCase alias kept for compatibility with tutorial notebooks.
    def updateSysParam(
        self,
        forw_vel: float,
        g: float = 9.81,
        min_forw_vel: float = 1.0,
    ) -> LinearBicycleSystem:
        return self.update_system_parameters(forw_vel, gravity=g, min_forward_speed=min_forw_vel)
