"""Episode-relative rear angle from physics-rate native position increments."""

import math
from collections.abc import Sequence

import torch


class WheelNetAngleTracker:
    """Unwrap the installed PhysX continuous joint's 4pi position branch.

    Each physics-step motion must be less than half the branch period (2pi).
    This is far above the task's motor speed and divergence limits at 400 Hz.
    Reset references must be captured after the selected joint states are reset.
    """

    def __init__(self, position_rad: torch.Tensor):
        self.angle_rad = torch.zeros_like(position_rad)
        self._previous_position_rad = position_rad.clone()

    def update(self, position_rad: torch.Tensor) -> None:
        delta_rad = position_rad - self._previous_position_rad
        period_rad = 4.0 * math.pi
        delta_rad -= period_rad * torch.round(delta_rad / period_rad)
        self.angle_rad.add_(delta_rad)
        self._previous_position_rad.copy_(position_rad)

    def reset(self, position_rad: torch.Tensor, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.angle_rad[ids] = 0.0
        self._previous_position_rad[ids] = position_rad[ids]
