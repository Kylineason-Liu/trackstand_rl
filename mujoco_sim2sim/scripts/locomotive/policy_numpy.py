# Ported from ~/bike_test/CycleRL/scripts/mujoco_paper/policy_numpy.py (imports/paths rewritten for trackstand_rl locomotive).
"""NumPy inference for exported skrl Paper-v0 PPO policies (no torch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(x))


class PaperPolicy:
    """8-D obs → 2-D mean actions with RunningStandardScaler + MLP[512,256,128,64]."""

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path, allow_pickle=True)
        self.mean = data["mean"].astype(np.float32).reshape(-1)
        # skrl RunningStandardScaler: (x - mean) / (sqrt(var) + 1e-8), clamped to ±5.
        self.inv_std = (1.0 / (np.sqrt(data["var"].astype(np.float32).reshape(-1)) + 1e-8)).astype(
            np.float32
        )
        self.obs_clip = 5.0
        self.layers = [
            (data["w0"].astype(np.float32), data["b0"].astype(np.float32)),
            (data["w1"].astype(np.float32), data["b1"].astype(np.float32)),
            (data["w2"].astype(np.float32), data["b2"].astype(np.float32)),
            (data["w3"].astype(np.float32), data["b3"].astype(np.float32)),
            (data["w4"].astype(np.float32), data["b4"].astype(np.float32)),
        ]
        self.steer_scale = float(data["steer_scale"])
        # Joint target clip. Older exports omit it; then the scale is the clip.
        if "steer_clip" in data.files:
            self.steer_clip = float(data["steer_clip"])
        else:
            self.steer_clip = self.steer_scale
        self.drive_scale = float(data["drive_scale"])
        self.drive_clip = float(data["drive_clip"])
        self.steer_kp = float(data["steer_kp"])
        self.steer_kd = float(data["steer_kd"])
        self.drive_kd = float(data["drive_kd"])
        self.steer_effort = float(data["steer_effort"])
        self.drive_effort = float(data["drive_effort"])
        self.obs_dim = int(data["obs_dim"])
        self.act_dim = int(data["act_dim"])

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = (obs.astype(np.float32).reshape(1, -1) - self.mean) * self.inv_std
        x = np.clip(x, -self.obs_clip, self.obs_clip)
        for w, b in self.layers[:-1]:
            x = elu(x @ w.T + b)
        w, b = self.layers[-1]
        return (x @ w.T + b).reshape(-1).astype(np.float32)

    def action_to_targets(self, action: np.ndarray) -> tuple[float, float]:
        """Map policy a ∈ R² to (steer_pos_rad, drive_vel_rad_s)."""
        a0 = float(action[0])
        a1 = float(action[1])
        steer = float(np.clip(a0 * self.steer_scale, -self.steer_clip, self.steer_clip))
        drive = float(np.clip(a1 * self.drive_scale, -self.drive_clip, self.drive_clip))
        return steer, drive
