"""Env class for Trackstand-rl-v0 — standalone.

Maintains per-episode state that reward functions cannot compute statelessly:

- ``wheel_net_angle`` (num_envs,): the drive wheel's ACCUMULATED rotation since
  reset. ``joint_pos`` WRAPS at +-pi in this stack (verified empirically: write
  10.0 rad, read back -2.5 rad), so the integration lives here.
- ``spawn_yaw`` (num_envs,): the robot's heading at (post-)reset, via the
  canonical ``heading_w`` (base-frame +x rotated to world). "Heading vs spawn"
  terms MUST be relative to this per-env baseline.

The drive-joint index is resolved from the ACTION config (no hardcoded joint
names), so this class works for any robot spelling of the drive joint.
"""

import torch
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedRLEnv


class TrackstandRLHybridEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv + per-env net wheel angle and spawn-yaw baselines."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.wheel_net_angle = torch.zeros(self.num_envs, device=self.device)
        self.spawn_yaw = torch.zeros(self.num_envs, device=self.device)
        drive_joint_name = self.cfg.actions.drive.joint_names[0]
        self._drive_idx = self.scene["robot"].find_joints(drive_joint_name)[0][0]

    def _heading_of(self):
        """Yaw of the base frame, computed from the RAW root quaternion.

        The raw pose buffer is updated synchronously by the reset-write kernels
        (write_root_link_pose_to_sim_index writes into the data buffer directly),
        so this is fresh in EVERY window — unlike ``data.heading_w``, whose
        kernel cache is NOT invalidated by ``_reset_pose()`` and which returns
        the pre-reset heading until the next physics step.
        """
        quat = self.scene["robot"].data.root_quat_w.torch  # (N, 4) xyzw, raw buffer
        return math_utils.euler_xyz_from_quat(quat)[2]

    def step(self, action):
        obs, rew, term, trunc, extras = super().step(action)
        # integrate this step's drive-wheel rotation; zero for envs that just reset
        dv = self.scene["robot"].data.joint_vel[:, self._drive_idx]
        self.wheel_net_angle = self.wheel_net_angle + dv * self.step_dt
        # capture the post-reset heading as each resetting env's new episode baseline
        yaw = self._heading_of()
        self.spawn_yaw[self.reset_terminated] = yaw[self.reset_terminated]
        self.wheel_net_angle[self.reset_terminated] = 0.0
        return obs, rew, term, trunc, extras

    def reset(self, seed=None, options=None):
        obs, extras = super().reset(seed=seed, options=options)
        self.wheel_net_angle.zero_()
        self.spawn_yaw = self._heading_of().clone()
        return obs, extras
