"""V2 with physics-rate wheel-net tracking before policy-rate terminations."""

import torch
import warp as wp

from isaaclab.envs import ManagerBasedRLEnv

from ..mdp.stationary.wheel_net import WheelNetAngleTracker
from .trackstand_rl_env import TrackstandRLEnvV1


class TrackstandRLEnvV2(TrackstandRLEnvV1):
    """Track actual rear rotation without v1's delayed endpoint-speed integral."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._wheel_net_callback_handle = None
        self._wheel_net_tracker = None
        super().__init__(cfg, render_mode, **kwargs)
        from isaaclab_physx.physics.physx_manager import IsaacEvents

        self._wheel_net_tracker = WheelNetAngleTracker(self._drive_position())
        self.wheel_net_angle = self._wheel_net_tracker.angle_rad
        self._wheel_net_callback_handle = self.sim.physics_manager.register_callback(
            self._update_wheel_net,
            event=IsaacEvents.POST_PHYSICS_STEP,
            name="trackstand_v2_wheel_net",
        )

    def _drive_position(self):
        # The post-physics callback precedes scene.update(): bypass lazy caches.
        position = self.scene["robot"].root_view.get_dof_positions()
        position = position if isinstance(position, torch.Tensor) else wp.to_torch(position)
        return position[:, self._drive_idx]

    def _update_wheel_net(self, _physics_dt):
        self._wheel_net_tracker.update(self._drive_position())

    def step(self, action):
        # Each physics callback updates wheel_net_angle before the manager checks
        # terminations. Skip the inherited post-step speed integral entirely.
        return ManagerBasedRLEnv.step(self, action)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self._wheel_net_tracker is not None:
            self._wheel_net_tracker.reset(self._drive_position(), env_ids)
            self.spawn_yaw[env_ids] = self._heading_of()[env_ids]

    def reset(self, seed=None, env_ids=None, options=None):
        # _reset_idx handles full, partial, failure and timeout resets alike.
        return ManagerBasedRLEnv.reset(self, seed=seed, env_ids=env_ids, options=options)

    def close(self):
        handle = getattr(self, "_wheel_net_callback_handle", None)
        if handle is not None:
            handle.deregister()
            self._wheel_net_callback_handle = None
        super().close()
