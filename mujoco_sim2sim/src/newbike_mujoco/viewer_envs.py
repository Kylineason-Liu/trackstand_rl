"""Independent CPU environments and render-only grid placement for the viewer."""

import math

import mujoco
import numpy as np

from .env import BikeBalanceEnv


class ViewerEnvGroup:
    """One physics model/data pair and seed stream per bike; no shared DR state.

    Environment zero retains the original seed. Further environments use seed+i.
    All physics runs at the original local origin. Grid offsets affect rendering
    only, so they do not alter observations, contacts, or termination distances.
    """

    def __init__(self, first_env: BikeBalanceEnv, num_envs: int, seed: int):
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.envs = [first_env]
        self.envs.extend(
            BikeBalanceEnv(model=first_env.model, seed=seed + i, enable_dr=first_env.enable_dr)
            for i in range(1, num_envs)
        )
        columns = math.ceil(math.sqrt(num_envs))
        self.offsets = np.array([
            (1.5 * (i % columns), 1.5 * (i // columns), 0.0) for i in range(num_envs)
        ])
        self._render_perturb = mujoco.MjvPerturb()

    def reset(self) -> np.ndarray:
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance every bike once, without resetting; return obs, done, motor torque.

        Torque is (N, 2) in N m, ordered [drive, steering]. The caller records
        terminal stats before resetting only the corresponding environment.
        """
        actions = np.asarray(actions)
        if actions.shape != (len(self.envs), 2):
            raise ValueError("Expected one two-channel action per environment")
        observations, done, torque = [], [], []
        for env, action in zip(self.envs, actions):
            obs, finished = env.step(action)
            observations.append(obs)
            done.append(finished)
            torque.append([
                env.data.actuator_force[env.model.actuator("drive_act").id],
                env.data.actuator_force[env.model.actuator("steering_act").id],
            ])
        return np.stack(observations), np.asarray(done, dtype=bool), np.asarray(torque)

    def configure_camera(self, camera: mujoco.MjvCamera) -> None:
        if len(self.envs) > 1:
            camera.lookat[:] = (self.offsets.min(axis=0) + self.offsets.max(axis=0)) / 2
            camera.lookat[2] = 0.2
            camera.distance = max(2.0, 1.5 * (np.ptp(self.offsets[:, :2], axis=0).max() + 1.5))
            camera.azimuth = 90.0
            camera.elevation = -60.0

    def update_scene(self, scene: mujoco.MjvScene, options: mujoco.MjvOption) -> None:
        """Append all bikes except env zero, which the passive viewer already draws.

        Model copies keep identical mesh/material IDs, so the primary model's
        render assets serve every bike. Only moving-body geoms are appended:
        duplicate floors would cause z-fighting and occlude the grid.
        """
        scene.ngeom = 0
        required = (len(self.envs) - 1) * self.envs[0].model.ngeom
        if required > scene.maxgeom:
            raise ValueError(
                f"Grid needs up to {required} render geoms; viewer capacity is {scene.maxgeom}. "
                "Reduce --num_envs."
            )
        for env, offset in zip(self.envs[1:], self.offsets[1:]):
            start = scene.ngeom
            mujoco.mjv_addGeoms(
                env.model, env.data, options, self._render_perturb,
                mujoco.mjtCatBit.mjCAT_DYNAMIC, scene,
            )
            for index in range(start, scene.ngeom):
                scene.geoms[index].pos[:] += offset
                # Copies are visual-only, not bodies in the primary model.
                # Do not let picking one accidentally perturb environment zero.
                scene.geoms[index].objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                scene.geoms[index].objid = -1
