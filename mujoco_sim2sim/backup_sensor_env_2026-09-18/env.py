"""MuJoCo environment replicating the NewBike-Balance-v1 deployment contract.

Faithful to the Isaac Sim task on the pieces the policy can observe/feel:
  - control at 50 Hz (0.02 s), physics timestep 0.01 s (2 substeps per control)
  - 13-dim observation, exact order and units of
    ``mdp/stationary/observations.py:balance_imu_state``
  - actions ``[steer_target, drive_torque]`` scaled by (1.57 rad, 6.28 N m)
  - reset: roll ~ U(-0.10, 0.10), roll rate ~ U(-0.25, 0.25) (the sim's
    initial_state event), spawn height 0.02 like the sim's spawn
  - termination envelope: time_out 1500 steps and fallen (|gy| > sin 30deg or
    inverted) ONLY — the yaw / wheel-net / drift / diverged terms were DELETED
"""

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np


def _rot_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a w-first quaternion (MuJoCo framequat)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _roll_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))


@dataclass
class EpisodeStats:
    """Aggregated per-episode metrics, play_eval-comparable."""

    length: int = 0
    timeout: bool = False
    death: str | None = None
    roll_sq: float = 0.0          # sum of roll^2 for RMS
    roll_max: float = 0.0
    drift_max: float = 0.0
    yaw_max: float = 0.0          # deg
    turn: float = 0.0             # deg of accumulated heading change
    rolls: list[float] = field(default_factory=list)
    drive_speed_sum: float = 0.0  # |rear-wheel qvel| sums (rad/s)
    drive_speed_max: float = 0.0
    steer_speed_sum: float = 0.0  # |steering qvel| sums (rad/s)
    steer_speed_max: float = 0.0


def compose_scene_xml(model_xml_path: str) -> str:
    """Wrap an arbitrary bike MJCF in a scene: checker floor + include.

    Writes the composed scene as a temp file NEXT TO the model (so the model's
    relative meshdir/mesh paths keep their VFS context) and returns its path.
    """
    model_xml_path = os.path.abspath(model_xml_path)
    scene_path = os.path.join(os.path.dirname(model_xml_path), ".scene_composed.xml")
    xml = f"""<mujoco model="composed_scene">
  <include file="{model_xml_path}"/>
  <asset>
    <texture name="texplane" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             width="512" height="512" gridsize="10 10"/>
    <material name="matplane" texture="texplane" texuniform="true" reflectance="0.2"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" friction="0.75 0 0" material="matplane"/>
  </worldbody>
</mujoco>"""
    with open(scene_path, "w") as f:
        f.write(xml)
    return scene_path


class BikeBalanceEnv:
    """Single bike + deployment policy loop. One instance = one env."""

    # deployment contract (NewBike-Balance-v1, from the env cfg on disk)
    STEER_SCALE = 1.57        # rad, position target
    DRIVE_SCALE = 6.28        # N m (torque) per unit action
    DRIVE_KV = 0.05           # N m s / rad (actuator damping -> velocity servo)
    # velocity-servo target: ctrl = a1 * DRIVE_SCALE / DRIVE_KV; the actuator's
    # ctrlrange clamps it to +-61.6 rad/s (the GIM3510-8 speed limit)
    MAX_STEPS = 1500      # 30 s at 50 Hz
    FALLEN_GY = np.sin(np.deg2rad(30.0))
    SPAWN_Z = 0.02          # the sim's spawn height (wheels start ~3.6 cm up)
    # obs-noise #8: per-step i.i.d. Gaussian on gravity (0.001) and angvel
    # (0.0012 rad/s), the sim's always-on PerChannelGaussianNoiseCfg stds
    OBS_NOISE_STD = (0.001, 0.001, 0.0012, 0.0012, 0.0012)

    def __init__(self, scene_path: str | None = None, scene_xml: str | None = None, seed: int = 0):
        if scene_xml is not None:
            self.model = mujoco.MjModel.from_xml_string(scene_xml)
        elif scene_path is not None:
            self.model = mujoco.MjModel.from_xml_path(scene_path)
        else:
            raise ValueError("BikeBalanceEnv needs scene_path or scene_xml")
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self._base_qposadr = int(self.model.joint("base_free").qposadr)
        self._base_qveladr = int(self.model.joint("base_free").dofadr)
        self._steer_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "steering_joint")
        self._rear_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rearwheel_joint")
        self._steer_dof = int(self.model.jnt_dofadr[self._steer_jid])
        self._rear_dof = int(self.model.jnt_dofadr[self._rear_jid])
        self._steer_qadr = int(self.model.jnt_qposadr[self._steer_jid])
        self._front_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "frontwheel_joint")
        self._front_dof = int(self.model.jnt_dofadr[self._front_jid])
        self.lock_steer = False  # set True to weld the headset at 0 (zero_agent demo)
        # reset perturbation ranges: (roll rad, roll-rate rad/s). Default = the
        # sim's 0950 protocol: the initial_state event is wrapped, so every
        # episode starts from the fixed spawn state (zeros). The viewer sets
        # stronger ranges for the zero_agent / free_fall demos.
        self.reset_ranges = ((0.0, 0.0), (0.0, 0.0))
        # termination envelope: time_out + fallen ONLY (the yaw / wheel-net /
        # drift / diverged terms were DELETED); the deploy_sim viewer task
        # additionally disables time_out, leaving just the roll-angle limit
        self.enabled_terms = {"time_out", "fallen"}
        self._imu_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        self._vel_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "vel")
        self._spawn_pos = np.zeros(2)
        self._yaw0 = 0.0
        self._yaw_prev = 0.0
        self._steer_cmd = 0.0     # slew-limited steering target (rad)
        self._step = 0
        self._done = False
        self._stats = EpisodeStats()

    # ------------------------------------------------------------------ reset
    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        adr = self._base_qposadr
        roll_range, roll_vel_range = self.reset_ranges
        roll = self.rng.uniform(*roll_range)
        roll_vel = self.rng.uniform(*roll_vel_range)
        q = np.array([
            np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0,  # w-first, roll about x
        ])
        d.qpos[adr:adr + 3] = (0.0, 0.0, self.SPAWN_Z)
        d.qpos[adr + 3:adr + 7] = q
        d.qvel[self._base_qveladr:self._base_qveladr + 3] = 0.0
        d.qvel[self._base_qveladr + 3:self._base_qveladr + 6] = (roll_vel, 0.0, 0.0)
        self._spawn_pos[:] = d.qpos[adr:adr + 2]
        self._yaw0 = _yaw_from_quat(q)
        self._yaw_prev = self._yaw0
        self._steer_cmd = 0.0
        self._step = 0
        self._done = False
        self._stats = EpisodeStats()
        mujoco.mj_forward(self.model, d)
        return self.observe()

    def _base_quat(self) -> np.ndarray:
        """USD base-frame orientation (w-first) from the free-joint qpos.

        Not the framequat sensor: MuJoCo's body-framequat returns the INERTIAL
        frame (body x iquat), which is rotated by the principal-axes quat here.
        """
        adr = self._base_qposadr
        return np.asarray(self.data.qpos[adr + 3:adr + 7])

    # -------------------------------------------------------------------- obs
    def observe(self) -> np.ndarray:
        d = self.data
        q = self._base_quat()
        R = _rot_matrix(q)
        g_b = R.T @ np.array([0.0, 0.0, -1.0])          # gravity in body frame
        ang_vel = np.asarray(d.sensordata[self.model.sensor_adr[self._imu_sid]:
                                           self.model.sensor_adr[self._imu_sid] + 3])
        vel_w = np.asarray(d.sensordata[self.model.sensor_adr[self._vel_sid]:
                                         self.model.sensor_adr[self._vel_sid] + 3])
        vel_b = R.T @ vel_w                               # body-frame linear velocity
        steer_pos = float(d.qpos[self.model.jnt_qposadr[self._steer_jid]])
        # the sim's drive_pos obs is the WRAPPED joint_pos (PhysX wraps at +-pi;
        # verified in the sim's wheel_net termination docstring) — the unbounded
        # MuJoCo hinge must be wrapped for the policy to see the same values
        drive_raw = float(d.qpos[self.model.jnt_qposadr[self._rear_jid]])
        drive_pos = (drive_raw + np.pi) % (2 * np.pi) - np.pi
        steer_vel = float(d.qvel[self.model.jnt_dofadr[self._steer_jid]])
        drive_vel = float(d.qvel[self.model.jnt_dofadr[self._rear_jid]])
        obs = np.array([
            g_b[1], g_b[2],
            ang_vel[0], ang_vel[1], ang_vel[2],
            vel_b[0], vel_b[1],
            steer_pos, drive_pos,
            steer_vel, drive_vel,
            np.sin(steer_pos), np.cos(steer_pos),
        ], dtype=np.float32)

        # --- #8 white noise (the sim's always-on obs corruption; per-step i.i.d.,
        #     applied after the term each step like the observation manager) ---
        # gravity sigma 0.001, body angvel sigma 0.0012 rad/s; the joint
        # channels carry zero noise in the sim's 13-entry std tuple
        obs[0] += self.rng.normal(0.0, self.OBS_NOISE_STD[0])
        obs[1] += self.rng.normal(0.0, self.OBS_NOISE_STD[1])
        obs[2:5] += self.rng.normal(0.0, self.OBS_NOISE_STD[2], 3)
        return obs

    # ------------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply the scaled deployment action, advance 2 physics substeps (0.02 s)."""
        d = self.data
        # steering: velocity_limit_sim 23.9 rad/s -> slew-limited position target
        desired = float(np.clip(action[0] * self.STEER_SCALE, -1.5708, 1.5708))
        self._steer_cmd += float(np.clip(desired - self._steer_cmd, -23.9 * 0.02, 23.9 * 0.02))
        d.ctrl[0] = self._steer_cmd
        # the sim's drive: JointEffortAction (scale 6.28, clip null) -- the
        # torque target is 6.28*a1 (the actuator's force range clamps it)
        d.ctrl[1] = float(action[1]) * self.DRIVE_SCALE
        for _ in range(8):  # 8 x 0.0025 s substeps per 0.02 s control period
            mujoco.mj_step(self.model, d)
            if self.lock_steer:
                # weld the headset only — freezing the wheels would turn the
                # bike into a rim-rocker with a stable lean equilibrium
                d.qpos[self._steer_qadr] = 0.0
                d.qvel[self._steer_dof] = 0.0
            # velocity_limit_sim: PhysX hard-brakes the joint at the motor's
            # speed limit; emulate with a per-substep clamp on BOTH motors
            if abs(d.qvel[self._steer_dof]) > 23.9:
                d.qvel[self._steer_dof] = np.clip(d.qvel[self._steer_dof], -23.9, 23.9)
            if abs(d.qvel[self._rear_dof]) > 61.6:
                d.qvel[self._rear_dof] = np.clip(d.qvel[self._rear_dof], -61.6, 61.6)
        self._step += 1
        self._done = self._check_terminations()
        return self.observe(), self._done

    def _check_terminations(self) -> bool:
        d = self.data
        q = self._base_quat()
        R = _rot_matrix(q)
        g_b = R.T @ np.array([0.0, 0.0, -1.0])
        st = self._stats
        st.length = self._step

        roll = _roll_from_quat(q)
        st.roll_sq += roll * roll
        st.roll_max = max(st.roll_max, abs(roll))
        st.rolls.append(roll)

        pos = d.qpos[self._base_qposadr:self._base_qposadr + 2]
        drift = float(np.linalg.norm(pos - self._spawn_pos))
        st.drift_max = max(st.drift_max, drift)

        yaw = _yaw_from_quat(q)
        delta = (yaw - self._yaw_prev + np.pi) % (2 * np.pi) - np.pi
        st.turn += abs(np.degrees(delta))
        self._yaw_prev = yaw
        dsp = abs(float(d.qvel[self._rear_dof]))
        ssp = abs(float(d.qvel[self._steer_dof]))
        st.drive_speed_sum += dsp
        st.drive_speed_max = max(st.drive_speed_max, dsp)
        st.steer_speed_sum += ssp
        st.steer_speed_max = max(st.steer_speed_max, ssp)
        yaw_dev = abs((yaw - self._yaw0 + np.pi) % (2 * np.pi) - np.pi)
        st.yaw_max = max(st.yaw_max, np.degrees(yaw_dev))

        if self._step >= self.MAX_STEPS and "time_out" in self.enabled_terms:
            st.timeout = True
            return True
        if "fallen" in self.enabled_terms and (abs(g_b[1]) > self.FALLEN_GY or g_b[2] > 0.0):
            st.death = "fallen"
            return True
        return False

    @property
    def stats(self) -> EpisodeStats:
        return self._stats
