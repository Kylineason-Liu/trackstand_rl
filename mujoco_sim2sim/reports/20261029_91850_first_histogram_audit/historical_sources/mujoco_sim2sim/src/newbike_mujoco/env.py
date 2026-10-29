"""MuJoCo environment replicating the NewBike-Balance-v1 deployment contract.

Faithful to the Isaac Sim task on the pieces the policy can observe/feel:
  - control at 50 Hz (0.02 s); physics timestep = the model's own — the
    trackstand augmentation sets 0.0025 s (8 substeps; a dt-0.01 attempt to
    match the sim's physics rate FAILED — MuJoCo's contact can't hold the
    balance there, measured 2026-09-18), the newbike scene keeps 0.0025 s;
    env.step derives the substep count from model.opt.timestep
  - 13-dim observation, exact order and units of
    ``mdp/stationary/observations.py:balance_imu_state``
  - actions ``[steer_target, drive_torque]`` scaled by (1.57 rad, 6.28 N m)
  - reset: fixed spawn state (zeros) by default; with enable_dr=True the
    sim's initial_state DR applies (roll ~ U(-0.10, 0.10), roll rate ~
    U(-0.25, 0.25)), plus the other BalanceEventsCfg DR terms — see the
    DR_* constants. Spawn ON the floor (base z -0.0156, wheels resting —
    a MuJoCo-only deviation; the sim spawns at 0.02 in the air)
  - termination envelope: time_out 1500 steps and fallen (|gy| > sin 30deg or
    inverted) ONLY — the yaw / wheel-net / drift / diverged terms were DELETED
"""

import copy
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
    CTRL_DT = 0.02        # s, the 50 Hz control period (one env step)
    FALLEN_GY = np.sin(np.deg2rad(30.0))
    # ground-contact spawn (user-requested 2026-09-18): at the reset pose both
    # wheel collisions rest ON the floor — contact-scan calibrated (first touch
    # at -0.01532, both wheels by -0.01568; -0.0156 gives <0.3 mm penetration
    # that the solver settles without a drop). MuJoCo-only deviation: the sim
    # spawns at z=0.02 in the air (~35 mm gap) every episode.
    SPAWN_Z = -0.0156
    # obs-noise #8: per-step i.i.d. Gaussian on gravity (0.001) and angvel
    # (0.0012 rad/s), the sim's always-on PerChannelGaussianNoiseCfg stds
    OBS_NOISE_STD = (0.001, 0.001, 0.0012, 0.0012, 0.0012)
    # domain randomization — the sim's BalanceEventsCfg terms, per-episode:
    #   randomize_mass            per-BODY mass scale U(0.8, 1.2); inertia scaled
    #                             by the same per-body ratio (the sim's
    #                             recompute_inertia=True default)
    #   randomize_com             ONE offset 3-vec, each axis U(+-0.005 m), added
    #                             to ALL bodies; inertia untouched (the sim's
    #                             set_coms_index does not touch it either)
    #   randomize_material        per-collider: 32 friction buckets sampled ONCE
    #                             at init (static U(0.5, 0.8), dynamic = min(s, d)
    #                             via make_consistent); each episode draws a
    #                             random bucket per collider. Since the sim's
    #                             tire material now carries
    #                             frictionCombineMode="min" and the ground is
    #                             pinned at 1.0, the PAIR friction equals the
    #                             tire value itself — s IS the ground-wheel
    #                             relative friction, centered on the nominal
    #                             pair 0.65 (the measured real-environment
    #                             value). MuJoCo mirrors this directly: the
    #                             floor is a pass-through 0.1, so MAX-combine
    #                             returns the tire value unchanged.
    #   randomize_actuator_gains  per-actuator INDEPENDENT stiffness/damping
    #                             multipliers U(0.9, 1.1). Measured actuator law:
    #                             steer F = kp*ctrl - kp*qpos - kv*qvel with kp in
    #                             gainprm[0]+biasprm[1] and kv in biasprm[2];
    #                             drive F = 1*ctrl - 0.05*qvel (kp = 0, its
    #                             torque gain is NOT stiffness and stays fixed).
    #   initial_state             roll U(+-0.10) rad, roll-rate U(+-0.25) rad/s
    #                             (applied via reset_ranges)
    #   reset_imu_turnon_residuals  per-episode gyro bias N(0, 2e-4) rad/s and
    #                             accel bias N(0, 0.002) g, added per-step in
    #                             observe() BEFORE the white noise (the sim:
    #                             term adds the bias, then the obs manager
    #                             corrupts)
    DR_MASS = (0.8, 1.2)
    DR_COM = 0.005
    DR_FRICTION = (0.5, 0.8)
    DR_FRICTION_BUCKETS = 32
    DR_GAINS = (0.9, 1.1)
    DR_INIT_ROLL = (-0.10, 0.10)
    DR_INIT_ROLL_RATE = (-0.25, 0.25)
    DR_GYRO_RES = 2e-4
    DR_ACCEL_RES = 0.002

    def __init__(self, scene_path: str | None = None, scene_xml: str | None = None,
                 seed: int = 0, enable_dr: bool = False, model: mujoco.MjModel | None = None):
        if model is not None:
            # Independent mutable model: episode DR must never affect another bike.
            self.model = copy.copy(model)
        elif scene_xml is not None:
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
        # episode starts from the fixed spawn state (zeros). With DR on, the
        # ranges become the sim's CURRENT initial_state DR (U(+-0.10), U(+-0.25)).
        # The viewer sets stronger ranges for the zero_agent / free_fall demos.
        self.reset_ranges = ((0.0, 0.0), (0.0, 0.0))
        # --- domain randomization (the sim's BalanceEventsCfg; per-episode) ---
        self.enable_dr = enable_dr
        if self.enable_dr:
            self.reset_ranges = (self.DR_INIT_ROLL, self.DR_INIT_ROLL_RATE)
        # bike bodies (all non-world bodies — the sim randomizes the whole robot)
        self._dr_body_ids = [i for i in range(1, self.model.nbody)]
        # bike colliders (non-world geoms that collide; the floor is body 0)
        self._dr_geom_ids = [i for i in range(self.model.ngeom)
                             if self.model.geom_bodyid[i] != 0
                             and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])]
        steer_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "steering_act")
        drive_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_act")
        if steer_act < 0 or drive_act < 0:
            raise ValueError("BikeBalanceEnv needs steering_act / drive_act actuators "
                             "(the deploy-contract augmentation)")
        self._dr_steer_act, self._dr_drive_act = steer_act, drive_act
        # the sim pre-samples 32 friction buckets ONCE at init (PhysX's unique-
        # material limit); make_consistent: dynamic = min(static, dynamic)
        self._dr_friction_buckets = self.rng.uniform(*self.DR_FRICTION,
                                                     size=(self.DR_FRICTION_BUCKETS, 2))
        self._dr_friction_buckets[:, 1] = np.minimum(self._dr_friction_buckets[:, 0],
                                                     self._dr_friction_buckets[:, 1])
        # per-episode IMU turn-on residual biases (zeros unless DR is on)
        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(2)
        self.refresh_dr_nominal()
        # the current episode's DR draws (None when DR is off), for inspection
        self.dr: dict | None = None
        # termination envelope: time_out + fallen ONLY (the yaw / wheel-net /
        # drift / diverged terms were DELETED); the deploy_sim viewer task
        # additionally disables time_out, leaving just the roll-angle limit
        self.enabled_terms = {"time_out", "fallen"}
        self._spawn_pos = np.zeros(2)
        self._yaw0 = 0.0
        self._yaw_prev = 0.0
        self._steer_cmd = 0.0     # slew-limited steering target (rad)
        self._step = 0
        self._done = False
        self._stats = EpisodeStats()

    def refresh_dr_nominal(self):
        """Re-capture the current model values as the DR baseline.

        Called at init; call again AFTER any post-init model edit (e.g. the
        viewer's --base_inertia scaling or the no-policy gain zeroing) so the
        restore-before-redraw in ``_apply_dr_episode`` doesn't undo it.
        """
        m = self.model
        self._dr_mass0 = m.body_mass[self._dr_body_ids].copy()
        self._dr_ipos0 = m.body_ipos[self._dr_body_ids].copy()
        self._dr_inertia0 = m.body_inertia[self._dr_body_ids].copy()
        self._dr_friction0 = m.geom_friction[self._dr_geom_ids].copy()
        acts = [self._dr_steer_act, self._dr_drive_act]
        self._dr_gainprm0 = m.actuator_gainprm[acts].copy()
        self._dr_biasprm0 = m.actuator_biasprm[acts].copy()

    def _apply_dr_episode(self):
        """Draw this episode's DR (the sim's BalanceEventsCfg terms).

        Restores the nominal snapshot first, so randomization never compounds
        across episodes — the sim applies every term on its defaults the same
        way (``_randomize_prop_by_op`` starts from the stored defaults).
        """
        m = self.model
        rng = self.rng
        # restore nominal
        m.body_mass[self._dr_body_ids] = self._dr_mass0
        m.body_ipos[self._dr_body_ids] = self._dr_ipos0
        m.body_inertia[self._dr_body_ids] = self._dr_inertia0
        m.geom_friction[self._dr_geom_ids] = self._dr_friction0
        acts = [self._dr_steer_act, self._dr_drive_act]
        m.actuator_gainprm[acts] = self._dr_gainprm0
        m.actuator_biasprm[acts] = self._dr_biasprm0

        # randomize_mass: per-body INDEPENDENT scale; inertia scaled by the
        # same per-body ratio (the sim's recompute_inertia)
        s_mass = rng.uniform(*self.DR_MASS, size=len(self._dr_body_ids))
        m.body_mass[self._dr_body_ids] *= s_mass
        m.body_inertia[self._dr_body_ids] *= s_mass[:, None]

        # randomize_com: ONE offset shared by ALL bodies (the sim samples a
        # single 3-vec per env); inertia intentionally NOT adjusted
        com_offset = rng.uniform(-self.DR_COM, self.DR_COM, size=3)
        m.body_ipos[self._dr_body_ids] += com_offset

        # randomize_material: per-collider random bucket -> tire friction = s
        # directly (the sim's min-combine makes the tire value the pair; the
        # MuJoCo floor is a pass-through 0.1 so MAX-combine returns s)
        bucket_ids = rng.integers(0, self.DR_FRICTION_BUCKETS, size=len(self._dr_geom_ids))
        s_fric = self._dr_friction_buckets[bucket_ids, 0]
        m.geom_friction[self._dr_geom_ids, 0] = s_fric

        # randomize_actuator_gains: independent stiffness/damping multipliers
        # per actuator. Steer: kp = gainprm[0] (ctrl) & biasprm[1] (qpos) —
        # scale BOTH; kv = biasprm[2]. Drive: kp = 0 (0*s stays 0) and its
        # torque gain gainprm[0] is the effort-action scale, NOT stiffness —
        # the sim leaves it fixed, so do we.
        steer, drive = self._dr_steer_act, self._dr_drive_act
        s_stiff, s_damp = rng.uniform(*self.DR_GAINS, size=2)
        m.actuator_gainprm[steer, 0] *= s_stiff
        m.actuator_biasprm[steer, 1] *= s_stiff
        m.actuator_biasprm[steer, 2] *= s_damp
        m.actuator_biasprm[drive, 2] *= rng.uniform(*self.DR_GAINS)

        # reset_imu_turnon_residuals: per-episode constant biases (Gaussian,
        # like the sim's torch.randn), applied per-step in observe()
        self._gyro_bias = rng.normal(0.0, self.DR_GYRO_RES, size=3)
        self._accel_bias = rng.normal(0.0, self.DR_ACCEL_RES, size=2)

        self.dr = {
            "mass_scale": s_mass.copy(),
            "com_offset": com_offset.copy(),
            "tire_friction": m.geom_friction[self._dr_geom_ids, 0].copy(),
            "steer_kp_scale": float(s_stiff),
            "steer_kv_scale": float(s_damp),
            "drive_kv_scale": float(m.actuator_biasprm[drive, 2] / self._dr_biasprm0[1, 2]),
            "gyro_bias": self._gyro_bias.copy(),
            "accel_bias": self._accel_bias.copy(),
        }

    # ------------------------------------------------------------------ reset
    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if self.enable_dr:
            self._apply_dr_episode()
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
        # direct state API (mjdata.h: qvel "velocity (nv x 1)") — no sensors:
        # the freejoint's qvel[3:6] IS the base body's angular velocity in the
        # body frame (what the gyro reported, identically), and qvel[0:3] is the
        # WORLD-frame velocity of the base-frame origin, so one rotation gives
        # the sim's root_lin_vel_b. (The old velocimeter read R.T @ qvel[0:3]
        # sensor-frame and the extra R.T here double-rotated it — masked because
        # balance velocities are ~0.01 m/s.)
        adr = self._base_qveladr
        ang_vel = np.asarray(d.qvel[adr + 3:adr + 6])
        vel_b = R.T @ np.asarray(d.qvel[adr:adr + 3])
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

        # --- DR IMU turn-on residuals (per-episode constants, set by
        #     _apply_dr_episode — the sim's reset_imu_turnon_residuals). Added
        #     BEFORE the white noise, like the sim: the term adds the bias,
        #     then the observation manager corrupts. Zeros when DR is off. ---
        obs[0:2] += self._accel_bias
        obs[2:5] += self._gyro_bias

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
        # substeps = the model's own timestep filling one 0.02 s control period
        # (trackstand augmentation: 0.01 s x 2 = the sim's exact physics rate;
        # newbike scene: 0.0025 s x 8)
        n_sub = max(1, round(self.CTRL_DT / self.model.opt.timestep))
        for _ in range(n_sub):
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
