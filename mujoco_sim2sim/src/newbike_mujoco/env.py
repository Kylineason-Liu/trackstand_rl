"""Trackstand-Balance-v1 task contract in MuJoCo (not a PhysX emulator).

50-Hz observations/policy/slew/reward/terminations; configurable physics/PD
rate (default 400 Hz) and passive mechanical losses. See reports/20261022_v1_alignment
for the audited differences and unavoidable engine-level approximations.
"""

import copy
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import contract as c


def _rot_matrix(q):
    """Body-to-world matrix from MuJoCo wxyz; Isaac's equivalent is xyzw."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def _yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def _roll_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))


def physx_revolute_angle(angle):
    """Observed native PhysX branch: (-2pi, 2pi), period4pi, NOT +/-pi.

    Verified with current v1 direct-state and continuous-rotation probes in
    both directions. MuJoCo's physical hinge coordinate remains cumulative.
    This is an observation adapter, not an encoder or wheel-net counter.
    """
    return 2.0 * np.arctan2(np.sin(angle * .5), np.cos(angle * .5))


@dataclass
class EpisodeStats:
    length: int = 0
    timeout: bool = False
    death: str | None = None
    terminations: dict = field(default_factory=dict)
    reward: float = 0.0
    roll_sq: float = 0.0
    roll_max: float = 0.0
    drift_max: float = 0.0
    drift_sum: float = 0.0
    yaw_max: float = 0.0
    yaw_sum: float = 0.0
    turn: float = 0.0
    rolls: list = field(default_factory=list)
    drive_speed_sum: float = 0.0
    drive_speed_max: float = 0.0
    steer_speed_sum: float = 0.0
    steer_speed_max: float = 0.0


class BikeBalanceEnv:
    """One independent bike. Explicit reset after done keeps terminal telemetry.

    Unlike Isaac's vectorized wrapper this API returns (obs, done) and leaves
    reset to its caller. Reward/overlapping flags are available through
    last_reward/termination_flags/stats. ViewerEnvGroup resets individual bikes.
    """

    STEER_SCALE, DRIVE_SCALE = c.STEER_SCALE, c.DRIVE_SCALE
    CTRL_DT, MAX_STEPS, SPAWN_Z = c.CONTROL_DT, c.MAX_STEPS, c.SPAWN_Z
    FALLEN_GY, OBS_NOISE_STD = np.sin(c.FALL_ANGLE), c.OBS_NOISE_STD
    DR_MASS, DR_COM, DR_FRICTION = c.DR_MASS, c.DR_COM, c.DR_FRICTION
    DR_FRICTION_BUCKETS, DR_GAINS = c.DR_FRICTION_BUCKETS, c.DR_GAINS
    DR_INIT_ROLL, DR_INIT_ROLL_RATE = c.DR_INIT_ROLL, c.DR_INIT_ROLL_RATE
    DR_GYRO_RES, DR_ACCEL_RES = c.DR_GYRO_RES, c.DR_ACCEL_RES

    def __init__(self, scene_path=None, scene_xml=None, seed=0, enable_dr=False, model=None,
                 friction_buckets=None, velocity_limit_mode="project", physics_hz=None):
        self.physics_dt, self.decimation = c.physics_timing(physics_hz)
        if model is not None:
            self.model = copy.copy(model)
        elif scene_xml is not None:
            self.model = mujoco.MjModel.from_xml_string(scene_xml)
        elif scene_path is not None:
            from .model import load_trackstand_model
            self.model = load_trackstand_model(scene_path, physics_hz=physics_hz)
        else:
            raise ValueError("BikeBalanceEnv needs scene_path, scene_xml or model")
        if velocity_limit_mode not in ("project", "none"):
            raise ValueError("velocity_limit_mode must be project or none")
        self.velocity_limit_mode = velocity_limit_mode
        m = self.model
        m.opt.timestep = self.physics_dt
        self.data = mujoco.MjData(m)
        self.rng = np.random.default_rng(seed)
        self._base_qposadr = int(m.joint("base_free").qposadr[0])
        self._base_qveladr = int(m.joint("base_free").dofadr[0])
        self._root_body_id = m.body("base_link").id
        self._steer_jid, self._rear_jid, self._front_jid = (
            m.joint(name).id for name in ("steering_joint", "rearwheel_joint", "frontwheel_joint")
        )
        self._steer_dof, self._rear_dof, self._front_dof = (
            int(m.jnt_dofadr[jid]) for jid in (self._steer_jid, self._rear_jid, self._front_jid)
        )
        self._steer_qadr, self._rear_qadr = (int(m.jnt_qposadr[jid]) for jid in (self._steer_jid, self._rear_jid))
        self._dr_steer_act, self._dr_drive_act = (m.actuator(name).id for name in ("steering_act", "drive_act"))
        self._acts = [self._dr_steer_act, self._dr_drive_act]
        self._dofs = [self._steer_dof, self._rear_dof]
        self._limits = np.array([c.STEER_EFFORT_LIMIT, c.DRIVE_EFFORT_LIMIT])
        self._configure_motors()
        # Joint damping is PASSIVE B, not controller Kd. Frictionloss is a
        # solver constraint with equal static/moving magnitude in this model.
        m.dof_damping[self._dofs] = [c.STEER_VISCOUS_FRICTION, c.DRIVE_VISCOUS_FRICTION]
        m.dof_frictionloss[self._dofs] = [c.STEER_FRICTION, c.DRIVE_FRICTION]
        m.jnt_stiffness[[self._steer_jid, self._rear_jid]] = 0.
        self.kp, self.kd = np.array([c.STEER_KP, c.DRIVE_KP]), np.array([c.STEER_KD, c.DRIVE_KD])
        self.motors_enabled, self.lock_steer = True, False
        self.enable_dr = bool(enable_dr)
        self.reset_ranges = (c.DR_INIT_ROLL, c.DR_INIT_ROLL_RATE) if enable_dr else ((0., 0.), (0., 0.))
        # Exclude the massless export wrapper from physical-body randomization.
        self._dr_body_ids = np.flatnonzero(m.body_mass > 0.)
        # USD link frames are all identity-oriented at q=0 (base.usda); the
        # verified CAD MJCF uses rotated local body frames. Convert one shared
        # Isaac-local CoM offset into EACH corresponding CAD-local frame.
        reference = mujoco.MjData(m)
        mujoco.mj_forward(m, reference)
        self._com_isaac_to_mj = reference.xmat[self._dr_body_ids].reshape(-1, 3, 3).transpose(0, 2, 1).copy()
        self._dr_geom_ids = np.array([i for i in range(m.ngeom) if m.geom_bodyid[i] != 0
                                    and (m.geom_contype[i] or m.geom_conaffinity[i])], dtype=int)
        if friction_buckets is None:
            friction_buckets = self.rng.uniform(*c.DR_FRICTION, size=(c.DR_FRICTION_BUCKETS, 2))
            friction_buckets[:, 1] = np.minimum(friction_buckets[:, 0], friction_buckets[:, 1])
        self._dr_friction_buckets = np.asarray(friction_buckets).copy()
        if self._dr_friction_buckets.shape != (c.DR_FRICTION_BUCKETS, 2):
            raise ValueError("Expected a shared 32 x 2 static/dynamic friction bank")
        self._gyro_bias, self._accel_bias = np.zeros(3), np.zeros(2)
        self.dr = None
        self.enabled_terms = set(c.TERMINATIONS)
        self.refresh_dr_nominal()
        self._spawn_pos = np.zeros(2)
        self._yaw0 = self._yaw_prev = self._steer_cmd = self.wheel_net_angle = 0.
        self._step, self._done, self._stats = 0, False, EpisodeStats()
        self.raw_action, self._prev_action = np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
        self.computed_torque, self.applied_torque = np.zeros(2), np.zeros(2)
        self.physics_torques = np.zeros((self.decimation, 2))
        self.last_reward, self.reward_terms = 0., {}
        self.termination_flags = dict.fromkeys(c.TERMINATIONS, False)

    def _configure_motors(self):
        m = self.model
        if m.nu != 2:
            raise ValueError("V1 expects exactly two motor actuators; no hidden extra actuators")
        for act, joint, limit in zip(self._acts, (self._steer_jid, self._rear_jid), self._limits):
            if m.actuator_trntype[act] != mujoco.mjtTrn.mjTRN_JOINT or m.actuator_trnid[act, 0] != joint:
                raise ValueError("Motor actuator must address the matching named joint")
            m.actuator_dyntype[act] = mujoco.mjtDyn.mjDYN_NONE
            m.actuator_gaintype[act] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_biastype[act] = mujoco.mjtBias.mjBIAS_NONE
            m.actuator_gainprm[act] = 0.
            m.actuator_gainprm[act, 0] = 1.
            m.actuator_biasprm[act] = 0.
            m.actuator_gear[act] = [1., 0., 0., 0., 0., 0.]
            m.actuator_ctrllimited[act] = False
            m.actuator_forcelimited[act] = True
            m.actuator_forcerange[act] = [-limit, limit]
        m.jnt_actfrclimited[[self._steer_jid, self._rear_jid]] = False

    def refresh_dr_nominal(self):
        """Snapshot deliberate pre-reset edits; draws start from defaults."""
        m = self.model
        self._dr_mass0 = m.body_mass[self._dr_body_ids].copy()
        self._dr_ipos0 = m.body_ipos[self._dr_body_ids].copy()
        self._dr_inertia0 = m.body_inertia[self._dr_body_ids].copy()
        self._dr_friction0 = m.geom_friction[self._dr_geom_ids].copy()
        self._kp0, self._kd0 = self.kp.copy(), self.kd.copy()

    def _apply_dr_episode(self):
        m, rng = self.model, self.rng
        m.body_mass[self._dr_body_ids] = self._dr_mass0
        m.body_ipos[self._dr_body_ids] = self._dr_ipos0
        m.body_inertia[self._dr_body_ids] = self._dr_inertia0
        m.geom_friction[self._dr_geom_ids] = self._dr_friction0
        self.kp[:], self.kd[:] = self._kp0, self._kd0
        self._gyro_bias[:], self._accel_bias[:] = 0., 0.
        self.dr = None
        if self.enable_dr:
            mass = rng.uniform(*c.DR_MASS, size=len(self._dr_body_ids))
            offset = rng.uniform(-c.DR_COM, c.DR_COM, 3)
            buckets = rng.integers(c.DR_FRICTION_BUCKETS, size=len(self._dr_geom_ids))
            friction = self._dr_friction_buckets[buckets]
            kp_scale, kd_scale = rng.uniform(*c.DR_GAINS, size=(2, 2))
            m.body_mass[self._dr_body_ids] *= mass
            m.body_inertia[self._dr_body_ids] *= mass[:, None]
            local_offsets = self._com_isaac_to_mj @ offset
            m.body_ipos[self._dr_body_ids] += local_offsets
            # MuJoCo has ONE sliding coefficient, not separate static/dynamic
            # contact coefficients. Use static for the cone; retain both in dr.
            m.geom_friction[self._dr_geom_ids, 0] = friction[:, 0]
            self.kp *= kp_scale
            self.kd *= kd_scale  # Rear zero stays zero, not a torque gain randomization.
            self._gyro_bias[:] = rng.normal(0., c.DR_GYRO_RES, 3)
            self._accel_bias[:] = rng.normal(0., c.DR_ACCEL_RES, 2)
            self.dr = dict(mass_scale=mass, com_offset=offset, com_offset_mj_body=local_offsets,
                           tire_friction=friction[:, 0].copy(),
                           tire_dynamic_friction=friction[:, 1].copy(), bucket_ids=buckets,
                           steer_kp_scale=float(kp_scale[0]), steer_kv_scale=float(kd_scale[0]),
                           drive_kp_scale=float(kp_scale[1]), drive_kv_scale=float(kd_scale[1]),
                           gyro_bias=self._gyro_bias.copy(), accel_bias=self._accel_bias.copy())
        # Runtime mass/CoM changes require derived mass/inertia constants too.
        mujoco.mj_setConst(m, self.data)

    def reset(self):
        self._apply_dr_episode()
        mujoco.mj_resetData(self.model, self.data)
        d, adr = self.data, self._base_qposadr
        roll, rate = (self.rng.uniform(*interval) for interval in self.reset_ranges)
        d.qpos[adr:adr+3] = (0., 0., self.SPAWN_Z)
        d.qpos[adr+3:adr+7] = (np.cos(roll/2), np.sin(roll/2), 0., 0.)
        d.qvel[:] = 0.
        # World X in the reset event; pure X roll leaves local X unchanged.
        d.qvel[self._base_qveladr+3] = rate
        d.qpos[[self._steer_qadr, self._rear_qadr, int(self.model.jnt_qposadr[self._front_jid])]] = 0.
        # Isaac reset_root_state_uniform writes COM linear velocity, not link-
        # origin velocity. MuJoCo free-joint linear qvel belongs to the origin.
        # v_com = v_origin + omega_world x (com-origin); reset v_com to zero.
        mujoco.mj_kinematics(self.model, d)
        com_offset_w = d.xipos[self._root_body_id]-d.qpos[adr:adr+3]
        omega_w = _rot_matrix(self._base_quat()) @ d.qvel[self._base_qveladr+3:self._base_qveladr+6]
        d.qvel[self._base_qveladr:self._base_qveladr+3] = -np.cross(omega_w, com_offset_w)
        self._spawn_pos[:] = 0.
        self._yaw0 = self._yaw_prev = _yaw_from_quat(self._base_quat())
        self._steer_cmd = float(np.clip(d.qpos[self._steer_qadr], -c.STEER_POSITION_LIMIT, c.STEER_POSITION_LIMIT))
        self.wheel_net_angle = 0.
        self.raw_action[:], self._prev_action[:] = 0., 0.
        self.computed_torque[:], self.applied_torque[:], self.physics_torques[:] = 0., 0., 0.
        self._step, self._done, self._stats = 0, False, EpisodeStats()
        self.last_reward, self.reward_terms = 0., {}
        self.termination_flags = dict.fromkeys(c.TERMINATIONS, False)
        mujoco.mj_forward(self.model, d)
        return self.observe()

    def _base_quat(self):
        adr = self._base_qposadr
        return self.data.qpos[adr+3:adr+7]

    def observe(self):
        d = self.data
        gravity = _rot_matrix(self._base_quat()).T @ np.array([0., 0., -1.])
        steer = d.qpos[self._steer_qadr]
        omega = d.qvel[self._base_qveladr+3:self._base_qveladr+6]
        obs = np.array([*gravity[1:3], *omega, steer, physx_revolute_angle(d.qpos[self._rear_qadr]),
                        d.qvel[self._steer_dof], d.qvel[self._rear_dof], np.sin(steer), np.cos(steer)],
                       dtype=np.float32)
        obs[:2] += self._accel_bias
        obs[2:5] += self._gyro_bias
        # Isaac repairs inside the term BEFORE manager noise, with no finite cap.
        obs = np.nan_to_num(obs, nan=0., posinf=0., neginf=0.)
        obs += (self.rng.normal(size=11) * np.asarray(self.OBS_NOISE_STD)).astype(np.float32)
        return obs

    def step(self, action, observe_terminal=True):
        """Process once at 50 Hz; recompute/clamp torque at each physics substep."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected two finite normalized policy actions")
        self._prev_action[:], self.raw_action[:] = self.raw_action, action
        desired = float(np.clip(action[0]*np.float32(c.STEER_SCALE), -c.STEER_POSITION_LIMIT, c.STEER_POSITION_LIMIT))
        delta = np.clip(desired-self._steer_cmd, -c.STEER_SLEW_RATE*self.CTRL_DT, c.STEER_SLEW_RATE*self.CTRL_DT)
        self._steer_cmd += float(delta)
        rear_request = float(action[1]*np.float32(c.DRIVE_SCALE))
        d, m = self.data, self.model
        for sub in range(self.decimation):
            self.computed_torque[:] = (
                self.kp[0]*(self._steer_cmd-d.qpos[self._steer_qadr])-self.kd[0]*d.qvel[self._steer_dof],
                rear_request,
            )
            self.applied_torque[:] = np.clip(self.computed_torque, -self._limits, self._limits)
            if not self.motors_enabled:
                self.applied_torque[:] = 0.
            d.ctrl[self._acts] = self.applied_torque
            mujoco.mj_step(m, d)
            self.physics_torques[sub] = d.actuator_force[self._acts]
            if self.lock_steer:
                d.qpos[self._steer_qadr], d.qvel[self._steer_dof] = 0., 0.
            # MuJoCo has no native PhysX-equivalent soft velocity constraint.
            # Retain the old guard as a labelled, selectable approximation;
            # never confuse it with target slew or a torque-speed envelope.
            if self.velocity_limit_mode == "project":
                d.qvel[self._dofs] = np.clip(d.qvel[self._dofs],
                                            [-c.STEER_SPEED_LIMIT, -c.DRIVE_SPEED_LIMIT],
                                            [c.STEER_SPEED_LIMIT, c.DRIVE_SPEED_LIMIT])
        # mj_step integrates qpos/qvel but leaves some derived kinematics at
        # the previous boundary; refresh rendering/contact diagnostics only.
        mujoco.mj_forward(m, d)
        self._step += 1
        self._done = self._check_terminations()
        self._compute_reward()
        # Match training's intentional ordering: wheel_net is checked BEFORE
        # this endpoint speed is integrated (one-policy-step lag).
        self.wheel_net_angle += float(d.qvel[self._rear_dof])*self.CTRL_DT
        # Auto-reset callers should sample only the new episode observation,
        # like Isaac, not consume an extra terminal white-noise draw first.
        observation = self.observe() if observe_terminal or not self._done else np.zeros(11, dtype=np.float32)
        return observation, self._done

    def _compute_reward(self):
        gravity_y = (_rot_matrix(self._base_quat()).T @ np.array([0., 0., -1.]))[1]
        roll_deg = np.degrees(np.arcsin(np.clip(abs(gravity_y), 0., 1.)))
        drift = np.linalg.norm(self.data.qpos[self._base_qposadr:self._base_qposadr+2])
        self.reward_terms = dict(survival=5.*float(roll_deg < 30.),
                                 drift_gauss=2.*np.exp(-drift*drift/(2*.25**2)),
                                 rate_penalty=-.1*np.linalg.norm(self.raw_action-self._prev_action))
        self.last_reward = float(sum(self.reward_terms.values())*self.CTRL_DT)
        self._stats.reward += self.last_reward

    def _check_terminations(self):
        d, st = self.data, self._stats
        q = self._base_quat()
        rotation = _rot_matrix(q)
        gravity = rotation.T @ np.array([0., 0., -1.])
        pos = d.qpos[self._base_qposadr:self._base_qposadr+2]
        drift = float(np.linalg.norm(pos-self._spawn_pos))
        yaw, roll = _yaw_from_quat(q), _roll_from_quat(q)
        yaw_dev = abs(np.arctan2(np.sin(yaw-self._yaw0), np.cos(yaw-self._yaw0)))
        adr = self._base_qveladr
        angular = d.qvel[adr+3:adr+6]
        com_offset_w = d.xipos[self._root_body_id]-d.qpos[self._base_qposadr:self._base_qposadr+3]
        com_velocity_w = d.qvel[adr:adr+3]+np.cross(rotation @ angular, com_offset_w)
        # Isaac's root_lin_vel_b alias means ROOT-COM velocity in actor axes.
        linear = rotation.T @ com_velocity_w
        joints = d.qvel[[self._steer_dof, self._rear_dof, self._front_dof]]
        diverged = not np.isfinite(np.concatenate((linear, angular, joints))).all()
        diverged |= (np.linalg.norm(linear) > c.MAX_LINEAR_SPEED or np.linalg.norm(angular) > c.MAX_ANGULAR_SPEED
                     or np.max(np.abs(joints)) > c.MAX_JOINT_SPEED)
        flags = dict(time_out=self._step >= self.MAX_STEPS,
                     fallen=abs(gravity[1]) > self.FALLEN_GY or gravity[2] > 0., diverged=diverged,
                     drifted=drift > c.DRIFT_LIMIT, yaw_deviation=yaw_dev > c.YAW_LIMIT,
                     wheel_net=abs(self.wheel_net_angle) > c.WHEEL_NET_LIMIT)
        self.termination_flags = {name: bool(value and name in self.enabled_terms) for name, value in flags.items()}
        st.terminations = self.termination_flags.copy()
        st.timeout = self.termination_flags["time_out"]
        st.death = next((name for name, value in self.termination_flags.items() if name != "time_out" and value), None)
        st.length, st.roll_max, st.drift_max = self._step, max(st.roll_max, abs(roll)), max(st.drift_max, drift)
        st.roll_sq += roll*roll
        st.rolls.append(roll)
        st.yaw_max = max(st.yaw_max, np.degrees(yaw_dev))
        st.yaw_sum += np.degrees(yaw_dev)
        st.drift_sum += drift
        st.turn += abs(np.degrees(np.arctan2(np.sin(yaw-self._yaw_prev), np.cos(yaw-self._yaw_prev))))
        self._yaw_prev = yaw
        for label, dof in (("drive", self._rear_dof), ("steer", self._steer_dof)):
            speed = abs(float(d.qvel[dof]))
            setattr(st, label+"_speed_sum", getattr(st, label+"_speed_sum")+speed)
            setattr(st, label+"_speed_max", max(getattr(st, label+"_speed_max"), speed))
        return any(self.termination_flags.values())

    @property
    def stats(self):
        return self._stats
