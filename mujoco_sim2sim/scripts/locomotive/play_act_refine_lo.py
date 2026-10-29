#!/usr/bin/env python3
# Ported from ~/bike_test/CycleRL/scripts/mujoco_paper/play_act_refine_lo.py (imports/paths rewritten for trackstand_rl locomotive).
"""Standalone MuJoCo play for CycleRL Paper-v0 ``act_refine_lo`` (no Isaac / Newton).

Loads the exported NumPy policy + ``new_bike.xml``. Tire collision is a ring of
convex wedges of the CAD tread (the mesh PhysX convex-decomposes in training),
not the MJCF ellipsoid and not a cylinder. Runs ImplicitActuator-equivalent PD
on steer/drive, and opens a MuJoCo viewer.

Default checkpoint export:
  mujoco_sim2sim/logs/locomotive/act_refine_lo_agent_100000.npz
  (from 2026-09-02_01-28-28 / agent_100000.pt)

Usage (system Python with mujoco + numpy — no torch)::

    python3 mujoco_sim2sim/scripts/locomotive/play_act_refine_lo.py
    python3 mujoco_sim2sim/scripts/locomotive/play_act_refine_lo.py --vx 2.0 --duration 30
    python3 mujoco_sim2sim/scripts/locomotive/play_act_refine_lo.py --headless --duration 10

Interactive play opens a box-joystick GUI (↑ speed, ←→ turn). Disable with ``--no-gui``.

If the GLFW viewer fails (GLX / NVIDIA driver mismatch), use EGL video instead::

    python3 mujoco_sim2sim/scripts/locomotive/play_act_refine_lo.py --video mujoco_sim2sim/logs/locomotive/play.gif --duration 10

Re-export policy after a new train (needs Isaac Lab python for torch)::

    ./isaaclab.sh -p mujoco_sim2sim/scripts/locomotive/export_policy_npz.py \\
        --checkpoint ~/bike_test/CycleRL/logs/skrl/cyclerl_logs/CycleRL-NewBike-Paper-v0/\\
2026-09-02_01-28-28_ppo_torch_bicycle_ppo/checkpoints/agent_100000.pt
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# MuJoCo picks the GL backend at import time. Interactive GLFW needs a working
# GLX context; this machine often has NVIDIA driver/library mismatch, so video
# mode forces EGL before importing mujoco.
if any(a == "--video" or a.startswith("--video=") for a in sys.argv[1:]):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parents[2]  # trackstand_rl root
sys.path.insert(0, str(_SCRIPT_DIR))

from policy_numpy import PaperPolicy  # noqa: E402
from control_gui import CommandState, JoystickPanel  # noqa: E402
from tire_from_mesh import (  # noqa: E402
    _STIFF_SOLIMP,
    _STIFF_SOLREF,
    ellipsoid_plant,
    install_cad_tires,
)
from trackstand_plant import is_trackstand_xml, make_trackstand_spec  # noqa: E402

_DEFAULT_NPZ = _REPO / "mujoco_sim2sim/logs/locomotive/act_refine_lo_agent_100000.npz"
_DEFAULT_XML = Path("/home/lmbike2/bike_models/thubike_new/new_bike.xml")
_TRACKSTAND_XML = (
    _REPO
    / "mujoco_sim2sim/models"
    / "trackstand_rl_0916_bike/trackstand_rl_0916.xml"
)
_ISAACLAB = Path("/home/lmbike2/IsaacLab/isaaclab.sh")
_ISAAC_PYTHON = Path("/home/lmbike2/IsaacLab/_isaac_sim/python.sh")
_CKPT_NPZ_DIR = _REPO / "mujoco_sim2sim/logs/locomotive/from_ckpt"

# Match Paper-v0 training: sim.dt=0.01, decimation=2 → 50 Hz policy.
_POLICY_DT = 0.02
_PHYS_DT = 0.001
_WHEEL_RADIUS = 0.0725
_HEADING_STIFFNESS = 0.5  # UniformVelocityCommand heading_control_stiffness
# Isaac clips the heading P-controller output to ranges.ang_vel_z. Training uses
# (-0.5, 0.5); the paper eval suite narrows it to ±0.174 (eval_scenarios).
_TRAIN_YAW_RATE_CLIP = 0.5
_DEFAULT_VX_MIN = 0.8
_DEFAULT_VX_MAX = 3.0
# Full-stick yaw rate. The policy's yaw channel std is ~0.09 rad/s and the
# steer output saturates near ±0.4 rad/s, so a larger stick does not turn harder.
_DEFAULT_TURN_RATE = 0.4

def _wrap_pi(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))


def _require_id(model: mujoco.MjModel, obj: int, names: tuple[str, ...]) -> int:
    for name in names:
        idx = mujoco.mj_name2id(model, obj, name)
        if idx >= 0:
            return idx
    raise KeyError(f"none of {names} in model")


def _is_wheel_geom(name: str) -> bool:
    return name in ("rear_wheel_collision", "front_wheel_collision") or name.startswith(
        ("rear_tire_", "front_tire_")
    )


def zero_lateral_com(model: mujoco.MjModel) -> float:
    """Match Isaac ``zero_lateral_com``: mass-weighted ``body_ipos`` y → 0.

    On this MJCF every body origin is at y = 0, so this is the vehicle CoM.
    Returns the y that was removed (metres).
    """
    mass = np.asarray(model.body_mass[1:], dtype=np.float64)
    total = float(np.sum(mass))
    com_y = float(np.sum(mass * np.asarray(model.body_ipos[1:, 1], dtype=np.float64)) / total)
    model.body_ipos[1:, 1] -= com_y
    return com_y


def _patch_play_plant(model: mujoco.MjModel, policy: PaperPolicy) -> None:
    """Runtime plant tweaks so act_refine_lo can balance outside PhysX."""
    model.opt.timestep = _PHYS_DT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.noslip_iterations = 8

    # Critically damped tire contact on every wheel geom, including the MJCF
    # ellipsoids. Hull install also sets this; repeating it here keeps the
    # ellipsoid cell on the same constraint as the CoACD cell.
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not _is_wheel_geom(name):
            continue
        model.geom_solref[gid] = _STIFF_SOLREF
        model.geom_solimp[gid] = _STIFF_SOLIMP
        model.geom_friction[gid, 0] = 1.0

    # Match exported ImplicitActuator effort limits. Tire meshes are installed
    # before compile; this only sets the solver and motor limits.
    # Isaac's steering joint has no passive damping: ImplicitActuator kd is the only
    # damping term. The MJCF adds 0.05 N·m·s/rad on top of kd=0.2; drop it.
    jid_s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steering_joint")
    model.dof_damping[model.jnt_dofadr[jid_s]] = 0.0

    aid_s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cmd_steering_f")
    aid_d = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cmd_rearwheel_f")
    model.actuator_ctrlrange[aid_s] = [-policy.steer_effort, policy.steer_effort]
    model.actuator_forcerange[aid_s] = [-policy.steer_effort, policy.steer_effort]
    model.actuator_ctrlrange[aid_d] = [-policy.drive_effort, policy.drive_effort]
    model.actuator_forcerange[aid_d] = [-policy.drive_effort, policy.drive_effort]


class PaperBikeSim:
    def __init__(
        self,
        xml_path: Path,
        policy: PaperPolicy,
        vx_cmd: float = 2.0,
        *,
        tires: str = "convex",
        zero_com: bool = False,
    ):
        if is_trackstand_xml(xml_path):
            # Cylinder tires, Isaac inertia, CoM left where the CAD puts it (Isaac
            # disables zero_lateral_com here); nothing is subtracted.
            spec, self.tire, self.com_contact_offset = make_trackstand_spec()
            self.com_y_removed = 0.0
            self.tires = "cylinder"
            zero_com = False
        else:
            spec = mujoco.MjSpec.from_file(str(xml_path))
            if tires == "ellipsoid":
                self.tire = ellipsoid_plant(spec)
            elif tires == "convex":
                self.tire = install_cad_tires(spec)
            else:
                raise ValueError(f"tires must be 'convex' or 'ellipsoid', got {tires!r}")
            self.tires = tires
            self.com_y_removed = 0.0
            self.com_contact_offset = float("nan")
        self.model = spec.compile()
        self.policy = policy
        _patch_play_plant(self.model, policy)
        if zero_com:
            self.com_y_removed = zero_lateral_com(self.model)
        self.zero_com = bool(zero_com)
        self.data = mujoco.MjData(self.model)
        self.vx_cmd = float(vx_cmd)
        self.heading_target = 0.0
        self.yaw_rate_clip = _TRAIN_YAW_RATE_CLIP
        self.paused = False
        self.steer_kp = float(policy.steer_kp)

        self.jid_steer = _require_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, ("steering_joint",))
        self.jid_drive = _require_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, ("rearwheel_joint", "rear_wheel_joint")
        )
        self.jid_front = _require_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, ("frontwheel_joint", "front_wheel_joint")
        )
        self.bid_base = _require_id(self.model, mujoco.mjtObj.mjOBJ_BODY, ("base_link",))
        self.aid_steer = _require_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, ("cmd_steering_f",))
        self.aid_drive = _require_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, ("cmd_rearwheel_f",))

        self.qadr_steer = self.model.jnt_qposadr[self.jid_steer]
        self.qadr_drive = self.model.jnt_qposadr[self.jid_drive]
        self.qadr_front = self.model.jnt_qposadr[self.jid_front]
        self.dadr_steer = self.model.jnt_dofadr[self.jid_steer]
        self.dadr_drive = self.model.jnt_dofadr[self.jid_drive]
        self.dadr_front = self.model.jnt_dofadr[self.jid_front]
        self.qadr_free = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        ]
        self.dadr_free = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        ]

        self.steer_target = 0.0
        self.drive_target = 0.0
        # Stick mode: the drive motor tracks vx* directly. Steer stays with the policy.
        self.drive_remote = False
        self.lock_steer = False
        self._n_sub = max(1, int(round(_POLICY_DT / _PHYS_DT)))
        self.obs_noise_std = 0.0
        # Nominal plant values for domain-randomization restore.
        self._nom_floor_friction = self.model.geom_friction[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        ].copy()
        self._wheel_gids = []
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if _is_wheel_geom(name):
                self._wheel_gids.append(gid)
        self._nom_wheel_friction = {
            gid: self.model.geom_friction[gid].copy() for gid in self._wheel_gids if gid >= 0
        }
        self._nom_body_mass = self.model.body_mass.copy()
        self._nom_body_inertia = self.model.body_inertia.copy()
        self._nom_body_ipos = self.model.body_ipos.copy()
        self._nom_steer_kp = float(self.steer_kp)
        self._nom_steer_kd = float(self.policy.steer_kd)
        self._nom_drive_kd = float(self.policy.drive_kd)

    def restore_nominal_plant(self) -> None:
        """Undo domain-randomization mutations on the MuJoCo model / PD gains."""
        fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.model.geom_friction[fid] = self._nom_floor_friction
        for gid, fr in self._nom_wheel_friction.items():
            self.model.geom_friction[gid] = fr
        self.model.body_mass[:] = self._nom_body_mass
        self.model.body_inertia[:] = self._nom_body_inertia
        self.model.body_ipos[:] = self._nom_body_ipos
        self.steer_kp = self._nom_steer_kp
        self.policy.steer_kd = self._nom_steer_kd
        self.policy.drive_kd = self._nom_drive_kd
        self.obs_noise_std = 0.0

    def apply_domain(
        self,
        *,
        friction: float | None = None,
        mass_scale: float = 1.0,
        com_y: float = 0.0,
        com_z: float = 0.0,
        gain_scale: float = 1.0,
        imu_noise_std: float = 0.0,
    ) -> None:
        """Apply Table-I-style domain shifts (call ``restore_nominal_plant`` after)."""
        self.restore_nominal_plant()
        if friction is not None:
            fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            self.model.geom_friction[fid, 0] = float(friction)
            for gid in self._wheel_gids:
                if gid >= 0:
                    self.model.geom_friction[gid, 0] = float(friction)
        if abs(mass_scale - 1.0) > 1e-12:
            # Isaac randomize_mass scales base_link only (inertia recomputed with it).
            self.model.body_mass[self.bid_base] *= float(mass_scale)
            self.model.body_inertia[self.bid_base] *= float(mass_scale)
        if abs(com_y) > 1e-12 or abs(com_z) > 1e-12:
            self.model.body_ipos[self.bid_base, 1] += float(com_y)
            self.model.body_ipos[self.bid_base, 2] += float(com_z)
        self.steer_kp = self._nom_steer_kp * float(gain_scale)
        self.policy.steer_kd = self._nom_steer_kd * float(gain_scale)
        self.policy.drive_kd = self._nom_drive_kd * float(gain_scale)
        self.obs_noise_std = float(imu_noise_std)

    def apply_lateral_push(self, vy: float) -> None:
        """Add world-frame lateral velocity impulse to the free joint."""
        self.data.qvel[self.dadr_free + 1] += float(vy)

    def reset(self, vx: float | None = None, roll_rad: float = 0.0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        vx = self.vx_cmd if vx is None else float(vx)
        # CAD front and rear crowns differ by ~2.7 mm. Pitch nose-up so both sit
        # on the ground; the MJCF z only seats the front ellipsoid.
        z0 = float(self.tire.z0)
        if abs(roll_rad) > 1e-6:
            z0 += 0.005
        pitch = float(self.tire.pitch_rad)
        self.data.qpos[self.qadr_free : self.qadr_free + 3] = [0.0, 0.0, z0]
        q_pitch = np.array([math.cos(0.5 * pitch), 0.0, math.sin(0.5 * pitch), 0.0])
        q_roll = np.array([math.cos(0.5 * roll_rad), math.sin(0.5 * roll_rad), 0.0, 0.0])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, q_pitch, q_roll)
        self.data.qpos[self.qadr_free + 3 : self.qadr_free + 7] = q
        # freejoint qvel: lin(world) then ang(body)
        v_world = np.zeros(3)
        mujoco.mju_rotVecQuat(v_world, np.array([vx, 0.0, 0.0]), q)
        self.data.qvel[self.dadr_free : self.dadr_free + 3] = v_world
        self.data.qvel[self.dadr_free + 3 : self.dadr_free + 6] = 0.0
        per_rear = self.tire.rear_spin_per_vx or (1.0 / self.tire.radius)
        per_front = self.tire.front_spin_per_vx or per_rear
        omega = vx * per_rear
        self.data.qvel[self.dadr_drive] = omega
        self.data.qvel[self.dadr_front] = vx * per_front
        self.data.qpos[self.qadr_steer] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.steer_target = 0.0
        self.drive_target = omega
        self.heading_target = self._heading()

    def _heading(self) -> float:
        xmat = self.data.xmat[self.bid_base].reshape(3, 3)
        return float(math.atan2(xmat[1, 0], xmat[0, 0]))

    def roll_rad(self) -> float:
        """Isaac ``euler_xyz_from_quat`` roll of ``base_link`` (eval_logger convention)."""
        w, x, y, z = (float(v) for v in self.data.xquat[self.bid_base])
        return float(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))

    def forward_vel(self) -> float:
        """``root_lin_vel_b[:, 0]``: base_link CoM velocity in the body frame.

        freejoint qvel[0:3] is the body-origin velocity in world, qvel[3:6] the
        angular velocity in the body frame. Isaac reports the CoM velocity, so add
        ω × r_com. (mj_objectVelocity(..., flg_local=1) is wrong with this COM offset.)
        """
        xmat = self.data.xmat[self.bid_base].reshape(3, 3)
        v_origin = np.asarray(self.data.qvel[self.dadr_free : self.dadr_free + 3], dtype=np.float64)
        omega_w = xmat @ np.asarray(self.data.qvel[self.dadr_free + 3 : self.dadr_free + 6], dtype=np.float64)
        r_com = self.data.xipos[self.bid_base] - self.data.xpos[self.bid_base]
        return float((xmat.T @ (v_origin + np.cross(omega_w, r_com)))[0])

    def yaw_rate_cmd(self) -> float:
        """Heading P-controller output, clipped like Isaac's ``ranges.ang_vel_z``."""
        raw = _HEADING_STIFFNESS * _wrap_pi(self.heading_target - self._heading())
        return float(np.clip(raw, -self.yaw_rate_clip, self.yaw_rate_clip))

    def clean_obs(self) -> np.ndarray:
        """Match Paper ``imu_orientation_and_velocity`` + command block (8-D), noise-free."""
        xmat = self.data.xmat[self.bid_base].reshape(3, 3)
        g_body = xmat.T @ np.array([0.0, 0.0, -1.0])
        gravity_y = float(g_body[1])
        wx, wy, wz = (float(v) for v in self.data.qvel[self.dadr_free + 3 : self.dadr_free + 6])
        steer = _wrap_pi(float(self.data.qpos[self.qadr_steer]))
        return np.array(
            [gravity_y, wx, wy, wz, self.forward_vel(), steer, self.vx_cmd, self.yaw_rate_cmd()],
            dtype=np.float32,
        )

    def _obs(self) -> np.ndarray:
        """Policy input: ``clean_obs`` plus the configured observation noise."""
        obs = self.clean_obs()
        if self.obs_noise_std > 0.0:
            # Noise on the 6-D IMU/encoder block only (not command channels).
            obs[:6] += np.random.normal(0.0, self.obs_noise_std, size=6).astype(np.float32)
        return obs

    def _apply_pd(self) -> None:
        steer_pos = float(self.data.qpos[self.qadr_steer])
        steer_vel = float(self.data.qvel[self.dadr_steer])
        drive_vel = float(self.data.qvel[self.dadr_drive])
        p = self.policy
        tau_s = self.steer_kp * (self.steer_target - steer_pos) + p.steer_kd * (0.0 - steer_vel)
        # Remote drive uses a stiffer velocity loop so the wheel holds vx*.
        drive_kd = 40.0 if self.drive_remote else p.drive_kd
        tau_d = drive_kd * (self.drive_target - drive_vel)
        self.data.ctrl[self.aid_steer] = float(np.clip(tau_s, -p.steer_effort, p.steer_effort))
        self.data.ctrl[self.aid_drive] = float(np.clip(tau_d, -p.drive_effort, p.drive_effort))

    def step_with_obs(self, obs: np.ndarray) -> np.ndarray:
        """Run one 50 Hz policy step on ``obs``. Returns the raw policy action.

        Isaac does not clip the raw action; JointPosition/JointVelocity actions clip
        the scaled joint targets (``action_to_targets``).
        """
        action = self.policy.act(obs)
        if self.lock_steer:
            # Ablation, as play_with_metrics --ablate_steer lock: steer action 0 → fork held straight.
            action = np.array(action, dtype=np.float64, copy=True)
            action[0] = 0.0
        self.steer_target, policy_drive = self.policy.action_to_targets(action)
        if self.drive_remote:
            self.drive_target = self._remote_drive_rate(self.forward_vel())
        else:
            self.drive_target = policy_drive
        for _ in range(self._n_sub):
            self._apply_pd()
            mujoco.mj_step(self.model, self.data)
        return action

    def policy_step(self) -> dict:
        obs = self._obs()
        action = self.step_with_obs(obs)
        # Report the true post-step state; re-reading _obs() would add fresh noise.
        return {
            "obs": obs,
            "action": action,
            "steer_target": self.steer_target,
            "drive_target": self.drive_target,
            "vx": self.forward_vel(),
            "roll_deg": math.degrees(self.roll_rad()),
            "heading_deg": math.degrees(self._heading()),
        }

    def _remote_drive_rate(self, vx_body: float) -> float:
        """Hub rate (rad/s) that holds body speed on vx*.

        Feedforward is the no-slip rate. A trim on the body-speed error takes
        out slip. The result stays inside the policy drive clip.
        """
        per = self.tire.rear_spin_per_vx or (1.0 / max(self.tire.radius, 1e-6))
        err = self.vx_cmd - float(vx_body)
        rate = per * (self.vx_cmd + 1.0 * err)
        clip = abs(float(self.policy.drive_clip))
        return float(np.clip(rate, -clip, clip))

    def apply_commands(self, cmd: CommandState, turn_rate_max: float) -> None:
        """Push GUI joystick into policy command channels."""
        self.vx_cmd = float(cmd.vx_cmd)
        self.paused = bool(cmd.paused)
        self.drive_remote = bool(cmd.drive_remote)
        # Map stick X → desired yaw rate via heading error / stiffness.
        # yaw_rate_cmd = stiffness * (heading_target - heading) ≈ turn * turn_rate_max
        if abs(cmd.turn) > 1e-3 and _HEADING_STIFFNESS > 1e-6:
            self.heading_target = self._heading() + (cmd.turn * turn_rate_max) / _HEADING_STIFFNESS
        else:
            # Hold current heading when stick is centered.
            self.heading_target = self._heading()


def _resolve_checkpoint(path: Path) -> Path:
    """A .pt file, a checkpoints directory, or a training run directory."""
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise SystemExit(f"Checkpoint path not found: {path}")
    folder = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    best = folder / "best_agent.pt"
    if best.is_file():
        return best
    agents = []
    for pt in folder.glob("agent_*.pt"):
        stem = pt.stem.replace("agent_", "")
        if stem.isdigit():
            agents.append((int(stem), pt))
    if not agents:
        raise SystemExit(f"No best_agent.pt or agent_*.pt in {folder}")
    return max(agents)[1]


def _run_dir(ckpt: Path) -> Path | None:
    """Resolve the training run directory from a .pt, checkpoints/, or run path."""
    path = Path(ckpt)
    if path.is_dir() and (path / "params" / "env.yaml").is_file():
        return path
    if path.is_dir() and path.name == "checkpoints" and (path.parent / "params" / "env.yaml").is_file():
        return path.parent
    if path.parent.name == "checkpoints":
        return path.parent.parent
    if (path.parent / "params" / "env.yaml").is_file():
        return path.parent
    return None


def _scales_from_run(run: Path | None) -> dict[str, float]:
    """Action scales from the run's env.yaml. Falls back to the Paper ±20° contract."""
    scales = {
        "steer_scale": 0.3490658503988659,
        "steer_clip": 0.3490658503988659,
        "drive_scale": 41.37931034482759,
        "drive_kd": 5.0,  # CycleRL implicit rear Kd; aligned runs override it below
    }
    if run is None:
        return scales
    env_yaml = run / "params" / "env.yaml"
    if not env_yaml.is_file():
        return scales
    text = env_yaml.read_text(encoding="utf-8")
    actions = text.split("\nactions:\n", 1)
    if len(actions) < 2:
        return scales
    block = actions[1].split("\nevents:\n", 1)[0]

    def term(name: str) -> str:
        """Text of one action term (``  <name>:`` up to the next 2-space key)."""
        if f"\n  {name}:\n" not in "\n" + block:
            return ""
        body = ("\n" + block).split(f"\n  {name}:\n", 1)[1]
        end = re.search(r"\n  [A-Za-z_]", body)
        return body[: end.start()] if end else body

    def scale_of(text: str) -> float | None:
        match = re.search(r"\n    scale:\s*([0-9.eE+-]+)", "\n" + text)
        return float(match.group(1)) if match else None

    steer_text, drive_text = term("steering"), term("drive")
    steer, drive = scale_of(steer_text), scale_of(drive_text)
    if steer is not None:
        scales["steer_scale"] = steer
        # ``clip: null`` (or no clip key) means Isaac never clips the steer target.
        clip = re.search(
            r"\n    clip:\s*\n\s+steering_joint:[^\n]*\n\s+-\s*([0-9.eE+-]+)", "\n" + steer_text
        )
        scales["steer_clip"] = abs(float(clip.group(1))) if clip else float("inf")
    if drive is not None:
        scales["drive_scale"] = drive
    # The rear motor gain is trained per run (Kd 5 CycleRL, Kd 0.2 Locomotive-Cylinder-Aligned-v0).
    kd = _drive_kd_from_env_yaml(text)
    if kd is not None:
        scales["drive_kd"] = kd
    return scales


def _drive_kd_from_env_yaml(text: str) -> float | None:
    """``scene.robot.actuators.drive.damping`` from a run's env.yaml, if present."""
    import yaml

    try:
        cfg = yaml.unsafe_load(text)
        return float(cfg["scene"]["robot"]["actuators"]["drive"]["damping"])
    except Exception:  # noqa: BLE001 - old/partial env.yaml: keep the default
        return None


def _zero_com_from_run(run: Path | None) -> bool:
    """True when the run's env.yaml applied ``zero_lateral_com`` (CoM-z recipes)."""
    if run is None:
        return False
    env_yaml = run / "params" / "env.yaml"
    if not env_yaml.is_file():
        return False
    text = env_yaml.read_text(encoding="utf-8")
    if "\n  zero_lateral_com:\n" not in text:
        return False
    block = text.split("\n  zero_lateral_com:\n", 1)[1]
    # The event's own keys are indented four spaces; the next event ends the block.
    end = re.search(r"\n  [A-Za-z_]", block)
    block = block[: end.start()] if end else block
    mode = re.search(r"\n    mode:\s*(\w+)", block)
    return bool(mode and mode.group(1) != "disabled")


def _xml_for_checkpoint(ckpt: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if "trackstand" in str(ckpt).lower():
        return _TRACKSTAND_XML
    return _DEFAULT_XML


def _export_checkpoint(ckpt: Path, scales: dict[str, float]) -> Path:
    """Cache an npz next to the MuJoCo logs. Exports with Isaac Lab python when needed."""
    run = _run_dir(ckpt)
    run_name = run.name if run is not None else ckpt.parent.name
    out = _CKPT_NPZ_DIR / run_name / f"{ckpt.stem}.npz"
    if out.is_file() and out.stat().st_mtime >= ckpt.stat().st_mtime:
        cached = np.load(out, allow_pickle=True)
        stored = {
            "steer_scale": float(cached["steer_scale"]),
            "steer_clip": float(cached["steer_clip"]) if "steer_clip" in cached.files else float("nan"),
            "drive_scale": float(cached["drive_scale"]),
            "drive_kd": float(cached["drive_kd"]),
        }
        if all(np.isclose(stored[k], scales[k], rtol=1e-5) or stored[k] == scales[k] for k in stored):
            return out
        print(f"[INFO] cached {out.name} has scales {stored}, run has {scales}; re-exporting")
    out.parent.mkdir(parents=True, exist_ok=True)
    export_py = _SCRIPT_DIR / "export_policy_npz.py"
    export_args = [
        str(export_py),
        f"--checkpoint={ckpt}",
        f"--output={out}",
        f"--steer-scale={scales['steer_scale']}",
        f"--steer-clip={scales['steer_clip']}",
        f"--drive-scale={scales['drive_scale']}",
        f"--drive-kd={scales['drive_kd']}",
    ]
    if _ISAAC_PYTHON.is_file():
        cmd = [str(_ISAAC_PYTHON), *export_args]
        cwd = str(_REPO)
    elif _ISAACLAB.is_file():
        cmd = [str(_ISAACLAB), "-p", *export_args]
        cwd = str(_ISAACLAB.parent)
    else:
        raise SystemExit(f"No torch launcher found ({_ISAAC_PYTHON})")
    print(f"[INFO] exporting {ckpt.name} → {out}")
    subprocess.check_call(cmd, cwd=cwd)
    if not out.is_file():
        raise SystemExit(f"Export did not write {out}")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone MuJoCo play for act_refine_lo.")
    p.add_argument("--npz", type=Path, default=None, help="Exported policy npz.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="skrl .pt, a checkpoints/ folder, or a training run folder. "
        "A folder uses best_agent.pt when it is present.",
    )
    p.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="MuJoCo MJCF. Trackstand checkpoints use the cylinder plant when this is omitted.",
    )
    p.add_argument("--vx", type=float, default=2.0, help="Initial forward speed command (m/s).")
    p.add_argument("--vx-min", type=float, default=_DEFAULT_VX_MIN, help="Joystick min vx* (m/s).")
    p.add_argument("--vx-max", type=float, default=_DEFAULT_VX_MAX, help="Joystick max vx* (m/s).")
    p.add_argument(
        "--turn-rate",
        type=float,
        default=_DEFAULT_TURN_RATE,
        help="Yaw-rate command at full left/right stick (rad/s).",
    )
    p.add_argument("--duration", type=float, default=0.0, help="Seconds to run (0 = forever).")
    p.add_argument("--headless", action="store_true", help="No viewer; print status.")
    p.add_argument("--no-gui", action="store_true", help="Disable joystick control window.")
    p.add_argument(
        "--video",
        type=Path,
        default=None,
        help="EGL offscreen render to .gif (avoids broken GLX viewer).",
    )
    p.add_argument("--width", type=int, default=640, help="Video frame width.")
    p.add_argument("--height", type=int, default=360, help="Video frame height.")
    p.add_argument("--roll-init-deg", type=float, default=0.0, help="Initial roll (deg).")
    p.add_argument(
        "--zero-com",
        action="store_true",
        help="Apply zero_lateral_com (CoM-z recipes). Read from the run's env.yaml with --checkpoint.",
    )
    p.add_argument(
        "--drive-remote",
        action="store_true",
        help="Joystick sets the drive motor speed. The policy only steers.",
    )
    p.add_argument(
        "--lock-steer",
        action="store_true",
        help="Ablation: zero the steer action (fork held straight); the policy still drives.",
    )
    return p.parse_args()


_VIEWER_HINT = """\
GLFW/GLX window failed (often NVIDIA driver/library mismatch).

Workarounds:
  # Offscreen GIF via EGL (no interactive window)
  python3 mujoco_sim2sim/scripts/locomotive/play_act_refine_lo.py --video mujoco_sim2sim/logs/locomotive/play.gif --duration 10

  # Physics-only
  python3 mujoco_sim2sim/scripts/locomotive/play_act_refine_lo.py --headless --duration 10

  # Fix interactive viewer: reboot or reload NVIDIA driver so nvidia-smi works,
  # then retry without --video.
"""


# GLFW codes. The viewer callback receives a press (and repeat), not a release.
_VIEWER_LEFT = (263, 65)  # Left, A — positive turn, left yaw
_VIEWER_RIGHT = (262, 68)  # Right, D
_VIEWER_UP = (265, 87)  # Up, W
_VIEWER_DOWN = (264, 83)  # Down, S


def _viewer_key(cmd: CommandState, vx_min: float, vx_max: float):
    """Arrow / WASD from the MuJoCo window. Left is a left turn."""

    def _on_key(keycode: int) -> None:
        if keycode in _VIEWER_LEFT:
            cmd.key_turn = 1.0
            cmd.key_turn_until = time.time() + 0.45
        elif keycode in _VIEWER_RIGHT:
            cmd.key_turn = -1.0
            cmd.key_turn_until = time.time() + 0.45
        elif keycode in _VIEWER_UP:
            cmd.vx_cmd = float(min(vx_max, cmd.vx_cmd + 0.1))
        elif keycode in _VIEWER_DOWN:
            cmd.vx_cmd = float(max(vx_min, cmd.vx_cmd - 0.1))
        else:
            return
        cmd.knob_dirty = True

    return _on_key


def _follow_cam(viewer, body_id: int) -> None:
    """Track ``body_id`` from a side/rear chase pose (matches MJCF chasis_camera)."""
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = int(body_id)
        viewer.cam.fixedcamid = -1
        viewer.cam.distance = 1.6
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 90.0  # look from -Y toward +X travel


def _make_follow_mjv_camera(body_id: int) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = int(body_id)
    cam.distance = 1.6
    cam.elevation = -15.0
    cam.azimuth = 90.0
    return cam


def _run_video(sim: PaperBikeSim, out_path: Path, max_steps: int, width: int, height: int) -> None:
    """Render policy rollout to an animated GIF using EGL (no GLFW window)."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow required for --video (pip install pillow)") from exc

    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    cam = _make_follow_mjv_camera(sim.bid_base)
    frames: list = []
    # Subsample to keep GIF small: 10 Hz from 50 Hz policy.
    every = 5
    step_i = 0
    try:
        while step_i < max_steps:
            info = sim.policy_step()
            step_i += 1
            if step_i % every == 0:
                renderer.update_scene(sim.data, camera=cam)
                frames.append(Image.fromarray(renderer.render()))
            if step_i % 25 == 0:
                print(
                    f"t={step_i * _POLICY_DT:6.2f}s  "
                    f"vx={info['vx']:+.2f}  roll={info['roll_deg']:+.1f}°  "
                    f"hdg={info['heading_deg']:+.1f}°"
                )
            if abs(info["roll_deg"]) > 45.0:
                print(f"[FAIL] crashed at t={step_i * _POLICY_DT:.2f}s roll={info['roll_deg']:+.1f}°")
                break
        if not frames:
            raise SystemExit("No frames captured")
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 * _POLICY_DT * every),
            loop=0,
            optimize=False,
        )
        print(f"[PASS] wrote {out_path} ({len(frames)} frames)")
    finally:
        renderer.close()


def main() -> None:
    args = _parse_args()
    zero_com = bool(args.zero_com)
    if args.checkpoint is not None:
        ckpt = _resolve_checkpoint(args.checkpoint)
        scales = _scales_from_run(_run_dir(ckpt))
        args.npz = _export_checkpoint(ckpt, scales)
        args.xml = _xml_for_checkpoint(ckpt, args.xml)
        zero_com = zero_com or _zero_com_from_run(_run_dir(ckpt))
        print(
            f"[INFO] checkpoint={ckpt.name} steer_scale={scales['steer_scale']:.4f} "
            f"drive_scale={scales['drive_scale']:.2f} zero_lateral_com={zero_com}"
        )
    if args.npz is None:
        args.npz = _DEFAULT_NPZ
    if args.xml is None:
        args.xml = _DEFAULT_XML
    if not args.npz.is_file():
        raise SystemExit(
            f"Missing policy npz: {args.npz}\n"
            "Export with:\n"
            "  ./isaaclab.sh -p mujoco_sim2sim/scripts/locomotive/export_policy_npz.py "
            "--checkpoint <agent_100000.pt>\n"
            "Or pass --checkpoint <run-or-checkpoints-folder>."
        )
    if not args.xml.is_file():
        raise SystemExit(f"Missing MJCF: {args.xml}")

    policy = PaperPolicy(args.npz)
    sim = PaperBikeSim(args.xml, policy, vx_cmd=args.vx, zero_com=zero_com)
    sim.lock_steer = bool(args.lock_steer)
    sim.reset(roll_rad=math.radians(args.roll_init_deg))
    print(
        f"[INFO] policy={args.npz.name} xml={args.xml.name} "
        f"vx*={args.vx} steer_scale={policy.steer_scale:.4f} "
        f"drive_scale={policy.drive_scale:.2f} "
        f"plant={sim.tires} geoms={sim.tire.n_hulls} "
        f"R={sim.tire.radius*1000:.1f}mm "
        f"half_width={sim.tire.half_width*1000:.2f}mm "
        f"zero_lateral_com={sim.zero_com} com_y_removed={sim.com_y_removed*1000:.2f}mm "
        f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '(default/glfw)')}"
    )

    max_steps = int(args.duration / _POLICY_DT) if args.duration > 0 else None
    step_i = 0

    if args.video is not None:
        if max_steps is None:
            max_steps = int(10.0 / _POLICY_DT)  # default 10 s for video
            print(f"[INFO] --video default duration=10s ({max_steps} policy steps)")
        _run_video(sim, args.video, max_steps, args.width, args.height)
        return

    if args.headless:
        while max_steps is None or step_i < max_steps:
            info = sim.policy_step()
            step_i += 1
            if step_i % 25 == 0:
                print(
                    f"t={step_i * _POLICY_DT:6.2f}s  "
                    f"vx={info['vx']:+.2f}  roll={info['roll_deg']:+.1f}°  "
                    f"hdg={info['heading_deg']:+.1f}°  "
                    f"a=[{info['action'][0]:+.2f},{info['action'][1]:+.2f}]"
                )
            if abs(info["roll_deg"]) > 45.0:
                print(f"[FAIL] crashed at t={step_i * _POLICY_DT:.2f}s roll={info['roll_deg']:+.1f}°")
                break
        else:
            print("[PASS] finished without lean crash")
        return

    try:
        from mujoco import viewer as mj_viewer
    except ImportError as exc:
        raise SystemExit("mujoco.viewer unavailable; use --headless or --video") from exc

    cmd = CommandState(vx_cmd=float(args.vx), drive_remote=bool(args.drive_remote))
    sim.drive_remote = bool(args.drive_remote)
    try:
        viewer_cm = mj_viewer.launch_passive(
            sim.model, sim.data, key_callback=_viewer_key(cmd, args.vx_min, args.vx_max)
        )
    except Exception as exc:
        raise SystemExit(f"{exc}\n\n{_VIEWER_HINT}") from exc

    if not args.no_gui:
        vx0 = float(np.clip(args.vx, args.vx_min, args.vx_max))
        cmd.vx_cmd = vx0
        sim.vx_cmd = vx0
        JoystickPanel(cmd, vx_min=args.vx_min, vx_max=args.vx_max).start()
        print(
            f"[INFO] joystick: drag the pad, or arrows/WASD in either window. "
            f"← left, → right, ↑/↓ speed [{args.vx_min:.1f},{args.vx_max:.1f}] m/s, "
            f"full stick ±{args.turn_rate:.2f} rad/s, "
            f"drive={'stick' if args.drive_remote else 'policy'}"
        )

    with viewer_cm as viewer:
        _follow_cam(viewer, sim.bid_base)
        next_policy = time.time()
        while viewer.is_running() and (max_steps is None or step_i < max_steps):
            if cmd.quit_requested:
                break
            if cmd.reset_requested:
                cmd.reset_requested = False
                sim.vx_cmd = float(cmd.vx_cmd)
                sim.reset(roll_rad=math.radians(args.roll_init_deg))
                _follow_cam(viewer, sim.bid_base)
            if not args.no_gui:
                sim.apply_commands(cmd, turn_rate_max=float(args.turn_rate))
            if not sim.paused:
                now = time.time()
                if now >= next_policy:
                    info = sim.policy_step()
                    next_policy = now + _POLICY_DT
                    step_i += 1
                    if step_i % 50 == 0:
                        print(
                            f"t={step_i * _POLICY_DT:6.2f}s  "
                            f"vx={info['vx']:+.2f}/{sim.vx_cmd:.2f}  "
                            f"roll={info['roll_deg']:+.1f}°  "
                            f"hdg={info['heading_deg']:+.1f}°→{math.degrees(sim.heading_target):+.1f}°"
                        )
                    if abs(info["roll_deg"]) > 45.0:
                        print("[WARN] lean > 45° — resetting")
                        sim.reset(roll_rad=math.radians(args.roll_init_deg))
                        _follow_cam(viewer, sim.bid_base)
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
