"""Compare deployment with actual installed Isaac terms; no physics or CAN.

Run from the repository with ~/IsaacLab/_isaac_sim/python.sh -B <this file>.
"""

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "isaaclab_train"), str(ROOT / "deploy/source/trackstand_deploy")]

import gymnasium as gym
import isaaclab.utils.math as math_utils
from isaaclab.actuators import IdealPDActuator
from isaaclab.utils.types import ArticulationActions

from trackstand_deploy import contracts as c

PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"


def resolved_class(value):
    if isinstance(value, str):
        module, name = value.split(":")
        return getattr(importlib.import_module(module), name)
    return value


importlib.import_module(PACKAGE)
cfg = gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"]()
assert (cfg.sim.dt, cfg.decimation) == (.0025, 8)
assert cfg.actions.steering.scale == c.STEERING_ACTION_SCALE_RAD
assert cfg.actions.steering.clip == {"steering_joint": (-1.57, 1.57)}
assert cfg.actions.steering.max_target_velocity_rad_s == c.STEERING_VELOCITY_LIMIT_RAD_S
assert cfg.actions.drive.scale == c.DRIVE_ACTION_SCALE_NM
for name, kp, kd, limit in (("steering", 10., .2, 5.6), ("drive", 0., 0., 6.28)):
    actuator = cfg.scene.robot.actuators[name]
    assert resolved_class(actuator.class_type) is IdealPDActuator
    assert (actuator.stiffness, actuator.damping, actuator.effort_limit) == (kp, kd, limit)

rng = np.random.default_rng(42)
count = 1000
quat = torch.from_numpy(rng.normal(size=(count, 4)).astype(np.float32))
quat /= quat.norm(dim=-1, keepdim=True)
omega = rng.normal(size=(count, 3)).astype(np.float32)
steer = rng.uniform(-1.57, 1.57, count)
rear = rng.uniform(-40, 40, count)
rear_native = np.array([c.physx_revolute_angle(value) for value in rear])
speeds = rng.normal(size=(count, 2)).astype(np.float32) * 150
data = SimpleNamespace(
    root_quat_w=SimpleNamespace(torch=quat),
    root_link_ang_vel_w=SimpleNamespace(torch=math_utils.quat_apply(quat, torch.from_numpy(omega))),
    joint_pos=torch.tensor(np.column_stack((steer, rear_native)), dtype=torch.float32),
    joint_vel=torch.from_numpy(speeds),
)
robot = SimpleNamespace(data=data, find_joints=lambda name: ([0 if name == "steering_joint" else 1], [name]))
env = SimpleNamespace(scene={"robot": robot}, device="cpu", num_envs=count)
term = cfg.observations["policy"].imu
expected = term.func(env, **term.params).numpy()
actual = np.array([c.build_observation(quat[i].numpy(), omega[i], steer[i], rear[i], *speeds[i])
                   for i in range(count)])
np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

# Execute the actual v1 action class, including resets, on small in-memory buffers.
Action = resolved_class(cfg.actions.steering.class_type)
action = object.__new__(Action)
action.cfg = cfg.actions.steering
action._raw_actions = torch.zeros(1, 1)
action._processed_actions = torch.zeros(1, 1)
action._last_target = torch.zeros(1, 1)
action._scale, action._offset = 1.57, 0.
action._clip = torch.tensor([[[-1.57, 1.57]]])
action._max_target_step_rad = 23.9 * .02
action._joint_ids = [0]
action._asset = SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.zeros(1, 1))))
limiter = c.SteeringTargetLimiter()
slew_error = 0.
for index in range(1000):
    if index % 31 == 0:
        reset_position = float(rng.uniform(-.05, .05))
        action._asset.data.joint_pos.torch[:] = reset_position
        action.reset()
        limiter.reset(reset_position)
    raw = np.float32(rng.uniform(-1, 1))
    action.process_actions(torch.tensor([[raw]]))
    target, _ = c.scale_policy_action(np.array([raw, 0], dtype=np.float32))
    slew_error = max(slew_error, abs(limiter.update(target) - action._processed_actions.item()))
assert slew_error < 2e-6

# Execute the installed IdealPD equation and clipping, not a transcribed formula.
pd_errors = {}
for name, kp, kd, limit in (("steering", 10., .2, 5.6), ("drive", 0., 0., 6.28)):
    pd = object.__new__(IdealPDActuator)
    pd.stiffness, pd.damping, pd.effort_limit = (torch.tensor(value) for value in (kp, kd, limit))
    q, v, target, ff = (torch.from_numpy(rng.normal(size=(1000, 1)).astype(np.float32) * 10)
                        for _ in range(4))
    if name == "drive":
        target[:] = 0
    command = ArticulationActions(joint_positions=target, joint_velocities=torch.zeros_like(v), joint_efforts=ff)
    output = pd.compute(command, q, v).joint_efforts.numpy()
    reference = np.clip(kp * (target.numpy() - q.numpy()) - kd * v.numpy() + ff.numpy(), -limit, limit)
    np.testing.assert_array_equal(output, reference)
    pd_errors[name] = float(np.max(np.abs(output - reference)))

# Save true BetaMean references for verification by Raspberry-Pi-style ONNX Runtime.
checkpoint = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_11-18-29/model_2500.pt"
state = dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["actor_state_dict"])
assert state.pop("distribution._beta_policy_version").item() == 1
assert not any(key.startswith("obs_normalizer.") for key in state)
indices = sorted({int(key.split(".")[1]) for key in state})
layers = []
for index in indices:
    weight = state[f"mlp.{index}.weight"]
    layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
    if index != indices[-1]:
        layers.append(torch.nn.ELU())
network = torch.nn.Sequential(*layers).eval()
network.load_state_dict({key.removeprefix("mlp."): value for key, value in state.items()})
BetaMean = importlib.import_module(f"{PACKAGE}.agents.beta_distribution").BetaMean
observations = rng.normal(size=(4096, 11)).astype(np.float32)
observations[:1000] = actual
with torch.no_grad():
    actions = BetaMean()(network(torch.from_numpy(observations)).reshape(-1, 2, 2)).numpy()
assert np.isfinite(actions).all() and np.all(np.abs(actions) <= 1)
np.savez_compressed(HERE / "policy_reference.npz", observations=observations, actions=actions)
result = {"seed": 42, "observation_matched_states": count,
          "observation_max_abs_error_per_channel": np.max(np.abs(actual - expected), axis=0).tolist(),
          "slew_actions_with_periodic_resets": 1000, "slew_max_abs_error_rad": slew_error,
          "ideal_pd_max_abs_error_nm": pd_errors, "beta_reference_inputs": 4096,
          "actual_isaac_terms_invoked": True, "physics_or_hardware_started": False,
          "rear_coordinate_note": "Uses previously measured PhysX branch; not a new live engine probe"}
(HERE / "isaac_parity.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
