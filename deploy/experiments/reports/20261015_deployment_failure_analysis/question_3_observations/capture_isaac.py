"""Capture the actual policy inputs of unchanged play_eval.py; Isaac Sim Python required.

Only a new report directory is written. No extra observation-manager computation,
noise draw, control action, or physics step is added during evaluation. Synthetic
rear-angle probes run AFTER evaluation and are labelled separately.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "isaaclab_train").is_dir())
TRAIN = ROOT / "isaaclab_train"
RUN = TRAIN / "logs/rsl_rl/trackstand_rl_rsl/2026-09-22_19-45-06"
SCRIPT = TRAIN / "scripts/rsl_rl/play_eval.py"
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--capture-dir", type=Path, required=True)
parser.add_argument("--deploy-velocity", action="store_true", help="Isolated in-memory vx/vy observation ablation")
options, remaining = parser.parse_known_args()
OUT = options.capture_dir.resolve()
if OUT.exists():
    raise FileExistsError(f"Use a fresh capture directory: {OUT}")
OUT.mkdir(parents=True)
checkpoint = OUT / "model_91850.pt"
shutil.copy2(RUN / checkpoint.name, checkpoint)
invocation = sys.argv.copy()
sys.argv = [str(SCRIPT), *remaining, "--checkpoint", str(checkpoint)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


protected = [p for folder in (TRAIN / "source", TRAIN / "scripts") for p in folder.rglob("*")
             if p.is_file() and p.suffix in {".py", ".yaml", ".usda"}]
protected += [RUN / checkpoint.name, Path(__file__).resolve()]
before = {str(p): sha(p) for p in protected}
sys.path[:0] = [str(TRAIN), str(SCRIPT.parent)]
ns = {"__name__": "__observation_capture__", "__file__": str(SCRIPT)}
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), ns)

import numpy as np
import torch

from isaaclab.utils import math as math_utils
from isaaclab.utils.io import dump_yaml


class Recorder:
    def __init__(self, env):
        self.env = env.unwrapped
        self.robot = self.env.scene["robot"]
        self.joints = [self.robot.find_joints(name)[0][0] for name in ("steering_joint", "rear_wheel_joint")]
        self.count = 0
        self.saved = False
        if options.deploy_velocity:
            term = self.env.observation_manager._group_obs_term_cfgs["policy"][0]
            original_term = term.func

            def estimated_velocity(*args, **kwargs):
                value = original_term(*args, **kwargs)
                value[:, 5] = .07 * value[:, 10]
                value[:, 6] = 0
                return value

            term.func = estimated_velocity
        shape = (ns["args_cli"].num_steps, self.env.num_envs)
        widths = {"observation": 13, "raw_action": 2, "clean_observation": 13,
                  "root_quaternion_xyzw": 4, "root_position_w_m": 3,
                  "linear_velocity_world_m_s": 3, "angular_velocity_world_rad_s": 3,
                  "imu_bias": 5}
        self.arrays = {key: torch.empty((*shape, width), device=self.env.device) for key, width in widths.items()}
        self.arrays["done"] = torch.empty(shape, dtype=torch.bool, device=self.env.device)
        dump_yaml(str(OUT / "effective_env.yaml"), self.env.cfg)

    def capture(self, policy, observations):
        if self.count >= len(self.arrays["observation"]):
            raise RuntimeError("Unexpected extra inference")
        groups = policy.obs_groups
        if groups != ["policy"]:
            raise RuntimeError(f"Unexpected actor input groups: {groups}")
        obs = observations["policy"]
        r = self.robot
        q = r.data.root_quat_w.torch
        lin_w = r.data.root_link_lin_vel_w.torch
        ang_w = r.data.root_link_ang_vel_w.torch
        g = torch.zeros_like(lin_w)
        g[:, 2] = -1
        gravity = math_utils.quat_apply_inverse(q, g)
        ang = math_utils.quat_apply_inverse(q, ang_w)
        lin = math_utils.quat_apply_inverse(q, lin_w)
        pos = r.data.joint_pos[:, self.joints]
        vel = r.data.joint_vel[:, self.joints]
        clean = torch.cat((gravity[:, 1:], ang, lin[:, :2], pos, vel,
                           torch.sin(pos[:, :1]), torch.cos(pos[:, :1])), dim=1)
        bias = torch.zeros((self.env.num_envs, 5), device=self.env.device)
        if hasattr(self.env, "_imu_tilt_bias"):
            bias[:, :2] = self.env._imu_tilt_bias
            bias[:, 2:] = self.env._imu_gyro_bias
        clean = torch.nan_to_num(clean + torch.nn.functional.pad(bias, (0, 8)), nan=0, posinf=0, neginf=0)
        clean = torch.clamp(clean, -100, 100)
        if options.deploy_velocity:
            clean[:, 5] = .07 * clean[:, 10]
            clean[:, 6] = 0
        action = policy(observations)
        values = {"observation": obs, "raw_action": action, "clean_observation": clean,
                  "root_quaternion_xyzw": q, "root_position_w_m": r.data.root_pos_w.torch,
                  "linear_velocity_world_m_s": lin_w, "angular_velocity_world_rad_s": ang_w,
                  "imu_bias": bias}
        for key, value in values.items():
            self.arrays[key][self.count] = value
        return action

    def step(self, wrapper, actions):
        result = original_step(wrapper, actions)
        self.arrays["done"][self.count] = result[2].bool()
        self.count += 1
        if self.count % 500 == 0:
            print(f"[OBS CAPTURE] {self.count} steps", flush=True)
        return result

    def save(self):
        if self.saved or not self.count:
            return
        arrays = {key: value[:self.count].cpu().numpy() for key, value in self.arrays.items()}
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError("Non-finite observations/state")
        np.savez_compressed(OUT / "observations.npz", **arrays)
        noise = arrays["observation"] - arrays["clean_observation"]
        metadata = {"invocation": invocation, "checkpoint_sha256": sha(checkpoint),
                    "deploy_velocity_ablation": options.deploy_velocity,
                    "num_steps": self.count, "num_envs": self.env.num_envs,
                    "dr": ns["args_cli"].dr, "seed": ns["args_cli"].seed,
                    "noise_std_measured": noise.reshape(-1, 13).std(axis=0).tolist(),
                    "noise_abs_max": np.abs(noise).reshape(-1, 13).max(axis=0).tolist(),
                    "capture": "Exact actor input before inference; clean state includes episode bias, excludes white noise"}
        (OUT / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.saved = True

    def probe_rear_angles(self):
        """Known joint-state writes and one physics step, AFTER the saved rollout."""
        targets = torch.tensor([2.8, 3.5, 7.1, -2.8, -3.5, -7.1], device=self.env.device)
        if self.env.num_envs < len(targets):
            raise RuntimeError("Rear-angle probe needs at least six environments")
        ids = torch.arange(len(targets), device=self.env.device)
        positions = self.robot.data.joint_pos[ids].clone()
        velocities = torch.zeros_like(positions)
        positions[:, self.joints[1]] = targets
        self.robot.write_joint_state_to_sim(positions, velocities, env_ids=ids)
        read_before = self.robot.data.joint_pos[ids, self.joints[1]].clone()
        # Clear previous targets so the diagnostic is not a replayed policy action.
        self.robot.set_joint_effort_target(torch.zeros_like(self.robot.data.joint_pos))
        self.robot.set_joint_position_target(self.robot.data.joint_pos.clone())
        self.robot.set_joint_velocity_target(torch.zeros_like(self.robot.data.joint_pos))
        self.env.scene.write_data_to_sim()
        self.env.sim.step(render=False)
        self.env.scene.update(self.env.physics_dt)
        read_after = self.robot.data.joint_pos[ids, self.joints[1]].clone()
        term = self.env.observation_manager._group_obs_term_cfgs["policy"][0]
        observation_after = term.func(self.env, **term.params)[ids, 8]
        result = {"scope": "Post-evaluation synthetic state probe; not a balancing trajectory or motor calibration",
                  "requested_joint_position_rad": targets.cpu().tolist(),
                  "immediate_joint_position_rad": read_before.cpu().tolist(),
                  "after_one_physics_step_rad": read_after.cpu().tolist(),
                  "actual_observation_term_rear_position_rad": observation_after.cpu().tolist()}
        (OUT / "rear_angle_probe.json").write_text(json.dumps(result, indent=2) + "\n")


original_make = ns["gym"].make
original_step = ns["RslRlVecEnvWrapper"].step
original_inference = ns["OnPolicyRunner"].get_inference_policy
recorder = None


def make(*args, **kwargs):
    global recorder
    env = original_make(*args, **kwargs)
    recorder = Recorder(env)
    close_original = env.close

    def close():
        recorder.save()
        recorder.probe_rear_angles()
        close_original()

    env.close = close
    return env


def inference(runner, *args, **kwargs):
    policy = original_inference(runner, *args, **kwargs)
    return lambda observations: recorder.capture(policy, observations)


ns["gym"].make = make
ns["RslRlVecEnvWrapper"].step = lambda env, actions: recorder.step(env, actions)
ns["OnPolicyRunner"].get_inference_policy = inference
try:
    ns["main"]()
finally:
    if recorder is not None:
        recorder.save()
    ns["gym"].make = original_make
    ns["RslRlVecEnvWrapper"].step = original_step
    ns["OnPolicyRunner"].get_inference_policy = original_inference
    changed = {path: [digest, sha(Path(path))] for path, digest in before.items() if digest != sha(Path(path))}
    (OUT / "source_integrity.json").write_text(json.dumps({"checked_files": len(before), "changed_files": changed}, indent=2) + "\n")
    ns["simulation_app"].close()
    if changed:
        raise RuntimeError(f"Protected files changed: {changed}")
