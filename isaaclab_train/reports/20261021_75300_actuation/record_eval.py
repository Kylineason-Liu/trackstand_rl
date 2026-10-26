"""Capture model_75300 through the unchanged evaluator, without altering controls.

Records the explicit PD input target, final submitted motor torque and separate
solver joint-axis effort. Outputs are isolated from the original training run.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

TRAIN = Path(__file__).resolve().parents[2]
ROOT = TRAIN.parent
RUN = TRAIN / "logs/rsl_rl/trackstand_rl_rsl/2026-10-21_00-41-38"
SCRIPT = TRAIN / "scripts/rsl_rl/play_eval.py"
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--capture-dir", type=Path, required=True)
options, remaining = parser.parse_known_args()
OUT = options.capture_dir.resolve()
OUT.mkdir(parents=True, exist_ok=False)
source_checkpoint = RUN / "model_75300.pt"
checkpoint = OUT / source_checkpoint.name
shutil.copy2(source_checkpoint, checkpoint)
invocation = sys.argv.copy()
sys.argv = [str(SCRIPT), *remaining, "--checkpoint", str(checkpoint)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


paths = [path for directory in (TRAIN / "source", TRAIN / "scripts", TRAIN / "tests")
         for path in directory.rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".usda"}]
paths += [source_checkpoint, RUN / "params/env.yaml", RUN / "params/agent.yaml", Path(__file__)]
paths += [ROOT / "progress_record.md", ROOT / "progress_record.txt"]
paths += [RUN / name for name in ("eval.txt", "torque_profile.png", "torque_profile.npz") if (RUN / name).exists()]
before = {str(path): sha(path) for path in paths}
sys.path[:0] = [str(TRAIN), str(SCRIPT.parent)]
namespace = {"__name__": "__75300_capture__", "__file__": str(SCRIPT)}
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)

import numpy as np
import torch
import warp as wp
import yaml
from isaaclab.actuators import IdealPDActuator
from isaaclab.utils import class_to_dict
from isaaclab.utils.io import dump_yaml
import isaaclab.utils.math as math_utils


class SavedLoader(yaml.SafeLoader):
    pass


SavedLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))
SavedLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice", lambda loader, node: slice(*loader.construct_sequence(node))
)


def normalize(value):
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    if isinstance(value, slice):
        return [value.start, value.stop, value.step]
    return value


def tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return value.torch if hasattr(value, "torch") else wp.to_torch(value)


class Recorder:
    def __init__(self, env):
        self.env = env.unwrapped
        self.robot = self.env.scene["robot"]
        self.joints = [self.robot.find_joints(name)[0][0] for name in ("steering_joint", "rear_wheel_joint")]
        self.steering = self.env.action_manager.get_term("steering")
        self.actuators = [self.robot.actuators[name] for name in ("steering", "drive")]
        assert all(type(actuator) is IdealPDActuator for actuator in self.actuators)
        self.steps, self.decimation, self.n = namespace["args_cli"].num_steps, self.env.cfg.decimation, self.env.num_envs
        shape = (self.steps, self.decimation, self.n, 2)
        self.physics = {key: torch.empty(shape, device=self.env.device) for key in (
            "position_before_rad", "velocity_before_rad_s", "position_after_rad", "velocity_after_rad_s",
            "projected_joint_effort_nm", "computed_motor_torque_nm", "clipped_motor_torque_nm",
            "submitted_motor_torque_nm", "controller_kp", "controller_kd",
        )}
        self.target = torch.empty(shape[:-1], device=self.env.device)
        self.body = torch.empty((*shape[:-1], 3), device=self.env.device)
        self.raw = torch.empty((self.steps, self.n, 2), device=self.env.device)
        self.obs = torch.empty((self.steps, self.n, 11), device=self.env.device)
        self.reward = torch.empty((self.steps, self.n), device=self.env.device)
        self.done = torch.empty((self.steps, self.n), dtype=torch.bool, device=self.env.device)
        self.truncated = torch.empty_like(self.done)
        self.term_names = list(self.env.termination_manager.active_terms)
        self.terms = torch.empty((self.steps, self.n, len(self.term_names)), dtype=torch.bool, device=self.env.device)
        self.completed, self.sub, self.obs_count = 0, 0, 0
        self.active, self.saved = False, False
        dump_yaml(str(OUT / "effective_env.yaml"), self.env.cfg)
        expected = normalize(yaml.load((RUN / "params/env.yaml").read_text(), Loader=SavedLoader))
        effective = normalize(class_to_dict(self.env.cfg))
        checks = {key: expected[key] == effective[key]
                  for key in ("actions", "observations", "rewards", "terminations", "decimation", "episode_length_s")}
        checks["physics_dt"] = expected["sim"]["dt"] == effective["sim"]["dt"]
        checks["actuators"] = expected["scene"]["robot"]["actuators"] == effective["scene"]["robot"]["actuators"]
        checks["spawn"] = expected["scene"]["robot"]["init_state"] == effective["scene"]["robot"]["init_state"]
        checks["dr_flag"] = effective["domain_randomization"] == namespace["args_cli"].dr
        if not namespace["args_cli"].dr:
            checks["nominal_events"] = expected["events"] == effective["events"]
        else:
            names = ("randomize_mass", "randomize_com", "randomize_material", "randomize_actuator_gains",
                     "reset_imu_turnon_residuals")
            checks["full_dr_reset_events_enabled"] = all(effective["events"][name] is not None for name in names)
            checks["full_dr_initial_roll"] = effective["events"]["initial_state"]["params"]["pose_range"]["roll"] == [-0.1, 0.1]
            checks["full_dr_initial_roll_rate"] = effective["events"]["initial_state"]["params"]["velocity_range"]["roll"] == [-0.25, 0.25]
        (OUT / "configuration_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        if not all(checks.values()):
            raise RuntimeError(f"Configuration mismatch: {checks}")
        assert self.env.step_dt == 0.02 and self.env.physics_dt == 0.0025 and self.decimation == 8
        assert hasattr(self.robot.root_view, "get_dof_projected_joint_forces")
        write_original, update_original = self.env.scene.write_data_to_sim, self.env.scene.update

        def write():
            write_original()
            if not self.active:
                return
            if self.sub >= self.decimation:
                raise RuntimeError("Unexpected extra physics write")
            values = {
                "position_before_rad": self.robot.data.joint_pos,
                "velocity_before_rad_s": self.robot.data.joint_vel,
                "computed_motor_torque_nm": self.robot.data.computed_torque,
                "clipped_motor_torque_nm": self.robot.data.applied_torque,
                "submitted_motor_torque_nm": self.robot._joint_effort_target_sim,
            }
            for key, value in values.items():
                self.physics[key][self.completed, self.sub] = tensor(value)[:, self.joints]
            for key, attr in (("controller_kp", "stiffness"), ("controller_kd", "damping")):
                self.physics[key][self.completed, self.sub] = torch.cat(
                    [getattr(actuator, attr) for actuator in self.actuators], dim=-1
                )
            self.target[self.completed, self.sub] = self.steering.processed_actions[:, 0]

        def update(*args, **kwargs):
            update_original(*args, **kwargs)
            if not self.active:
                return
            for key, value in (
                ("position_after_rad", self.robot.data.joint_pos),
                ("velocity_after_rad_s", self.robot.data.joint_vel),
                ("projected_joint_effort_nm", self.robot.root_view.get_dof_projected_joint_forces()),
            ):
                self.physics[key][self.completed, self.sub] = tensor(value)[:, self.joints]
            roll, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w.torch)
            relative_yaw = torch.atan2(torch.sin(yaw - self.env.spawn_yaw), torch.cos(yaw - self.env.spawn_yaw))
            drift = torch.norm(self.robot.data.root_pos_w.torch[:, :2] - self.env.scene.env_origins[:, :2], dim=-1)
            self.body[self.completed, self.sub] = torch.stack((roll, drift, relative_yaw), dim=-1)
            self.sub += 1

        self.env.scene.write_data_to_sim, self.env.scene.update = write, update

    def step(self, wrapped, actions):
        assert self.completed < self.steps and self.obs_count == self.completed + 1
        self.raw[self.completed] = actions
        self.sub, self.active = 0, True
        try:
            result = original_step(wrapped, actions)
        finally:
            self.active = False
        assert self.sub == self.decimation
        self.reward[self.completed] = result[1].reshape(-1)
        self.done[self.completed] = result[2].bool()
        self.truncated[self.completed] = self.env.reset_time_outs
        self.terms[self.completed] = torch.stack(
            [self.env.termination_manager.get_term(name) for name in self.term_names], dim=-1
        )
        self.completed += 1
        if self.completed % 500 == 0:
            print(f"[75300 CAPTURE] {self.completed}/{self.steps}", flush=True)
        return result

    def save(self):
        if self.saved or not self.completed:
            return
        if self.completed != self.steps or self.obs_count != self.steps:
            raise RuntimeError("Incomplete capture: no successful-data archive will be written")
        arrays = {key: value.cpu().numpy() for key, value in self.physics.items()}
        arrays.update(raw_action=self.raw.cpu().numpy(), observation=self.obs.cpu().numpy(),
                      done=self.done.cpu().numpy(), truncated=self.truncated.cpu().numpy(),
                      termination_flags=self.terms.cpu().numpy(), reward=self.reward.cpu().numpy(),
                      held_steering_target_rad=self.target.cpu().numpy(),
                      body_roll_drift_yaw=self.body.cpu().numpy())
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError("Non-finite capture")
        np.savez_compressed(OUT / "motor_samples.npz", **arrays)
        metadata = {
            "capture_completed": True, "invocation": invocation, "checkpoint_source": str(source_checkpoint),
            "checkpoint_sha256": sha(checkpoint), "motor_order": ["steering", "rear"],
            "seed": namespace["args_cli"].seed, "dr": namespace["args_cli"].dr,
            "training_dr": False, "pure_survival": namespace["args_cli"].pure_survival,
            "num_envs": self.n, "completed_policy_steps": self.completed, "decimation": self.decimation,
            "policy_dt_s": self.env.step_dt, "physics_dt_s": self.env.physics_dt,
            "termination_names": self.term_names, "body_columns": ["roll_rad", "drift_m", "relative_yaw_rad"],
            "target_meaning": "Held action-term steering target entering explicit PD, NOT native PhysX position target",
            "torque_meaning": "submitted_motor_torque is the explicit command; projected_joint_effort is the separate solver active joint-axis wrench component, not isolated motor-only torque",
            "timing": "Commands at physics-step start; responses at step end, before automatic resets",
        }
        (OUT / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.saved = True


original_make = namespace["gym"].make
original_step = namespace["RslRlVecEnvWrapper"].step
original_runner = namespace["OnPolicyRunner"]
recorder = None


def make(*args, **kwargs):
    global recorder
    env = original_make(*args, **kwargs)
    recorder = Recorder(env)
    close_original = env.close

    def close():
        recorder.save()
        close_original()

    env.close = close
    return env


def runner(*args, **kwargs):
    dump_yaml(str(OUT / "effective_agent.yaml"), args[1])
    instance = original_runner(*args, **kwargs)
    get_policy = instance.get_inference_policy

    def get_recorded_policy(*args, **kwargs):
        policy = get_policy(*args, **kwargs)

        def recorded(observation):
            assert recorder.obs_count == recorder.completed
            recorder.obs[recorder.completed] = observation["policy"]
            recorder.obs_count += 1
            return policy(observation)

        return recorded

    instance.get_inference_policy = get_recorded_policy
    return instance


namespace["gym"].make = make
namespace["RslRlVecEnvWrapper"].step = lambda wrapped, actions: recorder.step(wrapped, actions)
namespace["OnPolicyRunner"] = runner
success = False
try:
    assert namespace["args_cli"].task == "Trackstand-Balance-v1"
    namespace["main"]()
    assert recorder is not None and recorder.saved
    success = True
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    namespace["gym"].make, namespace["RslRlVecEnvWrapper"].step = original_make, original_step
    namespace["OnPolicyRunner"] = original_runner
    changed = {path: [digest, sha(path)] for path, digest in before.items() if digest != sha(path)}
    (OUT / "source_integrity.json").write_text(json.dumps(
        {"checked_files": len(before), "changed_files": changed, "evaluation_completed": success}, indent=2
    ) + "\n")
    namespace["simulation_app"].close()
