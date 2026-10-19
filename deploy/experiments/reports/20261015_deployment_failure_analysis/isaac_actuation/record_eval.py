"""Record unchanged play_eval.py, including solver-measured joint-axis effort.

Use Isaac Sim Python. All evaluator flags are passed through. A checkpoint copy
isolates evaluator outputs from the training run. This wrapper adds telemetry,
not a new actuator implementation, force clamp, or dynamics model.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "isaaclab_train").is_dir())
TRAIN = ROOT / "isaaclab_train"
RUN = TRAIN / "logs/rsl_rl/trackstand_rl_rsl/2026-09-22_19-45-06"
SCRIPT = TRAIN / "scripts/rsl_rl/play_eval.py"
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--capture-dir", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, default=RUN / "model_91850.pt")
options, remaining = parser.parse_known_args()
OUT = options.capture_dir.resolve()
if OUT.exists():
    raise FileExistsError(f"Use a fresh capture directory: {OUT}")
OUT.mkdir(parents=True)
checkpoint = OUT / options.checkpoint.name
shutil.copy2(options.checkpoint, checkpoint)
invocation = sys.argv.copy()
sys.argv = [str(SCRIPT), *remaining, "--checkpoint", str(checkpoint)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source_paths = sorted({
    path for directory in (TRAIN / "source", TRAIN / "scripts")
    for path in directory.rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".usda"}
})
before = {str(path): sha(path) for path in [*source_paths, options.checkpoint, Path(__file__).resolve()]}
sys.path[:0] = [str(TRAIN), str(SCRIPT.parent)]
namespace = {"__name__": "__joint_effort_capture__", "__file__": str(SCRIPT)}
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)

import numpy as np
import torch
import warp as wp
import yaml

from isaaclab.utils import class_to_dict
from isaaclab.utils.io import dump_yaml


class SavedLoader(yaml.SafeLoader):
    pass


SavedLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
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
    if isinstance(value, str):
        return value.replace("CycleRL.tasks.manager_based.cyclerl", "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand")
    return value


def tensor(value):
    if hasattr(value, "torch"):
        return value.torch
    return wp.to_torch(value) if isinstance(value, wp.array) else value


class Recorder:
    def __init__(self, env):
        self.env = env.unwrapped
        self.robot = self.env.scene["robot"]
        self.joints = [self.robot.find_joints(name)[0][0] for name in ("steering_joint", "rear_wheel_joint")]
        self.steps = namespace["args_cli"].num_steps
        self.decimation = self.env.cfg.decimation
        self.n = self.env.num_envs
        shape = (self.steps, self.decimation, self.n, 2)
        self.physics = {key: torch.empty(shape, device=self.env.device) for key in (
            "position_before_rad", "velocity_before_rad_s", "position_after_rad", "velocity_after_rad_s",
            "projected_joint_effort_nm", "computed_torque_estimate_nm", "clipped_torque_estimate_nm",
            "submitted_effort_nm", "submitted_position_rad",
        )}
        self.raw = torch.empty((self.steps, self.n, 2), device=self.env.device)
        self.done = torch.empty((self.steps, self.n), dtype=torch.bool, device=self.env.device)
        self.completed = 0
        self.sub = 0
        self.active = False
        self.saved = False
        dump_yaml(str(OUT / "effective_env.yaml"), self.env.cfg)
        expected = normalize(yaml.load((RUN / "params/env.yaml").read_text(), Loader=SavedLoader))
        effective = normalize(class_to_dict(self.env.cfg))
        # Strict equality for behavior-defining sections; relocated USD and new
        # framework metadata are recorded separately in the complete config.
        checks = {}
        for key in ("actions", "observations", "rewards", "terminations"):
            checks[key] = expected[key] == effective[key]
        for key in ("decimation", "episode_length_s"):
            checks[key] = expected[key] == effective[key]
        checks["physics_dt"] = expected["sim"]["dt"] == effective["sim"]["dt"]
        checks["actuators"] = expected["scene"]["robot"]["actuators"] == effective["scene"]["robot"]["actuators"]
        # DR-off intentionally removes randomization events; DR-on must match
        # the saved reset terms as well as the other task behavior.
        if namespace["args_cli"].dr:
            checks["events"] = expected["events"] == effective["events"]
        (OUT / "configuration_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        if not all(checks.values()):
            raise RuntimeError(f"Configuration mismatch; inspect effective_env.yaml: {checks}")
        if not hasattr(self.robot.root_view, "get_dof_projected_joint_forces"):
            raise RuntimeError("This PhysX version exposes no projected-effort sensor; estimates will not be substituted")
        scene = self.env.scene
        write_original, update_original = scene.write_data_to_sim, scene.update

        def write():
            write_original()
            if not self.active:
                return
            if self.sub >= self.decimation:
                raise RuntimeError("Unexpected extra physics substep")
            for key, value in (
                ("position_before_rad", self.robot.data.joint_pos),
                ("velocity_before_rad_s", self.robot.data.joint_vel),
                ("computed_torque_estimate_nm", self.robot.data.computed_torque),
                ("clipped_torque_estimate_nm", self.robot.data.applied_torque),
                ("submitted_effort_nm", self.robot._joint_effort_target_sim),
                ("submitted_position_rad", self.robot._joint_pos_target_sim),
            ):
                self.physics[key][self.completed, self.sub] = tensor(value)[:, self.joints]

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
            self.sub += 1

        scene.write_data_to_sim, scene.update = write, update

    def step(self, env, actions):
        if self.completed >= self.steps:
            raise RuntimeError("Unexpected extra policy step")
        self.raw[self.completed] = actions
        self.sub = 0
        self.active = True
        try:
            result = original_step(env, actions)
        finally:
            self.active = False
        if self.sub != self.decimation:
            raise RuntimeError(f"Expected {self.decimation} physics samples, got {self.sub}")
        self.done[self.completed] = result[2].bool()
        self.completed += 1
        if self.completed % 500 == 0:
            print(f"[EFFORT CAPTURE] {self.completed}/{self.steps} steps", flush=True)
        return result

    def save(self):
        if self.saved or not self.completed:
            return
        arrays = {key: value[:self.completed].cpu().numpy() for key, value in self.physics.items()}
        arrays.update(raw_action=self.raw[:self.completed].cpu().numpy(), done=self.done[:self.completed].cpu().numpy())
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError("Non-finite capture; refusing to produce apparently valid traces")
        np.savez_compressed(OUT / "motor_samples.npz", **arrays)
        metadata = {
            "invocation": invocation, "checkpoint_source": str(options.checkpoint), "checkpoint_sha256": sha(checkpoint),
            "motor_order": ["steering", "rear"], "seed": namespace["args_cli"].seed,
            "dr": namespace["args_cli"].dr, "pure_survival": namespace["args_cli"].pure_survival,
            "num_envs": self.n, "completed_policy_steps": self.completed,
            "policy_dt_s": self.env.step_dt, "physics_dt_s": self.env.physics_dt, "decimation": self.decimation,
            "measurement": "PhysX get_dof_projected_joint_forces after every physics step, before automatic reset",
            "measurement_scope": "solver-measured active joint-axis effort; not an independent motor-only shaft sensor",
            "estimates": "computed/clipped torque estimates retained separately and NOT used as actual measured effort",
        }
        (OUT / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.saved = True


original_make = namespace["gym"].make
original_step = namespace["RslRlVecEnvWrapper"].step
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


namespace["gym"].make = make
namespace["RslRlVecEnvWrapper"].step = lambda env, actions: recorder.step(env, actions)
try:
    namespace["main"]()
finally:
    if recorder is not None:
        recorder.save()
    namespace["gym"].make = original_make
    namespace["RslRlVecEnvWrapper"].step = original_step
    changed = {path: [digest, sha(path)] for path, digest in before.items() if digest != sha(path)}
    (OUT / "source_integrity.json").write_text(json.dumps({"checked_files": len(before), "changed_files": changed}, indent=2) + "\n")
    namespace["simulation_app"].close()
    if changed:
        raise RuntimeError(f"Source changed during evaluation: {changed}")
