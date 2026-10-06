"""Instrument the unchanged play_eval.py; capture both motors at each physics substep."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
ROOT = TRAIN.parent
RUN = TRAIN / "logs/rsl_rl/trackstand_rl_rsl/2026-09-22_19-45-06"
SCRIPT = TRAIN / "scripts/rsl_rl/play_eval.py"


def manifest():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    selected = [ROOT / p for p in paths if p and Path(p).suffix in {".py", ".yaml", ".usda", ".xml", ".md"}]
    selected.append(RUN / "model_91850.pt")
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in selected if p.is_file()}


before = manifest()
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(SCRIPT.parent))
sys.argv[0] = str(SCRIPT)
namespace = {"__name__": "__motor_audit_play_eval__", "__file__": str(SCRIPT)}
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)

import numpy as np
import torch
import yaml
import warp as wp
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
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, str):
        if value.endswith("/tasks/manager_based/trackstand/assets/trackstand_rl.usda"):
            return "<relocated-trackstand-usd>"
        if value == "CycleRL.tasks.manager_based.cyclerl.mdp.stationary.noise:per_channel_gaussian_noise":
            return "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand.mdp.stationary.noise:per_channel_gaussian_noise"
    return value


def differences(expected, actual, path=""):
    result = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            field = f"{path}.{key}" if path else key
            if key not in expected or key not in actual:
                result.append({"field": field, "saved": expected.get(key, "<MISSING>"), "effective": actual.get(key, "<MISSING>")})
            else:
                result.extend(differences(expected[key], actual[key], field))
    elif expected != actual:
        result.append({"field": path, "saved": expected, "effective": actual})
    return result


audits = {}


def audit(kind, cfg):
    expected = normalize(yaml.load((RUN / f"params/{kind}.yaml").read_text(), Loader=SavedLoader))
    effective = normalize(class_to_dict(cfg))
    allowed = {"env": {"scene.num_envs", "scene.terrain.num_envs", "log_dir", "domain_randomization"},
               "agent": {"load_checkpoint"}}[kind]
    diff = differences(expected, effective)
    unexpected = [item for item in diff if item["field"] not in allowed]
    audits[kind] = {"allowed_differences": [item for item in diff if item["field"] in allowed],
                    "unexpected_differences": unexpected,
                    "normalized_relocations": ["robot USD path", "noise module path"]}
    dump_yaml(str(OUT / f"effective_{kind}.yaml"), cfg)
    (OUT / "configuration_audit.json").write_text(json.dumps(audits, indent=2, default=str))
    if unexpected:
        raise RuntimeError(f"Saved {kind} configuration mismatch: {json.dumps(unexpected, default=str)}")
    print(f"[MOTOR AUDIT] {kind} matches saved 09221945 configuration", flush=True)


def tensor(value):
    if hasattr(value, "torch"):
        return value.torch
    if isinstance(value, wp.array):
        return wp.to_torch(value)
    return value


class MotorRecorder:
    def __init__(self, env):
        self.env = env
        self.robot = env.unwrapped.scene["robot"]
        self.joints = [self.robot.find_joints(name)[0][0] for name in ("steering_joint", "rear_wheel_joint")]
        self.steps = namespace["args_cli"].num_steps
        self.n = env.unwrapped.num_envs
        self.decimation = env.unwrapped.cfg.decimation
        self.device = env.unwrapped.device
        self.t = -1
        self.sub = 0
        self.active = False
        self.completed = 0
        shape = (self.steps, self.decimation, self.n, 2)
        self.physics = {name: torch.zeros(shape, device=self.device) for name in (
            "position_before_rad", "velocity_before_rad_s", "position_after_rad", "velocity_after_rad_s",
            "computed_torque_nm", "applied_torque_estimate_nm", "physx_effort_target_nm")}
        self.policy = {name: torch.zeros((self.steps, self.n, 2), device=self.device) for name in (
            "raw_action", "processed_target", "position_before_policy_rad", "velocity_before_policy_rad_s",
            "stiffness_nm_per_rad", "damping_nm_s_per_rad")}
        self.done = torch.zeros((self.steps, self.n), device=self.device, dtype=torch.bool)
        self.episode_id = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.episodes = torch.zeros((self.steps, self.n), device=self.device, dtype=torch.long)
        self.limits = {name: tensor(getattr(self.robot.data, name)).detach().cpu().numpy()[:, self.joints]
                       for name in ("joint_pos_limits", "joint_vel_limits", "joint_effort_limits")}
        self.metadata = {"motor_order": ["steering", "drive"], "num_envs": self.n, "policy_steps": self.steps,
                         "seed": namespace["args_cli"].seed, "dr": namespace["args_cli"].dr,
                         "policy_dt_s": env.unwrapped.step_dt, "physics_dt_s": env.unwrapped.physics_dt,
                         "decimation": self.decimation, "torque_semantics": "computed/applied are ImplicitActuator estimates, not measured motor torques",
                         "physx_effort_target_semantics": "feed-forward effort buffer submitted to PhysX; excludes implicit drive PD contribution",
                         "capture": "physics pre/post states captured before automatic reset; estimates captured after actuator.compute and before physics integration",
                         "checkpoint_sha256": hashlib.sha256(Path(namespace["args_cli"].checkpoint).read_bytes()).hexdigest()}
        scene = env.unwrapped.scene
        original_write = scene.write_data_to_sim
        original_update = scene.update

        def write():
            original_write()
            if not self.active or self.sub >= self.decimation:
                return
            target = torch.stack((env.unwrapped.action_manager.get_term("steering").processed_actions[:, 0],
                                  env.unwrapped.action_manager.get_term("drive").processed_actions[:, 0]), dim=1)
            if self.sub == 0:
                self.policy["processed_target"][self.t] = target
            for key, value in (
                ("position_before_rad", self.robot.data.joint_pos),
                ("velocity_before_rad_s", self.robot.data.joint_vel),
                ("computed_torque_nm", self.robot.data.computed_torque),
                ("applied_torque_estimate_nm", self.robot.data.applied_torque),
                ("physx_effort_target_nm", self.robot._joint_effort_target_sim),
            ):
                self.physics[key][self.t, self.sub] = tensor(value)[:, self.joints]

        def update(*args, **kwargs):
            original_update(*args, **kwargs)
            if not self.active or self.sub >= self.decimation:
                return
            self.physics["position_after_rad"][self.t, self.sub] = tensor(self.robot.data.joint_pos)[:, self.joints]
            self.physics["velocity_after_rad_s"][self.t, self.sub] = tensor(self.robot.data.joint_vel)[:, self.joints]
            self.sub += 1

        scene.write_data_to_sim = write
        scene.update = update
        original_close = env.close

        def close():
            if self.completed:
                self.save()
            original_close()

        env.close = close

    def step(self, original, env, actions):
        self.t += 1
        self.sub = 0
        self.active = True
        self.policy["raw_action"][self.t] = actions
        self.episodes[self.t] = self.episode_id
        for name, value in (("position_before_policy_rad", self.robot.data.joint_pos),
                            ("velocity_before_policy_rad_s", self.robot.data.joint_vel),
                            ("stiffness_nm_per_rad", self.robot.data.joint_stiffness),
                            ("damping_nm_s_per_rad", self.robot.data.joint_damping)):
            self.policy[name][self.t] = tensor(value)[:, self.joints]
        result = original(env, actions)
        self.active = False
        if self.sub != self.decimation:
            raise RuntimeError(f"Captured {self.sub} physics steps, expected {self.decimation}")
        self.done[self.t] = result[2].bool()
        self.episode_id += result[2].long()
        self.completed = self.t + 1
        if self.completed % 500 == 0:
            print(f"[MOTOR AUDIT] captured {self.completed}/{self.steps} policy steps", flush=True)
        return result

    def save(self):
        data = {name: value[:self.completed].detach().cpu().numpy() for name, value in {**self.physics, **self.policy}.items()}
        data.update(done=self.done[:self.completed].cpu().numpy(), episode_id=self.episodes[:self.completed].cpu().numpy())
        data.update(self.limits)
        np.savez_compressed(OUT / "motor_limit_samples.npz", **data)
        self.metadata["captured_policy_steps"] = self.completed
        self.metadata["captured_physics_env_samples"] = self.completed * self.decimation * self.n
        (OUT / "experiment.json").write_text(json.dumps(self.metadata, indent=2))
        print("[MOTOR AUDIT] full motor traces saved", flush=True)


original_make = namespace["gym"].make
original_runner = namespace["OnPolicyRunner"]
original_step = namespace["RslRlVecEnvWrapper"].step
recorder = None


def audited_make(*args, **kwargs):
    env = original_make(*args, **kwargs)
    audit("env", kwargs["cfg"])
    global recorder
    recorder = MotorRecorder(env)
    return env


def audited_runner(env, config, *args, **kwargs):
    audit("agent", config)
    return original_runner(env, config, *args, **kwargs)


def recorded_step(env, actions):
    return recorder.step(original_step, env, actions)


namespace["gym"].make = audited_make
namespace["OnPolicyRunner"] = audited_runner
namespace["RslRlVecEnvWrapper"].step = recorded_step
try:
    namespace["main"]()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
finally:
    namespace["gym"].make = original_make
    namespace["OnPolicyRunner"] = original_runner
    namespace["RslRlVecEnvWrapper"].step = original_step
    after = manifest()
    changed = {p: [before.get(p), after.get(p)] for p in before if before.get(p) != after.get(p)}
    (OUT / "source_integrity.json").write_text(json.dumps({"checked_files": len(before), "changed_files": changed}, indent=2))
    if changed:
        raise RuntimeError(f"Unexpected source changes: {changed}")
    namespace["simulation_app"].close()
