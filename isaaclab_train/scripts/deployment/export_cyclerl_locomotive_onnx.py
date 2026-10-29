#!/usr/bin/env python3
"""Export CycleRL skrl riding policies to self-contained ONNX plus a locomotive manifest.

The trained tensor contract is preserved exactly: 8 observations in CycleRL's
order, skrl RunningStandardScaler (epsilon outside the square root, +/-5 clip),
[512, 256, 128, 64] ELU MLP and the two RAW Gaussian mean actions. Scaling and
clipping stay in deployment, as in Isaac's JointPosition/JointVelocity terms.

Each run's saved ``params/env.yaml`` is the source of the per-model contract
(action scales/clips, command ranges, timing, actuators). The script refuses a
run whose observation/action terms differ from the supported layout.

Run with the Isaac Lab interpreter (torch, skrl, onnx, onnxruntime), e.g.:
    ~/IsaacLab/_isaac_sim/python.sh scripts/deployment/export_cyclerl_locomotive_onnx.py \
        --output-dir ../deploy/source/trackstand_deploy/models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml
from skrl.resources.preprocessors.torch import RunningStandardScaler as SkrlRunningStandardScaler


CYCLERL_LOGS = os.path.expanduser(
    "~/bike_test/CycleRL/logs/skrl/cyclerl_logs/CycleRL-Trackstand-Cylinder-Paper-v0"
)
TASK = "CycleRL-Trackstand-Cylinder-Paper-v0"
OBSERVATION_NAMES = (
    "gravity_y", "ang_vel_x", "ang_vel_y", "ang_vel_z", "speed", "steer", "vel_cmd", "yaw_rate_cmd",
)
ACTION_NAMES = ("steering_position", "rear_wheel_velocity")
HIDDEN_LAYERS = (512, 256, 128, 64)

# Ranked by flat-ground MuJoCo balance success among runs with 100% Isaac success
# (cylinder_top5_rerun / cylinder_top5_limitfix comparison.md, paper suite).
MODELS = (
    (1, "1900_drive03", "cylinder_top5_rerun", "2026-10-19_16-30-17", 1.000, 1.000),
    (2, "1900", "cylinder_top5_rerun", "2026-10-19_19-04-32", 1.000, 0.882),
    (3, "1900_drive03", "cylinder_top5_limitfix", "2026-10-20_03-18-29", 1.000, 0.825),
    (4, "deploy_10deg", "cylinder_top5_limitfix", "2026-10-20_04-32-20", 1.000, 0.687),
    (5, "act_refine_lo", "cylinder_top5_rerun", "2026-10-19_15-30-18", 1.000, 0.647),
)


class _PermissiveLoader(yaml.SafeLoader):
    """Read Isaac's dumped env.yaml, whose python/tuple tags are plain data here."""


def _construct_any(loader, _suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_PermissiveLoader.add_multi_constructor("", _construct_any)


class Policy(nn.Module):
    """skrl standardizer + mean network; identical math to GaussianMixin act() means."""

    def __init__(self, mean: torch.Tensor, variance: torch.Tensor, epsilon: float, clip: float):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std_plus_eps", torch.sqrt(variance.float()) + epsilon)
        self.clip = float(clip)
        layers, width = [], len(OBSERVATION_NAMES)
        for hidden in HIDDEN_LAYERS:
            layers += [nn.Linear(width, hidden), nn.ELU()]
            width = hidden
        layers.append(nn.Linear(width, len(ACTION_NAMES)))
        self.net_container = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        scaled = torch.clamp((observation - self.mean) / self.std_plus_eps, -self.clip, self.clip)
        return self.net_container(scaled)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_from_env(env: dict) -> dict:
    """Extract and validate the per-run deployment contract from the saved env cfg."""
    terms = env["observations"]["policy"]
    imu, commands = terms["imu_state"], terms["commands"]
    if not imu["func"].endswith(":imu_orientation_and_velocity"):
        raise ValueError(f"unsupported imu observation {imu['func']}")
    if not commands["func"].endswith(":command_tracking_error") or commands["params"].get("use_vel_error"):
        raise ValueError("commands must be raw v* and yaw-rate command (use_vel_error=False)")
    for name in ("lpf_alpha", "dropout_prob"):
        if float(imu["params"].get(name, 0.0)) != 0.0:
            raise ValueError(f"imu_state {name} must be zero for this deployment")
    for term in (imu, commands):
        if term.get("scale") is not None or term.get("clip") is not None:
            raise ValueError("observation scale/clip is not supported")
    steering, drive = env["actions"]["steering"], env["actions"]["drive"]
    if not steering["class_type"].endswith(":JointPositionAction") or steering["joint_names"] != ["steering_joint"]:
        raise ValueError("steering action must be JointPositionAction on steering_joint")
    if not drive["class_type"].endswith(":JointVelocityAction") or drive["joint_names"] != ["rear_wheel_joint"]:
        raise ValueError("drive action must be JointVelocityAction on rear_wheel_joint")
    for action in (steering, drive):
        if float(action.get("offset", 0.0)) != 0.0:
            raise ValueError("action offsets are not supported")
    robot = env["scene"]["robot"]
    for joint in ("steering_joint", "rear_wheel_joint"):
        if float(robot["init_state"]["joint_pos"].get(".*", 0.0)) != 0.0 or float(
                robot["init_state"]["joint_vel"].get(".*", 0.0)) != 0.0:
            raise ValueError(f"default joint state for {joint} must be zero (use_default_offset)")
    velocity, heading = env["commands"]["velocity"], env["commands"]["steering"]
    actuators = robot["actuators"]
    steer_scale = float(steering["scale"])
    drive_scale = float(drive["scale"])
    return {
        "observation_names": list(OBSERVATION_NAMES),
        "action_names": list(ACTION_NAMES),
        "steering_action_scale_rad": steer_scale,
        "steering_action_clip_rad": [float(v) for v in steering["clip"]["steering_joint"]],
        "drive_action_scale_rad_s": drive_scale,
        "drive_action_clip_rad_s": [float(v) for v in drive["clip"]["rear_wheel_joint"]],
        # The env's drive scale is 3 m/s / wheel radius; speed obs uses the same radius.
        "wheel_radius_m": round(3.0 / drive_scale, 9),
        "vel_cmd_range_m_s": [float(v) for v in velocity["ranges"]["lin_vel_x"]],
        "yaw_rate_cmd_range_rad_s": [float(v) for v in heading["ranges"]["ang_vel_z"]],
        "heading_control_stiffness": float(heading["heading_control_stiffness"]),
        "command_resampling_time_s": [float(v) for v in velocity["resampling_time_range"]],
        "physics_dt_s": float(env["sim"]["dt"]),
        "decimation": int(env["decimation"]),
        "policy_rate_hz": round(1.0 / (float(env["sim"]["dt"]) * int(env["decimation"])), 9),
        "episode_length_s": float(env["episode_length_s"]),
        "lean_termination_rad": float(env["terminations"]["base_crash"]["params"]["limit_angle"]),
        # Episodes start rolling: the policy never trained from a standstill.
        "initial_root_velocity_x_m_s": [float(v) for v in
                                        env["events"]["initial_state"]["params"]["velocity_range"]["x"]],
        "initial_roll_rad": [float(v) for v in
                             env["events"]["initial_state"]["params"]["pose_range"]["roll"]],
        "training_actuators": {
            name: {key: actuators[name].get(key) for key in (
                "class_type", "stiffness", "damping", "effort_limit", "velocity_limit")}
            for name in ("steering", "drive")
        },
        "usd_path": robot["spawn"]["usd_path"],
    }


def reference_observations(contract: dict) -> np.ndarray:
    """Deterministic probes, including values beyond the scaler's +/-5 sigma clip."""
    vmin, vmax = contract["vel_cmd_range_m_s"]
    rng = np.random.default_rng(20261029)
    rows = [
        np.zeros(8),
        [0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 1.5, 0.0],
        [-0.17, 0.3, 0.0, 0.05, 1.0, 0.05, vmin, 0.2],
        [0.17, -0.3, 0.02, -0.05, 2.0, -0.05, vmax, -0.2],
        [-0.7, 4.0, 1.0, 2.0, 0.1, 0.34, vmax, 0.5],
        [0.7, -4.0, -1.0, -2.0, 3.0, -0.34, vmin, -0.5],
    ]
    rows += list(rng.normal(0.0, [0.1, 0.5, 0.1, 0.2, 0.6, 0.1, 0.6, 0.25], size=(10, 8))
                 + np.array([0, 0, 0, 0, 1.5, 0, 1.5, 0]))
    return np.asarray(rows, dtype=np.float32)


def export(rank, recipe, campaign, run, isaac_bsr, mujoco_bsr, output_dir, opset):
    run_dir = os.path.join(CYCLERL_LOGS, f"{run}_ppo_torch_bicycle_ppo")
    checkpoint = os.path.join(run_dir, "checkpoints", "best_agent.pt")
    with open(os.path.join(run_dir, "params", "env.yaml")) as stream:
        contract = contract_from_env(yaml.load(stream, Loader=_PermissiveLoader))
    with open(os.path.join(run_dir, "params", "agent.yaml")) as stream:
        agent = yaml.load(stream, Loader=_PermissiveLoader)
    policy_cfg = agent["models"]["policy"]
    if (policy_cfg["class"] != "GaussianMixin" or policy_cfg["network"][0]["layers"] != list(HIDDEN_LAYERS)
            or policy_cfg["network"][0]["activations"] != "elu" or policy_cfg.get("clip_mean_actions")
            or agent["agent"]["observation_preprocessor"] != "RunningStandardScaler"
            or agent["agent"].get("observation_preprocessor_kwargs")):
        raise ValueError(f"{run}: unsupported agent architecture/preprocessor")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    scaler_state = state["observation_preprocessor"]
    reference_scaler = SkrlRunningStandardScaler(size=len(OBSERVATION_NAMES), device="cpu")
    reference_scaler.load_state_dict(scaler_state)
    model = Policy(scaler_state["running_mean"], scaler_state["running_variance"],
                   reference_scaler.epsilon, reference_scaler.clip_threshold)
    weights = {k: v for k, v in state["policy"].items() if k != "log_std_parameter"}
    model.net_container.load_state_dict(
        {k.removeprefix("net_container."): v for k, v in weights.items()}, strict=True)
    model.eval()

    probes = reference_observations(contract)
    with torch.no_grad():
        inputs = torch.from_numpy(probes)
        expected = model.net_container(reference_scaler(inputs, train=False)).numpy()
        actual = model(inputs).numpy()
    if float(np.max(np.abs(expected - actual))) > 1e-6:
        raise RuntimeError(f"{run}: exported graph differs from skrl preprocessing")

    name = f"locomotive_{recipe}_{run}"
    onnx_path = os.path.join(output_dir, f"{name}.onnx")
    torch.onnx.export(
        model, torch.zeros(1, len(OBSERVATION_NAMES)), onnx_path, export_params=True, opset_version=opset,
        do_constant_folding=True, input_names=["observation"], output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}}, dynamo=False,
    )
    graph = onnx.load(onnx_path, load_external_data=True)
    onnx.save(graph, onnx_path, save_as_external_data=False)  # One file: the SHA covers the weights.
    if os.path.exists(onnx_path + ".data"):
        os.remove(onnx_path + ".data")
    onnx.checker.check_model(onnx.load(onnx_path))
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = session.run(["action"], {"observation": probes})[0]
    error = float(np.max(np.abs(ort_out - expected)))
    if error > 1e-5:
        raise RuntimeError(f"{run}: ONNX Runtime error {error:.2e}")

    return {
        "rank": rank,
        "recipe": recipe,
        "campaign": campaign,
        "task": TASK,
        "run": f"{run}_ppo_torch_bicycle_ppo",
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "onnx": os.path.basename(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "onnx_opset": opset,
        "weights_embedded": True,
        "observation_size": len(OBSERVATION_NAMES),
        "action_size": len(ACTION_NAMES),
        "distribution": "Gaussian mean (raw, unbounded; scaled and clipped by deployment)",
        "observation_preprocessor": {
            "type": "skrl RunningStandardScaler (embedded in ONNX)",
            "formula": "clamp((x - mean) / (sqrt(var) + epsilon), -clip, clip)",
            "epsilon": reference_scaler.epsilon, "clip": reference_scaler.clip_threshold,
        },
        "network": {"hidden_layers": list(HIDDEN_LAYERS), "activation": "elu"},
        "evaluation": {"isaac_paper_flat_bsr": isaac_bsr, "mujoco_paper_flat_bsr": mujoco_bsr,
                       "source": f"CycleRL logs/skrl/reward_sweeps/{campaign}/comparison.md"},
        "contract": contract,
        "reference_io": {"observation": probes.tolist(), "raw_action": expected.astype(float).tolist(),
                         "tolerance": 1e-5, "onnxruntime_max_error": error},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    entries = [export(*model, args.output_dir, args.opset) for model in MODELS]
    for entry in entries:
        if not math.isclose(entry["contract"]["policy_rate_hz"], 50.0):
            raise RuntimeError(f"{entry['run']}: policy rate is not 50 Hz")
    manifest = {
        "manifest_version": 1,
        "description": "CycleRL riding policies on the trackstand_rl_0916 cylinder plant; "
                       "original 8-in/2-out tensor contract, realigned to this bike's deployment.",
        "default_model": entries[0]["onnx"],
        "models": entries,
    }
    path = os.path.join(args.output_dir, "locomotive_policy_manifest.json")
    with open(path, "w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    for entry in entries:
        print(f"[{entry['rank']}] {entry['onnx']} sha256={entry['onnx_sha256'][:12]} "
              f"ort_err={entry['reference_io']['onnxruntime_max_error']:.1e}")
    print(f"manifest: {path}")


if __name__ == "__main__":
    main()
