"""Record one unchanged v2 episode for steering-position/rear-velocity replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
ROOT = TRAIN.parent
RUN = TRAIN / "logs/rsl_rl/trackstand_rl_v2_rsl/2026-11-02_21-49-32"
SOURCE = RUN / "model_26600.pt"
TASK = "Trackstand-Balance-v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(paths):
    return {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in paths}


def prepare(args):
    destination = OUT / SOURCE.name
    if destination.exists():
        if sha(destination) != sha(SOURCE):
            raise ValueError("Staged checkpoint differs from original")
    else:
        shutil.copy2(SOURCE, destination)
    inputs = [p for top in (TRAIN / "source", TRAIN / "scripts")
              for p in sorted(top.rglob("*")) if p.is_file() and p.suffix in (".py", ".yaml", ".usda")]
    original_files = [p for p in sorted(RUN.rglob("*")) if p.is_file()]
    before = {"source": manifest(inputs), "original_run": manifest(original_files)}
    (OUT / "input_manifest_before.json").write_text(json.dumps(before, indent=2) + "\n")
    information = dict(task=TASK, checkpoint_source=str(SOURCE), checkpoint_copy=str(destination),
                       checkpoint_sha256=sha(SOURCE), recorder_sha256=sha(__file__),
                       repository_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                                 text=True).strip(),
                       repository_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                                                 text=True),
                       arguments=vars(args), invocation=sys.argv, source_files=len(inputs),
                       original_run_files=len(original_files), capture_completed=False)
    (OUT / "preflight.json").write_text(json.dumps(information, indent=2) + "\n")
    return destination, before, information


def normalize(value):
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    return value


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            name = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                result.append(dict(path=name, saved=left.get(key), effective=right.get(key)))
            else:
                result.extend(differences(left[key], right[key], name))
        return result
    return [] if left == right else [dict(path=path, saved=left, effective=right)]


def check_config(kind, value, allowed):
    import yaml

    class Loader(yaml.FullLoader):
        pass

    Loader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice",
                           lambda loader, node: slice(*loader.construct_sequence(node)))
    expected = normalize(yaml.load((RUN / f"params/{kind}.yaml").read_text(), Loader=Loader))
    effective = normalize(value)
    delta = differences(expected, effective)
    (OUT / f"effective_{kind}.json").write_text(json.dumps(effective, indent=2) + "\n")
    (OUT / f"{kind}_configuration_differences.json").write_text(json.dumps(delta, indent=2) + "\n")
    unexpected = [item for item in delta if item["path"] not in allowed]
    if unexpected:
        raise RuntimeError(f"Unexpected {kind} configuration mismatch: {unexpected}")
    print(f"[CAPTURE] {kind} configuration matches training; allowed differences: {delta}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=30.)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 1 or not 0 < args.duration <= 30 or abs(args.duration * 50 - round(args.duration * 50)) > 1e-8:
        parser.error("positive num-envs and duration on the 50-Hz grid up to 30 seconds required")
    checkpoint, before, metadata = prepare(args)
    if args.prepare_only:
        print(json.dumps(metadata, indent=2))
        return 0
    if (OUT / "motor_samples.npz").exists():
        raise FileExistsError("Capture already exists; do not overwrite a completed experiment")
    sys.path.insert(0, str(TRAIN))
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=args.device).app
    env = None
    started = time.monotonic()
    try:
        import importlib.metadata
        import platform
        import gymnasium as gym
        import numpy as np
        import torch
        import warp as wp
        from isaaclab.actuators import IdealPDActuator
        from isaaclab.utils.io import dump_yaml
        import isaaclab.utils.math as math_utils
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab_tasks.utils import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner
        __import__("source.trackstand_rl.trackstand_rl.tasks")
        from scripts.domain_randomization import configure_domain_randomization

        def tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            return value.torch if hasattr(value, "torch") else wp.to_torch(value)

        cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
        agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        configure_domain_randomization(cfg, False)
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
        cfg.log_dir = str(OUT)
        agent.seed, agent.device, agent.max_iterations = args.seed, args.device, 100000
        agent.load_checkpoint = str(checkpoint)
        agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version("rsl-rl-lib"))
        raw = gym.make(TASK, cfg=cfg)
        check_config("env", raw.unwrapped.cfg.to_dict(), {"scene.num_envs", "scene.terrain.num_envs", "log_dir"})
        env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        check_config("agent", agent.to_dict(), {"load_checkpoint"})
        dump_yaml(str(OUT / "effective_env.yaml"), cfg)
        dump_yaml(str(OUT / "effective_agent.yaml"), agent)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped
        robot = base.scene["robot"]
        joints = [robot.find_joints(name)[0][0] for name in ("steering_joint", "rear_wheel_joint")]
        steering, drive = (base.action_manager.get_term(name) for name in ("steering", "drive"))
        actuators = [robot.actuators[name] for name in ("steering", "drive")]
        assert all(type(act) is IdealPDActuator for act in actuators)
        assert base.step_dt == .02 and abs(base.physics_dt - 1 / 1200) < 1e-12 and cfg.decimation == 24
        assert drive._integral_decimation == 6 and drive.cfg.integral_rate_hz == 200
        steps, decimation = round(args.duration / base.step_dt), cfg.decimation
        shape = (steps, decimation, 2)
        physics = {name: torch.empty(shape, device=base.device) for name in (
            "position_before_rad", "velocity_before_rad_s", "position_after_rad", "velocity_after_rad_s",
            "computed_motor_torque_nm", "applied_motor_torque_nm", "submitted_motor_torque_nm",
            "solver_joint_axis_effort_nm", "held_target", "controller_kp", "controller_kd",
        )}
        scalar = {name: torch.empty((steps, decimation), device=base.device) for name in (
            "drive_integral_torque_nm", "drive_integrated_error_rad", "drive_previous_error_rad_s",
            "drive_integral_initialized", "drive_integral_limited_updates", "wheel_net_angle_rad",
        )}
        body = torch.empty((steps, decimation, 3), device=base.device)
        actions_record = torch.empty((steps, 2), device=base.device)
        observations_record = torch.empty((steps, 11), device=base.device)
        done_record = torch.empty((steps, args.num_envs), device=base.device, dtype=torch.bool)
        term_names = list(base.termination_manager.active_terms)
        flags_record = torch.empty((steps, args.num_envs, len(term_names)), device=base.device, dtype=torch.bool)
        rewards_record = torch.empty((steps, args.num_envs), device=base.device)
        assert set(term_names) == {"time_out", "fallen", "diverged", "drifted", "yaw_deviation", "wheel_net"}
        state = dict(active=False, completed=0, sub=0)
        original_write, original_update = base.scene.write_data_to_sim, base.scene.update

        def write(*positional, **keywords):
            result = original_write(*positional, **keywords)
            if not state["active"]:
                return result
            t, s = state["completed"], state["sub"]
            assert s < decimation
            for name, value in (
                ("position_before_rad", robot.data.joint_pos),
                ("velocity_before_rad_s", robot.data.joint_vel),
                ("computed_motor_torque_nm", robot.data.computed_torque),
                ("applied_motor_torque_nm", robot.data.applied_torque),
                ("submitted_motor_torque_nm", robot._joint_effort_target_sim),
            ):
                physics[name][t, s] = tensor(value)[0, joints]
            physics["held_target"][t, s] = torch.stack((steering.processed_actions[0, 0], drive.processed_actions[0, 0]))
            for name, attribute in (("controller_kp", "stiffness"), ("controller_kd", "damping")):
                physics[name][t, s] = torch.cat([getattr(act, attribute)[0] for act in actuators])
            for name, attribute in (
                ("drive_integral_torque_nm", "torque_nm"), ("drive_integrated_error_rad", "error_rad"),
                ("drive_previous_error_rad_s", "previous_error"),
                ("drive_integral_initialized", "initialized"),
            ):
                scalar[name][t, s] = getattr(drive.integral, attribute)[0, 0]
            scalar["drive_integral_limited_updates"][t, s] = drive.integral.limited_updates[0]
            return result

        def update(*positional, **keywords):
            result = original_update(*positional, **keywords)
            if not state["active"]:
                return result
            t, s = state["completed"], state["sub"]
            for name, value in (
                ("position_after_rad", robot.data.joint_pos), ("velocity_after_rad_s", robot.data.joint_vel),
                ("solver_joint_axis_effort_nm", robot.root_view.get_dof_projected_joint_forces()),
            ):
                physics[name][t, s] = tensor(value)[0, joints]
            quat = tensor(robot.data.root_quat_w)
            roll, _, yaw = math_utils.euler_xyz_from_quat(quat)
            relative = torch.atan2(torch.sin(yaw - base.spawn_yaw), torch.cos(yaw - base.spawn_yaw))
            drift = torch.norm(tensor(robot.data.root_pos_w)[:, :2] - base.scene.env_origins[:, :2], dim=-1)
            body[t, s] = torch.stack((roll[0], drift[0], relative[0]))
            scalar["wheel_net_angle_rad"][t, s] = base.wheel_net_angle[0]
            state["sub"] += 1
            return result

        base.scene.write_data_to_sim, base.scene.update = write, update
        observations = env.get_observations()
        try:
            with torch.inference_mode():
                for step in range(steps):
                    observations_record[step] = observations["policy"][0]
                    actions = policy(observations)
                    actions_record[step] = actions[0]
                    state["sub"], state["active"] = 0, True
                    try:
                        observations, rewards, done, extras = env.step(actions)
                    finally:
                        state["active"] = False
                    assert state["sub"] == decimation, "Unexpected extra/missing physics capture intervals"
                    done_record[step] = done.bool().reshape(-1)
                    rewards_record[step] = rewards.reshape(-1)
                    flags_record[step] = base.termination_manager._last_episode_dones
                    state["completed"] += 1
                    if (step + 1) % 250 == 0:
                        print(f"[CAPTURE] env0 first episode: {(step + 1) * base.step_dt:.2f} s", flush=True)
                    if bool(done[0]):
                        break
        finally:
            base.scene.write_data_to_sim, base.scene.update = original_write, original_update
        completed = state["completed"]
        arrays = {name: value[:completed].cpu().numpy() for name, value in physics.items()}
        arrays.update({name: value[:completed].cpu().numpy() for name, value in scalar.items()})
        arrays.update(raw_action=actions_record[:completed].cpu().numpy(),
                      observation=observations_record[:completed].cpu().numpy(),
                      done=done_record[:completed].cpu().numpy(),
                      termination_flags=flags_record[:completed].cpu().numpy(),
                      reward=rewards_record[:completed].cpu().numpy(), body_roll_drift_yaw=body[:completed].cpu().numpy())
        assert all(np.isfinite(value).all() for value in arrays.values())
        np.savez_compressed(OUT / "motor_samples.npz", **arrays)
        duration = completed * base.step_dt
        metadata.update(capture_completed=True, num_envs=args.num_envs, recorded_environment=0,
                        recorded_episode=1, completed_policy_steps=completed, captured_duration_s=duration,
                        physics_samples=completed * decimation, policy_hz=50., physics_hz=1200., integral_hz=200.,
                        policy_dt_s=base.step_dt, physics_dt_s=base.physics_dt, decimation=decimation,
                        dr=False, observation_white_noise=True, termination_names=term_names,
                        motor_order=["steering", "rear"], target_units=["rad", "rad/s"],
                        target_meaning=["held steering position after clip and 50-Hz slew",
                                        "held rear velocity after scale and clip"],
                        body_columns=["roll_rad", "drift_m", "relative_yaw_rad"],
                        timing="Commands at physics-step start, response at step end before automatic reset",
                        raw_archive_sha256=sha(OUT / "motor_samples.npz"), wall_time_s=time.monotonic() - started,
                        versions={"python": platform.python_version(), "torch": torch.__version__,
                                  "isaaclab": importlib.metadata.version("isaaclab"),
                                  "rsl_rl": importlib.metadata.version("rsl-rl-lib")})
        (OUT / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # Explicitly export actual state and policy targets as separate levels.
        position = arrays["position_after_rad"].reshape(-1, 2)
        initial = arrays["position_before_rad"][0, 0]
        position = np.unwrap(np.vstack((initial, position)), period=4 * np.pi, axis=0) - initial
        velocity = np.vstack((arrays["velocity_before_rad_s"][0, 0], arrays["velocity_after_rad_s"].reshape(-1, 2)))
        with (OUT / "joint_velocity_reference_1200hz.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time_s", "steering_velocity_rad_s", "drive_velocity_rad_s",
                             "steering_position_rad", "drive_position_rad"))
            for i, (v, q) in enumerate(zip(velocity, position)):
                writer.writerow((i * base.physics_dt, *v, *q))
        held = arrays["held_target"][:, 0]
        with (OUT / "policy_targets_50hz.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time_s", "raw_steering_action", "raw_drive_action", "scaled_steering_position_rad",
                             "held_steering_position_target_rad", "drive_velocity_target_rad_s"))
            for i, (a, h) in enumerate(zip(arrays["raw_action"], held)):
                writer.writerow((i * base.step_dt, *a, float(a[0] * np.float32(1.57)), *h))
        fields = ["command_time_s", "response_time_s", "held_steering_position_target_rad", "drive_velocity_target_rad_s"]
        for suffix in ("position_before_rad", "velocity_before_rad_s", "position_after_rad", "velocity_after_rad_s",
                       "computed_motor_torque_nm", "applied_motor_torque_nm", "submitted_motor_torque_nm",
                       "solver_joint_axis_effort_nm"):
            fields.extend(("steering_" + suffix, "drive_" + suffix))
        fields.append("drive_integral_torque_nm")
        with (OUT / "motor_physics_1200hz.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for i in range(completed * decimation):
                t, s = divmod(i, decimation)
                row = [i * base.physics_dt, (i + 1) * base.physics_dt, *arrays["held_target"][t, s]]
                for name in ("position_before_rad", "velocity_before_rad_s", "position_after_rad", "velocity_after_rad_s",
                             "computed_motor_torque_nm", "applied_motor_torque_nm", "submitted_motor_torque_nm",
                             "solver_joint_axis_effort_nm"):
                    row.extend(arrays[name][t, s])
                writer.writerow([*row, arrays["drive_integral_torque_nm"][t, s]])
        after = {group: manifest([ROOT / name for name in entries]) for group, entries in before.items()}
        assert after == before, "Original training/source inputs changed during capture"
        (OUT / "input_manifest_after.json").write_text(json.dumps(after, indent=2) + "\n")
        status = dict(completed=True, duration_s=duration, policy_steps=completed,
                      physics_samples=completed * decimation, original_inputs_preserved=True)
        (OUT / "execution_status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2), flush=True)
        return 0
    except BaseException:
        (OUT / "execution_status.json").write_text(json.dumps(dict(completed=False, traceback=traceback.format_exc()),
                                                              indent=2) + "\n")
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
