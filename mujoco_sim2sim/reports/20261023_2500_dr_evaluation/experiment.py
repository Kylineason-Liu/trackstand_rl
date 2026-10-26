"""Read-only runtime evaluation and version-aware DR comparison.

The 800-Hz variants patch constants in this process only. Historical v0 loads
the pinned Git snapshot in memory; it never checks out or rewrites a model.
"""

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time
import types
from unittest.mock import patch

import mujoco
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "mujoco_sim2sim/src"))
from newbike_mujoco import contract as c

CHECKPOINT = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_11-18-29/model_2500.pt"
V0_ONNX = ROOT / "mujoco_sim2sim/policies/trackstand_rl_rsl_2026-09-22_19-45-06_model_91850.onnx"
SNAPSHOT = "e5f5d7f8d4652e83e649fbdfc2eea3f9dea8a465"


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False) + "\n")


def protected_hashes():
    paths = list((ROOT / "mujoco_sim2sim/src").rglob("*.py"))
    paths += list((ROOT / "mujoco_sim2sim/scripts").glob("*.py"))
    paths += list((ROOT / "mujoco_sim2sim/tests").glob("*.py"))
    paths += list((ROOT / "mujoco_sim2sim/models/trackstand_rl_0916_bike").rglob("*"))
    paths += list((ROOT / "isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand").rglob("*.py"))
    paths += list(CHECKPOINT.parent.joinpath("params").glob("*.yaml"))
    paths += [CHECKPOINT, V0_ONNX, ROOT / ".git/index", ROOT / "progress_record.md", ROOT / "progress_record.txt"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file() and "__pycache__" not in p.parts}


def preflight():
    import torch
    import yaml

    target = OUT / "protected_before.json"
    if target.exists():
        assert json.loads(target.read_text()) == protected_hashes(), "Protected files changed since initial capture"
    else:
        write_json(target, protected_hashes())
    cfg = yaml.load((CHECKPOINT.parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    agent = yaml.safe_load((CHECKPOINT.parent / "params/agent.yaml").read_text())
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert cfg["domain_randomization"] == "true"
    assert cfg["sim"]["dt"] == "0.0025" and cfg["decimation"] == "8"
    assert cfg["episode_length_s"] == "30.0"
    assert agent["actor"]["obs_normalization"] is False
    state = saved["actor_state_dict"]
    assert tuple(state["mlp.0.weight"].shape) == (32, 11)
    assert tuple(state["mlp.4.weight"].shape) == (4, 32)
    assert state["distribution._beta_policy_version"].item() == 1
    result = dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                  checkpoint_iteration=saved["iter"], mujoco_version=mujoco.__version__,
                  torch_version=torch.__version__, domain_randomization_training=True,
                  saved_observation=cfg["observations"]["policy"], saved_actions=cfg["actions"],
                  saved_actuators=cfg["scene"]["robot"]["actuators"], saved_agent=agent,
                  physics_hz=400, policy_hz=50, episode_horizon_s=30,
                  historical_snapshot=SNAPSHOT, protected_file_count=len(protected_hashes()))
    write_json(OUT / "configuration_checks.json", result)
    print("Preflight passed:", result["checkpoint_sha256"], flush=True)


def current(variant):
    from newbike_mujoco.env import BikeBalanceEnv
    from newbike_mujoco.viewer_envs import ViewerEnvGroup

    hz = 800 if variant.startswith("diagnostic800") else 400
    survival = variant.endswith("survival")
    folder = OUT / variant
    folder.mkdir(exist_ok=True)
    draws = []
    counters = np.zeros(64, dtype=int)
    original_init, original_reset = ViewerEnvGroup.__init__, BikeBalanceEnv.reset

    def group_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for index, env in enumerate(self.envs):
            env._evaluation_audit_id = index

    def reset(self):
        obs = original_reset(self)
        index = self._evaluation_audit_id
        counters[index] += 1
        draws.append(dict(env=index, episode=int(counters[index]), dr=self.dr,
                          kp=self.kp.copy(), kd=self.kd.copy(),
                          qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy()))
        return obs

    argv = [str(ROOT / "mujoco_sim2sim/scripts/run_deploy_eval.py"), "--pt", str(CHECKPOINT),
            "--mjcf", str(c.DEFAULT_MJCF), "--dr", "--num_envs", "64", "--num_steps", "3000",
            "--seed", "42", "--output_dir", str(folder)]
    if survival:
        argv.append("--pure_survival")
    print("Invocation:", " ".join(argv), flush=True)
    print(f"Experiment rates: physics/PD={hz}Hz, policy=50Hz; process-local change={hz != 400}", flush=True)
    with patch.object(c, "PHYSICS_DT", 1 / hz), patch.object(c, "DECIMATION", hz // 50), \
            patch.object(ViewerEnvGroup, "__init__", group_init), patch.object(BikeBalanceEnv, "reset", reset), \
            patch.object(sys, "argv", argv):
        namespace = runpy.run_path(argv[0], run_name="evaluation_module")
        namespace["main"]()
    write_json(folder / "dr_draws.json", draws)
    write_json(folder / "experiment.json", dict(variant=variant, command=argv, physics_hz=hz,
               pd_hz=hz, policy_hz=50, process_local_rate_override=hz != 400,
               dr=True, pure_survival=survival, seeds=list(range(42, 106)), num_steps=3000,
               recorded_resets=len(draws), completed=True))


def historical_module(package, short, path):
    source = subprocess.check_output(["git", "show", f"{SNAPSHOT}:{path}"], cwd=ROOT, text=True)
    mod = types.ModuleType(package + "." + short)
    sys.modules[mod.__name__] = mod
    exec(compile(source, f"<{SNAPSHOT}:{path}>", "exec"), mod.__dict__)
    return mod, source


def historical():
    package = "trackstand_historical_v0"
    pkg = types.ModuleType(package)
    pkg.__path__ = []
    sys.modules[package] = pkg
    env_module, env_source = historical_module(package, "env", "mujoco_sim2sim/src/newbike_mujoco/env.py")
    policy_module, policy_source = historical_module(package, "policy", "mujoco_sim2sim/src/newbike_mujoco/policy.py")
    group_module, group_source = historical_module(package, "viewer_envs", "mujoco_sim2sim/src/newbike_mujoco/viewer_envs.py")
    viewer_source = subprocess.check_output(["git", "show", f"{SNAPSHOT}:mujoco_sim2sim/scripts/run_deploy_viewer.py"],
                                            cwd=ROOT, text=True)
    tree = ast.parse(viewer_source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "augment_trackstand_model"
                or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_TRACKSTAND_JOINT_RENAMES"
                                                     for t in n.targets)]
    scope = {"mujoco": mujoco}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<historical augmentation>", "exec"), scope)
    spec = mujoco.MjSpec.from_file(str(c.DEFAULT_MJCF))
    scope["augment_trackstand_model"](spec, str(c.DEFAULT_MJCF.parent))
    model = spec.compile()
    first = env_module.BikeBalanceEnv(model=model, seed=42, enable_dr=True)
    group = group_module.ViewerEnvGroup(first, 64, 42)
    policy = policy_module.DeployedPolicy(onnx_path=str(V0_ONNX))
    obs = group.reset()
    records, resets, draws = [], [], []
    episode_ids = np.ones(64, dtype=int)
    raw, positions, velocities, observations, torques, flags = [], [], [], [], [], []
    folder = OUT / "historical_v0_91850_survival"
    folder.mkdir(exist_ok=True)

    def capture_draw(index):
        env = group.envs[index]
        draws.append(dict(env=index, episode=int(episode_ids[index]), dr=env.dr,
                          qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy()))

    for i in range(64):
        capture_draw(i)
    started = time.monotonic()
    for step in range(3000):
        actions = policy.act_batch(obs)
        observations.append(obs.copy())
        raw.append(actions.copy())
        obs, done, torque = group.step(actions)
        torques.append(torque.copy())
        positions.append([env.data.qpos[[env._steer_qadr, int(env.model.jnt_qposadr[env._rear_jid])]].copy()
                          for env in group.envs])
        velocities.append([env.data.qvel[[env._steer_dof, env._rear_dof]].copy() for env in group.envs])
        flags.append([[env.stats.timeout, bool(env.stats.death)] for env in group.envs])
        for i in np.flatnonzero(done):
            st = group.envs[i].stats
            records.append(dict(env=int(i), episode=int(episode_ids[i]), global_step=step + 1, **asdict(st)))
            resets.append((step + 1, int(i)))
            print(f"env {i}: episode {episode_ids[i]}: {st.length*.02:.2f}s {'timeout' if st.timeout else st.death}", flush=True)
            episode_ids[i] += 1
            obs[i] = group.envs[i].reset()
            capture_draw(i)
        if step % 500 == 499:
            print("Historical progress", step + 1, "steps", len(records), "completed episodes", flush=True)
    np.savez_compressed(folder / "evaluation.npz", observation=np.asarray(observations), raw_action=np.asarray(raw),
                        position_rad=np.asarray(positions), velocity_rad_s=np.asarray(velocities),
                        motor_torque_drive_steering_nm=np.asarray(torques), termination_flags=np.asarray(flags),
                        reset_events=np.asarray(resets), control_dt_s=.02, physics_dt_s=model.opt.timestep)
    write_json(folder / "metadata.json", dict(episodes=records, completed=True, num_envs=64,
               num_steps=3000, policy=str(V0_ONNX), policy_sha256=hashlib.sha256(V0_ONNX.read_bytes()).hexdigest(),
               physics_hz=1/model.opt.timestep, pd_hz=1/model.opt.timestep, policy_hz=50, dr=True,
               terms=["time_out", "fallen"], seeds=list(range(42, 106)), snapshot=SNAPSHOT,
               wall_time_s=time.monotonic()-started, source_sha256={
                   name: hashlib.sha256(s.encode()).hexdigest() for name, s in
                   (("env", env_source), ("policy", policy_source), ("viewer_envs", group_source), ("viewer", viewer_source))}))
    write_json(folder / "dr_draws.json", draws)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("preflight", "current400_full", "current400_survival",
                        "diagnostic800_full", "diagnostic800_survival", "historical_v0"))
    args = parser.parse_args()
    preflight()
    if args.variant == "historical_v0":
        historical()
    elif args.variant != "preflight":
        current(args.variant)
    assert protected_hashes() == json.loads((OUT / "protected_before.json").read_text())
    print("Protected production/model/checkpoint/ledger/index files unchanged", flush=True)


if __name__ == "__main__":
    main()
