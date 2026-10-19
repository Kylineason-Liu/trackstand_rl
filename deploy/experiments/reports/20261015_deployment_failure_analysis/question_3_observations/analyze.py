"""Offline observation-contract audit and fixed-input counterfactuals for model 91850.

Writes only new report artifacts. Simulations use unchanged runtime classes;
synthetic parity probes are explicitly separated from observed trajectories.
"""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import onnxruntime as ort
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "isaaclab_train").is_dir())
HERE = Path(__file__).resolve().parent
MU = ROOT / "mujoco_sim2sim"
BIKE = ROOT / "deploy/experiments/reports/deployment_20261012_180607"
TRAIN = ROOT / "isaaclab_train"
RUN = TRAIN / "logs/rsl_rl/trackstand_rl_rsl/2026-09-22_19-45-06"
ONNX = MU / "policies/trackstand_rl_rsl_2026-09-22_19-45-06_model_91850.onnx"
FEATURES = ["gravity_y", "gravity_z", "omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s",
            "vx_m_s", "vy_m_s", "steer_pos_rad", "rear_pos_rad", "steer_vel_rad_s",
            "rear_vel_rad_s", "sin_steer", "cos_steer"]
SCALES = np.array([1.57, 6.28])
LIMITS = np.array([np.pi / 2, 6.28])
torch.set_num_threads(1)
sys.path[:0] = [str(MU / "src"), str(ROOT / "deploy/source/trackstand_deploy")]
from newbike_mujoco.env import BikeBalanceEnv, _rot_matrix
from trackstand_deploy.contracts import build_observation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def actor_models():
    sd = torch.load(RUN / "model_91850.pt", map_location="cpu", weights_only=False)["actor_state_dict"]
    actor = torch.nn.Sequential(torch.nn.Linear(13, 32), torch.nn.ELU(), torch.nn.Linear(32, 32),
                                torch.nn.ELU(), torch.nn.Linear(32, 2))
    actor.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")}, strict=True)
    actor.eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ONNX), options, providers=["CPUExecutionProvider"])

    def infer(x):
        x = np.asarray(x, dtype=np.float32)
        return np.concatenate([session.run(["action"], {"observation": x[i:i + 4096]})[0]
                               for i in range(0, len(x), 4096)])

    return actor, infer


def make_model():
    viewer = module("observation_audit_viewer", MU / "scripts/run_deploy_viewer.py")
    xml = MU / "models/trackstand_rl_0916_bike/trackstand_rl_0916.xml"
    spec = mujoco.MjSpec.from_file(str(xml))
    viewer.augment_trackstand_model(spec, str(xml.parent))
    return spec.compile()


def clean_mujoco(env):
    d = env.data
    adr = env._base_qveladr
    rotation = _rot_matrix(env._base_quat())
    gravity = rotation.T @ np.array([0., 0., -1.])
    vel = rotation.T @ d.qvel[adr:adr + 3]
    steer = d.qpos[env._steer_qadr]
    rear = d.qpos[env.model.jnt_qposadr[env._rear_jid]]
    return np.array([gravity[1], gravity[2], *d.qvel[adr + 3:adr + 6], *vel[:2],
                     steer, (rear + np.pi) % (2 * np.pi) - np.pi,
                     d.qvel[env._steer_dof], d.qvel[env._rear_dof], np.sin(steer), np.cos(steer)], np.float32)


def collect_mujoco(out, infer, num_envs, steps, estimator=False, rear_period_4pi=False):
    if out.exists():
        raise FileExistsError(out)
    m = make_model()
    result = {}
    for dr in (False, True):
        label = "dr" if dr else "no_dr"
        envs = [BikeBalanceEnv(model=m, seed=42 + i, enable_dr=dr) for i in range(num_envs)]
        obs = np.stack([env.reset() for env in envs])
        observations = np.empty((steps, num_envs, 13), np.float32)
        clean = np.empty_like(observations)
        raw = np.empty((steps, num_envs, 2), np.float32)
        done = np.empty((steps, num_envs), bool)
        unwrapped = np.empty((steps, num_envs), np.float64)
        lengths = np.zeros(num_envs, int)
        episodes = []
        for t in range(steps):
            if estimator:
                obs[:, 5] = .07 * obs[:, 10]
                obs[:, 6] = 0
            if rear_period_4pi:
                for i, env in enumerate(envs):
                    rear = env.data.qpos[env.model.jnt_qposadr[env._rear_jid]]
                    obs[i, 8] = (rear + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
            observations[t] = obs
            for i, env in enumerate(envs):
                clean[t, i] = clean_mujoco(env)
                unwrapped[t, i] = env.data.qpos[env.model.jnt_qposadr[env._rear_jid]]
            raw[t] = infer(obs)
            for i, env in enumerate(envs):
                obs[i], done[t, i] = env.step(raw[t, i])
                lengths[i] += 1
                if done[t, i]:
                    episodes.append({"env": i, "seed": 42 + i, "end_step": t, "steps": int(lengths[i]),
                                     "result": "time_out" if env.stats.timeout else env.stats.death})
                    lengths[i] = 0
                    obs[i] = env.reset()
            if (t + 1) % 500 == 0:
                print(f"[MU OBS] estimator={estimator} {label} {t + 1}/{steps}", flush=True)
        for name, array in (("observation", observations), ("clean_observation", clean),
                            ("raw_action", raw), ("done", done), ("unwrapped_rear_rad", unwrapped)):
            result[label + "_" + name] = array
        result[label + "_episodes_json"] = np.array(json.dumps(episodes))
    np.savez_compressed(out, **result)


def parity_probes(infer):
    """Run the actual Isaac observation function on mock state, not a rewritten formula."""
    math_utils = module("installed_isaac_math_for_audit", Path("/home/lmbike/IsaacLab/source/isaaclab/isaaclab/utils/math.py"))
    source = TRAIN / "source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/mdp/stationary/observations.py"
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("balance_imu_state", "_accumulate_motor_speeds")]
    ns = {"torch": torch, "math_utils": math_utils, "ManagerBasedRLEnv": object,
          "SceneEntityCfg": lambda name: SimpleNamespace(name=name)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), ns)
    rng = np.random.default_rng(42)
    n = 1000
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    lv, av = rng.uniform(-2, 2, (n, 3)), rng.uniform(-3, 3, (n, 3))
    pos, vel = rng.uniform(-1.5, 1.5, (n, 2)), rng.uniform(-20, 20, (n, 2))
    pos[:, 1] *= 2
    tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    data = SimpleNamespace(root_quat_w=SimpleNamespace(torch=tensor(q)),
                           root_link_lin_vel_w=SimpleNamespace(torch=tensor(lv)),
                           root_link_ang_vel_w=SimpleNamespace(torch=tensor(av)),
                           joint_pos=tensor(pos), joint_vel=tensor(vel))
    asset = SimpleNamespace(data=data, find_joints=lambda name: ([0 if name == "steer" else 1], [name]))
    mock = SimpleNamespace(scene={"robot": asset}, device="cpu", num_envs=n)
    isaac = ns["balance_imu_state"](mock, steer_joint_name="steer", drive_joint_name="rear").numpy()
    env = BikeBalanceEnv(model=make_model(), seed=42)
    env.OBS_NOISE_STD = (0.,) * 5
    mu, dep, angular_errors = [], [], []
    for i in range(n):
        env.data.qpos[3:7] = q[i, [3, 0, 1, 2]]
        rotation = _rot_matrix(env.data.qpos[3:7])
        env.data.qvel[:3] = lv[i]
        env.data.qvel[3:6] = rotation.T @ av[i]
        env.data.qpos[env._steer_qadr] = pos[i, 0]
        env.data.qpos[env.model.jnt_qposadr[env._rear_jid]] = pos[i, 1]
        env.data.qvel[[env._steer_dof, env._rear_dof]] = vel[i]
        mu.append(env.observe())
        dep.append(build_observation(q[i], rotation.T @ av[i], *pos[i], *vel[i], .07))
        mujoco.mj_forward(env.model, env.data)
        spatial = np.empty(6)
        mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY,
                                env.model.body("model_root").id, spatial, 1)
        angular_errors.append(np.max(np.abs(spatial[:3] - env.data.qvel[3:6])))
    mu, dep = np.array(mu), np.array(dep)
    common = [i for i in range(13) if i not in (5, 6)]
    return {"synthetic_states": n, "not_physical_runs": True,
            "isaac_vs_mujoco_max_abs_per_channel": np.abs(isaac - mu).max(axis=0).tolist(),
            "isaac_vs_deploy_max_abs_per_channel": np.abs(isaac - dep).max(axis=0).tolist(),
            "deploy_non_velocity_channel_max_error": float(np.abs(isaac[:, common] - dep[:, common]).max()),
            "isaac_vs_mujoco_raw_max_error": np.abs(infer(isaac) - infer(mu)).max(axis=0).tolist(),
            "mujoco_body_angular_velocity_api_error": float(max(angular_errors))}


def starts_and_ages(done):
    starts = np.zeros_like(done, bool)
    starts[0] = True
    starts[1:] = done[:-1]
    ticks = np.zeros_like(done, dtype=np.int64)
    for t in range(1, len(done)):
        ticks[t] = np.where(starts[t], 0, ticks[t - 1] + 1)
    return starts, ticks * .02


def action_stats(raw, selected=None):
    if selected is not None:
        raw = raw[selected]
    raw = raw.reshape(-1, 2)
    if not len(raw):
        return {"n": 0}
    scaled = raw * SCALES
    over = np.abs(scaled) > LIMITS + 1e-5
    return {"n": len(raw), "oob_count": over.sum(axis=0).tolist(),
            "oob_percent": (over.mean(axis=0) * 100).tolist(),
            "raw_abs_max": np.abs(raw).max(axis=0).tolist(),
            "raw_abs_p95": np.quantile(np.abs(raw), .95, axis=0).tolist(),
            "raw_rms": np.sqrt(np.mean(raw.astype(float) ** 2, axis=0)).tolist()}


def compare_replay(obs, raw, infer):
    changes = {}
    substitutions = {"deploy_vx_only": {5: .07 * obs[:, 10]}, "deploy_vy_only": {6: 0},
                     "deploy_vx_vy": {5: .07 * obs[:, 10], 6: 0},
                     "wrap_rear_only": {8: (obs[:, 8] + np.pi) % (2 * np.pi) - np.pi},
                     "remove_observed_vx": {5: 0}, "zero_rear_phase": {8: 0}}
    for name, mapping in substitutions.items():
        x = obs.copy()
        for index, value in mapping.items():
            x[:, index] = value
        alternative = infer(x)
        delta = alternative - raw
        changes[name] = {**action_stats(alternative),
                         "raw_delta_rms": np.sqrt(np.mean(delta.astype(float) ** 2, axis=0)).tolist(),
                         "raw_delta_abs_max": np.abs(delta).max(axis=0).tolist()}
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-mujoco", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--collect-phase", action="store_true")
    parser.add_argument("--refresh-report", action="store_true", help="Replace this script's summary/plots, never input captures")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    actor, infer = actor_models()
    if args.collect_mujoco:
        collect_mujoco(HERE / "mujoco_observations.npz", infer, args.num_envs, args.steps)
        collect_mujoco(HERE / "mujoco_estimator_ablation.npz", infer, min(args.num_envs, 8), args.steps, True)
    if args.collect_phase:
        collect_mujoco(HERE / "mujoco_phase_ablation.npz", infer, min(args.num_envs, 8), args.steps,
                      rear_period_4pi=True)
    if not args.analyze:
        return
    if (HERE / "summary.json").exists() and not args.refresh_report:
        raise FileExistsError("Refusing to overwrite summary.json")
    datasets = {}
    for label, folder in (("isaac_no_dr", "isaac_no_dr_gpu"), ("isaac_dr", "isaac_dr_gpu")):
        with np.load(HERE / folder / "observations.npz") as source:
            data = dict(source)
        starts, ages = starts_and_ages(data["done"])
        datasets[label] = {"obs": data["observation"].reshape(-1, 13),
                           "raw": data["raw_action"].reshape(-1, 2),
                           "starts": starts.ravel(), "ages": ages.ravel(), "shape": data["done"].shape,
                           "clean": data["clean_observation"].reshape(-1, 13)}
    mu = dict(np.load(HERE / "mujoco_observations.npz"))
    for dr in (False, True):
        prefix = "dr" if dr else "no_dr"
        starts, ages = starts_and_ages(mu[prefix + "_done"])
        datasets["mujoco_" + prefix] = {"obs": mu[prefix + "_observation"].reshape(-1, 13),
                                       "raw": mu[prefix + "_raw_action"].reshape(-1, 2),
                                       "starts": starts.ravel(), "ages": ages.ravel(),
                                       "shape": starts.shape, "clean": mu[prefix + "_clean_observation"].reshape(-1, 13),
                                       "unwrapped_rear_rad": mu[prefix + "_unwrapped_rear_rad"].ravel()}
    records = np.genfromtxt(BIKE / "policy_samples.csv", delimiter=",", names=True)
    bike = np.column_stack([records[name] for name in records.dtype.names[1:14]]).astype(np.float32)
    summary_bike = json.loads((BIKE / "summary.json").read_text())
    raw = np.column_stack((records["raw_steer"], records["raw_rear"])).astype(np.float32)
    ages, starts = np.empty(len(bike)), np.zeros(len(bike), bool)
    for attempt in summary_bike["attempts"]:
        mask = ((records["bag_time_s"] >= attempt["first_s"] - 1e-7)
                & (records["bag_time_s"] <= attempt["last_policy_s"] + 1e-7))
        assert mask.sum() == attempt["policy_count"]
        ages[mask] = records["bag_time_s"][mask] - attempt["first_s"]
        starts[np.flatnonzero(mask)[0]] = True
    datasets["bike"] = {"obs": bike, "raw": raw, "starts": starts, "ages": ages, "shape": (len(bike),)}
    result = {"model": "09221945/model_91850", "onnx_sha256": sha(ONNX),
              "checkpoint_sha256": sha(RUN / "model_91850.pt"),
              "parity_probes": parity_probes(infer), "datasets": {}, "closed_loop_ablation": {},
              "closed_loop_isaac_ablation": {}}
    reference = datasets["isaac_dr"]["obs"]
    qlo, qhi = np.quantile(reference, [.001, .999], axis=0)
    field_rows = []
    replay_arrays = {}
    for label, data in datasets.items():
        obs, raw = data["obs"], data["raw"]
        replay = infer(obs)
        with torch.no_grad():
            pt = actor(torch.from_numpy(obs)).numpy()
        score = {"all": action_stats(raw), "first_action": action_stats(raw, data["starts"]),
                 "after_0_5s": action_stats(raw, data["ages"] >= .5),
                 "first_0_2s": action_stats(raw, data["ages"] < .2),
                 "actor_replay_max_error": np.abs(replay - raw).max(axis=0).tolist(),
                 "pt_vs_onnx_max_error": np.abs(pt - replay).max(axis=0).tolist(),
                 "fixed_input_replays": compare_replay(obs, replay, infer)}
        if "unwrapped_rear_rad" in data:
            candidate = obs.copy()
            candidate[:, 8] = (data["unwrapped_rear_rad"] + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
            candidate_raw = infer(candidate)
            score["candidate_4pi_rear_fixed_replay"] = {
                **action_stats(candidate_raw),
                "raw_delta_rms": np.sqrt(np.mean((candidate_raw - replay).astype(float) ** 2, axis=0)).tolist()}
        deploy_input = obs.copy(); deploy_input[:, 5] = .07 * obs[:, 10]; deploy_input[:, 6] = 0
        replay_arrays[label + "_deploy_velocity_raw"] = infer(deploy_input).reshape(*data["shape"], 2)
        if "clean" in data:
            clean = data["clean"]
            clean_raw = infer(clean)
            score["without_white_noise"] = {**action_stats(clean_raw),
                "raw_delta_rms": np.sqrt(np.mean((clean_raw - replay).astype(float) ** 2, axis=0)).tolist()}
            vel_error = .07 * clean[:, 10] - clean[:, 5]
            score["wheel_vx_estimator_error_m_s"] = {"rms": float(np.sqrt(np.mean(vel_error ** 2))),
                "abs_p95": float(np.quantile(np.abs(vel_error), .95)), "abs_max": float(np.abs(vel_error).max()),
                "true_vx_rms": float(np.sqrt(np.mean(clean[:, 5] ** 2))),
                "estimate_vx_rms": float(np.sqrt(np.mean((.07 * clean[:, 10]) ** 2))),
                "correlation": float(np.corrcoef(clean[:, 5], .07 * clean[:, 10])[0, 1])}
            for title, mask in (("first_0_5s", data["ages"] < .5), ("after_0_5s", data["ages"] >= .5)):
                score["wheel_vx_estimator_error_m_s"][title] = {
                    "n": int(mask.sum()), "rms": float(np.sqrt(np.mean(vel_error[mask] ** 2))),
                    "abs_p95": float(np.quantile(np.abs(vel_error[mask]), .95))}
        for i, field in enumerate(FEATURES):
            x = obs[:, i]
            row = {"dataset": label, "feature": field, "min": float(x.min()), "max": float(x.max()),
                   "median": float(np.median(x)), "abs_p95": float(np.quantile(np.abs(x), .95)),
                   "rms": float(np.sqrt(np.mean(x.astype(float) ** 2))),
                   "outside_isaac_dr_p0_1_p99_9_percent": float(100 * np.mean((x < qlo[i]) | (x > qhi[i])))}
            field_rows.append(row)
        result["datasets"][label] = score
    ablation = dict(np.load(HERE / "mujoco_estimator_ablation.npz"))
    phase = dict(np.load(HERE / "mujoco_phase_ablation.npz"))
    for mode in ("no_dr", "dr"):
        for name, source in (("true_velocity", mu), ("wheel_velocity_estimate", ablation), ("rear_period_4pi", phase)):
            entries = json.loads(str(source[mode + "_episodes_json"]))
            entries = [e for e in entries if e["env"] < 8]
            # Initial capture used stats.death, which is None for a timeout.
            for entry in entries:
                if entry["result"] is None:
                    assert entry["steps"] == BikeBalanceEnv.MAX_STEPS
                    entry["result"] = "time_out"
            actions = source[mode + "_raw_action"][:, :8]
            lengths = np.array([e["steps"] for e in entries]) * .02
            result["closed_loop_ablation"][mode + "_" + name] = {
                **action_stats(actions), "num_envs": 8, "seeds": list(range(42, 50)),
                "completed_episodes": len(entries), "mean_completed_length_s": float(lengths.mean()),
                "timeout_count": sum(e["result"] == "time_out" for e in entries),
                "episode_results": entries}
        for name, suffix in (("true_velocity", "_gpu"), ("wheel_velocity_estimate", "_estimator_gpu")):
            folder = HERE / ("isaac_" + mode + suffix)
            values = dict(np.load(folder / "observations.npz"))
            _, ages = starts_and_ages(values["done"])
            lengths = ages[values["done"]] + .02
            result["closed_loop_isaac_ablation"][mode + "_" + name] = {
                **action_stats(values["raw_action"]), "num_envs": 64, "seed": 42,
                "completed_episodes": len(lengths), "mean_completed_length_s": float(lengths.mean()),
                "min_completed_length_s": float(lengths.min()), "max_completed_length_s": float(lengths.max()),
                "horizon_reached_count": int(np.isclose(lengths, 30, atol=1e-6).sum()),
                "evaluation_summary": str((folder / "eval.txt").relative_to(HERE))}
    ideal = np.zeros((1, 13), np.float32); ideal[0, [1, 12]] = [-1, 1]
    result["ideal_upright_raw"] = infer(ideal)[0].tolist()
    # Explicit wrapping counterexample, with all other inputs identical.
    cases = np.repeat(ideal, 2, axis=0); cases[0, 8] = 3.5; cases[1, 8] = 3.5 - 2 * np.pi
    result["rear_angle_only_example"] = {"rear_position_rad": cases[:, 8].tolist(), "raw": infer(cases).tolist()}
    with (HERE / "feature_distributions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(field_rows[0])); writer.writeheader(); writer.writerows(field_rows)
    np.savez_compressed(HERE / "counterfactual_actions.npz", **replay_arrays)
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(datasets, result, field_rows)
    print(json.dumps({k: {"all": v["all"], "wheel_error": v.get("wheel_vx_estimator_error_m_s")}
                      for k, v in result["datasets"].items()}, indent=2))


def save(fig, name):
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{name}.{extension}", dpi=150)
    plt.close(fig)


def plot(datasets, result, rows):
    labels = list(datasets)
    colors = ["#4477aa", "#228833", "#ee7733", "#aa3377", "#cc3311"]
    fig, axes = plt.subplots(4, 4, figsize=(17, 13))
    for i, field in enumerate(FEATURES):
        ax = axes.ravel()[i]
        for label, color in zip(labels, colors):
            sorted_x = np.sort(datasets[label]["obs"][:, i])
            selected = np.linspace(0, len(sorted_x) - 1, min(len(sorted_x), 2000)).astype(int)
            ax.plot(sorted_x[selected], (selected + 1) / len(sorted_x), color=color, label=label)
        ax.set_title(f"[{i}] {field}"); ax.grid(alpha=.2)
    for ax in axes.ravel()[13:]:
        ax.axis("off")
    axes.ravel()[13].legend(*axes[0, 0].get_legend_handles_labels(), loc="center")
    fig.suptitle("91850 actual policy-input distributions (ECDF)\nDifferent state visitation, not matched physical motion; bike includes stand contact")
    save(fig, "observation_distributions")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = ["isaac_no_dr", "isaac_dr", "mujoco_no_dr", "mujoco_dr"]
    for j in range(2):
        original = [result["datasets"][name]["all"]["oob_percent"][j] for name in names]
        altered = [result["datasets"][name]["fixed_input_replays"]["deploy_vx_vy"]["oob_percent"][j] for name in names]
        x = np.arange(len(names))
        axes[j].bar(x - .18, original, .36, label="Recorded true velocity")
        axes[j].bar(x + .18, altered, .36, label="Same states, deployment vx/vy")
        axes[j].set_xticks(x, names, rotation=20); axes[j].set_ylabel("OOB policy samples (%)")
        axes[j].set_title(["Steering position demand", "Rear feed-forward demand"][j]); axes[j].legend(); axes[j].grid(axis="y", alpha=.2)
    fig.suptitle("Fixed-input counterfactual: change only vx=0.07*rear_speed and vy=0\nNot an alternate trajectory or survival test")
    save(fig, "velocity_substitution_effect")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for j, name in enumerate(("isaac_dr", "mujoco_dr", "bike")):
        data = datasets[name]
        mask = (data["ages"] >= .5)
        # Deterministic evenly spaced plotting samples; all rows used in statistics.
        indices = np.flatnonzero(mask)
        indices = indices[np.linspace(0, len(indices) - 1, min(len(indices), 8000)).astype(int)]
        for k in range(2):
            axes[k, j].scatter(data["obs"][indices, 8], data["raw"][indices, k], s=2, alpha=.25)
            axes[k, j].set_title(name); axes[k, j].grid(alpha=.2)
            axes[k, j].set_ylabel(["Raw steering", "Raw rear"][k]); axes[k, j].set_xlabel("Rear-position observation (rad)")
    fig.suptitle("Different input/output operating regions after 0.5 s (not a causal single-feature plot)")
    save(fig, "rear_phase_operating_regions")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    variants = ["true_velocity", "wheel_velocity_estimate", "rear_period_4pi"]
    titles = ["Original", "Deployment vx/vy", "4-pi rear candidate"]
    for j, mode in enumerate(("no_dr", "dr")):
        scores = [result["closed_loop_ablation"][mode + "_" + key] for key in variants]
        axes[0, j].bar(titles, [s["mean_completed_length_s"] for s in scores], color=["#4477aa", "#cc3311", "#228833"])
        axes[0, j].set_title(mode); axes[0, j].set_ylabel("Mean completed episode (s)")
        for k, score in enumerate(scores):
            axes[0, j].text(k, score["mean_completed_length_s"],
                            f"{score['mean_completed_length_s']:.2f}s\n{score['timeout_count']}/{score['completed_episodes']} timeouts",
                            ha="center", va="bottom", fontsize=8)
        x = np.arange(3)
        axes[1, j].bar(x - .18, [s["oob_percent"][0] for s in scores], .36, label="Steering")
        axes[1, j].bar(x + .18, [s["oob_percent"][1] for s in scores], .36, label="Rear")
        axes[1, j].set_xticks(x, titles); axes[1, j].set_ylabel("OOB policy samples (%)"); axes[1, j].legend()
    for ax in axes.ravel():
        ax.grid(axis="y", alpha=.2); ax.margins(y=.25)
    fig.suptitle("Closed-loop MuJoCo input-contract ablations: 8 seeds, 60 s/environment\nSame model, dynamics, noise settings and control/physics rates; final unfinished episodes excluded from mean")
    save(fig, "closed_loop_observation_ablation")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, mode in enumerate(("no_dr", "dr")):
        scores = [result["closed_loop_isaac_ablation"][mode + "_" + key]
                  for key in ("true_velocity", "wheel_velocity_estimate")]
        names = ["Original true vx/vy", "Deployment vx/vy"]
        axes[0, j].bar(names, [s["mean_completed_length_s"] for s in scores], color=["#4477aa", "#cc3311"])
        axes[0, j].set_title(mode); axes[0, j].set_ylabel("Mean completed episode (s)")
        for k, score in enumerate(scores):
            axes[0, j].text(k, score["mean_completed_length_s"],
                            f"{score['mean_completed_length_s']:.3f}s\n{score['horizon_reached_count']}/{score['completed_episodes']} reached 30s",
                            ha="center", va="bottom", fontsize=8)
        x = np.arange(2)
        axes[1, j].bar(x - .18, [s["oob_percent"][0] for s in scores], .36, label="Steering")
        axes[1, j].bar(x + .18, [s["oob_percent"][1] for s in scores], .36, label="Rear")
        axes[1, j].set_xticks(x, names); axes[1, j].set_ylabel("OOB policy samples (%)"); axes[1, j].legend()
    for ax in axes.ravel():
        ax.grid(axis="y", alpha=.2); ax.margins(y=.22)
    fig.suptitle("Controlled Isaac ablation: change ONLY two observation channels\nSame 91850 checkpoint, seed 42, 64 envs x 3000 steps, physics/actions/terminations unchanged")
    save(fig, "isaac_closed_loop_observation_ablation")
    data = datasets["isaac_dr"]
    obs = data["obs"].reshape(*data["shape"], 13)
    true_raw = data["raw"].reshape(*data["shape"], 2)
    alternative = obs.reshape(-1, 13).copy()
    alternative[:, 5] = .07 * alternative[:, 10]; alternative[:, 6] = 0
    _, infer = actor_models()
    alternative_raw = infer(alternative).reshape(*data["shape"], 2)
    step, env_id = np.unravel_index(np.abs(alternative_raw[..., 1] - true_raw[..., 1]).argmax(), true_raw.shape[:2])
    begins = np.flatnonzero(data["starts"].reshape(data["shape"])[:step + 1, env_id])
    begin = begins[-1]
    end = min(begin + 51, len(obs))
    times = np.arange(end - begin) * .02
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(times, obs[begin:end, env_id, 5], label="True root body vx")
    axes[0].plot(times, .07 * obs[begin:end, env_id, 10], label="Deployment wheel-speed estimate")
    axes[0].set_ylabel("vx (m/s)")
    for i, ax in enumerate(axes[1:]):
        ax.step(times, true_raw[begin:end, env_id, i] * SCALES[i], where="post", label="Original recorded policy demand")
        ax.step(times, alternative_raw[begin:end, env_id, i] * SCALES[i], where="post", label="Replay with deployment vx/vy")
        ax.axhline(LIMITS[i], color="gray", ls=":"); ax.axhline(-LIMITS[i], color="gray", ls=":")
        ax.set_ylabel(["Steering target (rad)", "Rear feed-forward (N m)"][i])
    for ax in axes:
        ax.grid(alpha=.2); ax.legend()
    axes[-1].set_xlabel("Episode time (s)")
    fig.suptitle(f"Isaac DR env {env_id}, episode beginning step {begin}: exact same recorded states\nExample selected for largest rear-action change from substituting deployment vx/vy; NOT a new closed-loop run")
    save(fig, "isaac_velocity_mismatch_example")


if __name__ == "__main__":
    main()
