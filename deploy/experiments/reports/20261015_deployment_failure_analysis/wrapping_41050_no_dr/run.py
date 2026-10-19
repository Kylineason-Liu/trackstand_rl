"""Controlled 41050 MuJoCo rear-angle encoding tests; no runtime source edits.

Nominal tests use current viewer physics, no DR, 64 seeds x 3000 steps.
Separately labelled stress tests change the fixed initial physical rear phase
equally for every encoding and stop after each seed's first episode. Only the
rear-angle observation encoding differs between variants in each comparison.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "isaaclab_train").is_dir())
MU = ROOT / "mujoco_sim2sim"
ONNX = ROOT / "isaaclab_train/scripts/deployment/trackstand_rl_rsl_2026-09-17_19-46-37_model_41050.onnx"
PT = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-09-17_19-46-37/model_41050.pt"
XML = MU / "models/trackstand_rl_0916_bike/trackstand_rl_0916.xml"
sys.path.insert(0, str(MU / "src"))
from newbike_mujoco.env import BikeBalanceEnv
from newbike_mujoco.policy import DeployedPolicy

MODES = ("wrapped_pi", "native_unwrapped", "candidate_4pi")
SCALES = np.array([1.57, 6.28])
LIMITS = np.array([np.pi / 2, 6.28])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model():
    spec = importlib.util.spec_from_file_location("wrap41050_viewer", MU / "scripts/run_deploy_viewer.py")
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    model = mujoco.MjSpec.from_file(str(XML))
    viewer.augment_trackstand_model(model, str(XML.parent))
    return model.compile()


def encode(angle, mode):
    if mode == "wrapped_pi":
        return (angle + np.pi) % (2 * np.pi) - np.pi
    if mode == "candidate_4pi":
        return (angle + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
    if mode == "native_unwrapped":
        return angle
    raise ValueError(mode)


class EncodingEnv(BikeBalanceEnv):
    def __init__(self, *args, encoding, initial_rear_rad=0., **kwargs):
        self.encoding = encoding
        self.initial_rear_rad = initial_rear_rad
        super().__init__(*args, **kwargs)

    def observe(self):
        value = super().observe()  # Preserve all existing noise draws and channels.
        if self.encoding != "wrapped_pi":
            rear = self.data.qpos[self.model.jnt_qposadr[self._rear_jid]]
            value[8] = encode(rear, self.encoding)
        return value

    def reset(self):
        value = super().reset()
        if self.initial_rear_rad:
            self.data.qpos[self.model.jnt_qposadr[self._rear_jid]] = self.initial_rear_rad
            mujoco.mj_forward(self.model, self.data)
            # Only the rear joint coordinate changed. Other observed root/joint
            # coordinates are unchanged, so do not make an extra noise draw.
            value[8] = encode(self.initial_rear_rad, self.encoding)
        return value


def episode(env, seed, end_tick):
    s = env.stats
    return {"seed": seed, "end_tick": end_tick, "length_steps": s.length, "length_s": s.length * .02,
            "result": "timeout" if s.timeout else s.death,
            "roll_rms_deg": float(np.rad2deg(np.sqrt(s.roll_sq / s.length))),
            "roll_max_deg": float(np.rad2deg(s.roll_max)), "drift_max_m": s.drift_max,
            "yaw_max_deg": s.yaw_max, "steer_speed_max_rad_s": s.steer_speed_max,
            "rear_speed_max_rad_s": s.drive_speed_max}


def collect(model, policy, mode, n, steps, phase=0., first_episode_only=False):
    envs = [EncodingEnv(model=model, seed=42 + i, enable_dr=False,
                        encoding=mode, initial_rear_rad=phase) for i in range(n)]
    obs = np.stack([env.reset() for env in envs])
    width = model.nq + model.nv
    records = {"observation": np.full((steps, n, 13), np.nan, np.float32),
               "raw_action": np.full((steps, n, 2), np.nan, np.float32),
               "state_before": np.full((steps, n, width), np.nan, np.float64),
               "roll_after_rad": np.full((steps, n), np.nan, np.float64),
               "valid": np.zeros((steps, n), bool), "done": np.zeros((steps, n), bool),
               "starts": np.zeros((steps, n), bool)}
    active = np.ones(n, bool)
    starts = np.ones(n, bool)
    episodes = []
    for t in range(steps):
        ids = np.flatnonzero(active)
        if not len(ids):
            records = {key: value[:t] for key, value in records.items()}
            break
        actions = policy.act_batch(obs[ids])
        records["valid"][t, ids] = True
        records["starts"][t, ids] = starts[ids]
        records["observation"][t, ids] = obs[ids]
        records["raw_action"][t, ids] = actions
        for i, action in zip(ids, actions):
            env = envs[i]
            records["state_before"][t, i] = np.r_[env.data.qpos, env.data.qvel]
            obs[i], done = env.step(action)
            records["roll_after_rad"][t, i] = env.stats.rolls[-1]
            records["done"][t, i] = done
            starts[i] = done
            if done:
                episodes.append(episode(env, 42 + int(i), t))
                if first_episode_only:
                    active[i] = False
                else:
                    obs[i] = env.reset()
        if (t + 1) % 1000 == 0:
            print(f"[CAPTURE] phase={phase:+g} mode={mode} {t + 1}/{steps}", flush=True)
    if first_episode_only:
        assert len(episodes) == n
    rear_adr = int(model.jnt_qposadr[model.joint("rearwheel_joint").id])
    valid = records["valid"]
    raw = records["raw_action"][valid]
    observation = records["observation"][valid]
    rear = records["state_before"][..., rear_adr][valid]
    lengths = np.array([e["length_s"] for e in episodes])
    over = np.abs(raw * SCALES) > LIMITS + 1e-5
    within = valid[1:] & valid[:-1] & ~records["starts"][1:]
    jumps = np.abs(np.diff(records["observation"][..., 8], axis=0)) > np.pi
    metric = {"encoding": mode, "initial_rear_phase_rad": phase, "first_episode_only": first_episode_only,
              "num_envs": n, "seeds": list(range(42, 42 + n)), "valid_policy_samples": int(valid.sum()),
              "completed_episodes": len(episodes), "timeouts": sum(e["result"] == "timeout" for e in episodes),
              "falls": sum(e["result"] == "fallen" for e in episodes),
              "mean_completed_episode_s": float(lengths.mean()), "min_episode_s": float(lengths.min()),
              "max_episode_s": float(lengths.max()),
              "roll_rms_all_steps_deg": float(np.rad2deg(np.sqrt(np.mean(records["roll_after_rad"][valid] ** 2)))),
              "rear_native_min_rad": float(rear.min()), "rear_native_max_rad": float(rear.max()),
              "native_rear_abs_over_pi_samples": int(np.sum(np.abs(rear) > np.pi)),
              "encoding_changes_native_samples": int(np.sum(np.abs(observation[:, 8] - rear) > 1e-5)),
              "rear_observation_branch_jumps": int(np.sum(jumps & within)),
              "raw_abs_max": np.abs(raw).max(axis=0).tolist(), "oob_count": over.sum(axis=0).tolist(),
              "oob_percent": (100 * over.mean(axis=0)).tolist(), "episodes": episodes}
    return records, metric


def compare(first, second):
    result = {"initial_physical_states_identical": bool(np.array_equal(first["state_before"][0], second["state_before"][0])),
              "initial_other_12_channels_identical": bool(np.array_equal(
                  np.delete(first["observation"][0], 8, axis=1), np.delete(second["observation"][0], 8, axis=1))),
              "all_valid_masks_identical": bool(np.array_equal(first["valid"], second["valid"]))}
    for name in ("observation", "raw_action", "state_before", "roll_after_rad", "done"):
        result[name + "_bitwise_identical"] = bool(np.array_equal(first[name], second[name], equal_nan=True))
    return result


def make_plots(out, summary, base, stress):
    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg", "pdf"):
            fig.savefig(out / f"{name}.{extension}", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    scores = [summary["nominal"][mode] for mode in MODES]
    labels = ["Current ±π", "No wrapping", "4π candidate"]
    axes[0, 0].bar(labels, [s["mean_completed_episode_s"] for s in scores])
    axes[0, 0].set_ylabel("Mean completed episode (s)")
    for i, score in enumerate(scores):
        axes[0, 0].text(i, score["mean_completed_episode_s"],
                        f"{score['mean_completed_episode_s']:.3f}s\n{score['timeouts']}/{score['completed_episodes']} timeouts",
                        ha="center", va="bottom", fontsize=8)
    axes[0, 0].margins(y=.2)
    for i in (0, 21, 63):
        axes[0, 1].plot(np.arange(len(base["wrapped_pi"]["valid"])) * .02,
                        base["wrapped_pi"]["state_before"][:, i, summary["rear_qpos_index"]], label=f"seed {42 + i}", lw=.8)
    axes[0, 1].axhline(np.pi, ls="--", color="red", label="±π branch boundary")
    axes[0, 1].axhline(-np.pi, ls="--", color="red")
    axes[0, 1].set_xlabel("Simulation time (s), resets included")
    axes[0, 1].set_ylabel("Native rear angle (rad)"); axes[0, 1].legend(fontsize=8)
    for j, mode in enumerate(("native_unwrapped", "candidate_4pi")):
        a = base[mode]["raw_action"] - base["wrapped_pi"]["raw_action"]
        axes[1, 0].plot(np.arange(len(a)) * .02, np.abs(a).max(axis=(1, 2)), label=mode, lw=.8)
        delta = np.abs(base[mode]["observation"] - base["wrapped_pi"]["observation"]).max(axis=(1, 2))
        axes[1, 1].plot(np.arange(len(delta)) * .02, delta, label=mode, lw=.8)
    axes[1, 0].set_ylabel("Max absolute raw-action difference across 64 envs")
    axes[1, 1].set_ylabel("Max absolute observation difference across 64 envs")
    for ax in axes[1]:
        ax.set_xlabel("Simulation time (s)"); ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.suptitle("41050 no-DR nominal test: only rear-angle encoding differs\n64 seeds (42–105) × 3000 steps; true body velocities, unchanged viewer physics/noise/controls")
    save(fig, "nominal_wrapping_comparison")
    phases = list(stress)
    fig, axes = plt.subplots(2, len(phases), figsize=(5 * len(phases), 8), squeeze=False)
    for column, phase in enumerate(phases):
        scores = [summary["boundary_stress"][phase][mode] for mode in MODES]
        axes[0, column].bar(labels, [s["mean_completed_episode_s"] for s in scores])
        for i, score in enumerate(scores):
            axes[0, column].text(i, score["mean_completed_episode_s"],
                                f"{score['mean_completed_episode_s']:.2f}s\n{score['timeouts']}/{score['completed_episodes']}",
                                ha="center", va="bottom", fontsize=8)
        axes[0, column].set_title(f"Fixed initial rear phase {float(phase):+.1f} rad")
        axes[0, column].set_ylabel("Mean first-episode duration (s)"); axes[0, column].margins(y=.2)
        for mode, label in zip(MODES, labels):
            values = stress[phase][mode]
            valid = values["valid"][:, 0]
            t = np.arange(len(valid))[valid] * .02
            axes[1, column].plot(t, np.rad2deg(values["roll_after_rad"][valid, 0]), label=label)
        axes[1, column].axhline(30, color="gray", ls=":"); axes[1, column].axhline(-30, color="gray", ls=":")
        axes[1, column].set_ylabel("Seed 42 roll (deg)"); axes[1, column].set_xlabel("First-episode time (s)")
        axes[1, column].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.suptitle("Separately labelled boundary stress: identical physical starting phase within each column\n16 matched seeds; no DR; this is NOT the original zero-phase nominal condition")
    save(fig, "boundary_stress_comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    protected = [MU / "scripts/run_deploy_viewer.py", MU / "src/newbike_mujoco/env.py",
                 MU / "src/newbike_mujoco/policy.py", XML, ONNX, ONNX.with_suffix(".onnx.data"), PT,
                 ROOT / "progress_record.md", ROOT / "progress_record.txt"]
    protected += sorted(XML.parent.rglob("*.stl"))
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    model = make_model()
    policy = DeployedPolicy(onnx_path=str(ONNX), pt_path=str(PT))
    options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
    policy._ort_sess = ort.InferenceSession(str(ONNX), options, providers=["CPUExecutionProvider"])
    summary = {"model": "09171946/model_41050", "domain_randomization": False,
               "observation_white_noise": list(BikeBalanceEnv.OBS_NOISE_STD),
               "policy_dt_s": BikeBalanceEnv.CTRL_DT, "physics_dt_s": model.opt.timestep,
               "model_nq": model.nq, "model_nv": model.nv,
               "rear_qpos_index": int(model.jnt_qposadr[model.joint("rearwheel_joint").id]),
               "onnx_sha256": sha(ONNX), "checkpoint_sha256": sha(PT),
               "nominal": {}, "nominal_comparisons": {}, "boundary_stress": {}, "boundary_comparisons": {}}
    begin = time.monotonic()
    baseline = {}
    for mode in MODES:
        data, metric = collect(model, policy, mode, 64, 3000)
        baseline[mode] = data
        summary["nominal"][mode] = metric
        np.savez_compressed(out / ("nominal_" + mode + ".npz"), **data)
    for mode in MODES[1:]:
        summary["nominal_comparisons"][mode] = compare(baseline["wrapped_pi"], baseline[mode])
    stress = {}
    for phase in (2.8, 3.5, -3.5):
        key = str(phase); stress[key] = {}; summary["boundary_stress"][key] = {}; summary["boundary_comparisons"][key] = {}
        for mode in MODES:
            data, metric = collect(model, policy, mode, 16, 1500, phase, first_episode_only=True)
            stress[key][mode] = data
            summary["boundary_stress"][key][mode] = metric
            np.savez_compressed(out / (f"phase_{phase:+g}_" + mode + ".npz"), **data)
        for mode in MODES[1:]:
            summary["boundary_comparisons"][key][mode] = compare(stress[key]["wrapped_pi"], stress[key][mode])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ["condition", "encoding", "mean_completed_episode_s", "completed_episodes", "timeouts", "falls",
              "rear_native_min_rad", "rear_native_max_rad", "native_rear_abs_over_pi_samples",
              "rear_observation_branch_jumps", "oob_percent"]
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        groups = [("nominal", summary["nominal"]), *[("phase=" + p, v) for p, v in summary["boundary_stress"].items()]]
        for condition, variants in groups:
            for metric in variants.values():
                writer.writerow({"condition": condition, **{k: metric[k] for k in fields[1:]}})
    make_plots(out, summary, baseline, stress)
    integrity = {"before_hashes": before, "changed_files": {str(p.relative_to(ROOT)): sha(p) for p in protected
                 if sha(p) != before[str(p.relative_to(ROOT))]}, "wall_time_s": time.monotonic() - begin,
                 "no_hardware_or_runtime_source_changes": True}
    (out / "source_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    if integrity["changed_files"]:
        raise RuntimeError(integrity)
    print(json.dumps({"nominal": {k: {f: v[f] for f in fields[1:]} for k, v in summary["nominal"].items()},
                      "wall_time_s": integrity["wall_time_s"]}, indent=2))


if __name__ == "__main__":
    main()
