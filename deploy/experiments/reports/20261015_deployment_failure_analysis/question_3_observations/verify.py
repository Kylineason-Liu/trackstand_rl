"""Independently check captures/replays; no input capture or runtime file is modified."""

import importlib.util
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("observation_analysis_verifier", HERE / "analyze.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
actor, infer = audit.actor_models()
summary = json.loads((HERE / "summary.json").read_text())
checks = {}
datasets = {}
for mode in ("no_dr", "dr"):
    capture = dict(np.load(HERE / ("isaac_" + mode + "_gpu") / "observations.npz"))
    old = np.load(HERE.parent / "isaac_actuation" / mode / "motor_samples.npz")
    label = "isaac_" + mode
    checks[label + "_raw_identical_to_previous_evaluation"] = np.array_equal(capture["raw_action"], old["raw_action"])
    checks[label + "_done_identical_to_previous_evaluation"] = np.array_equal(capture["done"], old["done"])
    checks[label + "_capture_shape"] = capture["observation"].shape == (3000, 64, 13)
    checks[label + "_noise_only_first_five"] = np.array_equal(capture["observation"][..., 5:], capture["clean_observation"][..., 5:])
    checks[label + "_source_integrity"] = not json.loads((HERE / (label + "_gpu") / "source_integrity.json").read_text())["changed_files"]
    datasets[label] = capture
    probe = json.loads((HERE / (label + "_gpu") / "rear_angle_probe.json").read_text())
    checks[label + "_actual_rear_term_above_pi"] = probe["actual_observation_term_rear_position_rad"][1] > np.pi
    changed = dict(np.load(HERE / ("isaac_" + mode + "_estimator_gpu") / "observations.npz"))
    ablation_folder = HERE / ("isaac_" + mode + "_estimator_gpu")
    base_cfg = (HERE / (label + "_gpu") / "effective_env.yaml").read_text()
    altered_cfg = (ablation_folder / "effective_env.yaml").read_text()
    checks[label + "_ablation_base_configuration_identical_except_output_directory"] = (
        re.sub(r"^log_dir:.*$", "", base_cfg, flags=re.MULTILINE)
        == re.sub(r"^log_dir:.*$", "", altered_cfg, flags=re.MULTILINE))
    checks[label + "_ablation_source_integrity"] = not json.loads((ablation_folder / "source_integrity.json").read_text())["changed_files"]
    checks[label + "_ablation_metadata_declares_override"] = json.loads((ablation_folder / "experiment.json").read_text())["deploy_velocity_ablation"] is True
    common = [i for i in range(13) if i not in (5, 6)]
    checks[label + "_ablation_other_initial_channels_identical"] = np.array_equal(
        capture["observation"][0][:, common], changed["observation"][0][:, common])
    checks[label + "_ablation_initial_physical_states_identical"] = all(
        np.array_equal(capture[key][0], changed[key][0]) for key in
        ("root_quaternion_xyzw", "root_position_w_m", "linear_velocity_world_m_s", "angular_velocity_world_rad_s", "imu_bias"))
    checks[label + "_ablation_estimator_applied"] = bool(
        np.allclose(changed["observation"][..., 5], .07 * changed["observation"][..., 10], atol=1e-6)
        and np.all(changed["observation"][..., 6] == 0))
    _, age = audit.starts_and_ages(changed["done"])
    lengths = age[changed["done"]] + .02
    ablation_score = summary["closed_loop_isaac_ablation"][mode + "_wheel_velocity_estimate"]
    checks[label + "_ablation_episode_counts"] = len(lengths) == ablation_score["completed_episodes"]
    checks[label + "_ablation_episode_mean"] = bool(np.isclose(lengths.mean(), ablation_score["mean_completed_length_s"]))
    if mode == "no_dr":
        checks["isaac_nominal_ablation_initial_full_inputs_actions_identical"] = bool(
            np.array_equal(capture["observation"][0], changed["observation"][0])
            and np.array_equal(capture["raw_action"][0], changed["raw_action"][0]))
        checks["isaac_nominal_ablation_every_completed_episode_11_steps"] = bool(np.allclose(lengths, .22))
    mu_label = "mujoco_" + mode
    with np.load(HERE / "mujoco_observations.npz") as source:
        datasets[mu_label] = {"observation": source[mode + "_observation"], "raw_action": source[mode + "_raw_action"]}
    base = datasets[mu_label]
    for kind in ("estimator", "phase"):
        source = np.load(HERE / ("mujoco_" + kind + "_ablation.npz"))
        checks[mu_label + "_" + kind + "_same_initial_obs"] = np.array_equal(source[mode + "_observation"][0], base["observation"][0, :8])
        checks[mu_label + "_" + kind + "_same_initial_action"] = np.array_equal(source[mode + "_raw_action"][0], base["raw_action"][0, :8])
        rows = json.loads(str(source[mode + "_episodes_json"]))
        checks[mu_label + "_" + kind + "_valid_terminations"] = all(
            e["result"] == "fallen" or e["steps"] == 1500 for e in rows)

for label, values in datasets.items():
    obs = values["observation"].reshape(-1, 13)
    raw = values["raw_action"].reshape(-1, 2)
    checks[label + "_finite"] = bool(np.isfinite(obs).all() and np.isfinite(raw).all())
    predicted = infer(obs)
    checks[label + "_actor_replay"] = bool(np.allclose(raw, predicted, atol=2e-5, rtol=1e-5))
    over = np.abs(raw.astype(float) * [1.57, 6.28]) > np.array([np.pi / 2, 6.28]) + 1e-5
    checks[label + "_counts"] = over.sum(axis=0).tolist() == summary["datasets"][label]["all"]["oob_count"]

actual = np.genfromtxt(audit.BIKE / "policy_samples.csv", delimiter=",", names=True)
obs = np.column_stack([actual[name] for name in actual.dtype.names[1:14]]).astype(np.float32)
recorded = np.column_stack((actual["raw_steer"], actual["raw_rear"]))
checks["bike_actor_replay"] = bool(np.allclose(infer(obs), recorded, atol=2e-5, rtol=1e-5))
checks["bike_velocity_estimator_reconstruction"] = bool(np.allclose(obs[:, 5], .07 * obs[:, 10], atol=4e-7))
checks["bike_lateral_velocity_zero"] = bool(np.all(obs[:, 6] == 0))
checks["synthetic_formula_parity_below_1e_6"] = max(summary["parity_probes"]["isaac_vs_mujoco_max_abs_per_channel"]) < 1e-6

for png in HERE.glob("*.png"):
    from PIL import Image
    with Image.open(png) as image:
        image.verify()
    checks[png.name + "_image_valid"] = True
result = {"checks": checks, "all_passed": all(checks.values())}
(HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not result["all_passed"]:
    raise SystemExit(1)
