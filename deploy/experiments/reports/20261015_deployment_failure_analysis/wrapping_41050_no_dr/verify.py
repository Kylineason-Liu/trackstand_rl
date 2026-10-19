"""Independently verify 41050 wrapping captures and the unmodified-env baseline."""

import importlib.util
import csv
import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("wrap41050_verification", HERE / "run.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=HERE / "results")
OUT = parser.parse_args().results.resolve()
summary = json.loads((OUT / "summary.json").read_text())
checks = {}

torch.set_num_threads(1)
sd = torch.load(study.PT, map_location="cpu", weights_only=False)["actor_state_dict"]
actor = torch.nn.Sequential(torch.nn.Linear(13, 32), torch.nn.ELU(), torch.nn.Linear(32, 32),
                            torch.nn.ELU(), torch.nn.Linear(32, 2))
actor.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")}, strict=True)
actor.eval()
max_actor_error = 0.
for path in OUT.glob("*.npz"):
    data = dict(np.load(path))
    mask = data["valid"]
    obs = data["observation"][mask]
    action = data["raw_action"][mask]
    checks[path.stem + "_valid_samples_finite"] = bool(
        np.isfinite(obs).all() and np.isfinite(action).all() and np.isfinite(data["state_before"][mask]).all())
    with torch.no_grad():
        predicted = actor(torch.from_numpy(obs)).numpy()
    error = float(np.abs(predicted - action).max())
    max_actor_error = max(max_actor_error, error)
    checks[path.stem + "_checkpoint_actor_matches_recorded_onnx"] = bool(np.allclose(predicted, action, atol=2e-5, rtol=1e-5))
    if path.stem.startswith("nominal_"):
        mode = path.stem.removeprefix("nominal_")
        metric = summary["nominal"][mode]
    else:
        prefix, mode = path.stem.split("_", 2)[1:]
        metric = summary["boundary_stress"][str(float(prefix))][mode]
    checks[path.stem + "_sample_count"] = int(mask.sum()) == metric["valid_policy_samples"]
    checks[path.stem + "_episode_count"] = int(data["done"].sum()) == metric["completed_episodes"]
    raw_oob = np.abs(action * [1.57, 6.28]) > np.array([np.pi / 2, 6.28]) + 1e-5
    checks[path.stem + "_oob_counts"] = raw_oob.sum(axis=0).tolist() == metric["oob_count"]
    native = data["state_before"][..., summary["rear_qpos_index"]][mask]
    encoded = study.encode(native, metric["encoding"]).astype(np.float32)
    checks[path.stem + "_only_requested_angle_mapping"] = bool(np.array_equal(encoded, obs[:, 8]))
    lengths = np.zeros(mask.shape[1], dtype=int)
    completed = []
    for t in range(len(mask)):
        lengths[mask[t]] += 1
        completed.extend(lengths[data["done"][t]].tolist())
        lengths[data["done"][t]] = 0
    checks[path.stem + "_episode_lengths"] = sorted(completed) == sorted(e["length_steps"] for e in metric["episodes"])

model = study.make_model()
policy = study.DeployedPolicy(onnx_path=str(study.ONNX), pt_path=str(study.PT))
options = study.ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
policy._ort_sess = study.ort.InferenceSession(str(study.ONNX), options, providers=["CPUExecutionProvider"])
for seed in (42, 105):
    plain = study.BikeBalanceEnv(model=model, seed=seed, enable_dr=False)
    wrapped = study.EncodingEnv(model=model, seed=seed, enable_dr=False, encoding="wrapped_pi")
    left, right = plain.reset(), wrapped.reset()
    exact = np.array_equal(left, right)
    for _ in range(128):
        actions = policy.act_batch(np.stack((left, right)))
        left, ld = plain.step(actions[0]); right, rd = wrapped.step(actions[1])
        exact &= (ld == rd and np.array_equal(left, right)
                  and np.array_equal(plain.data.qpos, wrapped.data.qpos)
                  and np.array_equal(plain.data.qvel, wrapped.data.qvel))
        if ld:
            break
    checks[f"wrapped_subclass_matches_unmodified_env_seed_{seed}"] = bool(exact)

for phase, comparisons in summary["boundary_comparisons"].items():
    for mode, comparison in comparisons.items():
        checks[f"phase_{phase}_{mode}_same_initial_physical_state"] = comparison["initial_physical_states_identical"]
        checks[f"phase_{phase}_{mode}_same_initial_other_12_channels"] = comparison["initial_other_12_channels_identical"]

for mode, comparison in summary["nominal_comparisons"].items():
    checks[mode + "_nominal_same_initial_physical_state"] = comparison["initial_physical_states_identical"]

checks["encoding_examples"] = bool(np.isclose(study.encode(3.5, "wrapped_pi"), 3.5 - 2 * np.pi)
                                    and study.encode(3.5, "native_unwrapped") == 3.5
                                    and np.isclose(study.encode(7.1, "candidate_4pi"), 7.1 - 4 * np.pi))
integrity = json.loads((OUT / "source_integrity.json").read_text())
checks["protected_sources_models_and_records_unchanged"] = not integrity["changed_files"]
with (OUT / "first_encoding_differences.csv").open() as stream:
    first_differences = list(csv.DictReader(stream))
checks["96_first_encoding_differences_recorded"] = len(first_differences) == 96
checks["physical_states_other_channels_prefix_identical_at_first_difference"] = all(
    row[field] == "True" for row in first_differences for field in
    ("physical_state_identical", "other_12_channels_identical", "preceding_actions_and_states_identical"))
checks["phase_2_8_same_start_then_later_divergence"] = all(
    int(row["first_different_tick"]) > 0 for row in first_differences if row["initial_phase_rad"] == "2.8")
for path in OUT.glob("*.png"):
    from PIL import Image
    with Image.open(path) as image:
        image.verify()
    checks[path.name + "_valid"] = True
result = {"checks": checks, "all_passed": all(checks.values()),
          "max_checkpoint_vs_onnx_error": max_actor_error}
(OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not result["all_passed"]:
    raise SystemExit(1)
