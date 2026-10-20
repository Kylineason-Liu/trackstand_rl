"""Observe pre-clamp v0 values while running the unchanged play_eval.py.

The original observation function bytecode runs once per ordinary invocation.
A private torch proxy records its nan_to_num/clamp arguments and forwards both
operations unchanged. No extra observation calls, RNG draws or unclipped policy
inputs are introduced. All ordinary evaluator output goes to a temporary copy
of the checkpoint and is retained here, never in the original training run.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
import types

TRAIN = Path(__file__).resolve().parents[2]
ROOT = TRAIN.parent
SCRIPT = TRAIN / "scripts/rsl_rl/play_eval.py"
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "run")
probe_args, eval_args = parser.parse_known_args()
OUT = probe_args.output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
invocation = sys.argv.copy()


def manifest():
    paths = []
    directories = (
        TRAIN / "source", TRAIN / "scripts", TRAIN / "tests", ROOT / "deploy/source", ROOT / "mujoco_sim2sim/src",
    )
    for directory in directories:
        paths.extend(directory.rglob("*.py"))
    paths.extend(ROOT / name for name in ("AGENTS.md", "progress_record.md", "progress_record.txt"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}


before = manifest()
sys.path[:0] = [str(TRAIN), str(SCRIPT.parent)]
sys.argv = [str(SCRIPT), *eval_args]
namespace = {"__name__": "__observation_audit_play_eval__", "__file__": str(SCRIPT)}
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)

import numpy as np
import torch
from isaaclab.utils.io import dump_yaml

FIELDS = (
    ("gravity_y", "dimensionless"), ("gravity_z", "dimensionless"),
    ("ang_vel_x", "rad/s"), ("ang_vel_y", "rad/s"), ("ang_vel_z", "rad/s"),
    ("lin_vel_x", "m/s"), ("lin_vel_y", "m/s"),
    ("steering_pos", "rad"), ("drive_pos", "rad"),
    ("steering_vel", "rad/s"), ("drive_vel", "rad/s"),
    ("sin_steer", "dimensionless"), ("cos_steer", "dimensionless"),
)
raw_calls, sanitized_calls, clamped_calls, policy_inputs, policy_call_ids = [], [], [], [], []
original_make = namespace["gym"].make
original_runner = namespace["OnPolicyRunner"]
temporary = None


class TorchTap:
    def __getattr__(self, name):
        return getattr(torch, name)

    def nan_to_num(self, data, *args, **kwargs):
        assert data.ndim == 2 and data.shape[1] == 13
        raw_calls.append(data.detach().clone())
        output = torch.nan_to_num(data, *args, **kwargs)
        sanitized_calls.append(output.detach().clone())
        return output

    def clamp(self, data, *args, **kwargs):
        assert args == (-100.0, 100.0) and not kwargs
        output = torch.clamp(data, *args, **kwargs)
        clamped_calls.append(output.detach().clone())
        return output


def observed_make(*args, **kwargs):
    cfg = kwargs["cfg"]
    term = cfg.observations["policy"].imu
    original = term.func
    assert original.__name__ == "balance_imu_state"
    dump_yaml(str(OUT / "effective_env.yaml"), cfg)
    private_globals = dict(original.__globals__, torch=TorchTap())
    observed = types.FunctionType(
        original.__code__, private_globals, original.__name__, original.__defaults__, original.__closure__,
    )
    observed.__annotations__, observed.__kwdefaults__ = original.__annotations__, original.__kwdefaults__
    assert observed.__code__ is original.__code__
    term.func = observed
    return original_make(*args, **kwargs)


class PolicyTap:
    def __init__(self, policy):
        self.policy = policy

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def __call__(self, observations):
        policy_inputs.append(observations["policy"].detach().clone())
        policy_call_ids.append(len(raw_calls)-1)
        if len(policy_inputs) % 500 == 0:
            print(f"[OBS AUDIT] policy steps {len(policy_inputs)}/{namespace['args_cli'].num_steps}", flush=True)
        return self.policy(observations)


def observed_runner(env, config, *args, **kwargs):
    dump_yaml(str(OUT / "effective_agent.yaml"), config)
    runner = original_runner(env, config, *args, **kwargs)
    get_policy = runner.get_inference_policy

    def get_observed_policy(*args, **kwargs):
        return PolicyTap(get_policy(*args, **kwargs))

    runner.get_inference_policy = get_observed_policy
    return runner


def statistics(values):
    finite = np.isfinite(values)
    selected = values[finite]
    return {
        "min": float(selected.min()) if selected.size else None,
        "max": float(selected.max()) if selected.size else None,
        "max_abs": float(np.max(np.abs(selected))) if selected.size else None,
        "above_100": int(np.count_nonzero(selected > 100.0)),
        "below_minus_100": int(np.count_nonzero(selected < -100.0)),
        "abs_above_100": int(np.count_nonzero(np.abs(selected) > 100.0)),
        "nonfinite": int(np.count_nonzero(~finite)),
        "samples": int(values.size),
    }


try:
    cli = namespace["args_cli"]
    assert cli.task == "Trackstand-Balance-v0" and cli.checkpoint
    source_checkpoint = Path(cli.checkpoint).resolve()
    before[str(source_checkpoint)] = hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
    temporary = tempfile.TemporaryDirectory(prefix="trackstand_observation_caps_")
    local_checkpoint = Path(temporary.name) / source_checkpoint.name
    shutil.copy2(source_checkpoint, local_checkpoint)
    cli.checkpoint = str(local_checkpoint)
    namespace["gym"].make, namespace["OnPolicyRunner"] = observed_make, observed_runner
    namespace["main"]()
    assert len(policy_inputs) == cli.num_steps
    assert len(raw_calls) == len(sanitized_calls) == len(clamped_calls)
    raw = torch.stack(raw_calls).cpu().numpy()
    sanitized = torch.stack(sanitized_calls).cpu().numpy()
    clamped = torch.stack(clamped_calls).cpu().numpy()
    actual = torch.stack(policy_inputs).cpu().numpy()
    call_ids = np.array(policy_call_ids)
    selected_raw = raw[call_ids]
    np.testing.assert_array_equal(sanitized, np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0))
    np.testing.assert_array_equal(clamped, np.clip(sanitized, -100.0, 100.0))
    # Only the first five channels receive manager white noise; confirm alignment.
    np.testing.assert_array_equal(actual[:, :, 5:], clamped[call_ids, :, 5:])
    rows, channels = [], []
    for column, (name, unit) in enumerate(FIELDS):
        pre = statistics(selected_raw[:, :, column])
        all_calls = statistics(raw[:, :, column])
        policy = statistics(actual[:, :, column])
        channels.append({"index_1based": column+1, "name": name, "unit": unit,
                         "preclamp_for_policy_inputs": pre, "all_observation_calls_preclamp": all_calls,
                         "actual_policy_input": policy})
        rows.append({"index": column+1, "name": name, "unit": unit, "raw_min": pre["min"],
                     "raw_max": pre["max"], "raw_max_abs": pre["max_abs"],
                     "raw_above_100": pre["above_100"], "raw_below_minus_100": pre["below_minus_100"],
                     "raw_nonfinite": pre["nonfinite"], "samples_per_channel": pre["samples"],
                     "actual_policy_input_max_abs": policy["max_abs"],
                     "actual_policy_input_abs_above_100": policy["abs_above_100"]})
    with (OUT / "observation_ranges.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(OUT / "observation_samples.npz", raw_before_sanitize_all_calls=raw,
                        sanitized_before_clamp_all_calls=sanitized, clamped_all_calls=clamped,
                        actual_policy_inputs=actual, policy_observation_call_ids=call_ids,
                        channel_names=np.array([field[0] for field in FIELDS]))
    summary = {
        "passed": True, "command": invocation, "task": cli.task, "checkpoint": str(source_checkpoint),
        "checkpoint_sha256": before[str(source_checkpoint)], "num_envs": cli.num_envs,
        "policy_steps": cli.num_steps, "seed": cli.seed, "domain_randomization": cli.dr,
        "pure_survival": cli.pure_survival, "policy_rate_hz": 50, "physics_rate_hz": 400,
        "observation_function_calls": len(raw_calls), "policy_input_batches": len(policy_inputs),
        "samples_per_channel_used_by_policy": int(selected_raw.shape[0]*selected_raw.shape[1]),
        "original_observation_bytecode_and_math_preserved": True,
        "no_extra_observation_or_rng_calls": True, "policy_still_receives_original_clipped_noisy_inputs": True,
        "channels": channels,
        "scope": "Current v0 and one checkpoint/seed/DR regime; policy observation times, not every physics substep",
    }
    (OUT / "observation_statistics.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([row["name"] for row in rows], [row["raw_max_abs"] for row in rows], color="#2878b5")
    ax.axhline(100.0, color="#c43c39", linestyle="--", label="shared numeric clamp threshold: 100")
    ax.set_yscale("log")
    ax.set_ylabel("Max |observation| (mixed units)")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"v0 / {source_checkpoint.stem} / {'DR' if cli.dr else 'no DR'} / seed {cli.seed}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "observation_maxima.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    for artifact in ("eval.txt", "torque_profile.png", "torque_profile.npz"):
        shutil.copy2(Path(temporary.name) / artifact, OUT / artifact)
    print("[OBS AUDIT] PASS " + json.dumps(rows), flush=True)
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    namespace["gym"].make, namespace["OnPolicyRunner"] = original_make, original_runner
    after = manifest()
    if "source_checkpoint" in globals():
        after[str(source_checkpoint)] = hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
    changed = {path: [digest, after.get(path)] for path, digest in before.items() if after.get(path) != digest}
    integrity = {"checked_files": len(before), "changed_files": changed}
    (OUT / "source_integrity.json").write_text(json.dumps(integrity, indent=2)+"\n")
    if temporary is not None:
        temporary.cleanup()
    namespace["simulation_app"].close()
    if changed:
        raise RuntimeError(f"Source/checkpoint changed unexpectedly: {changed}")
