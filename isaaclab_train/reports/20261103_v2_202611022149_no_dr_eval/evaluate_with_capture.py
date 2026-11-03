"""Run unchanged play_eval.py with observation-only provenance/episode capture."""

import atexit
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import runpy
import sys
import time

TRAIN = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = TRAIN / "logs/rsl_rl/trackstand_rl_v2_rsl/2026-11-02_21-49-32"
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(TRAIN / "scripts/rsl_rl"))

from isaaclab.app import AppLauncher
import isaaclab.app


def normalized(value):
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {"__slice__": [value.start, value.stop, value.step]}
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    return value


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            name = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                result.append({"path": name, "saved": left.get(key), "resolved": right.get(key)})
            else:
                result += differences(left[key], right[key], name)
        return result
    if left != right:
        return [{"path": path, "saved": left, "resolved": right}]
    return []


def dump_config(kind, value, permitted_paths):
    import yaml

    class SavedConfigLoader(yaml.FullLoader):
        pass

    SavedConfigLoader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:builtins.slice",
        lambda loader, node: slice(*loader.construct_sequence(node)),
    )
    resolved = normalized(value)
    saved = normalized(yaml.load((RUN / f"params/{kind}.yaml").read_text(), Loader=SavedConfigLoader))
    diff = differences(saved, resolved)
    (OUT / f"resolved_{kind}.json").write_text(json.dumps(resolved, indent=2) + "\n")
    (OUT / f"{kind}_config_comparison.json").write_text(json.dumps(diff, indent=2) + "\n")
    unexpected = [d for d in diff if d["path"] not in permitted_paths]
    if unexpected:
        raise RuntimeError(f"Unexpected {kind} config changes versus training YAML: {unexpected}")
    print(f"[CAPTURE] {kind} saved/resolved config checked; allowed differences: {diff}", flush=True)


class CaptureLauncher(AppLauncher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_capture()


def install_capture():
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for name in ("isaaclab", "isaacsim", "rsl-rl-lib", "gymnasium", "tensorboard", "warp-lang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions.update(cuda_runtime=torch.version.cuda, cuda_device=torch.cuda.get_device_name(0))
    (OUT / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    original_make = gym.make

    def captured_make(id, *args, **kwargs):
        result = original_make(id, *args, **kwargs)
        if id == "Trackstand-Balance-v2":
            # Training saves cfg after construction. Compare at the same point:
            # construction resolves prim/spawn paths, terrain count and camera metadata.
            dump_config("env", result.unwrapped.cfg.to_dict(), {"scene.num_envs", "scene.terrain.num_envs", "log_dir"})
        return result

    gym.make = captured_make
    original_runner_init = OnPolicyRunner.__init__

    def captured_runner_init(self, env, train_cfg, *args, **kwargs):
        dump_config("agent", train_cfg, {"load_checkpoint"})
        return original_runner_init(self, env, train_cfg, *args, **kwargs)

    OnPolicyRunner.__init__ = captured_runner_init
    original_step = RslRlVecEnvWrapper.step
    original_close = RslRlVecEnvWrapper.close
    state = {"steps": 0, "episodes": [], "finished": False}

    def captured_step(self, actions):
        if "lengths" not in state:
            state["lengths"] = np.zeros(self.num_envs, dtype=np.int64)
            state["returns"] = np.zeros(self.num_envs, dtype=np.float64)
            state["episode_index"] = np.zeros(self.num_envs, dtype=np.int64)
            state["term_names"] = list(self.unwrapped.termination_manager.active_terms)
            state["step_dt"] = self.unwrapped.step_dt
            state["start_monotonic"] = time.monotonic()
        result = original_step(self, actions)
        _, rewards, dones, extras = result
        state["steps"] += 1
        state["lengths"] += 1
        state["returns"] += rewards.detach().cpu().numpy().reshape(-1)
        ended = dones.detach().cpu().numpy().reshape(-1).astype(bool)
        if ended.any():
            flags = self.unwrapped.termination_manager._last_episode_dones.detach().cpu().numpy()
            timeout = extras["time_outs"].detach().cpu().numpy().astype(bool)
            for env_id in np.flatnonzero(ended):
                state["episodes"].append({
                    "env_id": int(env_id),
                    "episode_index": int(state["episode_index"][env_id]),
                    "end_policy_step": state["steps"],
                    "length_steps": int(state["lengths"][env_id]),
                    "length_seconds": float(state["lengths"][env_id] * state["step_dt"]),
                    "return": float(state["returns"][env_id]),
                    "timeout": bool(timeout[env_id]),
                    **{name: bool(flags[env_id, i]) for i, name in enumerate(state["term_names"])},
                })
            state["lengths"][ended] = 0
            state["returns"][ended] = 0
            state["episode_index"][ended] += 1
        return result

    def save_capture():
        if state["finished"] or "lengths" not in state:
            return
        state["finished"] = True
        episodes = state["episodes"]
        if episodes:
            with (OUT / "completed_episodes.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
                writer.writeheader()
                writer.writerows(episodes)
        summary = {
            "policy_steps": state["steps"],
            "num_envs": len(state["lengths"]),
            "environment_transitions": state["steps"] * len(state["lengths"]),
            "step_dt": state["step_dt"],
            "completed_episodes": len(episodes),
            "first_episodes_completed": sum(e["episode_index"] == 0 for e in episodes),
            "timeouts": sum(e["timeout"] for e in episodes),
            "termination_counts": {name: sum(e[name] for e in episodes) for name in state["term_names"]},
            "overlapping_termination_episodes": sum(sum(e[name] for name in state["term_names"]) > 1 for e in episodes),
            "censored_trajectory_count": int((state["lengths"] > 0).sum()),
            "final_in_progress_lengths_steps": state["lengths"].tolist(),
            "final_episode_indices": state["episode_index"].tolist(),
            "evaluation_loop_wall_seconds": time.monotonic() - state["start_monotonic"],
            "capture_does_not_modify_actions_or_step_results": True,
        }
        if episodes:
            lengths = np.array([e["length_steps"] for e in episodes])
            summary.update(mean_length_steps=float(lengths.mean()), min_length_steps=int(lengths.min()),
                           max_length_steps=int(lengths.max()), std_length_steps=float(lengths.std()))
        (OUT / "episode_capture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("[CAPTURE] Per-episode records and censored trajectories saved.", flush=True)

    def captured_close(self):
        save_capture()
        return original_close(self)

    RslRlVecEnvWrapper.step = captured_step
    RslRlVecEnvWrapper.close = captured_close
    atexit.register(save_capture)


isaaclab.app.AppLauncher = CaptureLauncher
sys.argv[0] = str(TRAIN / "scripts/rsl_rl/play_eval.py")
runpy.run_path(sys.argv[0], run_name="__main__")
