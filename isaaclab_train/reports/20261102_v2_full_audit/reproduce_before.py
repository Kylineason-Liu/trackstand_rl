"""Reproduce inherited pose/joint containment gaps before the v2 audit fixes."""

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

TRAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAIN))
PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"
terms = importlib.import_module(f"{PACKAGE}.mdp.stationary.terminations")
rewards = importlib.import_module(f"{PACKAGE}.mdp.stationary.rewards")
data = SimpleNamespace(root_pos_w=torch.zeros(4, 3), joint_pos=torch.zeros(4, 3),
                       root_lin_vel_b=torch.zeros(4, 3), root_ang_vel_b=torch.zeros(4, 3),
                       joint_vel=torch.zeros(4, 3))
data.root_pos_w[0, 0] = float("nan")
data.joint_pos[1, 0] = float("nan")
data.joint_vel[2, 0] = float("nan")
env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
env.scene = type("Scene", (dict,), {})(env.scene)
env.scene.env_origins = torch.zeros(4, 3)
result = dict(injected=["NaN root X position", "NaN joint position", "NaN joint speed", "healthy"],
              inherited_diverged=terms.nonfinite_or_extreme_state(env).tolist(),
              inherited_drift_reward_finite=torch.isfinite(rewards.balance_drift_gaussian(env)).tolist())
assert result["inherited_diverged"] == [False, False, True, False]
assert result["inherited_drift_reward_finite"] == [False, True, True, True]
(Path(__file__).parent / "before_numerical_gaps.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
