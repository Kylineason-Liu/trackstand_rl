"""Check v2 deterministic inference against checkpoint weights and exact BetaMean."""

import ast
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT / "mujoco_sim2sim/src"))
from newbike_mujoco_v2 import contract as c
from newbike_mujoco_v2.policy import DeployedPolicy
from newbike_mujoco_v2.policy_contract import validate_policy_contract


def main():
    torch.set_num_threads(1)
    beta_source = ROOT / (
        "isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/"
        "agents/beta_distribution.py"
    )
    tree = ast.parse(beta_source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BetaMean")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(beta_source), "exec"), namespace)
    beta = namespace["BetaMean"]()
    checkpoint = torch.load(c.DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    weights = checkpoint["actor_state_dict"]
    policy = DeployedPolicy(pt_path=c.DEFAULT_CHECKPOINT)
    rng = np.random.default_rng(20261103)
    x = rng.normal(size=(1024, 11)).astype(np.float32)
    x[:, 0] *= 0.5
    x[:, 1] = -1.0
    x[:, 5] *= 1.57
    x[:, 6] *= 2.0
    x[:, 8] *= 10.0
    x[:, 9], x[:, 10] = np.sin(x[:, 5]), np.cos(x[:, 5])
    with torch.no_grad():
        value = torch.from_numpy(x)
        for index in (0, 2, 4):
            value = nn.functional.linear(value, weights[f"mlp.{index}.weight"], weights[f"mlp.{index}.bias"])
            if index < 4:
                value = nn.functional.elu(value)
        expected = beta(value.reshape(-1, 2, 2)).numpy()
    actual = policy.act_batch(x)
    np.testing.assert_array_equal(actual, expected)
    rejected = []
    invalid = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_00-41-38/model_75300.pt"
    try:
        validate_policy_contract(invalid)
    except ValueError as error:
        rejected.append(dict(case="v1 torque metadata", message=str(error)))
    else:
        raise AssertionError("V1 torque-action policy must be rejected despite matching Beta/11 shapes")
    for label, bad in (("nonfinite", np.full((1, 11), np.nan, dtype=np.float32)),
                       ("wrong input dimension", np.zeros((1, 13), dtype=np.float32))):
        try:
            policy.act_batch(bad)
        except ValueError as error:
            rejected.append(dict(case=label, message=str(error)))
        else:
            raise AssertionError(label)
    result = dict(passed=True, checkpoint=str(c.DEFAULT_CHECKPOINT),
                  checkpoint_sha256=hashlib.sha256(c.DEFAULT_CHECKPOINT.read_bytes()).hexdigest(),
                  architecture=[11, 32, 32, 4], activation="ELU", deterministic_transform="exact Isaac BetaMean",
                  tested_observations=len(x), maximum_absolute_action_error=float(np.max(np.abs(actual - expected))),
                  actual_action_minimum=actual.min(axis=0).tolist(), actual_action_maximum=actual.max(axis=0).tolist(),
                  metadata=policy.contract_metadata, rejected_cases=rejected,
                  scope="CPU inference and contract checks only; no physics or hardware",
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (beta_source, ROOT / "mujoco_sim2sim/src/newbike_mujoco_v2/policy.py",
                                              ROOT / "mujoco_sim2sim/src/newbike_mujoco_v2/policy_contract.py")})
    (OUT / "independent_policy_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
