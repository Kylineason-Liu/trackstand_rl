"""Compare the new NumPy PI against the exact Isaac class, without starting Kit."""

import ast
from collections.abc import Sequence
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
ISAAC = ROOT / (
    "isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/"
    "mdp/stationary/velocity_pi.py"
)
NUMPY = ROOT / "mujoco_sim2sim/src/newbike_mujoco_v2/control.py"


def exact_reference():
    tree = ast.parse(ISAAC.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VelocityIntegral")
    namespace = {"torch": torch, "math": math, "Sequence": Sequence}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ISAAC), "exec"), namespace)
    return namespace["VelocityIntegral"]


def state(value, reference=False):
    def scalar(name):
        item = getattr(value, name)
        return item.item() if reference else item

    return np.array([scalar(name) for name in (
        "torque_nm", "error_rad", "previous_error", "initialized", "limited_updates",
    )], dtype=np.float64)


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("independent_v2_control", NUMPY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference_class = exact_reference()
    fixture = np.load(OUT / "pi_reference_fixture.npz")
    errors = fixture["error_rad_s"]
    resets = fixture["reset_before"]
    cases = []
    for gain, p_gain, label in ((3.3, 0.3, "current_v2"), (3.3, 0.0, "I_bound_without_P"),
                                (0.0, 0.3, "P_only_optional")):
        actual = module.VelocityIntegral(gain)
        reference = reference_class(torch.zeros((1, 1), dtype=torch.float64), gain)
        maximum = np.zeros(5)
        clip_count = 0
        for index, error in enumerate(errors):
            if resets[index]:
                actual.reset()
                reference.reset()
            actual.update(float(error), 0.005, p_gain, 6.28)
            reference.update(torch.tensor([[error]], dtype=torch.float64), 0.005,
                             torch.tensor([[p_gain]], dtype=torch.float64),
                             torch.tensor([[6.28]], dtype=torch.float64))
            left, right = state(actual), state(reference, reference=True)
            maximum = np.maximum(maximum, np.abs(left - right))
            np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12)
            assert abs(actual.torque_nm) <= 6.28
            clip_count += abs(actual.torque_nm) == 6.28
            if resets[index]:
                assert actual.torque_nm == 0.0
        cases.append(dict(name=label, gain_nm_per_rad=gain, p_nm_s_per_rad=p_gain,
                          updates=len(errors), resets=int(resets.sum()), max_absolute_state_error=maximum.tolist(),
                          exact_integral_bound_samples=clip_count))
    result = dict(passed=True, reference="Exact AST-extracted Isaac VelocityIntegral; CPU float64",
                  state_fields=fixture["state_fields"].tolist(), cases=cases,
                  sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (ISAAC, NUMPY)},
                  scope="Controller-law/state comparison only; no MuJoCo/Isaac simulation or hardware",
                  numpy_version=np.__version__, torch_version=torch.__version__)
    (OUT / "independent_pi_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
