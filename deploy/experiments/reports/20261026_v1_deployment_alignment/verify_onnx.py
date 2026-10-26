"""Verify packaged deployment inference against the actual Isaac BetaMean capture."""

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "deploy/source/trackstand_deploy"))

from trackstand_deploy.policy import OnnxPolicy

manifest = json.loads((ROOT / "deploy/source/trackstand_deploy/models/v1_policy_manifest.json").read_text())
model = ROOT / "deploy/source/trackstand_deploy/models" / manifest["onnx"]
assert hashlib.sha256(model.read_bytes()).hexdigest() == manifest["onnx_sha256"]
policy = OnnxPolicy(str(model))
reference = np.load(HERE / "policy_reference.npz")
started = time.perf_counter()
actions = np.array([policy.act(obs) for obs in reference["observations"]])
elapsed = time.perf_counter() - started
errors = np.abs(actions - reference["actions"])
worst = np.unravel_index(errors.argmax(), errors.shape)
assert errors.max() < 2e-5
result = {
    "inputs": len(actions), "reference": "Actual Isaac BetaMean, PyTorch float32 batched MLP",
    "runtime": "CPU ONNX Runtime single-observation deployment path, one intra/inter-op thread",
    "max_abs_action_error": float(errors.max()), "physical_state_inputs": 1000,
    "physical_state_max_abs_action_error": float(errors[:1000].max()),
    "remaining_inputs": "3096 unconstrained Gaussian stress vectors, including impossible gravity/sine values",
    "p99_abs_error": float(np.percentile(errors, 99)), "acceptance_abs_tolerance": 2e-5,
    "worst_input_index": int(worst[0]), "worst_channel": int(worst[1]),
    "actions_min": actions.min(axis=0).tolist(), "actions_max": actions.max(axis=0).tolist(),
    "model_sha256": manifest["onnx_sha256"], "cpu_single_observation_inference_total_s": elapsed,
    "note": "Float32 comparison, not bitwise equality, Pi timing, physics or hardware validation",
}
(HERE / "onnx_parity.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
