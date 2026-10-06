import ast
import hashlib
from pathlib import Path

import numpy as np

from trackstand_deploy.policy import OnnxPolicy


def test_deployed_onnx_contract_and_determinism():
    model = Path(__file__).parents[1] / "models" / "trackstand_rl_rsl_2026-09-22_19-45-06_model_91850.onnx"
    policy = OnnxPolicy(str(model))
    observation = np.zeros(13, dtype=np.float32)
    observation[1] = -1.0
    observation[12] = 1.0
    first = policy.act(observation)
    second = policy.act(observation)
    assert first.shape == (2,)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_default_policy_filename_and_checksum_match_the_bundled_model():
    package = Path(__file__).parents[1]
    tree = ast.parse((package / "trackstand_deploy" / "policy_controller.py").read_text())
    constants = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in ("MODEL_FILENAME", "MODEL_SHA256"):
                    constants[target.id] = ast.literal_eval(statement.value)
    assert "2026-09-22_19-45-06_model_91850" in constants["MODEL_FILENAME"]
    model = package / "models" / constants["MODEL_FILENAME"]
    assert hashlib.sha256(model.read_bytes()).hexdigest() == constants["MODEL_SHA256"]
