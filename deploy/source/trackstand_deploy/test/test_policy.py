from pathlib import Path

import numpy as np

from trackstand_deploy.policy import OnnxPolicy


def test_deployed_onnx_contract_and_determinism():
    model = Path(__file__).parents[1] / "models" / "trackstand_rl_rsl_2026-09-17_19-46-37_model_41050.onnx"
    policy = OnnxPolicy(str(model))
    observation = np.zeros(13, dtype=np.float32)
    observation[1] = -1.0
    observation[12] = 1.0
    first = policy.act(observation)
    second = policy.act(observation)
    assert first.shape == (2,)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)

