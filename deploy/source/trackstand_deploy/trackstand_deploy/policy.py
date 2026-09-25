"""Validated ONNX Runtime wrapper for the deployed deterministic actor."""

from __future__ import annotations

import numpy as np

from .contracts import ACTION_SIZE, OBSERVATION_SIZE


class OnnxPolicy:
    def __init__(self, model_path: str):
        import onnxruntime as ort

        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "observation" or inputs[0].shape[-1] != OBSERVATION_SIZE:
            raise ValueError("ONNX input must be observation[batch,13]")
        if len(outputs) != 1 or outputs[0].name != "action" or outputs[0].shape[-1] != ACTION_SIZE:
            raise ValueError("ONNX output must be action[batch,2]")

    def act(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(value)):
            raise ValueError("observation must contain 13 finite float values")
        action = self._session.run(["action"], {"observation": value.reshape(1, -1)})[0][0]
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
            raise RuntimeError("policy produced an invalid action")
        return action

