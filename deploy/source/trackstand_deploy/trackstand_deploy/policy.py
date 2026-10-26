"""Validated ONNX Runtime wrapper for the deployed deterministic actor."""

from __future__ import annotations

import numpy as np

from .contracts import ACTION_SIZE, OBSERVATION_SIZE


class OnnxPolicy:
    def __init__(self, model_path: str):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "observation" or len(inputs[0].shape) != 2
                or inputs[0].shape[-1] != OBSERVATION_SIZE or inputs[0].type != "tensor(float)"
                or isinstance(inputs[0].shape[0], int) and inputs[0].shape[0] != 1):
            raise ValueError("v1 ONNX input must be float32 observation[batch,11], accepting batch=1")
        if (len(outputs) != 1 or outputs[0].name != "action" or len(outputs[0].shape) != 2
                or outputs[0].shape[-1] != ACTION_SIZE or outputs[0].type != "tensor(float)"):
            raise ValueError("ONNX output must be action[batch,2]")

    def act(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(value)):
            raise ValueError("observation must contain 11 finite float values")
        output = np.asarray(self._session.run(["action"], {"observation": value.reshape(1, -1)})[0])
        if output.shape != (1, ACTION_SIZE) or not np.all(np.isfinite(output)) or np.any(np.abs(output) > 1.0):
            raise RuntimeError("v1 policy must produce two finite Beta actions within [-1, 1]")
        return output[0].astype(np.float32)

