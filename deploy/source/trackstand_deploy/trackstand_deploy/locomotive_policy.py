"""Validated ONNX Runtime wrapper for the 8-in/2-out CycleRL riding policies."""

from __future__ import annotations

import numpy as np

from .locomotive_contracts import ACTION_SIZE, OBSERVATION_SIZE, LocomotiveModelSpec


# Gaussian means are unbounded; a value this large means the model is not the trained actor.
RAW_ACTION_SANITY_LIMIT = 1.0e3


class LocomotiveOnnxPolicy:
    def __init__(self, model_path: str, spec: LocomotiveModelSpec):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "observation" or len(inputs[0].shape) != 2
                or inputs[0].shape[-1] != OBSERVATION_SIZE or inputs[0].type != "tensor(float)"
                or isinstance(inputs[0].shape[0], int) and inputs[0].shape[0] != 1):
            raise ValueError("locomotive ONNX input must be float32 observation[batch,8]")
        if (len(outputs) != 1 or outputs[0].name != "action" or len(outputs[0].shape) != 2
                or outputs[0].shape[-1] != ACTION_SIZE or outputs[0].type != "tensor(float)"):
            raise ValueError("locomotive ONNX output must be action[batch,2]")
        # Reproduce the exporter's skrl reference outputs before any motor can move.
        observations = np.asarray(spec.reference_observations, dtype=np.float32)
        expected = np.asarray(spec.reference_raw_actions, dtype=np.float32)
        actual = self._session.run(["action"], {"observation": observations})[0]
        if actual.shape != expected.shape or float(np.max(np.abs(actual - expected))) > spec.reference_tolerance:
            raise ValueError("locomotive ONNX does not reproduce its manifest reference actions")

    def act(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(value)):
            raise ValueError("observation must contain 8 finite float values")
        output = np.asarray(self._session.run(["action"], {"observation": value.reshape(1, -1)})[0])
        if (output.shape != (1, ACTION_SIZE) or not np.all(np.isfinite(output))
                or np.any(np.abs(output) > RAW_ACTION_SANITY_LIMIT)):
            raise RuntimeError("locomotive policy produced a non-finite or implausible raw action")
        return output[0].astype(np.float32)
