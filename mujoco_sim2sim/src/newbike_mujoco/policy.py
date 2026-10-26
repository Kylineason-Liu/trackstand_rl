"""Deterministic v1 Beta inference, ONNX or independently reconstructed PyTorch.

13-input Gaussian checkpoints (41050/91850) are deliberately rejected: they
cannot run the v1 observation/action contract. No action scale is baked here.
"""

import re

import numpy as np

from .contract import OBSERVATIONS


class DeployedPolicy:
    def __init__(self, onnx_path=None, pt_path=None):
        self.backend, self._ort_sess, self._torch_net = None, None, None
        self.path = None
        if onnx_path is not None:
            try:
                import onnxruntime as ort
            except ImportError:
                if pt_path is None:
                    raise RuntimeError("Install onnxruntime or supply a v1 --pt checkpoint") from None
            else:
                self._ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                inputs, outputs = self._ort_sess.get_inputs(), self._ort_sess.get_outputs()
                if (len(inputs) != 1 or len(outputs) != 1 or len(inputs[0].shape) != 2
                        or inputs[0].shape[1] != len(OBSERVATIONS) or len(outputs[0].shape) != 2
                        or outputs[0].shape[1] != 2 or inputs[0].type != "tensor(float)"):
                    raise ValueError("V1 ONNX must map float32 (N,11) observations to (N,2) Beta means")
                self._input_name = inputs[0].name
                self.backend, self.path = "onnxruntime", str(onnx_path)
        if self.backend is None and pt_path is not None:
            self._load_torch(pt_path)
            self.backend, self.path = "torch", str(pt_path)
        if self.backend is None:
            raise ValueError("Supply a Trackstand-Balance-v1 ONNX or PT policy")

    def _load_torch(self, path):
        import torch
        from torch import nn

        state = dict(torch.load(path, map_location="cpu", weights_only=True)["actor_state_dict"])
        marker = state.pop("distribution._beta_policy_version", None)
        if marker is None or marker.numel() != 1 or marker.item() != 1:
            raise ValueError("Expected v1 Beta checkpoint marker version1; Gaussian/v0 is incompatible")
        # Current v1 explicitly disables normalization. Never silently discard one.
        if any(key.startswith("obs_normalizer.") for key in state):
            raise ValueError("Current v1 uses no observation normalizer; this checkpoint has a different contract")
        if any(not re.fullmatch(r"mlp\.\d+\.(weight|bias)", key) for key in state):
            raise ValueError("Unsupported actor state keys; refusing to guess the inference transform")
        indices = sorted({int(key.split(".")[1]) for key in state})
        if not indices or indices != list(range(0, 2*len(indices), 2)):
            raise ValueError("Expected contiguous Linear/ELU v1 actor layers")
        layers = []
        for index in indices:
            weight, bias = state[f"mlp.{index}.weight"], state[f"mlp.{index}.bias"]
            if weight.ndim != 2 or bias.shape != (weight.shape[0],):
                raise ValueError("Malformed actor linear layer")
            layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
            if index != indices[-1]:
                layers.append(nn.ELU())
        if layers[0].in_features != len(OBSERVATIONS) or layers[-1].out_features != 4:
            raise ValueError("V1 Beta actor needs 11 inputs and four shape logits for two actions")
        self._torch_net = nn.Sequential(*layers).eval()
        self._torch_net.load_state_dict({key.removeprefix("mlp."): value for key, value in state.items()}, strict=True)

    def act(self, obs):
        return self.act_batch(np.asarray(obs, dtype=np.float32).reshape(1, -1))[0]

    def act_batch(self, observations):
        x = np.asarray(observations, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(OBSERVATIONS) or len(x) < 1 or not np.isfinite(x).all():
            raise ValueError("Expected a finite nonempty (N,11) v1 observation batch")
        if self.backend == "onnxruntime":
            batch = self._ort_sess.get_inputs()[0].shape[0]
            if isinstance(batch, int) and batch > 0:
                # Fixed batches of any size: pad only the last batch, discard its padding.
                chunks = []
                for start in range(0, len(x), batch):
                    chunk = x[start:start+batch]
                    padded = np.zeros((batch, x.shape[1]), dtype=np.float32)
                    padded[:len(chunk)] = chunk
                    chunks.append(self._ort_sess.run(None, {self._input_name: padded})[0][:len(chunk)])
                out = np.concatenate(chunks)
            else:
                out = self._ort_sess.run(None, {self._input_name: x})[0]
        else:
            import torch
            with torch.no_grad():
                logits = self._torch_net(torch.from_numpy(x)).reshape(-1, 2, 2)
                shapes = torch.nn.functional.softplus(logits)+1.
                out = (2.*(shapes[:, 0]/(shapes[:, 0]+shapes[:, 1]))-1.).numpy()
        if out.shape != (len(x), 2) or not np.isfinite(out).all() or np.any(np.abs(out) > 1.):
            raise ValueError("Policy did not return two finite bounded v1 actions; no post-hoc clipping applied")
        return np.asarray(out, dtype=np.float32)
