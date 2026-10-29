"""The deployed policy artifact: ONNX preferred, .pt torch fallback.

Runs the exact exported ONNX (policies/model_500.onnx) when onnxruntime is
installed; otherwise reconstructs the deterministic actor from the rsl_rl
checkpoint with torch — the same mean-action path play_eval uses.
"""

import numpy as np


class DeployedPolicy:
    def __init__(self, onnx_path: str | None = None, pt_path: str | None = None):
        self.backend = None
        self._ort_sess = None
        self._torch_net = None

        if onnx_path is not None:
            try:
                import onnxruntime as ort
                self._ort_sess = ort.InferenceSession(
                    onnx_path, providers=["CPUExecutionProvider"])
                self.backend = "onnxruntime"
            except ImportError:
                pass

        if self.backend is None and pt_path is not None:
            import torch
            import torch.nn as nn
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            sd = dict(ckpt["actor_state_dict"])
            for k in list(sd):
                if k.startswith("distribution.") or k.startswith("obs_normalizer."):
                    sd.pop(k)
            seq = []
            idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
            for j, i in enumerate(idxs):
                w = sd[f"mlp.{i}.weight"]
                seq.append(nn.Linear(w.shape[1], w.shape[0]))
                if j < len(idxs) - 1:
                    seq.append(nn.ELU())
            net = nn.Sequential(*seq)
            state = {}
            for j, i in enumerate(idxs):
                state[f"{j * 2}.weight"] = sd[f"mlp.{i}.weight"]
                state[f"{j * 2}.bias"] = sd[f"mlp.{i}.bias"]
            net.load_state_dict(state, strict=True)
            net.eval()
            self._torch_net = net
            self.backend = "torch"

        if self.backend is None:
            raise RuntimeError("No usable policy backend: onnxruntime missing and no .pt given.")

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic mean action for a (13,) observation -> (2,) raw action."""
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return self.act_batch(x)[0]

    def act_batch(self, observations: np.ndarray) -> np.ndarray:
        """Deterministic actions for (N, 13), retaining fixed-batch-one ONNX support."""
        x = np.asarray(observations, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 13 or x.shape[0] < 1:
            raise ValueError("Expected a nonempty (N, 13) observation batch")
        if self.backend == "onnxruntime":
            batch_dim = self._ort_sess.get_inputs()[0].shape[0]
            if batch_dim == 1 and len(x) > 1:
                # Older exports have a fixed input batch; do not change the artifact.
                out = np.concatenate([
                    self._ort_sess.run(None, {"observation": row[None, :]})[0] for row in x
                ], axis=0)
            else:
                out = self._ort_sess.run(None, {"observation": x})[0]
        else:
            import torch
            with torch.no_grad():
                out = self._torch_net(torch.from_numpy(x)).numpy()
        return out
