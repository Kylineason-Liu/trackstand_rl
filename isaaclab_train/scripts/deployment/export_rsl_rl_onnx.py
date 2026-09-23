#!/usr/bin/env python3
"""Export an rsl_rl PPO checkpoint to ONNX for embedded deployment.

Usage:
    ~/IsaacLab/isaaclab.sh -p scripts/deployment/export_rsl_rl_onnx.py \
        --checkpoint logs/rsl_rl/newbike_balance_rsl/2026-09-09_09-50-48/model_500.pt

The script:
  1. Loads an rsl_rl OnPolicyRunner checkpoint (model_*.pt) and takes its
     ``actor_state_dict`` — the deterministic inference path (mean actions),
     the same path ``play_eval`` uses.
  2. Reconstructs the MLP architecture from the state dict itself (no
     hardcoded layer sizes); ELU between the Linear layers, per this repo's
     agent config.
  3. Bakes in observation normalization ONLY if the checkpoint carries one
     (``obs_normalizer.running_mean`` etc.). The 09090950 baseline has none
     (``obs_normalization: false``), so its ONNX takes raw 13-dim obs.
  4. Exports ONNX (opset 18, dynamic batch) and verifies numeric fidelity
     PyTorch vs onnxruntime on random inputs.

Action scaling is NOT baked in — the ONNX outputs raw policy means and the
embedding side applies the env's action scales (for NewBike-Balance-v1:
steer x 1.57 rad position target, drive x 10.0 rad/s velocity target).
"""

import argparse
import os

import torch
import torch.nn as nn


class DeploymentPolicy(nn.Module):
    """Optional obs preprocessor + the policy MLP — the full inference graph."""

    def __init__(self, mlp: nn.Sequential, mean: torch.Tensor | None = None, var: torch.Tensor | None = None):
        super().__init__()
        self.mlp = mlp
        if mean is not None:
            assert var is not None  # guarded by the caller's mean/var parity check
            # matches rsl_rl EmpiricalNormalization: (x - mean) / sqrt(var + eps), eps = 1e-2
            self.register_buffer("mean", mean.to(torch.float32))
            self.register_buffer("inv_std", 1.0 / torch.sqrt(var.to(torch.float32) + 1e-2))
        else:
            self.mean = None
            self.inv_std = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.mean is not None:
            obs = (obs - self.mean) * self.inv_std
        return self.mlp(obs)


def main():
    parser = argparse.ArgumentParser(description="Export rsl_rl PPO policy to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl model_*.pt checkpoint.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX path. Default: scripts/deployment/<task>_<run>_<stem>.onnx")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version (13 fails under torch 2.10 dynamo: CastLike has no previous version).")
    args = parser.parse_args()

    # ---- 0. Derive output name ------------------------------------------------
    if args.output is None:
        cp = os.path.abspath(args.checkpoint)
        run_dir = os.path.basename(os.path.dirname(cp))                              # run dir
        task_dir = os.path.basename(os.path.dirname(os.path.dirname(cp)))            # task
        stem = os.path.splitext(os.path.basename(cp))[0]                             # model_500
        args.output = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"{task_dir}_{run_dir}_{stem}.onnx")
        print(f"[INFO] Output name derived from checkpoint: {args.output}")

    # ---- 1. Load checkpoint ----------------------------------------------------
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in ckpt:
        raise KeyError("Not an rsl_rl checkpoint: 'actor_state_dict' missing "
                       "(skrl best_agent.pt files need export_onnx.py instead).")
    print(f"[INFO] Checkpoint iteration: {ckpt.get('iter')}")

    sd = dict(ckpt["actor_state_dict"])

    # ---- 2. Strip distribution (deterministic export) ---------------------------
    dist_keys = [k for k in sd if k.startswith("distribution.")]
    for k in dist_keys:
        sd.pop(k)
    if dist_keys:
        print(f"[INFO] Dropped distribution params {dist_keys} — ONNX outputs deterministic means.")

    # ---- 3. Optional observation normalization -----------------------------------
    mean = sd.pop("obs_normalizer.running_mean", None)
    var = sd.pop("obs_normalizer.running_var", None)
    sd.pop("obs_normalizer.count", None)
    if (mean is None) != (var is None):
        raise ValueError("Checkpoint has mean/var mismatch in obs_normalizer.")
    print(f"[INFO] Observation normalization baked in: {mean is not None}")

    # ---- 4. Reconstruct MLP from the state dict ---------------------------------
    # remaining keys must be mlp.<i>.weight / mlp.<i>.bias with even indices
    layers: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
    for k in list(sd.keys()):
        if not k.startswith("mlp."):
            raise KeyError(f"Unexpected actor key: {k}")
        parts = k.split(".")
        if len(parts) != 3 or parts[2] not in ("weight", "bias") or int(parts[1]) % 2 != 0:
            raise KeyError(f"Unexpected actor key layout: {k}")
        idx = int(parts[1])
        w, b = layers.get(idx, (None, None))
        if parts[2] == "weight":
            w = sd.pop(k)
        else:
            b = sd.pop(k)
        layers[idx] = (w, b)

    idxs = sorted(layers)
    validated: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for i in idxs:
        w, b = layers[i]
        if w is None or b is None:
            raise KeyError(f"Incomplete mlp layer {i} in actor state dict.")
        validated[i] = (w, b)

    seq: list[nn.Module] = []
    for j, i in enumerate(idxs):
        w, _ = validated[i]
        seq.append(nn.Linear(w.shape[1], w.shape[0]))
        if j < len(idxs) - 1:
            seq.append(nn.ELU())  # this repo's agent cfg: activation elu
    mlp = nn.Sequential(*seq)
    mlp_state = {}
    for new_i, old_i in enumerate(idxs):
        mlp_state[f"{new_i * 2}.weight"] = validated[old_i][0]
        mlp_state[f"{new_i * 2}.bias"] = validated[old_i][1]
    mlp.load_state_dict(mlp_state, strict=True)

    input_dim = validated[idxs[0]][0].shape[1]
    output_dim = validated[idxs[-1]][0].shape[0]
    print(f"[INFO] Reconstructed MLP: {input_dim} -> "
          f"{' -> '.join(str(validated[i][0].shape[0]) for i in idxs[:-1])}"
          f" -> {output_dim} (ELU between layers)")

    deployment = DeploymentPolicy(mlp, mean, var)
    deployment.eval()

    # ---- 5. Export ONNX ----------------------------------------------------------
    dummy = torch.randn(1, input_dim)
    print(f"[INFO] Exporting ONNX (opset={args.opset}) to: {args.output}")
    torch.onnx.export(
        deployment,
        (dummy,),
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
    )
    print("[INFO] ONNX export successful.")

    # ---- 6. Sanity checks ---------------------------------------------------------
    import onnx
    model = onnx.load(args.output)
    onnx.checker.check_model(model)
    print("[INFO] ONNX model passes onnx.checker validation.")

    import numpy as np
    with torch.no_grad():
        pt_out = deployment(dummy).numpy()
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"observation": dummy.numpy()})[0]
        engine = "onnxruntime"
    except ImportError:
        from onnx.reference import ReferenceEvaluator
        ort_out = ReferenceEvaluator(model).run(None, {"observation": dummy.numpy()})[0]
        engine = "onnx.reference"
    err = float(np.max(np.abs(pt_out - ort_out)))
    print(f"[INFO] PyTorch <-> ONNX max absolute error ({engine}): {err:.2e}")
    if err > 1e-5:
        print("[WARN]  Error is larger than expected — verify numeric fidelity.")
    else:
        print("[INFO] Numeric fidelity verified.")
    print(f"[INFO] Action scales are NOT baked in — apply the env's scales on the embedding side"
          f" (NewBike-Balance-v1: steer x 1.57 rad pos, drive x 10.0 rad/s vel).")


if __name__ == "__main__":
    main()
