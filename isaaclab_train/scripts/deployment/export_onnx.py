#!/usr/bin/env python3
"""Export a skrl PPO Gaussian policy to ONNX for embedded deployment.

Usage:
    python scripts/deployment/export_onnx.py \
        --checkpoint logs/skrl/cyclerl_logs/NewBike-Balance-v0/<run>/checkpoints/best_agent.pt \
        --obs-dim 13 \
        --action-dim 2 \
        --output bicycle_policy.onnx

The script:
  1. Reconstructs the exact MLP architecture from the skrl PPO config
     ([512, 256, 128, 64] hidden layers, ELU activation).
  2. Loads the checkpoint and extracts policy weights + RunningStandardScaler.
  3. Wraps both into a single torch.nn.Module (preprocessor + policy net).
  4. Exports to ONNX with constant folding.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MLP matching the skrl PPO config
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """Feed-forward MLP: Linear -> ELU -> ... -> Linear (mean actions only)."""

    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RunningStandardScaler(nn.Module):
    """Stateless z-score normaliser from pre-trained statistics."""

    def __init__(self, mean: torch.Tensor, var: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", mean.to(torch.float32))
        self.register_buffer("inv_std", 1.0 / torch.sqrt(var.to(torch.float32) + eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.inv_std


class DeploymentPolicy(nn.Module):
    """Preprocessor + policy network — the full inference graph for ONNX."""

    def __init__(self, scaler: RunningStandardScaler, policy: PolicyNetwork):
        super().__init__()
        self.scaler = scaler
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy(self.scaler(obs))


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export skrl PPO policy to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--obs-dim", type=int, required=True, help="Observation dimension.")
    parser.add_argument("--action-dim", type=int, required=True, help="Action dimension.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX path. Default: derived from checkpoint path.")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[512, 256, 128, 64],
                        help="Hidden layer sizes (default matches skrl PPO config).")
    args = parser.parse_args()

    # ---- 0. Derive output name from checkpoint path ------------------------
    if args.output is None:
        cp = os.path.abspath(args.checkpoint)
        # path: .../<TaskName>-v0/<run_dir>/checkpoints/<file>.pt
        run_dir = os.path.basename(os.path.dirname(os.path.dirname(cp)))      # run dir
        task_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(cp))))  # task
        args.output = f"{task_dir}_{run_dir}.onnx"
        print(f"[INFO] Output name derived from checkpoint: {args.output}")

    # ---- 1. Load checkpoint ------------------------------------------------
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # ---- 2. Build policy network -------------------------------------------
    policy = PolicyNetwork(args.obs_dim, args.hidden_layers, args.action_dim)
    policy_state = ckpt["policy"]

    # Remove 'log_std_parameter' — it is NOT part of the forward pass
    # (the ONNX model outputs deterministic mean actions).
    policy_state = {k: v for k, v in policy_state.items() if k != "log_std_parameter"}
    # skrl >=2.x stores MLP weights under net_container.*; this script uses net.*
    policy_state = {
        k.replace("net_container.", "net.", 1): v for k, v in policy_state.items()
    }
    policy.load_state_dict(policy_state, strict=False)
    policy.eval()

    # ---- 3. Build preprocessor (if present) ---------------------------------
    pp = ckpt.get("observation_preprocessor")
    if pp is not None:
        scaler = RunningStandardScaler(pp["running_mean"], pp["running_variance"])
        scaler.eval()
    else:
        scaler = nn.Identity()

    deployment = DeploymentPolicy(scaler, policy)
    deployment.eval()

    # ---- 4. Export to ONNX --------------------------------------------------
    dummy = torch.randn(1, args.obs_dim)
    print(f"[INFO] Exporting ONNX (opset={args.opset}) to: {args.output}")
    torch.onnx.export(
        deployment,
        dummy,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch"},
            "action": {0: "batch"},
        },
    )
    print("[INFO] ONNX export successful.")

    # ---- 5. Quick sanity check ---------------------------------------------
    import onnx
    model = onnx.load(args.output)
    onnx.checker.check_model(model)
    print("[INFO] ONNX model passes onnx.checker validation.")

    # Run inference comparison (ONNX vs PyTorch).
    import onnxruntime as ort
    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = deployment(dummy).numpy()
    ort_out = sess.run(None, {"observation": dummy.numpy()})[0]
    err = float(np.max(np.abs(pt_out - ort_out)))
    print(f"[INFO] PyTorch ↔ ONNX max absolute error: {err:.2e}")
    if err > 1e-5:
        print("[WARN]  Error is larger than expected — verify numeric fidelity.")
    else:
        print("[INFO] Numeric fidelity verified.")


if __name__ == "__main__":
    import numpy as np
    main()
