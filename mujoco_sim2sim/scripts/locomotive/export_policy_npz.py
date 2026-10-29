#!/usr/bin/env python3
# Ported from ~/bike_test/CycleRL/scripts/mujoco_paper/export_policy_npz.py (imports/paths rewritten for trackstand_rl locomotive).
"""Export a skrl Paper-v0 PPO checkpoint to NumPy npz for standalone MuJoCo play.

Requires torch (use Isaac Lab python)::

    ./isaaclab.sh -p mujoco_sim2sim/scripts/locomotive/export_policy_npz.py \\
        --checkpoint ~/bike_test/CycleRL/logs/skrl/cyclerl_logs/CycleRL-NewBike-Paper-v0/\\
2026-09-02_01-28-28_ppo_torch_bicycle_ppo/checkpoints/agent_100000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]  # trackstand_rl root
_DEFAULT_CKPT = (
    Path.home()
    / "bike_test/CycleRL/logs/skrl/cyclerl_logs/CycleRL-NewBike-Paper-v0"
    / "2026-09-02_01-28-28_ppo_torch_bicycle_ppo/checkpoints/agent_100000.pt"
)
_DEFAULT_OUT = _REPO / "mujoco_sim2sim/logs/locomotive/act_refine_lo_agent_100000.npz"

# Scales from that run's params/env.yaml (act_refine_lo).
_STEER_SCALE = 0.3490658503988659  # ±20°
_DRIVE_SCALE = 41.37931034482759  # 3.0 / 0.0725
_DRIVE_CLIP = 45.0
_STEER_KP, _STEER_KD = 10.0, 0.2
_DRIVE_KD = 5.0
_STEER_EFFORT, _DRIVE_EFFORT = 5.6, 6.28


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CKPT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--steer-scale", type=float, default=_STEER_SCALE)
    parser.add_argument(
        "--steer-clip",
        type=float,
        default=None,
        help="Joint-position clip (rad). Defaults to --steer-scale.",
    )
    parser.add_argument("--drive-scale", type=float, default=_DRIVE_SCALE)
    parser.add_argument("--drive-kd", type=float, default=_DRIVE_KD, help="Rear velocity-loop Kd.")
    args = parser.parse_args()
    steer_clip = args.steer_scale if args.steer_clip is None else args.steer_clip

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {}
    for k, v in ckpt["policy"].items():
        if k == "log_std_parameter":
            continue
        state[k.replace("net_container.", "net.", 1)] = v.detach().cpu().numpy()

    wkeys = sorted(
        [k for k in state if k.endswith(".weight")],
        key=lambda s: int(s.split(".")[1]),
    )
    weights = [state[k] for k in wkeys]
    biases = [state[k.replace(".weight", ".bias")] for k in wkeys]
    if len(weights) != 5:
        raise SystemExit(f"Expected 5 Linear layers, got {len(weights)}: {wkeys}")

    pp = ckpt["observation_preprocessor"]
    mean = pp["running_mean"].detach().cpu().numpy().astype(np.float32)
    var = pp["running_variance"].detach().cpu().numpy().astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        mean=mean,
        var=var,
        w0=weights[0],
        b0=biases[0],
        w1=weights[1],
        b1=biases[1],
        w2=weights[2],
        b2=biases[2],
        w3=weights[3],
        b3=biases[3],
        w4=weights[4],
        b4=biases[4],
        steer_scale=np.float32(args.steer_scale),
        steer_clip=np.float32(steer_clip),
        drive_scale=np.float32(args.drive_scale),
        drive_clip=np.float32(_DRIVE_CLIP),
        steer_kp=np.float32(_STEER_KP),
        steer_kd=np.float32(_STEER_KD),
        drive_kd=np.float32(args.drive_kd),
        steer_effort=np.float32(_STEER_EFFORT),
        drive_effort=np.float32(_DRIVE_EFFORT),
        obs_dim=np.int32(mean.shape[-1]),
        act_dim=np.int32(weights[-1].shape[0]),
        checkpoint=np.array(str(args.checkpoint)),
    )
    print(f"[PASS] wrote {args.output} (obs_dim={mean.shape[-1]}, layers={[w.shape for w in weights]})")


if __name__ == "__main__":
    main()
