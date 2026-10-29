#!/usr/bin/env python3
# Ported from ~/bike_test/CycleRL/scripts/mujoco_paper/eval_paper_suite.py (imports/paths rewritten for trackstand_rl locomotive).
"""CycleRL paper eval suite in MuJoCo (sim-to-sim counterpart of the Isaac suite).

Mirrors ``eval_scenarios._configure_paper_table_vi`` mode for mode and scores
episodes with ``eval/paper_metrics.py`` (the module the Isaac ``eval_logger``
uses), so BSR / STE / VTE / BRT / CAT / MNT / MSS are defined identically.

Per mode it writes ``<out>/<mode>/summary.json`` (Isaac schema subset),
``scenario.json`` and ``episodes.json``, so ``scripts/eval/score_paper_table.py``
scores the MuJoCo directory unchanged and ``compare_isaac_mujoco.py`` builds the
transfer table.

Conditions (Isaac eval_scenarios, Froude-scaled to the NewBike):
  - 64 s episodes (3200 × 50 Hz), terminate at |gravity_y| > sin 45°.
  - Spawn: roll U(±10°), steer U(±20°) (speed_sweep ±2°), hub ω coupled to body vx.
  - paper_low/normal_speed: vx* ∈ [0.51,1.52] / [1.52,2.54], heading* ∈ ±10°,
    both resampled every U(3,5) s; spawn vx from the band.
  - other modes: vx* = 1.4 held, heading* = 0, spawn vx U(1.26, 1.4).
  - Heading P-controller output clipped to ±0.174 rad/s (eval suite range).
  - max_noise: ρ = 0.20 × per-signal s_max on all 8 channels; dropout: hold the
    last IMU block with p = 0.1 (never on the first two steps); gravel: μ = 0.45.
  - Ladders (one rung per episode): roll [-40°, 40°], noise ρ [0, 0.5],
    speed vx* [0.3, 2.5] (spawn vx = rung).
  - Terrain rows (rough / slope / steps) are not modelled in MuJoCo.

Usage (system Python with mujoco + numpy; torch only needed for the npz export)::

    python3 mujoco_sim2sim/scripts/locomotive/eval_paper_suite.py \\
        --checkpoint ~/bike_test/CycleRL/logs/skrl/cyclerl_logs/CycleRL-Trackstand-Cylinder-Paper-v0/<run>
    python3 mujoco_sim2sim/scripts/locomotive/eval_paper_suite.py --checkpoint <run> --modes paper_flat --episodes 64
"""

from __future__ import annotations

import os

# One BLAS thread per worker process; parallelism comes from the episode pool.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parents[2]  # trackstand_rl root
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_REPO / "isaaclab_train/scripts/eval_locomotive"))

import mujoco  # noqa: E402
import eval_paths  # noqa: E402
from play_act_refine_lo import (  # noqa: E402
    _POLICY_DT,
    PaperBikeSim,
    _export_checkpoint,
    _resolve_checkpoint,
    _run_dir,
    _scales_from_run,
    _wrap_pi,
    _xml_for_checkpoint,
    _zero_com_from_run,
)
from policy_numpy import PaperPolicy  # noqa: E402

_PM_PATH = _REPO / "isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/locomotive" / "eval/paper_metrics.py"
_spec = importlib.util.spec_from_file_location("cyclerl_paper_metrics", _PM_PATH)
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)

# --- Isaac eval_scenarios constants (keep in sync) ---------------------------
EPISODE_S = 64.0
EVAL_VX = 1.4
LOW_SPEED = (0.51, 1.52)
NORMAL_SPEED = (1.52, 2.54)
HEADING_BAND_RAD = math.radians(10.0)
EVAL_YAW_RATE_CLIP = 0.174
ROLL_INIT_RAD = math.radians(10.0)
STEER_INIT_DEG = 20.0
SPEED_SWEEP_STEER_INIT_DEG = 2.0
NOISE_MAX_RHO = 0.20
DROPOUT_PROB = 0.10
GRAVEL_MU = 0.45
ROLL_SWEEP_DEG = (-40.0, 40.0)
NOISE_SWEEP_RHO = (0.0, 0.5)
SPEED_SWEEP = (0.3, 2.5)
# Table I per-signal s_max: [gravity_y, wx, wy, wz, speed, steer, v*, ωz*].
PAPER_OBS_SMAX = np.array([0.7, 5.0, 5.0, 5.0, 2.5, 0.35, 2.5, 0.5], dtype=np.float64)
RESAMPLE_S = (3.0, 5.0)
CRASH_ROLL_DEG = 45.0
STABLE_ROLL_RAD = math.radians(3.0)  # eval_logger stable_roll_threshold_deg
STRAIGHT_WZ = 0.05  # eval_logger straight_wz_threshold

PAPER_MODES = (
    "paper_low_speed",
    "paper_normal_speed",
    "paper_no_noise",
    "paper_max_noise",
    "paper_dropout",
    "paper_flat",
    "paper_gravel",
    "paper_roll_sweep",
    "paper_noise_sweep",
    "speed_sweep",
)
TERRAIN_MODES = ("paper_rough", "paper_slope", "paper_steps")


@dataclass
class ModeSpec:
    mode: str
    description: str
    vx_band: tuple[float, float] | None = None  # resampled band (Table VI speed rows)
    vx_fixed: float = EVAL_VX
    heading_band: float = 0.0  # ± rad, resampled with the band modes
    spawn_vx: tuple[float, float] = (EVAL_VX * 0.9, EVAL_VX)
    steer_init_deg: float = STEER_INIT_DEG
    noise_rho: float = 0.0
    dropout: float = 0.0
    friction: float | None = None
    ladder: str | None = None  # roll | noise | speed
    ladder_range: tuple[float, float] = (0.0, 0.0)
    extra: dict = field(default_factory=dict)


def mode_spec(mode: str) -> ModeSpec:
    if mode == "paper_low_speed":
        return ModeSpec(mode, "Table VI low speed: vx* ∈ [0.51, 1.52] m/s, heading ±10°, resample 3–5 s.",
                        vx_band=LOW_SPEED, heading_band=HEADING_BAND_RAD, spawn_vx=LOW_SPEED)
    if mode == "paper_normal_speed":
        return ModeSpec(mode, "Table VI normal speed: vx* ∈ [1.52, 2.54] m/s, heading ±10°, resample 3–5 s.",
                        vx_band=NORMAL_SPEED, heading_band=HEADING_BAND_RAD, spawn_vx=NORMAL_SPEED)
    if mode in ("paper_no_noise", "paper_flat"):
        kind = "no noise" if mode == "paper_no_noise" else "flat plane"
        return ModeSpec(mode, f"Table VI {kind}: vx*=1.4 m/s, heading*=0, μ=1.")
    if mode == "paper_max_noise":
        return ModeSpec(mode, "Table VI max noise: ρ=0.20 × per-signal s_max, vx*=1.4 m/s.",
                        noise_rho=NOISE_MAX_RHO, extra={"noise_rho": NOISE_MAX_RHO})
    if mode == "paper_dropout":
        return ModeSpec(mode, "Table VI dropouts: hold last IMU packet with p=10%, vx*=1.4 m/s.",
                        dropout=DROPOUT_PROB, extra={"dropout_prob": DROPOUT_PROB})
    if mode == "paper_gravel":
        return ModeSpec(mode, "Table VI gravel: ground and tires μ=0.45, vx*=1.4 m/s.",
                        friction=GRAVEL_MU, extra={"friction": GRAVEL_MU})
    if mode == "paper_roll_sweep":
        return ModeSpec(mode, "CAT ladder: spawn roll [-40, 40]°, vx*=1.4 m/s.", ladder="roll",
                        ladder_range=ROLL_SWEEP_DEG,
                        extra={"roll_sweep_min_deg": ROLL_SWEEP_DEG[0], "roll_sweep_max_deg": ROLL_SWEEP_DEG[1]})
    if mode == "paper_noise_sweep":
        return ModeSpec(mode, "MNT ladder: ρ ∈ [0, 0.5] × per-signal s_max, vx*=1.4 m/s.", ladder="noise",
                        ladder_range=NOISE_SWEEP_RHO,
                        extra={"noise_rho_min": NOISE_SWEEP_RHO[0], "noise_rho_max": NOISE_SWEEP_RHO[1]})
    if mode == "speed_sweep":
        return ModeSpec(mode, "Speed sweep: vx* ladder [0.3, 2.5] m/s (spawn vx = rung), heading*=0.",
                        ladder="speed", ladder_range=SPEED_SWEEP, steer_init_deg=SPEED_SWEEP_STEER_INIT_DEG,
                        extra={"speed_sweep_min": SPEED_SWEEP[0], "speed_sweep_max": SPEED_SWEEP[1]})
    raise ValueError(f"Unknown or unsupported MuJoCo mode {mode!r} (terrain rows are not modelled)")


# --- worker -------------------------------------------------------------------
_SIM: PaperBikeSim | None = None


def _init_worker(xml: str, npz: str, zero_com: bool) -> None:
    global _SIM
    _SIM = PaperBikeSim(Path(xml), PaperPolicy(Path(npz)), zero_com=zero_com)


def _ladder_value(spec: ModeSpec, idx: int, n: int) -> float:
    lo, hi = spec.ladder_range
    return lo if n <= 1 else lo + (hi - lo) * idx / (n - 1)


def run_episode(job: tuple[ModeSpec, int, int, int]) -> dict:
    """One episode on the worker's plant. Mirrors the Isaac reset → step → log loop."""
    spec, idx, n, seed = job
    sim = _SIM
    assert sim is not None
    rng = np.random.default_rng([seed, PAPER_MODES.index(spec.mode), idx])
    n_steps = int(round(EPISODE_S / _POLICY_DT))

    # Plant condition and observation corruption.
    sim.apply_domain(friction=spec.friction)
    sim.yaw_rate_clip = EVAL_YAW_RATE_CLIP
    noise_rho = spec.noise_rho
    if spec.ladder == "noise":
        noise_rho = _ladder_value(spec, idx, n)
    noise_std = noise_rho * PAPER_OBS_SMAX

    # Commands at reset.
    def sample_vx() -> float:
        return float(rng.uniform(*spec.vx_band)) if spec.vx_band else spec.vx_fixed

    def sample_heading() -> float:
        return float(rng.uniform(-spec.heading_band, spec.heading_band)) if spec.heading_band else 0.0

    vx_cmd = sample_vx()
    if spec.ladder == "speed":
        vx_cmd = _ladder_value(spec, idx, n)
    heading_target = sample_heading()
    t_vx = float(rng.uniform(*RESAMPLE_S)) if spec.vx_band else math.inf
    t_hd = float(rng.uniform(*RESAMPLE_S)) if spec.heading_band else math.inf

    # Spawn state.
    if spec.vx_band:
        spawn_vx = float(rng.uniform(*spec.spawn_vx))
    elif spec.ladder == "speed":
        spawn_vx = vx_cmd
    else:
        spawn_vx = float(rng.uniform(*spec.spawn_vx))
    roll0 = float(rng.uniform(-ROLL_INIT_RAD, ROLL_INIT_RAD))
    if spec.ladder == "roll":
        roll0 = math.radians(_ladder_value(spec, idx, n))
    sim.vx_cmd = vx_cmd
    sim.reset(vx=spawn_vx, roll_rad=roll0)
    steer0 = math.radians(float(rng.uniform(-spec.steer_init_deg, spec.steer_init_deg)))
    sim.data.qpos[sim.qadr_steer] = steer0
    mujoco.mj_forward(sim.model, sim.data)
    sim.heading_target = heading_target

    crash_gy = math.sin(math.radians(CRASH_ROLL_DEG))
    stats = dict(bsr_balanced=True, brt_active=False, brt_steps=0, brt_sum=0.0, brt_events=0,
                 mbd_run=0, mbd_max=0, head_sq=0.0, vx_abs=0.0, roll_sq_straight=0.0,
                 straight=0, max_lean=0.0, steer_jitter=0.0)
    held_imu = None
    prev_a0 = None
    survived = True
    steps_done = 0
    for k in range(n_steps):
        t = k * _POLICY_DT
        if t >= t_vx:
            sim.vx_cmd = sample_vx()
            t_vx = t + float(rng.uniform(*RESAMPLE_S))
        if t >= t_hd:
            sim.heading_target = sample_heading()
            t_hd = t + float(rng.uniform(*RESAMPLE_S))

        obs = sim.clean_obs().astype(np.float64)
        if spec.dropout > 0.0:
            # Isaac _apply_dropout: never on the first two observations of an episode.
            if held_imu is not None and k > 1 and rng.random() < spec.dropout:
                obs[:6] = held_imu
            held_imu = obs[:6].copy()
        if noise_rho > 0.0:
            obs = obs + rng.normal(0.0, 1.0, size=8) * noise_std
        action = sim.step_with_obs(obs.astype(np.float32))
        steps_done = k + 1

        # eval_logger.step on the post-step state.
        roll = sim.roll_rad()
        roll_abs = abs(roll)
        yaw = sim._heading()
        wz_cmd = sim.yaw_rate_cmd()
        balanced = roll_abs < pm.PAPER_BSR_ROLL_LIMIT_RAD
        stats["bsr_balanced"] &= balanced
        stats["mbd_run"] = stats["mbd_run"] + 1 if balanced else 0
        stats["mbd_max"] = max(stats["mbd_max"], stats["mbd_run"])
        stats["brt_active"] |= roll_abs >= pm.PAPER_BRT_DISTURB_RAD
        if stats["brt_active"]:
            stats["brt_steps"] += 1
            if roll_abs < STABLE_ROLL_RAD:
                stats["brt_sum"] += stats["brt_steps"] * _POLICY_DT
                stats["brt_events"] += 1
                stats["brt_active"] = False
                stats["brt_steps"] = 0
        stats["head_sq"] += _wrap_pi(sim.heading_target - yaw) ** 2
        stats["vx_abs"] += abs(sim.forward_vel() - sim.vx_cmd)
        if abs(wz_cmd) < STRAIGHT_WZ:
            stats["roll_sq_straight"] += roll * roll
            stats["straight"] += 1
        stats["max_lean"] = max(stats["max_lean"], roll_abs)
        if prev_a0 is not None:
            stats["steer_jitter"] += abs(float(action[0]) - prev_a0)
        prev_a0 = float(action[0])

        gravity_y = float(sim.clean_obs()[0])
        if abs(gravity_y) > crash_gy:
            survived = False
            break

    n = max(steps_done, 1)
    return {
        "mode": spec.mode,
        "idx": idx,
        "complete": True,
        "survived": survived,
        "episode_length_s": steps_done * _POLICY_DT,
        "bsr_balanced": bool(stats["bsr_balanced"]),
        "brt_s": stats["brt_sum"] / stats["brt_events"] if stats["brt_events"] else float("nan"),
        "brt_events": stats["brt_events"],
        "max_balance_duration_s": stats["mbd_max"] * _POLICY_DT,
        "heading_rmse_deg": math.degrees(math.sqrt(stats["head_sq"] / n)),
        "vx_mae_ms": stats["vx_abs"] / n,
        "roll_rmse_straight_deg": (
            math.degrees(math.sqrt(stats["roll_sq_straight"] / stats["straight"])) if stats["straight"] else float("nan")
        ),
        "max_lean_deg": math.degrees(stats["max_lean"]),
        "mean_steer_action_jitter": stats["steer_jitter"] / max(steps_done - 1, 1),
        "spawn_roll_deg": math.degrees(roll0),
        "spawn_cmd_vx_ms": vx_cmd,
        "noise_rho": noise_rho if (spec.ladder == "noise" or noise_rho > 0.0) else float("nan"),
    }


# --- summary ------------------------------------------------------------------
def _mean(values: list[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(episodes: list[dict]) -> dict:
    eps = [SimpleNamespace(**e) for e in episodes]
    complete = [e for e in eps if e.complete]
    brt_vals = [e.brt_s for e in eps if e.brt_events and math.isfinite(e.brt_s)]
    return {
        "num_episodes": len(eps),
        "num_complete_episodes": len(complete),
        "num_censored_episodes": len(eps) - len(complete),
        "survival_rate": (sum(e.survived for e in complete) / len(complete)) if complete else 0.0,
        "mean_episode_length_s": _mean([e.episode_length_s for e in eps]),
        "bsr": pm.balance_success_rate(eps),
        "brt_s": _mean(brt_vals),
        "mbd_s": max((e.max_balance_duration_s for e in eps), default=0.0),
        "ste_deg": _mean([e.heading_rmse_deg for e in eps]),
        "vte_ms": _mean([e.vx_mae_ms for e in eps]),
        "mean_heading_rmse_deg": _mean([e.heading_rmse_deg for e in eps]),
        "mean_vx_mae_ms": _mean([e.vx_mae_ms for e in eps]),
        "mean_roll_rmse_straight_deg": _mean([e.roll_rmse_straight_deg for e in eps]),
        "mean_steer_action_jitter": _mean([e.mean_steer_action_jitter for e in eps]),
        "cat_deg": pm.analyze_critical_angle(eps),
        "mnt": pm.analyze_noise_tolerance(eps),
        "mss_ms": pm.analyze_min_sustaining_speed(eps),
    }


def _json_safe(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# --- driver -------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CycleRL paper eval suite in MuJoCo.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Run dir, checkpoints dir or .pt.")
    p.add_argument("--xml", type=Path, default=None, help="Override the MJCF (default: from the run).")
    p.add_argument("--modes", type=str, default=",".join(PAPER_MODES))
    p.add_argument("--episodes", type=int, default=1024, help="Episodes (ladder rungs) per mode; Isaac uses 1024 envs.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None, help="Default: <run>/eval_mujoco (dated sibling if it exists).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    skipped = [m for m in modes if m in TERRAIN_MODES]
    modes = [m for m in modes if m not in TERRAIN_MODES]
    specs = [mode_spec(m) for m in modes]

    ckpt = _resolve_checkpoint(args.checkpoint)
    run = _run_dir(ckpt)
    if run is None:
        raise SystemExit(f"Cannot resolve run dir (need params/env.yaml) for {ckpt}")
    scales = _scales_from_run(run)
    if scales["drive_scale"] <= 0 or scales["steer_scale"] <= 0:
        raise SystemExit(f"Invalid action scales from {run / 'params' / 'env.yaml'}: {scales}")
    npz = _export_checkpoint(ckpt, scales)
    xml = _xml_for_checkpoint(ckpt, args.xml)
    zero_com = _zero_com_from_run(run)

    requested = args.out_dir or (run / "eval_mujoco")
    out_dir, redirected = eval_paths.allocate_eval_dir(requested)
    eval_paths.note_eval_allocation(out_dir, requested=requested, redirected=redirected, checkpoint=str(ckpt))

    # Build once in the parent so plant errors surface before the pool starts.
    probe = PaperBikeSim(xml, PaperPolicy(npz), zero_com=zero_com)
    # Cylinder CAD reports a small contact-midplane offset; Isaac keeps that CoM
    # (zero_lateral_com disabled). Record both so transfer reports stay honest.
    plant = {
        "xml": str(xml),
        "plant": probe.tires,
        "zero_lateral_com": zero_com,
        "com_y_removed_m": probe.com_y_removed,
        "com_contact_offset_m": getattr(probe, "com_contact_offset", float("nan")),
        "tire_radius_m": float(probe.tire.radius),
        "policy_dt_s": _POLICY_DT,
        "npz": str(npz),
        "isaac_run": str(run),
        **scales,
    }
    print(f"[mujoco_suite] ckpt={ckpt} plant={plant}")
    if "Trackstand-Cylinder" in str(run) or "trackstand" in str(xml).lower():
        if abs(float(probe.tire.radius) - 0.07) > 1e-6:
            raise SystemExit(f"Cylinder plant radius {probe.tire.radius} != Isaac 0.07 m")
        if abs(scales["drive_scale"] - (3.0 / 0.07)) > 1e-3:
            raise SystemExit(
                f"Cylinder drive_scale {scales['drive_scale']} != Isaac 3/0.07 "
                f"({3.0 / 0.07}); env.yaml was not applied."
            )
        if zero_com:
            raise SystemExit("Cylinder Paper-v0 disables zero_lateral_com; refusing to shift CoM.")
    if skipped:
        print(f"[mujoco_suite] skipping terrain modes (not modelled): {', '.join(skipped)}")

    status = {"timestamp": datetime.now(timezone.utc).isoformat(), "checkpoint": str(ckpt),
              "simulator": "mujoco", "plant": plant, "modes": [], "results": []}
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(str(xml), str(npz), zero_com)) as pool:
        for spec in specs:
            t0 = time.time()
            jobs = [(spec, i, args.episodes, args.seed) for i in range(args.episodes)]
            episodes = pool.map(run_episode, jobs, chunksize=4)
            summary = summarize(episodes)
            mode_dir = out_dir / spec.mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            scenario = {"mode": spec.mode, "description": spec.description, "simulator": "mujoco",
                        "suite": "paper", "terrain": "plane", "episodes": args.episodes,
                        "seed": args.seed, **spec.extra, "plant": plant}
            (mode_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n")
            (mode_dir / "scenario.json").write_text(json.dumps(_json_safe(scenario), indent=2) + "\n")
            (mode_dir / "episodes.json").write_text(json.dumps(_json_safe(episodes)) + "\n")
            dt = time.time() - t0
            status["modes"].append(spec.mode)
            status["results"].append({"mode": spec.mode, "elapsed_s": round(dt, 1), "wrote_summary": True})
            print(
                f"[mujoco_suite] {spec.mode:20s} BSR {100 * summary['bsr']:5.1f}%  "
                f"STE {summary['ste_deg']:6.2f}°  VTE {summary['vte_ms']:.3f}  "
                f"CAT {summary['cat_deg']}  MNT {summary['mnt']}  MSS {summary['mss_ms']}  ({dt:.0f}s)"
            )
    (out_dir / "eval_all_metrics_status.json").write_text(json.dumps(_json_safe(status), indent=2) + "\n")
    eval_paths.publish_eval_dir(out_dir)
    print(f"[mujoco_suite] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
