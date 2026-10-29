# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/eval/eval_drive_accel.py (imports/paths rewritten for trackstand_rl locomotive).
"""Balance vs RC hub acceleration from drive_accel_sweep eval.

Policy steers only; hub ω* is owned by vx* and slewed at a per-env accel.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class DriveAccelAnalysis:
    """Aggregated drive-accel sweep results."""

    bins: list[dict[str, Any]] = field(default_factory=list)
    max_balance_accel_ms2: float | None = None
    min_fail_accel_ms2: float | None = None
    balance_crossing_ms2: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_episodes(episodes: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for ep in episodes:
        if not getattr(ep, "complete", True):
            continue
        accel = getattr(ep, "drive_accel_ms2", float("nan"))
        if math.isfinite(float(accel)):
            out.append(ep)
    return out


def _group_by_accel(episodes: Sequence[Any], round_decimals: int = 3) -> dict[float, list[Any]]:
    groups: dict[float, list[Any]] = {}
    scale = 10 ** round_decimals
    for ep in episodes:
        a = round(float(ep.drive_accel_ms2) * scale) / scale
        groups.setdefault(a, []).append(ep)
    return dict(sorted(groups.items()))


def _mean(vals: Sequence[float]) -> float:
    arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def _crossing(bins: Sequence[dict[str, Any]], key: str, target: float) -> float | None:
    if len(bins) < 2:
        return None
    ordered = sorted(bins, key=lambda b: b["accel_ms2"])
    for i in range(len(ordered) - 1):
        y0 = float(ordered[i][key])
        y1 = float(ordered[i + 1][key])
        if not (math.isfinite(y0) and math.isfinite(y1)) or y0 == y1:
            continue
        if (y0 - target) * (y1 - target) > 0:
            continue
        x0 = float(ordered[i]["accel_ms2"])
        x1 = float(ordered[i + 1]["accel_ms2"])
        t = (target - y0) / (y1 - y0)
        return float(x0 + t * (x1 - x0))
    return None


def analyze_drive_accel(
    episodes: Sequence[Any],
    *,
    survival_min: float = 0.80,
    stable_balance_min: float = 0.50,
    min_episodes_per_bin: int = 1,
    round_decimals: int = 3,
    a_min: float | None = None,
    a_max: float | None = None,
) -> DriveAccelAnalysis:
    """Bin episodes by RC hub accel and estimate the highest still-balancing rate."""
    eps = _finite_episodes(episodes)
    groups = _group_by_accel(eps, round_decimals=round_decimals)

    bins: list[dict[str, Any]] = []
    for accel, group in groups.items():
        if a_min is not None and accel < a_min - 1e-6:
            continue
        if a_max is not None and accel > a_max + 1e-6:
            continue
        n = len(group)
        survived = sum(1 for e in group if e.survived)
        survival = survived / n if n else 0.0
        bins.append(
            {
                "accel_ms2": accel,
                "n": n,
                "survival": survival,
                "mean_episode_length_s": _mean([e.episode_time_s for e in group]),
                "mean_stable_balance_ratio": _mean([e.stable_balance_ratio for e in group]),
                "mean_roll_rmse_deg": _mean([e.roll_rmse_deg for e in group]),
                "mean_steer_action_jitter": _mean([e.steer_action_jitter for e in group]),
                "mean_vx_rmse_ms": _mean([e.vx_rmse_ms for e in group]),
                "mean_velocity_ms": _mean([e.mean_velocity_ms for e in group]),
            }
        )

    def _balanced(b: dict[str, Any]) -> bool:
        return (
            b["survival"] >= survival_min
            and b["mean_stable_balance_ratio"] >= stable_balance_min
        )

    passing = [b for b in bins if b["n"] >= min_episodes_per_bin and _balanced(b)]
    failing = [b for b in bins if b["n"] >= min_episodes_per_bin and not _balanced(b)]

    analysis = DriveAccelAnalysis(
        bins=bins,
        max_balance_accel_ms2=float(max(b["accel_ms2"] for b in passing)) if passing else None,
        min_fail_accel_ms2=float(min(b["accel_ms2"] for b in failing)) if failing else None,
        balance_crossing_ms2=_crossing(bins, "survival", survival_min),
        params={
            "survival_min": survival_min,
            "stable_balance_min": stable_balance_min,
            "min_episodes_per_bin": min_episodes_per_bin,
            "n_episodes": len(eps),
            "a_min": a_min,
            "a_max": a_max,
        },
    )
    return analysis


def format_drive_accel_report(analysis: DriveAccelAnalysis) -> str:
    p = analysis.params
    lines = [
        "",
        f"{'DRIVE ACCEL SWEEP (RC hub, policy steers only)':^74}",
        f"{'─' * 74}",
        f"  Episodes analysed : {p.get('n_episodes', 0)}",
        f"  Balance gate      : survival ≥ {p.get('survival_min', 0.8):.0%}, "
        f"stable balance ≥ {p.get('stable_balance_min', 0.5):.0%}",
        "",
    ]

    def _fmt(val: float | None) -> str:
        return f"{val:.3f} m/s²" if val is not None and math.isfinite(val) else "n/a"

    lines.append(f"  Max balancing accel (highest a still passing) : {_fmt(analysis.max_balance_accel_ms2)}")
    lines.append(f"  Min failing accel                             : {_fmt(analysis.min_fail_accel_ms2)}")
    if analysis.balance_crossing_ms2 is not None:
        lines.append(f"  Survival crossing (interpolated)              : {_fmt(analysis.balance_crossing_ms2)}")

    if analysis.bins:
        lines += [
            "",
            f"{'a (m/s²)':>10} {'n':>4} {'surv':>7} {'len(s)':>7} {'stable':>7} "
            f"{'rollRMSE':>9} {'|Δa_st|':>9} {'vxRMSE':>8}",
            f"{'─' * 74}",
        ]
        for b in analysis.bins:
            lines.append(
                f"{b['accel_ms2']:10.3f} {b['n']:4d} {b['survival']:6.0%} "
                f"{b['mean_episode_length_s']:7.2f} {b['mean_stable_balance_ratio']:6.0%} "
                f"{b['mean_roll_rmse_deg']:9.2f} {b['mean_steer_action_jitter']:9.4f} "
                f"{b['mean_vx_rmse_ms']:8.3f}"
            )
    lines.append(f"{'─' * 74}")
    return "\n".join(lines)


def export_drive_accel(analysis: DriveAccelAnalysis, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(analysis.to_dict(), f, indent=2)
    return str(path)


def plot_drive_accel(analysis: DriveAccelAnalysis, save_path: str | Path) -> str | None:
    if not analysis.bins:
        return None

    import matplotlib.pyplot as plt

    xs = [b["accel_ms2"] for b in analysis.bins]
    surv = [b["survival"] for b in analysis.bins]
    roll = [b["mean_roll_rmse_deg"] for b in analysis.bins]
    vx = [b["mean_vx_rmse_ms"] for b in analysis.bins]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax0, ax1 = axes

    ax0.plot(xs, surv, "o-", color="#2ecc71", label="Survival")
    ax0.axhline(analysis.params.get("survival_min", 0.8), color="#2ecc71", ls="--", alpha=0.5)
    if analysis.max_balance_accel_ms2 is not None:
        ax0.axvline(analysis.max_balance_accel_ms2, color="#27ae60", ls=":", label="Max balancing accel")
    ax0.set_ylabel("Survival rate")
    ax0.set_ylim(-0.05, 1.05)
    ax0.legend(loc="lower left")
    ax0.grid(True, alpha=0.3)

    ax1.plot(xs, roll, "s-", color="#8e44ad", label="Roll RMSE (°)")
    ax1.plot(xs, vx, "^-", color="#2980b9", label="vx RMSE (m/s)")
    ax1.set_xlabel("RC drive accel (m/s²)")
    ax1.set_ylabel("Tracking RMSE")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    fig.suptitle("Drive accel sweep — RC hub, policy steers only")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)
