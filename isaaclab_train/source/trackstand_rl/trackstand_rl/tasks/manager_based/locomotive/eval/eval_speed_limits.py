# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/eval/eval_speed_limits.py (imports/paths rewritten for trackstand_rl locomotive).
"""Minimum balance speed and smooth-drive threshold from speed-sweep eval."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class SpeedLimitAnalysis:
    """Aggregated speed-sweep results."""

    bins: list[dict[str, Any]] = field(default_factory=list)
    min_balance_speed_ms: float | None = None
    smooth_drive_speed_ms: float | None = None
    balance_crossing_ms: float | None = None
    drive_smooth_crossing_ms: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_episodes(episodes: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for ep in episodes:
        if not getattr(ep, "complete", True):
            continue
        vx = getattr(ep, "spawn_cmd_vx_ms", float("nan"))
        if math.isfinite(float(vx)):
            out.append(ep)
    return out


def _group_by_speed(episodes: Sequence[Any], speed_round_decimals: int = 2) -> dict[float, list[Any]]:
    groups: dict[float, list[Any]] = {}
    scale = 10 ** speed_round_decimals
    for ep in episodes:
        vx = round(float(ep.spawn_cmd_vx_ms) * scale) / scale
        groups.setdefault(vx, []).append(ep)
    return dict(sorted(groups.items()))


def _mean(vals: Sequence[float]) -> float:
    arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def _crossing(bins: Sequence[dict[str, Any]], key: str, target: float, *, ascending_x: bool = True) -> float | None:
    """Interpolate x where ``key`` crosses ``target`` between adjacent bins."""
    if len(bins) < 2:
        return None
    ordered = sorted(bins, key=lambda b: b["vx_ms"], reverse=not ascending_x)
    for i in range(len(ordered) - 1):
        y0 = float(ordered[i][key])
        y1 = float(ordered[i + 1][key])
        if not (math.isfinite(y0) and math.isfinite(y1)) or y0 == y1:
            continue
        if (y0 - target) * (y1 - target) > 0:
            continue
        x0 = float(ordered[i]["vx_ms"])
        x1 = float(ordered[i + 1]["vx_ms"])
        t = (target - y0) / (y1 - y0)
        return float(x0 + t * (x1 - x0))
    return None


def _lowest_speed_passing(
    bins: Sequence[dict[str, Any]],
    predicate,
    *,
    min_episodes: int,
) -> float | None:
    passing = [b for b in bins if b["n"] >= min_episodes and predicate(b)]
    if not passing:
        return None
    return float(min(b["vx_ms"] for b in passing))


def analyze_speed_limits(
    episodes: Sequence[Any],
    *,
    survival_min: float = 0.80,
    drive_jitter_max: float = 0.08,
    drive_bang_ratio_max: float = 0.05,
    stable_balance_min: float = 0.50,
    min_episodes_per_bin: int = 1,
    speed_round_decimals: int = 2,
    v_min: float | None = None,
    v_max: float | None = None,
) -> SpeedLimitAnalysis:
    """Bin episodes by commanded vx* and estimate balance / smooth-drive speeds."""
    eps = _finite_episodes(episodes)
    groups = _group_by_speed(eps, speed_round_decimals=speed_round_decimals)

    bins: list[dict[str, Any]] = []
    for vx_ms, group in groups.items():
        if v_min is not None and vx_ms < v_min - 1e-6:
            continue
        if v_max is not None and vx_ms > v_max + 1e-6:
            continue
        n = len(group)
        survived = sum(1 for e in group if e.survived)
        survival = survived / n if n else 0.0
        mean_jitter = _mean([e.drive_action_jitter for e in group])
        mean_bang = _mean([getattr(e, "drive_bang_bang_ratio", float("nan")) for e in group])
        mean_stable = _mean([e.stable_balance_ratio for e in group])
        mean_roll_rmse = _mean([e.roll_rmse_deg for e in group])
        bins.append(
            {
                "vx_ms": vx_ms,
                "n": n,
                "survival": survival,
                "mean_stable_balance_ratio": mean_stable,
                "mean_roll_rmse_deg": mean_roll_rmse,
                "mean_drive_action_jitter": mean_jitter,
                "mean_drive_bang_bang_ratio": mean_bang,
            }
        )

    def _balanced(b: dict[str, Any]) -> bool:
        return (
            b["survival"] >= survival_min
            and b["mean_stable_balance_ratio"] >= stable_balance_min
        )

    def _smooth_drive(b: dict[str, Any]) -> bool:
        jitter_ok = math.isfinite(b["mean_drive_action_jitter"]) and b["mean_drive_action_jitter"] <= drive_jitter_max
        bang = b["mean_drive_bang_bang_ratio"]
        bang_ok = (not math.isfinite(bang)) or bang <= drive_bang_ratio_max
        return jitter_ok and bang_ok

    analysis = SpeedLimitAnalysis(
        bins=bins,
        min_balance_speed_ms=_lowest_speed_passing(bins, _balanced, min_episodes=min_episodes_per_bin),
        smooth_drive_speed_ms=_lowest_speed_passing(
            bins,
            lambda b: _balanced(b) and _smooth_drive(b),
            min_episodes=min_episodes_per_bin,
        ),
        balance_crossing_ms=_crossing(bins, "survival", survival_min, ascending_x=True),
        drive_smooth_crossing_ms=_crossing(
            bins, "mean_drive_action_jitter", drive_jitter_max, ascending_x=True
        ),
        params={
            "survival_min": survival_min,
            "stable_balance_min": stable_balance_min,
            "drive_jitter_max": drive_jitter_max,
            "drive_bang_ratio_max": drive_bang_ratio_max,
            "min_episodes_per_bin": min_episodes_per_bin,
            "n_episodes": len(eps),
            "v_min": v_min,
            "v_max": v_max,
        },
    )
    return analysis


def format_speed_limit_report(analysis: SpeedLimitAnalysis) -> str:
    """Human-readable report for console / markdown."""
    p = analysis.params
    lines = [
        "",
        f"{'SPEED LIMITS (balance + drive smoothness)':^74}",
        f"{'─' * 74}",
        f"  Episodes analysed : {p.get('n_episodes', 0)}",
        f"  Balance gate      : survival ≥ {p.get('survival_min', 0.8):.0%}, "
        f"stable balance ≥ {p.get('stable_balance_min', 0.5):.0%}",
        f"  Smooth drive gate : mean |Δa_drive| ≤ {p.get('drive_jitter_max', 0.08):.3f}, "
        f"bang-bang ratio ≤ {p.get('drive_bang_ratio_max', 0.05):.0%}",
        "",
    ]

    def _fmt_speed(val: float | None) -> str:
        return f"{val:.3f} m/s" if val is not None and math.isfinite(val) else "n/a"

    lines.append(f"  Min balance speed (lowest vx* still balancing) : {_fmt_speed(analysis.min_balance_speed_ms)}")
    lines.append(f"  Smooth drive threshold (lowest vx* w/o bang-bang): {_fmt_speed(analysis.smooth_drive_speed_ms)}")
    if analysis.balance_crossing_ms is not None:
        lines.append(f"  Balance survival crossing (interpolated)      : {_fmt_speed(analysis.balance_crossing_ms)}")
    if analysis.drive_smooth_crossing_ms is not None:
        lines.append(
            f"  Drive jitter crossing (interpolated)          : {_fmt_speed(analysis.drive_smooth_crossing_ms)}"
        )

    if analysis.bins:
        lines += ["", f"{'vx* (m/s)':>10} {'n':>4} {'surv':>7} {'stable':>7} {'|Δa_drv|':>9} {'bang%':>7}"]
        lines.append(f"{'─' * 74}")
        for b in analysis.bins:
            bang = b["mean_drive_bang_bang_ratio"]
            bang_s = f"{bang:.1%}" if math.isfinite(bang) else "  n/a"
            lines.append(
                f"{b['vx_ms']:10.3f} {b['n']:4d} {b['survival']:6.0%} "
                f"{b['mean_stable_balance_ratio']:6.0%} {b['mean_drive_action_jitter']:9.4f} {bang_s:>7}"
            )
    lines.append(f"{'─' * 74}")
    return "\n".join(lines)


def export_speed_limits(analysis: SpeedLimitAnalysis, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(analysis.to_dict(), f, indent=2)
    return str(path)


def plot_speed_limits(analysis: SpeedLimitAnalysis, save_path: str | Path) -> str | None:
    """Plot survival and drive jitter vs commanded speed."""
    if not analysis.bins:
        return None

    import matplotlib.pyplot as plt

    xs = [b["vx_ms"] for b in analysis.bins]
    surv = [b["survival"] for b in analysis.bins]
    jitter = [b["mean_drive_action_jitter"] for b in analysis.bins]
    bang = [b["mean_drive_bang_bang_ratio"] for b in analysis.bins]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax0, ax1 = axes

    ax0.plot(xs, surv, "o-", color="#2ecc71", label="Survival")
    ax0.axhline(analysis.params.get("survival_min", 0.8), color="#2ecc71", ls="--", alpha=0.5)
    if analysis.min_balance_speed_ms is not None:
        ax0.axvline(analysis.min_balance_speed_ms, color="#27ae60", ls=":", label="Min balance speed")
    ax0.set_ylabel("Survival rate")
    ax0.set_ylim(-0.05, 1.05)
    ax0.legend(loc="lower right")
    ax0.grid(True, alpha=0.3)

    ax1.plot(xs, jitter, "s-", color="#e67e22", label="Mean |Δa_drive|")
    ax1.axhline(analysis.params.get("drive_jitter_max", 0.08), color="#e67e22", ls="--", alpha=0.5)
    if any(math.isfinite(v) for v in bang):
        ax1.plot(xs, bang, "^-", color="#c0392b", alpha=0.8, label="Bang-bang step ratio")
        ax1.axhline(analysis.params.get("drive_bang_ratio_max", 0.05), color="#c0392b", ls="--", alpha=0.5)
    if analysis.smooth_drive_speed_ms is not None:
        ax1.axvline(analysis.smooth_drive_speed_ms, color="#d35400", ls=":", label="Smooth drive threshold")
    ax1.set_xlabel("Commanded vx* (m/s)")
    ax1.set_ylabel("Drive action metric")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    fig.suptitle("Speed sweep — balance vs drive smoothness")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)
