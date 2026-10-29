# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/eval/eval_heading_limits.py (imports/paths rewritten for trackstand_rl locomotive).
"""Heading-tracking envelope from heading_sweep eval.

Policy balances at fixed vx* while each env holds a fixed heading* on a ladder.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class HeadingLimitAnalysis:
    """Aggregated heading-sweep results."""

    bins: list[dict[str, Any]] = field(default_factory=list)
    max_balance_heading_abs_deg: float | None = None
    min_fail_heading_abs_deg: float | None = None
    balance_crossing_abs_deg: float | None = None
    # θ*≈0 (straight) bin — always present when the ladder straddles 0.
    straight_bin: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_episodes(episodes: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for ep in episodes:
        if not getattr(ep, "complete", True):
            continue
        h = getattr(ep, "spawn_cmd_heading_deg", float("nan"))
        if math.isfinite(float(h)):
            out.append(ep)
    return out


def _group_by_heading(episodes: Sequence[Any], round_decimals: int = 2) -> dict[float, list[Any]]:
    groups: dict[float, list[Any]] = {}
    scale = 10 ** round_decimals
    for ep in episodes:
        h = round(float(ep.spawn_cmd_heading_deg) * scale) / scale
        groups.setdefault(h, []).append(ep)
    return dict(sorted(groups.items()))


def _mean(vals: Sequence[float]) -> float:
    arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def _crossing_abs(bins: Sequence[dict[str, Any]], key: str, target: float) -> float | None:
    """Crossing on |heading*| axis (bins sorted by abs heading)."""
    if len(bins) < 2:
        return None
    ordered = sorted(bins, key=lambda b: abs(float(b["heading_deg"])))
    for i in range(len(ordered) - 1):
        y0 = float(ordered[i][key])
        y1 = float(ordered[i + 1][key])
        if not (math.isfinite(y0) and math.isfinite(y1)) or y0 == y1:
            continue
        if (y0 - target) * (y1 - target) > 0:
            continue
        x0 = abs(float(ordered[i]["heading_deg"]))
        x1 = abs(float(ordered[i + 1]["heading_deg"]))
        t = (target - y0) / (y1 - y0)
        return float(x0 + t * (x1 - x0))
    return None


def analyze_heading_limits(
    episodes: Sequence[Any],
    *,
    survival_min: float = 0.80,
    stable_balance_min: float = 0.50,
    heading_rmse_max_deg: float = 15.0,
    min_episodes_per_bin: int = 1,
    round_decimals: int = 2,
    h_min_deg: float | None = None,
    h_max_deg: float | None = None,
) -> HeadingLimitAnalysis:
    """Bin episodes by commanded heading* and estimate the largest |θ*| still tracking.

    Headline ``max_balance_heading_abs_deg`` requires **both** balance (survival +
    stable roll) **and** heading RMSE ≤ ``heading_rmse_max_deg`` — not survival alone.
    The ladder always includes θ*=0 (straight) when the sweep range straddles 0.
    """
    eps = _finite_episodes(episodes)
    groups = _group_by_heading(eps, round_decimals=round_decimals)

    bins: list[dict[str, Any]] = []
    for heading_deg, group in groups.items():
        if h_min_deg is not None and heading_deg < h_min_deg - 1e-6:
            continue
        if h_max_deg is not None and heading_deg > h_max_deg + 1e-6:
            continue
        n = len(group)
        survived = sum(1 for e in group if e.survived)
        survival = survived / n if n else 0.0
        bins.append(
            {
                "heading_deg": heading_deg,
                "heading_abs_deg": abs(heading_deg),
                "n": n,
                "survival": survival,
                "mean_episode_length_s": _mean([e.episode_time_s for e in group]),
                "mean_stable_balance_ratio": _mean([e.stable_balance_ratio for e in group]),
                "mean_roll_rmse_deg": _mean([e.roll_rmse_deg for e in group]),
                "mean_heading_rmse_deg": _mean([e.heading_rmse_deg for e in group]),
                "mean_yaw_drift_rate_deg_per_s": _mean(
                    [getattr(e, "yaw_drift_rate_deg_per_s", float("nan")) for e in group]
                ),
                "mean_lateral_deviation_rms_m": _mean(
                    [getattr(e, "lateral_deviation_rms_m", float("nan")) for e in group]
                ),
                "mean_steer_action_jitter": _mean([e.steer_action_jitter for e in group]),
                "mean_vx_rmse_ms": _mean([e.vx_rmse_ms for e in group]),
            }
        )

    def _balanced(b: dict[str, Any]) -> bool:
        return (
            b["survival"] >= survival_min
            and b["mean_stable_balance_ratio"] >= stable_balance_min
        )

    def _tracking_ok(b: dict[str, Any]) -> bool:
        h = b["mean_heading_rmse_deg"]
        return math.isfinite(h) and h <= heading_rmse_max_deg

    passing = [b for b in bins if b["n"] >= min_episodes_per_bin and _balanced(b)]
    failing = [b for b in bins if b["n"] >= min_episodes_per_bin and not _balanced(b)]
    # Headline envelope: must balance AND track heading* (not survival-only).
    tracking_ok = [b for b in passing if _tracking_ok(b)]

    straight_bin = None
    for b in bins:
        if abs(float(b["heading_deg"])) < 10 ** (-round_decimals) / 2:
            straight_bin = dict(b)
            straight_bin["balanced"] = _balanced(b)
            straight_bin["tracking_ok"] = _tracking_ok(b)
            break

    analysis = HeadingLimitAnalysis(
        bins=bins,
        max_balance_heading_abs_deg=(
            float(max(b["heading_abs_deg"] for b in tracking_ok)) if tracking_ok else None
        ),
        min_fail_heading_abs_deg=(
            float(min(b["heading_abs_deg"] for b in failing)) if failing else None
        ),
        balance_crossing_abs_deg=_crossing_abs(bins, "survival", survival_min),
        straight_bin=straight_bin,
        params={
            "survival_min": survival_min,
            "stable_balance_min": stable_balance_min,
            "heading_rmse_max_deg": heading_rmse_max_deg,
            "min_episodes_per_bin": min_episodes_per_bin,
            "n_episodes": len(eps),
            "h_min_deg": h_min_deg,
            "h_max_deg": h_max_deg,
        },
    )
    return analysis


def format_heading_limit_report(analysis: HeadingLimitAnalysis) -> str:
    p = analysis.params
    lines = [
        "",
        f"{'HEADING SWEEP (fixed vx*, heading* ladder)':^74}",
        f"{'─' * 74}",
        f"  Episodes analysed : {p.get('n_episodes', 0)}",
        f"  Balance gate      : survival ≥ {p.get('survival_min', 0.8):.0%}, "
        f"stable balance ≥ {p.get('stable_balance_min', 0.5):.0%}",
        f"  Tracking gate     : heading RMSE ≤ {p.get('heading_rmse_max_deg', 15.0):.1f}°",
        "",
    ]

    def _fmt(val: float | None) -> str:
        return f"{val:.2f}°" if val is not None and math.isfinite(val) else "n/a"

    lines.append(
        f"  Max balancing |heading*| (survival + tracking) : {_fmt(analysis.max_balance_heading_abs_deg)}"
    )
    lines.append(
        f"  Min failing |heading*| (survival/balance)      : {_fmt(analysis.min_fail_heading_abs_deg)}"
    )
    if analysis.balance_crossing_abs_deg is not None:
        lines.append(
            f"  Survival crossing on |θ*| (interpolated)       : {_fmt(analysis.balance_crossing_abs_deg)}"
        )

    sb = analysis.straight_bin
    if sb is not None:
        bal = "PASS" if sb.get("balanced") else "FAIL"
        trk = "PASS" if sb.get("tracking_ok") else "FAIL"
        lines += [
            "",
            "  Straight (θ*=0) bin:",
            f"    n={sb['n']}  surv={sb['survival']:.0%}  hdgRMSE={sb['mean_heading_rmse_deg']:.2f}°  "
            f"yawDrift={sb.get('mean_yaw_drift_rate_deg_per_s', float('nan')):.3f} °/s  "
            f"latRMS={sb.get('mean_lateral_deviation_rms_m', float('nan')):.3f} m",
            f"    balance={bal}  tracking={trk}",
        ]
    else:
        lines += ["", "  Straight (θ*=0) bin: missing (ladder did not include 0)"]

    if analysis.bins:
        lines += [
            "",
            f"{'θ* (°)':>10} {'n':>4} {'surv':>7} {'len(s)':>7} {'stable':>7} "
            f"{'rollRMSE':>9} {'hdgRMSE':>8} {'yaw°/s':>8} {'|Δa_st|':>9}",
            f"{'─' * 74}",
        ]
        for b in analysis.bins:
            mark = " ←straight" if abs(float(b["heading_deg"])) < 1e-6 else ""
            lines.append(
                f"{b['heading_deg']:10.2f} {b['n']:4d} {b['survival']:6.0%} "
                f"{b['mean_episode_length_s']:7.2f} {b['mean_stable_balance_ratio']:6.0%} "
                f"{b['mean_roll_rmse_deg']:9.2f} {b['mean_heading_rmse_deg']:8.2f} "
                f"{b.get('mean_yaw_drift_rate_deg_per_s', float('nan')):8.3f} "
                f"{b['mean_steer_action_jitter']:9.4f}{mark}"
            )
    lines.append(f"{'─' * 74}")
    return "\n".join(lines)


def export_heading_limits(analysis: HeadingLimitAnalysis, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(analysis.to_dict(), f, indent=2)
    return str(path)


def plot_heading_limits(analysis: HeadingLimitAnalysis, save_path: str | Path) -> str | None:
    if not analysis.bins:
        return None

    import matplotlib.pyplot as plt

    xs = [b["heading_deg"] for b in analysis.bins]
    surv = [b["survival"] for b in analysis.bins]
    hdg = [b["mean_heading_rmse_deg"] for b in analysis.bins]
    roll = [b["mean_roll_rmse_deg"] for b in analysis.bins]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax0, ax1 = axes

    ax0.plot(xs, surv, "o-", color="#2ecc71", label="Survival")
    ax0.axhline(analysis.params.get("survival_min", 0.8), color="#2ecc71", ls="--", alpha=0.5)
    if analysis.max_balance_heading_abs_deg is not None:
        for sign in (-1.0, 1.0):
            ax0.axvline(
                sign * analysis.max_balance_heading_abs_deg,
                color="#27ae60",
                ls=":",
                label="Max bal |θ*|" if sign > 0 else None,
            )
    ax0.set_ylabel("Survival rate")
    ax0.set_ylim(-0.05, 1.05)
    ax0.legend(loc="lower center")
    ax0.grid(True, alpha=0.3)

    ax1.plot(xs, hdg, "s-", color="#8e44ad", label="Heading RMSE (°)")
    ax1.plot(xs, roll, "^-", color="#2980b9", label="Roll RMSE (°)")
    ax1.axhline(analysis.params.get("heading_rmse_max_deg", 15.0), color="#8e44ad", ls="--", alpha=0.5)
    ax1.axvline(0.0, color="#7f8c8d", ls=":", alpha=0.6, label="Straight θ*=0")
    ax1.set_xlabel("Commanded heading* (°)")
    ax1.set_ylabel("Tracking RMSE (°)")
    ax1.legend(loc="upper center")
    ax1.grid(True, alpha=0.3)

    fig.suptitle("Heading sweep — fixed vx*, heading* ladder (incl. θ*=0)")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)
