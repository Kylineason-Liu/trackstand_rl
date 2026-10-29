# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/eval/eval_yaw_bias.py (imports/paths rewritten for trackstand_rl locomotive).
"""Straight-line yaw bias from yaw_bias_sweep eval.

Each env holds a fixed (vx*, teleop-style ωz*) on a speed × stick-deg grid.
Per commanded speed, find the bias that zeros mean yaw drift; report the mean
recommended ``--yaw_trim_deg`` across speeds for teleop.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class YawBiasAnalysis:
    """Aggregated yaw-bias sweep results (optionally across multiple vx*)."""

    bins: list[dict[str, Any]] = field(default_factory=list)
    per_speed: list[dict[str, Any]] = field(default_factory=list)
    best_bias_deg: float | None = None
    best_wz_rad_s: float | None = None
    zero_crossing_deg: float | None = None
    recommended_yaw_trim_deg: float | None = None
    mean_zero_crossing_deg: float | None = None
    mean_recommended_yaw_trim_deg: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_episodes(episodes: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for ep in episodes:
        if not getattr(ep, "complete", True):
            continue
        bias = getattr(ep, "yaw_bias_deg", float("nan"))
        if math.isfinite(float(bias)):
            out.append(ep)
    return out


def _episode_vx_ms(ep: Any) -> float:
    for attr in ("spawn_cmd_vx_ms", "mean_cmd_vx_ms", "end_cmd_vx_ms"):
        val = getattr(ep, attr, float("nan"))
        if math.isfinite(float(val)):
            return float(val)
    return float("nan")


def _group_by_bias(episodes: Sequence[Any], round_decimals: int = 2) -> dict[float, list[Any]]:
    groups: dict[float, list[Any]] = {}
    scale = 10 ** round_decimals
    for ep in episodes:
        b = round(float(ep.yaw_bias_deg) * scale) / scale
        groups.setdefault(b, []).append(ep)
    return dict(sorted(groups.items()))


def _group_by_vx(episodes: Sequence[Any], round_decimals: int = 2) -> dict[float, list[Any]]:
    groups: dict[float, list[Any]] = {}
    scale = 10 ** round_decimals
    for ep in episodes:
        vx = _episode_vx_ms(ep)
        if not math.isfinite(vx):
            continue
        key = round(vx * scale) / scale
        groups.setdefault(key, []).append(ep)
    return dict(sorted(groups.items()))


def _mean(vals: Sequence[float]) -> float:
    arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def _crossing_zero(bins: Sequence[dict[str, Any]]) -> float | None:
    """Interpolate bias_deg where mean yaw-drift rate crosses 0."""
    if len(bins) < 2:
        return None
    ordered = sorted(bins, key=lambda b: b["yaw_bias_deg"])
    for i in range(len(ordered) - 1):
        y0 = float(ordered[i]["mean_yaw_drift_rate_deg_per_s"])
        y1 = float(ordered[i + 1]["mean_yaw_drift_rate_deg_per_s"])
        if not (math.isfinite(y0) and math.isfinite(y1)) or y0 == y1:
            continue
        if y0 * y1 > 0:
            continue
        x0 = float(ordered[i]["yaw_bias_deg"])
        x1 = float(ordered[i + 1]["yaw_bias_deg"])
        t = (0.0 - y0) / (y1 - y0)
        return float(x0 + t * (x1 - x0))
    return None


def _trim_from_zero(zero: float | None, best_bias: float | None) -> float | None:
    # Teleop maps stick-right (+) → −ωz* (body RH / CCW). Sweep bins are in
    # raw ωz* stick-deg (positive → +ωz*). Flip sign for --yaw_trim_deg.
    if zero is not None and math.isfinite(zero):
        return float(-zero)
    if best_bias is not None and math.isfinite(best_bias):
        return float(-best_bias)
    return None


def _analyze_bias_bins(
    episodes: Sequence[Any],
    *,
    survival_min: float,
    min_episodes_per_bin: int,
    round_decimals: int,
    bias_min_deg: float | None,
    bias_max_deg: float | None,
    yaw_rate_scale: float,
    stick_full_scale_deg: float,
) -> dict[str, Any]:
    """Single-speed (or pooled) bias ladder → bins + zero-crossing + trim."""
    groups = _group_by_bias(episodes, round_decimals=round_decimals)
    scale = float(yaw_rate_scale) / max(float(stick_full_scale_deg), 1e-6)

    bins: list[dict[str, Any]] = []
    for bias_deg, group in groups.items():
        if bias_min_deg is not None and bias_deg < bias_min_deg - 1e-6:
            continue
        if bias_max_deg is not None and bias_deg > bias_max_deg + 1e-6:
            continue
        n = len(group)
        survived = sum(1 for e in group if e.survived)
        survival = survived / n if n else 0.0
        bins.append(
            {
                "yaw_bias_deg": bias_deg,
                "wz_cmd_rad_s": float(bias_deg) * scale,
                "n": n,
                "survival": survival,
                "mean_episode_length_s": _mean([e.episode_time_s for e in group]),
                "mean_yaw_drift_rate_deg_per_s": _mean(
                    [e.yaw_drift_rate_deg_per_s for e in group]
                ),
                "mean_lateral_deviation_rms_m": _mean(
                    [e.lateral_deviation_rms_m for e in group]
                ),
                "mean_roll_rmse_deg": _mean([e.roll_rmse_deg for e in group]),
                "mean_stable_balance_ratio": _mean(
                    [e.stable_balance_ratio for e in group]
                ),
                "mean_abs_spawn_cmd_wz": _mean(
                    [
                        abs(e.spawn_cmd_wz_rad_s)
                        for e in group
                        if math.isfinite(e.spawn_cmd_wz_rad_s)
                    ]
                ),
                "mean_vx_ms": _mean([_episode_vx_ms(e) for e in group]),
            }
        )

    usable = [
        b
        for b in bins
        if b["n"] >= min_episodes_per_bin
        and b["survival"] >= survival_min
        and math.isfinite(b["mean_yaw_drift_rate_deg_per_s"])
    ]
    best = None
    if usable:
        best = min(usable, key=lambda b: abs(float(b["mean_yaw_drift_rate_deg_per_s"])))

    zero = _crossing_zero(usable if usable else bins)
    raw_best = float(best["yaw_bias_deg"]) if best else None
    recommended = _trim_from_zero(zero, raw_best)

    return {
        "bins": bins,
        "best_bias_deg": raw_best,
        "best_wz_rad_s": float(best["wz_cmd_rad_s"]) if best else None,
        "zero_crossing_deg": zero,
        "recommended_yaw_trim_deg": recommended,
        "n_episodes": len(episodes),
        "n_usable_bins": len(usable),
    }


def analyze_yaw_bias(
    episodes: Sequence[Any],
    *,
    survival_min: float = 0.80,
    min_episodes_per_bin: int = 1,
    round_decimals: int = 2,
    bias_min_deg: float | None = None,
    bias_max_deg: float | None = None,
    yaw_rate_scale: float = 0.45,
    stick_full_scale_deg: float = 10.0,
    vx_round_decimals: int = 2,
) -> YawBiasAnalysis:
    """Bin by (vx*, yaw bias); mean recommended trim across commanded speeds."""
    eps = _finite_episodes(episodes)
    common = dict(
        survival_min=survival_min,
        min_episodes_per_bin=min_episodes_per_bin,
        round_decimals=round_decimals,
        bias_min_deg=bias_min_deg,
        bias_max_deg=bias_max_deg,
        yaw_rate_scale=yaw_rate_scale,
        stick_full_scale_deg=stick_full_scale_deg,
    )

    by_vx = _group_by_vx(eps, round_decimals=vx_round_decimals)
    per_speed: list[dict[str, Any]] = []
    for vx_ms, group in by_vx.items():
        slice_ = _analyze_bias_bins(group, **common)
        per_speed.append(
            {
                "vx_ms": float(vx_ms),
                **slice_,
            }
        )

    # Pooled-by-bias curve (all speeds mixed) for a single overview plot.
    pooled = _analyze_bias_bins(eps, **common)

    trims = [
        float(s["recommended_yaw_trim_deg"])
        for s in per_speed
        if s.get("recommended_yaw_trim_deg") is not None
        and math.isfinite(float(s["recommended_yaw_trim_deg"]))
    ]
    zeros = [
        float(s["zero_crossing_deg"])
        for s in per_speed
        if s.get("zero_crossing_deg") is not None
        and math.isfinite(float(s["zero_crossing_deg"]))
    ]
    mean_trim = float(np.mean(trims)) if trims else None
    mean_zero = float(np.mean(zeros)) if zeros else None

    # Headline trim = mean across speeds when multi-speed; else pooled.
    if len(per_speed) > 1 and mean_trim is not None:
        recommended = mean_trim
        zero = mean_zero
        best_bias = float(-mean_trim) if mean_trim is not None else pooled["best_bias_deg"]
        best_wz = None
        if best_bias is not None:
            scale = float(yaw_rate_scale) / max(float(stick_full_scale_deg), 1e-6)
            best_wz = float(best_bias) * scale
    else:
        recommended = pooled["recommended_yaw_trim_deg"]
        zero = pooled["zero_crossing_deg"]
        best_bias = pooled["best_bias_deg"]
        best_wz = pooled["best_wz_rad_s"]
        if mean_trim is None:
            mean_trim = recommended
        if mean_zero is None:
            mean_zero = zero

    return YawBiasAnalysis(
        bins=pooled["bins"],
        per_speed=per_speed,
        best_bias_deg=best_bias,
        best_wz_rad_s=best_wz,
        zero_crossing_deg=zero,
        recommended_yaw_trim_deg=recommended,
        mean_zero_crossing_deg=mean_zero,
        mean_recommended_yaw_trim_deg=mean_trim,
        params={
            "survival_min": survival_min,
            "min_episodes_per_bin": min_episodes_per_bin,
            "n_episodes": len(eps),
            "n_speeds": len(per_speed),
            "bias_min_deg": bias_min_deg,
            "bias_max_deg": bias_max_deg,
            "yaw_rate_scale": yaw_rate_scale,
            "stick_full_scale_deg": stick_full_scale_deg,
            "vx_values_ms": [s["vx_ms"] for s in per_speed],
        },
    )


def format_yaw_bias_report(analysis: YawBiasAnalysis) -> str:
    p = analysis.params
    n_speeds = int(p.get("n_speeds") or len(analysis.per_speed) or 0)
    title = (
        "YAW BIAS SWEEP (vx* × ωz* stick-deg grid → mean trim)"
        if n_speeds > 1
        else "YAW BIAS SWEEP (fixed vx*, ωz* stick-deg ladder)"
    )
    lines = [
        "",
        f"{title:^78}",
        f"{'─' * 78}",
        f"  Episodes analysed : {p.get('n_episodes', 0)}",
        f"  Speed levels      : {n_speeds}",
        f"  Survival filter   : ≥ {p.get('survival_min', 0.8):.0%} per bin",
        f"  Mapping           : ωz* = {p.get('yaw_rate_scale', 0.45):g} "
        f"* (bias_deg / {p.get('stick_full_scale_deg', 10.0):g})",
        "",
    ]

    def _fmt_deg(val: float | None) -> str:
        return f"{val:+.2f}°" if val is not None and math.isfinite(val) else "n/a"

    def _fmt_wz(val: float | None) -> str:
        return f"{val:+.4f} rad/s" if val is not None and math.isfinite(val) else "n/a"

    lines.append(
        f"  Mean recommended teleop --yaw_trim_deg : "
        f"{_fmt_deg(analysis.mean_recommended_yaw_trim_deg)}"
    )
    lines.append(
        f"  Mean zero-crossing bias (stick deg)    : "
        f"{_fmt_deg(analysis.mean_zero_crossing_deg)}"
    )
    lines.append(
        f"  Headline recommended --yaw_trim_deg    : "
        f"{_fmt_deg(analysis.recommended_yaw_trim_deg)}"
    )
    lines.append(
        "  (teleop: stick-right → −ωz*; trim = −sweep_bias so stick0 cancels model bias)"
    )

    if analysis.per_speed:
        lines += [
            "",
            f"{'vx*':>6} {'trim°':>8} {'zero°':>8} {'best°':>8} {'eps':>5} {'bins':>5}",
            f"{'─' * 78}",
        ]
        for s in analysis.per_speed:
            lines.append(
                f"{s['vx_ms']:6.2f} "
                f"{_fmt_deg(s.get('recommended_yaw_trim_deg')):>8} "
                f"{_fmt_deg(s.get('zero_crossing_deg')):>8} "
                f"{_fmt_deg(s.get('best_bias_deg')):>8} "
                f"{int(s.get('n_episodes', 0)):5d} "
                f"{int(s.get('n_usable_bins', 0)):5d}"
            )

    if analysis.bins:
        lines += [
            "",
            "Pooled across speeds (bias bins):",
            f"{'bias°':>8} {'ωz*':>9} {'n':>4} {'surv':>7} {'len(s)':>7} "
            f"{'yaw°/s':>9} {'latRMS':>8} {'rollRMSE':>9}",
            f"{'─' * 78}",
        ]
        for b in analysis.bins:
            lines.append(
                f"{b['yaw_bias_deg']:8.2f} {b['wz_cmd_rad_s']:9.4f} {b['n']:4d} "
                f"{b['survival']:6.0%} {b['mean_episode_length_s']:7.2f} "
                f"{b['mean_yaw_drift_rate_deg_per_s']:9.3f} "
                f"{b['mean_lateral_deviation_rms_m']:8.3f} "
                f"{b['mean_roll_rmse_deg']:9.2f}"
            )
    lines.append(f"{'─' * 78}")
    return "\n".join(lines)


def export_yaw_bias(analysis: YawBiasAnalysis, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(analysis.to_dict(), f, indent=2)
    return str(path)


def plot_yaw_bias(analysis: YawBiasAnalysis, save_path: str | Path) -> str | None:
    if not analysis.bins and not analysis.per_speed:
        return None

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax0, ax1, ax2 = axes

    # Per-speed drift curves when available; else pooled.
    if len(analysis.per_speed) > 1:
        cmap = plt.cm.viridis  # type: ignore[attr-defined]
        for i, s in enumerate(analysis.per_speed):
            bins = s.get("bins") or []
            if not bins:
                continue
            xs = [b["yaw_bias_deg"] for b in bins]
            drift = [b["mean_yaw_drift_rate_deg_per_s"] for b in bins]
            color = cmap(i / max(len(analysis.per_speed) - 1, 1))
            ax0.plot(xs, drift, "o-", color=color, ms=3, label=f"vx*={s['vx_ms']:.2f}")
            lat = [b["mean_lateral_deviation_rms_m"] for b in bins]
            surv = [b["survival"] for b in bins]
            ax1.plot(xs, lat, "o-", color=color, ms=3)
            ax2.plot(xs, surv, "o-", color=color, ms=3)
        ax0.legend(loc="best", fontsize=7, ncol=2)
    elif analysis.bins:
        xs = [b["yaw_bias_deg"] for b in analysis.bins]
        ax0.plot(
            xs,
            [b["mean_yaw_drift_rate_deg_per_s"] for b in analysis.bins],
            "o-",
            color="#e67e22",
            label="Yaw drift rate",
        )
        ax1.plot(
            xs,
            [b["mean_lateral_deviation_rms_m"] for b in analysis.bins],
            "o-",
            color="#9b59b6",
        )
        ax2.plot(
            xs,
            [b["survival"] for b in analysis.bins],
            "o-",
            color="#2ecc71",
        )

    ax0.axhline(0.0, color="#7f8c8d", ls="--", alpha=0.6)
    zero = analysis.mean_zero_crossing_deg or analysis.zero_crossing_deg
    trim = analysis.mean_recommended_yaw_trim_deg or analysis.recommended_yaw_trim_deg
    if zero is not None:
        ax0.axvline(
            zero,
            color="#3498db",
            ls="--",
            alpha=0.7,
            label=(
                f"zero bias={zero:+.2f}°"
                + (f" → trim={trim:+.2f}°" if trim is not None else "")
            ),
        )
        ax0.legend(loc="best", fontsize=8)
    ax0.set_ylabel("Yaw drift (°/s)")
    ax0.grid(True, alpha=0.3)

    ax1.set_ylabel("Lateral RMS (m)")
    ax1.grid(True, alpha=0.3)

    ax2.axhline(analysis.params.get("survival_min", 0.8), color="#2ecc71", ls="--", alpha=0.5)
    ax2.set_xlabel("Commanded yaw bias (stick deg)")
    ax2.set_ylabel("Survival")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    n_speeds = analysis.params.get("n_speeds", len(analysis.per_speed))
    fig.suptitle(f"Yaw bias sweep ({n_speeds} speed level(s) → mean trim)")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return str(save_path)
