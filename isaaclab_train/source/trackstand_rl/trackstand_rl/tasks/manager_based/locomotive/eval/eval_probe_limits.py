# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/eval/eval_probe_limits.py (imports/paths rewritten for trackstand_rl locomotive).
"""Envelope limits for mass / friction / kick / CoM ladder evals.

Mirrors ``eval_drive_accel``: bin episodes by the ladder field and report the
highest (mass, kick, |CoM|) or lowest (friction) still-balancing rung.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _finite_by_attr(episodes: Sequence[Any], attr: str) -> list[Any]:
    out: list[Any] = []
    for ep in episodes:
        if not getattr(ep, "complete", True):
            continue
        val = getattr(ep, attr, float("nan"))
        if math.isfinite(float(val)):
            out.append(ep)
    return out


def _first_episode_per_env(episodes: Sequence[Any]) -> list[Any]:
    """Keep the first logged episode per ``env_id`` (ladder rung).

    Crash-looping extreme rungs otherwise flood bins and dominate mode survival.
    """
    seen: set[Any] = set()
    out: list[Any] = []
    for ep in episodes:
        eid = getattr(ep, "env_id", None)
        if eid is None:
            out.append(ep)
            continue
        if eid in seen:
            continue
        seen.add(eid)
        out.append(ep)
    return out


def _group_by_attr(
    episodes: Sequence[Any], attr: str, round_decimals: int
) -> dict[float, list[Any]]:
    groups: dict[float, list[Any]] = {}
    scale = 10 ** round_decimals
    for ep in episodes:
        key = round(float(getattr(ep, attr)) * scale) / scale
        groups.setdefault(key, []).append(ep)
    return dict(sorted(groups.items()))


def _mean(vals: Sequence[float]) -> float:
    arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def _crossing(
    bins: Sequence[dict[str, Any]],
    x_key: str,
    y_key: str,
    target: float,
) -> float | None:
    if len(bins) < 2:
        return None
    ordered = sorted(bins, key=lambda b: b[x_key])
    for i in range(len(ordered) - 1):
        y0 = float(ordered[i][y_key])
        y1 = float(ordered[i + 1][y_key])
        if not (math.isfinite(y0) and math.isfinite(y1)) or y0 == y1:
            continue
        if (y0 - target) * (y1 - target) > 0:
            continue
        x0 = float(ordered[i][x_key])
        x1 = float(ordered[i + 1][x_key])
        t = (target - y0) / (y1 - y0)
        return float(x0 + t * (x1 - x0))
    return None


def _balanced(b: dict[str, Any], survival_min: float, stable_balance_min: float) -> bool:
    return (
        b["survival"] >= survival_min
        and b["mean_stable_balance_ratio"] >= stable_balance_min
    )


# ---------------------------------------------------------------------------
# Mass
# ---------------------------------------------------------------------------


@dataclass
class MassLimitAnalysis:
    bins: list[dict[str, Any]] = field(default_factory=list)
    max_balance_mass_scale: float | None = None
    min_fail_mass_scale: float | None = None
    balance_crossing_mass_scale: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_mass_limits(
    episodes: Sequence[Any],
    *,
    survival_min: float = 0.80,
    stable_balance_min: float = 0.50,
    min_episodes_per_bin: int = 1,
    round_decimals: int = 3,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> MassLimitAnalysis:
    eps = _first_episode_per_env(_finite_by_attr(episodes, "mass_scale"))
    groups = _group_by_attr(eps, "mass_scale", round_decimals)
    bins: list[dict[str, Any]] = []
    for scale, group in groups.items():
        if scale_min is not None and scale < scale_min - 1e-6:
            continue
        if scale_max is not None and scale > scale_max + 1e-6:
            continue
        n = len(group)
        survived = sum(1 for e in group if e.survived)
        bins.append(
            {
                "mass_scale": scale,
                "n": n,
                "survival": survived / n if n else 0.0,
                "mean_episode_length_s": _mean([e.episode_time_s for e in group]),
                "mean_stable_balance_ratio": _mean([e.stable_balance_ratio for e in group]),
                "mean_roll_rmse_deg": _mean([e.roll_rmse_deg for e in group]),
                "mean_vx_rmse_ms": _mean([e.vx_rmse_ms for e in group]),
            }
        )

    passing = [
        b
        for b in bins
        if b["n"] >= min_episodes_per_bin and _balanced(b, survival_min, stable_balance_min)
    ]
    failing = [
        b
        for b in bins
        if b["n"] >= min_episodes_per_bin and not _balanced(b, survival_min, stable_balance_min)
    ]
    return MassLimitAnalysis(
        bins=bins,
        max_balance_mass_scale=float(max(b["mass_scale"] for b in passing)) if passing else None,
        min_fail_mass_scale=float(min(b["mass_scale"] for b in failing)) if failing else None,
        balance_crossing_mass_scale=_crossing(bins, "mass_scale", "survival", survival_min),
        params={
            "survival_min": survival_min,
            "stable_balance_min": stable_balance_min,
            "min_episodes_per_bin": min_episodes_per_bin,
            "n_episodes": len(eps),
            "scale_min": scale_min,
            "scale_max": scale_max,
        },
    )


def format_mass_limits_report(analysis: MassLimitAnalysis) -> str:
    p = analysis.params

    def _fmt(val: float | None) -> str:
        return f"×{val:.3f}" if val is not None and math.isfinite(val) else "n/a"

    lines = [
        "",
        f"{'MASS SCALE SWEEP (CoM fixed at nominal)':^74}",
        f"{'─' * 74}",
        f"  Episodes analysed : {p.get('n_episodes', 0)}",
        f"  Balance gate      : survival ≥ {p.get('survival_min', 0.8):.0%}, "
        f"stable balance ≥ {p.get('stable_balance_min', 0.5):.0%}",
        "",
        f"  Max balancing mass scale : {_fmt(analysis.max_balance_mass_scale)}",
        f"  Min failing mass scale   : {_fmt(analysis.min_fail_mass_scale)}",
    ]
    if analysis.balance_crossing_mass_scale is not None:
        lines.append(f"  Survival crossing        : {_fmt(analysis.balance_crossing_mass_scale)}")
    if analysis.bins:
        lines += [
            "",
            f"{'scale':>8} {'n':>4} {'surv':>7} {'len(s)':>7} {'stable':>7} {'rollRMSE':>9} {'vxRMSE':>8}",
            f"{'─' * 74}",
        ]
        for b in analysis.bins:
            lines.append(
                f"{b['mass_scale']:8.3f} {b['n']:4d} {b['survival']:6.0%} "
                f"{b['mean_episode_length_s']:7.2f} {b['mean_stable_balance_ratio']:6.0%} "
                f"{b['mean_roll_rmse_deg']:9.2f} {b['mean_vx_rmse_ms']:8.3f}"
            )
    lines.append(f"{'─' * 74}")
    return "\n".join(lines)


def export_mass_limits(analysis: MassLimitAnalysis, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis.to_dict(), indent=2))
    return str(path)


def plot_mass_limits(analysis: MassLimitAnalysis, save_path: str | Path) -> str | None:
    if not analysis.bins:
        return None
    import matplotlib.pyplot as plt

    xs = [b["mass_scale"] for b in analysis.bins]
    surv = [b["survival"] for b in analysis.bins]
    roll = [b["mean_roll_rmse_deg"] for b in analysis.bins]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax0, ax1 = axes
    ax0.plot(xs, surv, "o-", color="#2ecc71", label="Survival")
    ax0.axhline(analysis.params.get("survival_min", 0.8), color="#2ecc71", ls="--", alpha=0.5)
    if analysis.max_balance_mass_scale is not None:
        ax0.axvline(analysis.max_balance_mass_scale, color="#27ae60", ls=":", label="Max balance scale")
    ax0.set_ylabel("Survival rate")
    ax0.set_ylim(-0.05, 1.05)
    ax0.legend(loc="lower left")
    ax0.grid(True, alpha=0.3)
    ax1.plot(xs, roll, "s-", color="#8e44ad", label="Roll RMSE (°)")
    ax1.set_xlabel("Mass scale (× nominal)")
    ax1.set_ylabel("Roll RMSE (°)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.suptitle("Mass sweep — scale ladder, CoM fixed")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


# ---------------------------------------------------------------------------
# Friction
# ---------------------------------------------------------------------------


@dataclass
class FrictionLimitAnalysis:
    bins: list[dict[str, Any]] = field(default_factory=list)
    min_friction_mu: float | None = None
    max_fail_friction_mu: float | None = None
    balance_crossing_mu: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_friction_limits(
    episodes: Sequence[Any],
    *,
    survival_min: float = 0.80,
    stable_balance_min: float = 0.50,
    min_episodes_per_bin: int = 1,
    round_decimals: int = 3,
    mu_min: float | None = None,
    mu_max: float | None = None,
) -> FrictionLimitAnalysis:
    """Lowest μ still balancing (descending envelope)."""
    eps = _first_episode_per_env(_finite_by_attr(episodes, "friction_mu"))
    groups = _group_by_attr(eps, "friction_mu", round_decimals)
    bins: list[dict[str, Any]] = []
    for mu, group in groups.items():
        if mu_min is not None and mu < mu_min - 1e-6:
            continue
        if mu_max is not None and mu > mu_max + 1e-6:
            continue
        n = len(group)
        survived = sum(1 for e in group if e.survived)
        bins.append(
            {
                "friction_mu": mu,
                "n": n,
                "survival": survived / n if n else 0.0,
                "mean_episode_length_s": _mean([e.episode_time_s for e in group]),
                "mean_stable_balance_ratio": _mean([e.stable_balance_ratio for e in group]),
                "mean_roll_rmse_deg": _mean([e.roll_rmse_deg for e in group]),
                "mean_drive_clip_sat_ratio": _mean([e.drive_clip_sat_ratio for e in group]),
                "mean_vx_ss_mae_ms": _mean([e.vx_ss_mae_ms for e in group]),
            }
        )

    passing = [
        b
        for b in bins
        if b["n"] >= min_episodes_per_bin and _balanced(b, survival_min, stable_balance_min)
    ]
    failing = [
        b
        for b in bins
        if b["n"] >= min_episodes_per_bin and not _balanced(b, survival_min, stable_balance_min)
    ]
    return FrictionLimitAnalysis(
        bins=bins,
        min_friction_mu=float(min(b["friction_mu"] for b in passing)) if passing else None,
        max_fail_friction_mu=float(max(b["friction_mu"] for b in failing)) if failing else None,
        balance_crossing_mu=_crossing(bins, "friction_mu", "survival", survival_min),
        params={
            "survival_min": survival_min,
            "stable_balance_min": stable_balance_min,
            "min_episodes_per_bin": min_episodes_per_bin,
            "n_episodes": len(eps),
            "mu_min": mu_min,
            "mu_max": mu_max,
        },
    )


def format_friction_limits_report(analysis: FrictionLimitAnalysis) -> str:
    p = analysis.params

    def _fmt(val: float | None) -> str:
        return f"μ={val:.3f}" if val is not None and math.isfinite(val) else "n/a"

    lines = [
        "",
        f"{'FRICTION SWEEP (tire μ ladder; plane at μ_min)':^74}",
        f"{'─' * 74}",
        f"  Episodes analysed : {p.get('n_episodes', 0)}",
        f"  Balance gate      : survival ≥ {p.get('survival_min', 0.8):.0%}, "
        f"stable balance ≥ {p.get('stable_balance_min', 0.5):.0%}",
        "",
        f"  Min balancing friction : {_fmt(analysis.min_friction_mu)}",
        f"  Max failing friction   : {_fmt(analysis.max_fail_friction_mu)}",
    ]
    if analysis.balance_crossing_mu is not None:
        lines.append(f"  Survival crossing      : {_fmt(analysis.balance_crossing_mu)}")
    if analysis.bins:
        lines += [
            "",
            f"{'μ':>8} {'n':>4} {'surv':>7} {'len(s)':>7} {'stable':>7} {'rollRMSE':>9} {'drvClip':>8}",
            f"{'─' * 74}",
        ]
        for b in analysis.bins:
            lines.append(
                f"{b['friction_mu']:8.3f} {b['n']:4d} {b['survival']:6.0%} "
                f"{b['mean_episode_length_s']:7.2f} {b['mean_stable_balance_ratio']:6.0%} "
                f"{b['mean_roll_rmse_deg']:9.2f} {b['mean_drive_clip_sat_ratio']:8.3f}"
            )
    lines.append(f"{'─' * 74}")
    return "\n".join(lines)


def export_friction_limits(analysis: FrictionLimitAnalysis, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis.to_dict(), indent=2))
    return str(path)


def plot_friction_limits(analysis: FrictionLimitAnalysis, save_path: str | Path) -> str | None:
    if not analysis.bins:
        return None
    import matplotlib.pyplot as plt

    xs = [b["friction_mu"] for b in analysis.bins]
    surv = [b["survival"] for b in analysis.bins]
    roll = [b["mean_roll_rmse_deg"] for b in analysis.bins]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax0, ax1 = axes
    ax0.plot(xs, surv, "o-", color="#2ecc71", label="Survival")
    ax0.axhline(analysis.params.get("survival_min", 0.8), color="#2ecc71", ls="--", alpha=0.5)
    if analysis.min_friction_mu is not None:
        ax0.axvline(analysis.min_friction_mu, color="#27ae60", ls=":", label="Min balance μ")
    ax0.set_ylabel("Survival rate")
    ax0.set_ylim(-0.05, 1.05)
    ax0.legend(loc="lower right")
    ax0.grid(True, alpha=0.3)
    ax1.plot(xs, roll, "s-", color="#8e44ad", label="Roll RMSE (°)")
    ax1.set_xlabel("Tire friction μ")
    ax1.set_ylabel("Roll RMSE (°)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    fig.suptitle("Friction sweep — tire μ ladder")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


# ---------------------------------------------------------------------------
# Kick
# ---------------------------------------------------------------------------


@dataclass
class KickLimitAnalysis:
    bins: list[dict[str, Any]] = field(default_factory=list)
    max_kick_force_n: float | None = None
    min_fail_kick_force_n: float | None = None
    balance_crossing_force_n: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_kick_limits(
    episodes: Sequence[Any],
    *,
    survival_min: float = 0.80,
    stable_balance_min: float = 0.50,
    min_episodes_per_bin: int = 1,
    round_decimals: int = 2,
    force_min: float | None = None,
    force_max: float | None = None,
) -> KickLimitAnalysis:
    eps = _first_episode_per_env(_finite_by_attr(episodes, "kick_force_n"))
    groups = _group_by_attr(eps, "kick_force_n", round_decimals)
    bins: list[dict[str, Any]] = []
    for force, group in groups.items():
        if force_min is not None and force < force_min - 1e-6:
            continue
        if force_max is not None and force > force_max + 1e-6:
            continue
        n = len(group)
        survived = sum(1 for e in group if e.survived)
        settle_n = sum(1 for e in group if getattr(e, "kick_settled", False))
        bins.append(
            {
                "kick_force_n": force,
                "n": n,
                "survival": survived / n if n else 0.0,
                "mean_episode_length_s": _mean([e.episode_time_s for e in group]),
                "mean_stable_balance_ratio": _mean([e.stable_balance_ratio for e in group]),
                "kick_settle_rate": settle_n / n if n else 0.0,
                "mean_kick_settle_time_s": _mean([e.kick_settle_time_s for e in group]),
                "mean_kick_peak_roll_deg": _mean([e.kick_peak_roll_deg for e in group]),
                "mean_kick_oscillations": _mean([float(e.kick_oscillations) for e in group]),
            }
        )

    passing = [
        b
        for b in bins
        if b["n"] >= min_episodes_per_bin and _balanced(b, survival_min, stable_balance_min)
    ]
    failing = [
        b
        for b in bins
        if b["n"] >= min_episodes_per_bin and not _balanced(b, survival_min, stable_balance_min)
    ]
    return KickLimitAnalysis(
        bins=bins,
        max_kick_force_n=float(max(b["kick_force_n"] for b in passing)) if passing else None,
        min_fail_kick_force_n=float(min(b["kick_force_n"] for b in failing)) if failing else None,
        balance_crossing_force_n=_crossing(bins, "kick_force_n", "survival", survival_min),
        params={
            "survival_min": survival_min,
            "stable_balance_min": stable_balance_min,
            "min_episodes_per_bin": min_episodes_per_bin,
            "n_episodes": len(eps),
            "force_min": force_min,
            "force_max": force_max,
        },
    )


def format_kick_limits_report(analysis: KickLimitAnalysis) -> str:
    p = analysis.params

    def _fmt(val: float | None) -> str:
        return f"{val:.2f} N" if val is not None and math.isfinite(val) else "n/a"

    lines = [
        "",
        f"{'KICK FORCE SWEEP (lateral impulse at t=2 s, 0.1 s)':^74}",
        f"{'─' * 74}",
        f"  Episodes analysed : {p.get('n_episodes', 0)}",
        f"  Balance gate      : survival ≥ {p.get('survival_min', 0.8):.0%}, "
        f"stable balance ≥ {p.get('stable_balance_min', 0.5):.0%}",
        "",
        f"  Max balancing kick force : {_fmt(analysis.max_kick_force_n)}",
        f"  Min failing kick force   : {_fmt(analysis.min_fail_kick_force_n)}",
    ]
    if analysis.balance_crossing_force_n is not None:
        lines.append(f"  Survival crossing        : {_fmt(analysis.balance_crossing_force_n)}")
    if analysis.bins:
        lines += [
            "",
            f"{'F (N)':>8} {'n':>4} {'surv':>7} {'settle':>7} {'peakφ':>7} {'osc':>5} {'t_set':>7}",
            f"{'─' * 74}",
        ]
        for b in analysis.bins:
            lines.append(
                f"{b['kick_force_n']:8.2f} {b['n']:4d} {b['survival']:6.0%} "
                f"{b['kick_settle_rate']:6.0%} {b['mean_kick_peak_roll_deg']:7.1f} "
                f"{b['mean_kick_oscillations']:5.1f} {b['mean_kick_settle_time_s']:7.2f}"
            )
    lines.append(f"{'─' * 74}")
    return "\n".join(lines)


def export_kick_limits(analysis: KickLimitAnalysis, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis.to_dict(), indent=2))
    return str(path)


def plot_kick_limits(analysis: KickLimitAnalysis, save_path: str | Path) -> str | None:
    if not analysis.bins:
        return None
    import matplotlib.pyplot as plt

    xs = [b["kick_force_n"] for b in analysis.bins]
    surv = [b["survival"] for b in analysis.bins]
    peak = [b["mean_kick_peak_roll_deg"] for b in analysis.bins]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax0, ax1 = axes
    ax0.plot(xs, surv, "o-", color="#2ecc71", label="Survival")
    ax0.axhline(analysis.params.get("survival_min", 0.8), color="#2ecc71", ls="--", alpha=0.5)
    if analysis.max_kick_force_n is not None:
        ax0.axvline(analysis.max_kick_force_n, color="#27ae60", ls=":", label="Max kick force")
    ax0.set_ylabel("Survival rate")
    ax0.set_ylim(-0.05, 1.05)
    ax0.legend(loc="lower left")
    ax0.grid(True, alpha=0.3)
    ax1.plot(xs, peak, "s-", color="#c0392b", label="Peak roll (°)")
    ax1.set_xlabel("Lateral kick force (N)")
    ax1.set_ylabel("Peak roll (°)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.suptitle("Kick sweep — lateral impulse force ladder")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


# ---------------------------------------------------------------------------
# CoM (lateral offset)
# ---------------------------------------------------------------------------


@dataclass
class ComLimitAnalysis:
    bins: list[dict[str, Any]] = field(default_factory=list)
    max_balance_com_abs_m: float | None = None
    min_fail_com_abs_m: float | None = None
    balance_crossing_com_abs_m: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_com_limits(
    episodes: Sequence[Any],
    *,
    survival_min: float = 0.80,
    stable_balance_min: float = 0.50,
    min_episodes_per_bin: int = 1,
    round_decimals: int = 4,
    offset_min_m: float | None = None,
    offset_max_m: float | None = None,
) -> ComLimitAnalysis:
    """Largest |CoM offset| still balancing on a signed lateral ladder."""
    eps = _first_episode_per_env(_finite_by_attr(episodes, "com_offset_m"))
    groups = _group_by_attr(eps, "com_offset_m", round_decimals)
    bins: list[dict[str, Any]] = []
    for offset, group in groups.items():
        if offset_min_m is not None and offset < offset_min_m - 1e-9:
            continue
        if offset_max_m is not None and offset > offset_max_m + 1e-9:
            continue
        n = len(group)
        survived = sum(1 for e in group if e.survived)
        bins.append(
            {
                "com_offset_m": offset,
                "com_abs_m": abs(offset),
                "n": n,
                "survival": survived / n if n else 0.0,
                "mean_episode_length_s": _mean([e.episode_time_s for e in group]),
                "mean_stable_balance_ratio": _mean([e.stable_balance_ratio for e in group]),
                "mean_roll_rmse_deg": _mean([e.roll_rmse_deg for e in group]),
                "mean_yaw_drift_rate_deg_per_s": _mean(
                    [getattr(e, "yaw_drift_rate_deg_per_s", float("nan")) for e in group]
                ),
                "mean_lateral_deviation_rms_m": _mean(
                    [getattr(e, "lateral_deviation_rms_m", float("nan")) for e in group]
                ),
            }
        )

    passing = [
        b
        for b in bins
        if b["n"] >= min_episodes_per_bin and _balanced(b, survival_min, stable_balance_min)
    ]
    failing = [
        b
        for b in bins
        if b["n"] >= min_episodes_per_bin and not _balanced(b, survival_min, stable_balance_min)
    ]

    # Crossing on |offset|: fold bins to abs, take worst survival at each |y|.
    abs_groups: dict[float, list[dict[str, Any]]] = {}
    for b in bins:
        key = round(float(b["com_abs_m"]) * (10 ** round_decimals)) / (10 ** round_decimals)
        abs_groups.setdefault(key, []).append(b)
    abs_bins: list[dict[str, Any]] = []
    for a, group in sorted(abs_groups.items()):
        survs = [float(g["survival"]) for g in group]
        abs_bins.append({"com_abs_m": a, "survival": min(survs) if survs else 0.0})

    return ComLimitAnalysis(
        bins=bins,
        max_balance_com_abs_m=float(max(b["com_abs_m"] for b in passing)) if passing else None,
        min_fail_com_abs_m=float(min(b["com_abs_m"] for b in failing)) if failing else None,
        balance_crossing_com_abs_m=_crossing(abs_bins, "com_abs_m", "survival", survival_min),
        params={
            "survival_min": survival_min,
            "stable_balance_min": stable_balance_min,
            "min_episodes_per_bin": min_episodes_per_bin,
            "n_episodes": len(eps),
            "offset_min_m": offset_min_m,
            "offset_max_m": offset_max_m,
        },
    )


def format_com_limits_report(analysis: ComLimitAnalysis) -> str:
    p = analysis.params

    def _fmt(val: float | None) -> str:
        return f"{val * 100:.2f} cm" if val is not None and math.isfinite(val) else "n/a"

    lines = [
        "",
        f"{'CoM OFFSET SWEEP (lateral y, mass nominal)':^74}",
        f"{'─' * 74}",
        f"  Episodes analysed : {p.get('n_episodes', 0)}",
        f"  Balance gate      : survival ≥ {p.get('survival_min', 0.8):.0%}, "
        f"stable balance ≥ {p.get('stable_balance_min', 0.5):.0%}",
        "",
        f"  Max balancing |CoM| : {_fmt(analysis.max_balance_com_abs_m)}",
        f"  Min failing |CoM|   : {_fmt(analysis.min_fail_com_abs_m)}",
    ]
    if analysis.balance_crossing_com_abs_m is not None:
        lines.append(f"  Survival crossing   : {_fmt(analysis.balance_crossing_com_abs_m)}")
    if analysis.bins:
        lines += [
            "",
            f"{'y (cm)':>8} {'|y|':>6} {'n':>4} {'surv':>7} {'len(s)':>7} {'stable':>7} {'rollRMSE':>9}",
            f"{'─' * 74}",
        ]
        for b in analysis.bins:
            lines.append(
                f"{b['com_offset_m'] * 100:8.2f} {b['com_abs_m'] * 100:6.2f} {b['n']:4d} "
                f"{b['survival']:6.0%} {b['mean_episode_length_s']:7.2f} "
                f"{b['mean_stable_balance_ratio']:6.0%} {b['mean_roll_rmse_deg']:9.2f}"
            )
    lines.append(f"{'─' * 74}")
    return "\n".join(lines)


def export_com_limits(analysis: ComLimitAnalysis, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis.to_dict(), indent=2))
    return str(path)


def plot_com_limits(analysis: ComLimitAnalysis, save_path: str | Path) -> str | None:
    if not analysis.bins:
        return None
    import matplotlib.pyplot as plt

    xs = [b["com_offset_m"] * 100.0 for b in analysis.bins]
    surv = [b["survival"] for b in analysis.bins]
    roll = [b["mean_roll_rmse_deg"] for b in analysis.bins]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax0, ax1 = axes
    ax0.plot(xs, surv, "o-", color="#2ecc71", label="Survival")
    ax0.axhline(analysis.params.get("survival_min", 0.8), color="#2ecc71", ls="--", alpha=0.5)
    if analysis.max_balance_com_abs_m is not None:
        lim = analysis.max_balance_com_abs_m * 100.0
        ax0.axvline(lim, color="#27ae60", ls=":", label="Max |CoM|")
        ax0.axvline(-lim, color="#27ae60", ls=":")
    ax0.set_ylabel("Survival rate")
    ax0.set_ylim(-0.05, 1.05)
    ax0.legend(loc="lower center")
    ax0.grid(True, alpha=0.3)
    ax1.plot(xs, roll, "s-", color="#8e44ad", label="Roll RMSE (°)")
    ax1.set_xlabel("Lateral CoM offset y (cm)")
    ax1.set_ylabel("Roll RMSE (°)")
    ax1.legend(loc="upper center")
    ax1.grid(True, alpha=0.3)
    fig.suptitle("CoM sweep — lateral offset ladder (mass nominal)")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


# ---------------------------------------------------------------------------
# Mass × CoM joint grid
# ---------------------------------------------------------------------------


@dataclass
class MassComLimitAnalysis:
    bins: list[dict[str, Any]] = field(default_factory=list)
    mass_frontier: list[dict[str, Any]] = field(default_factory=list)
    com_frontier: list[dict[str, Any]] = field(default_factory=list)
    # Probe targets live in params; field names kept for report compatibility.
    max_balance_mass_at_com_2cm: float | None = None
    max_balance_com_abs_at_mass_1_10: float | None = None
    probe_mass_1_10_com_2cm_survival: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Default joint-grid probe: inside the shared non-zero band at ×1.10
# (≈0.3–0.9 cm). Old 2 cm / 1.5 cm targets sit past tip for weak recipes → 0%.
_DEFAULT_PROBE_MASS_SCALE = 1.10
_DEFAULT_PROBE_COM_ABS_M = 0.005


def _nearest_bin_survival(
    bins: Sequence[dict[str, Any]], mass_target: float, com_abs_target: float
) -> float | None:
    if not bins:
        return None
    best = None
    best_d = float("inf")
    for b in bins:
        d = abs(float(b["mass_scale"]) - mass_target) + abs(float(b["com_abs_m"]) - com_abs_target)
        if d < best_d:
            best_d = d
            best = b
    return None if best is None else float(best["survival"])


def _interp_frontier_y(frontier: Sequence[dict[str, Any]], x_key: str, y_key: str, x_target: float) -> float | None:
    if not frontier:
        return None
    ordered = sorted(frontier, key=lambda b: b[x_key])
    xs = [float(b[x_key]) for b in ordered]
    ys = [float(b[y_key]) for b in ordered]
    # Do not extrapolate past the measured frontier.
    if x_target < xs[0] - 1e-12 or x_target > xs[-1] + 1e-12:
        return None
    if abs(x_target - xs[0]) <= 1e-12:
        return ys[0] if math.isfinite(ys[0]) else None
    if abs(x_target - xs[-1]) <= 1e-12:
        return ys[-1] if math.isfinite(ys[-1]) else None
    for i in range(len(xs) - 1):
        if xs[i] <= x_target <= xs[i + 1] and xs[i + 1] != xs[i]:
            t = (x_target - xs[i]) / (xs[i + 1] - xs[i])
            return float(ys[i] + t * (ys[i + 1] - ys[i]))
    return None


def analyze_mass_com_limits(
    episodes: Sequence[Any],
    *,
    survival_min: float = 0.80,
    stable_balance_min: float = 0.50,
    min_episodes_per_bin: int = 1,
    mass_round: int = 3,
    com_round: int = 4,
    scale_min: float | None = None,
    scale_max: float | None = None,
    offset_min_m: float | None = None,
    offset_max_m: float | None = None,
    probe_mass_scale: float = _DEFAULT_PROBE_MASS_SCALE,
    probe_com_abs_m: float = _DEFAULT_PROBE_COM_ABS_M,
) -> MassComLimitAnalysis:
    """2-D mass × |CoM| envelope from a joint grid eval."""
    raw = _first_episode_per_env(_finite_by_attr(episodes, "mass_scale"))
    eps = [e for e in raw if math.isfinite(float(getattr(e, "com_offset_m", float("nan"))))]

    # Group by (mass, |com|) — fold ±y into abs so signed grid densifies |com| bins.
    groups: dict[tuple[float, float], list[Any]] = {}
    ms = 10 ** mass_round
    cs = 10 ** com_round
    for ep in eps:
        mass = round(float(ep.mass_scale) * ms) / ms
        com_abs = round(abs(float(ep.com_offset_m)) * cs) / cs
        if scale_min is not None and mass < scale_min - 1e-6:
            continue
        if scale_max is not None and mass > scale_max + 1e-6:
            continue
        if offset_min_m is not None and float(ep.com_offset_m) < offset_min_m - 1e-9:
            continue
        if offset_max_m is not None and float(ep.com_offset_m) > offset_max_m + 1e-9:
            continue
        groups.setdefault((mass, com_abs), []).append(ep)

    bins: list[dict[str, Any]] = []
    for (mass, com_abs), group in sorted(groups.items()):
        n = len(group)
        survived = sum(1 for e in group if e.survived)
        bins.append(
            {
                "mass_scale": mass,
                "com_abs_m": com_abs,
                "n": n,
                "survival": survived / n if n else 0.0,
                "mean_episode_length_s": _mean([e.episode_time_s for e in group]),
                "mean_stable_balance_ratio": _mean([e.stable_balance_ratio for e in group]),
                "mean_roll_rmse_deg": _mean([e.roll_rmse_deg for e in group]),
            }
        )

    def _ok(b: dict[str, Any]) -> bool:
        return b["n"] >= min_episodes_per_bin and _balanced(b, survival_min, stable_balance_min)

    # Per mass level: max |com| still balancing.
    by_mass: dict[float, list[dict[str, Any]]] = {}
    for b in bins:
        by_mass.setdefault(float(b["mass_scale"]), []).append(b)
    mass_frontier: list[dict[str, Any]] = []
    for mass, group in sorted(by_mass.items()):
        passing = [b for b in group if _ok(b)]
        mass_frontier.append(
            {
                "mass_scale": mass,
                "max_balance_com_abs_m": (
                    float(max(b["com_abs_m"] for b in passing)) if passing else None
                ),
                "n_bins": len(group),
                "mean_survival": _mean([b["survival"] for b in group]),
            }
        )

    # Per |com| level: max mass still balancing.
    by_com: dict[float, list[dict[str, Any]]] = {}
    for b in bins:
        by_com.setdefault(float(b["com_abs_m"]), []).append(b)
    com_frontier: list[dict[str, Any]] = []
    for com_abs, group in sorted(by_com.items()):
        passing = [b for b in group if _ok(b)]
        com_frontier.append(
            {
                "com_abs_m": com_abs,
                "max_balance_mass_scale": (
                    float(max(b["mass_scale"] for b in passing)) if passing else None
                ),
                "n_bins": len(group),
                "mean_survival": _mean([b["survival"] for b in group]),
            }
        )

    # Interpolate frontiers at the headline probe (default ×1.10, 1.5 cm).
    # Old 2 cm target sat past every candidate's tip → always 0% / None.
    probe_mass = float(probe_mass_scale)
    probe_com_abs = float(probe_com_abs_m)
    mf_for_interp = [
        {"mass_scale": f["mass_scale"], "com_abs_m": f["max_balance_com_abs_m"]}
        for f in mass_frontier
        if f["max_balance_com_abs_m"] is not None
    ]
    cf_for_interp = [
        {"com_abs_m": f["com_abs_m"], "mass_scale": f["max_balance_mass_scale"]}
        for f in com_frontier
        if f["max_balance_mass_scale"] is not None
    ]

    return MassComLimitAnalysis(
        bins=bins,
        mass_frontier=mass_frontier,
        com_frontier=com_frontier,
        max_balance_mass_at_com_2cm=_interp_frontier_y(
            cf_for_interp, "com_abs_m", "mass_scale", probe_com_abs
        ),
        max_balance_com_abs_at_mass_1_10=_interp_frontier_y(
            mf_for_interp, "mass_scale", "com_abs_m", probe_mass
        ),
        probe_mass_1_10_com_2cm_survival=_nearest_bin_survival(
            bins, probe_mass, probe_com_abs
        ),
        params={
            "survival_min": survival_min,
            "stable_balance_min": stable_balance_min,
            "min_episodes_per_bin": min_episodes_per_bin,
            "n_episodes": len(eps),
            "scale_min": scale_min,
            "scale_max": scale_max,
            "offset_min_m": offset_min_m,
            "offset_max_m": offset_max_m,
            "probe_mass_scale": probe_mass,
            "probe_com_abs_m": probe_com_abs,
        },
    )


def format_mass_com_limits_report(analysis: MassComLimitAnalysis) -> str:
    p = analysis.params

    def _fmt_mass(val: float | None) -> str:
        return f"×{val:.3f}" if val is not None and math.isfinite(val) else "n/a"

    def _fmt_com(val: float | None) -> str:
        return f"{val * 100:.2f} cm" if val is not None and math.isfinite(val) else "n/a"

    def _fmt_surv(val: float | None) -> str:
        return f"{val:.0%}" if val is not None and math.isfinite(val) else "n/a"

    lines = [
        "",
        f"{'MASS × CoM GRID (lateral |y|)':^74}",
        f"{'─' * 74}",
        f"  Episodes analysed : {p.get('n_episodes', 0)} (first per env)",
        f"  Balance gate      : survival ≥ {p.get('survival_min', 0.8):.0%}, "
        f"stable balance ≥ {p.get('stable_balance_min', 0.5):.0%}",
        "",
        f"  Max mass @ |CoM|≈0.5 cm : {_fmt_mass(analysis.max_balance_mass_at_com_2cm)}",
        f"  Max |CoM| @ mass≈×1.10: {_fmt_com(analysis.max_balance_com_abs_at_mass_1_10)}",
        f"  Nearest (×1.10, 0.5cm) surv: {_fmt_surv(analysis.probe_mass_1_10_com_2cm_survival)}",
    ]
    if analysis.mass_frontier:
        lines += [
            "",
            f"{'mass':>8} {'max|CoM|cm':>11} {'mean surv':>10}",
            f"{'─' * 74}",
        ]
        for f in analysis.mass_frontier:
            com = f["max_balance_com_abs_m"]
            com_s = f"{com * 100:.2f}" if com is not None else "n/a"
            lines.append(
                f"{f['mass_scale']:8.3f} {com_s:>11} {f['mean_survival']:9.0%}"
            )
    lines.append(f"{'─' * 74}")
    return "\n".join(lines)


def export_mass_com_limits(analysis: MassComLimitAnalysis, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis.to_dict(), indent=2))
    return str(path)


def plot_mass_com_limits(analysis: MassComLimitAnalysis, save_path: str | Path) -> str | None:
    if not analysis.bins:
        return None
    import matplotlib.pyplot as plt

    masses = sorted({b["mass_scale"] for b in analysis.bins})
    coms = sorted({b["com_abs_m"] for b in analysis.bins})
    grid = np.full((len(coms), len(masses)), float("nan"))
    m_idx = {m: i for i, m in enumerate(masses)}
    c_idx = {c: i for i, c in enumerate(coms)}
    for b in analysis.bins:
        grid[c_idx[b["com_abs_m"]], m_idx[b["mass_scale"]]] = b["survival"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
        extent=[
            masses[0] if masses else 0,
            masses[-1] if masses else 1,
            (coms[0] if coms else 0) * 100,
            (coms[-1] if coms else 0) * 100,
        ],
    )
    fig.colorbar(im, ax=ax, label="Survival")
    # Frontier: max |CoM| vs mass
    xs = [f["mass_scale"] for f in analysis.mass_frontier if f["max_balance_com_abs_m"] is not None]
    ys = [
        f["max_balance_com_abs_m"] * 100
        for f in analysis.mass_frontier
        if f["max_balance_com_abs_m"] is not None
    ]
    if xs:
        ax.plot(xs, ys, "k-o", lw=1.5, ms=4, label="Balance frontier")
    ax.axvline(1.10, color="#333", ls="--", alpha=0.5, label="probe mass ×1.10")
    ax.axhline(2.0, color="#333", ls=":", alpha=0.5, label="probe |CoM| 2 cm")
    ax.set_xlabel("Mass scale (×)")
    ax.set_ylabel("|CoM y| (cm)")
    ax.set_title("Mass × CoM survival grid")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)
