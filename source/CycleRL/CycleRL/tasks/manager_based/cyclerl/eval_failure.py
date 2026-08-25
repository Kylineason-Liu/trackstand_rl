"""Survival-threshold analysis from per-episode spawn / command / fail context.

Given completed :class:`EpisodeMetrics`, estimate where survival drops as a
function of spawn velocity/yaw/roll, command vx/heading, terrain row, and
time-to-crash. No USD joint names — covariates come from the logger contract.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np


COVARIATES: tuple[tuple[str, str, str], ...] = (
    ("spawn_vx_ms", "Spawn vx", "m/s"),
    ("spawn_yaw_deg", "Spawn yaw", "°"),
    ("spawn_yaw_abs_deg", "Spawn |yaw|", "°"),
    ("spawn_roll_deg", "Spawn roll", "°"),
    ("spawn_cmd_vx_ms", "Command vx* at spawn", "m/s"),
    ("spawn_cmd_heading_deg", "Heading* at spawn", "°"),
    ("spawn_cmd_heading_abs_deg", "|heading*| at spawn", "°"),
    ("spawn_cmd_wz_rad_s", "ωz* at spawn", "rad/s"),
    ("spawn_terrain_level", "Terrain row (difficulty)", ""),
    ("spawn_terrain_type", "Terrain column (type)", ""),
    ("end_vx_ms", "vx at episode end", "m/s"),
    ("end_roll_deg", "Roll at episode end", "°"),
)


def _get(ep: Any, name: str) -> float:
    if name.endswith("_abs_deg"):
        base = name.replace("_abs_deg", "_deg")
        return abs(float(getattr(ep, base, float("nan"))))
    value = getattr(ep, name, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _quantile_bins(values: np.ndarray, survived: np.ndarray, n_bins: int = 8) -> list[dict[str, Any]]:
    finite = np.isfinite(values)
    v = values[finite]
    s = survived[finite]
    if v.size < 4:
        return []
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        return [{
            "lo": float(v.min()),
            "hi": float(v.max()),
            "n": int(v.size),
            "n_survived": int(s.sum()),
            "survival": float(s.mean()),
        }]
    inner = edges[1:-1]
    idx = np.clip(np.digitize(v, inner, right=True), 0, edges.size - 2)
    bins: list[dict[str, Any]] = []
    for i in range(edges.size - 1):
        mask = idx == i
        if not np.any(mask):
            continue
        bins.append({
            "lo": float(edges[i]),
            "hi": float(edges[i + 1]),
            "n": int(mask.sum()),
            "n_survived": int(s[mask].sum()),
            "survival": float(s[mask].mean()),
        })
    return bins


def _crossing_50(bins: Sequence[dict[str, Any]]) -> float | None:
    """Interpolate the covariate value where binned survival crosses 50%."""
    if len(bins) < 2:
        return None
    for i in range(len(bins) - 1):
        s0 = float(bins[i]["survival"])
        s1 = float(bins[i + 1]["survival"])
        if (s0 - 0.5) * (s1 - 0.5) > 0 or s0 == s1:
            continue
        x0 = 0.5 * (float(bins[i]["lo"]) + float(bins[i]["hi"]))
        x1 = 0.5 * (float(bins[i + 1]["lo"]) + float(bins[i + 1]["hi"]))
        t = (0.5 - s0) / (s1 - s0)
        return float(x0 + t * (x1 - x0))
    return None


def _mean_finite(vals: Iterable[float]) -> float:
    arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def _percentile_finite(vals: Iterable[float], q: float) -> float:
    arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def episodes_from_csv(path: str) -> list[Any]:
    """Load per-episode rows from an ``eval_report_*.csv`` into attribute objects."""
    import csv
    from types import SimpleNamespace

    bool_keys = {"survived", "complete", "kick_seen", "kick_settled"}
    int_keys = {
        "env_id", "recovery_count", "kick_oscillations", "episode_length",
        "spawn_terrain_level", "spawn_terrain_type", "end_terrain_level", "end_terrain_type",
    }
    str_keys = {"term_reason"}
    rows: list[Any] = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            parsed: dict[str, Any] = {}
            for key, val in raw.items():
                if key is None:
                    continue
                text = "" if val is None else str(val).strip()
                if key in str_keys:
                    parsed[key] = text
                elif key in bool_keys:
                    parsed[key] = text in ("1", "True", "true", "yes")
                elif text == "" or text.lower() in ("nan", "none", "null"):
                    parsed[key] = float("nan") if key not in int_keys else -1
                elif key in int_keys:
                    try:
                        parsed[key] = int(float(text))
                    except ValueError:
                        parsed[key] = -1
                else:
                    try:
                        parsed[key] = float(text)
                    except ValueError:
                        parsed[key] = text
            if "complete" not in parsed:
                parsed["complete"] = True
            # CSV header uses episode_length_s for EpisodeMetrics.episode_time_s.
            if "episode_time_s" not in parsed and "episode_length_s" in parsed:
                parsed["episode_time_s"] = parsed["episode_length_s"]
            rows.append(SimpleNamespace(**parsed))
    return rows


def analyze_survival_thresholds(
    episodes: Sequence[Any],
    *,
    n_bins: int = 8,
) -> dict[str, Any]:
    """Estimate crash timing and 50% survival crossings vs spawn/command covariates."""
    complete = [e for e in episodes if getattr(e, "complete", True)]
    crashed = [e for e in complete if not getattr(e, "survived", False)]
    survived = [e for e in complete if getattr(e, "survived", False)]
    flags = np.asarray([bool(e.survived) for e in complete], dtype=bool)

    term_counts: dict[str, int] = {}
    for ep in crashed:
        reason = str(getattr(ep, "term_reason", "") or "unknown")
        term_counts[reason] = term_counts.get(reason, 0) + 1
    if survived:
        term_counts["time_out"] = term_counts.get("time_out", 0) + len(survived)

    fail_times = [float(e.episode_time_s) for e in crashed]
    covariates: dict[str, Any] = {}
    for key, label, unit in COVARIATES:
        values = np.asarray([_get(e, key) for e in complete], dtype=float)
        bins = _quantile_bins(values, flags, n_bins=n_bins)
        failed_vals = [_get(e, key) for e in crashed]
        survived_vals = [_get(e, key) for e in survived]
        covariates[key] = {
            "label": label,
            "unit": unit,
            "failed_mean": _mean_finite(failed_vals),
            "survived_mean": _mean_finite(survived_vals),
            "threshold_50": _crossing_50(bins),
            "bins": bins,
        }

    return {
        "n_complete": len(complete),
        "n_survived": len(survived),
        "n_crashed": len(crashed),
        "survival_rate": (len(survived) / len(complete)) if complete else float("nan"),
        "term_counts": term_counts,
        "fail_time_s": {
            "mean": _mean_finite(fail_times),
            "median": _percentile_finite(fail_times, 50),
            "p10": _percentile_finite(fail_times, 10),
            "p90": _percentile_finite(fail_times, 90),
            "min": float(min(fail_times)) if fail_times else float("nan"),
            "max": float(max(fail_times)) if fail_times else float("nan"),
        },
        "covariates": covariates,
    }


def format_failure_report(analysis: dict[str, Any]) -> str:
    """Pretty-print :func:`analyze_survival_thresholds` for the eval summary."""
    if not analysis or not analysis.get("n_complete"):
        return "  (no complete episodes for failure-threshold analysis)"
    lines: list[str] = []
    n_c = int(analysis["n_crashed"])
    n_s = int(analysis["n_survived"])
    lines.append(f"  Crashes: {n_c}   Timeouts/survived: {n_s}   survival={analysis['survival_rate']:.1%}")
    terms = analysis.get("term_counts") or {}
    if terms:
        pretty = ", ".join(f"{k}={v}" for k, v in sorted(terms.items(), key=lambda kv: -kv[1]))
        lines.append(f"  Terminations: {pretty}")
    ft = analysis.get("fail_time_s") or {}
    if n_c and math.isfinite(ft.get("median", float("nan"))):
        lines.append(
            f"  Time-to-crash: median {ft['median']:.2f}s  "
            f"(p10 {ft['p10']:.2f}s, mean {ft['mean']:.2f}s, p90 {ft['p90']:.2f}s)"
        )
    lines.append("")
    lines.append(f"  {'covariate':<28s}  {'fail μ':>8s}  {'live μ':>8s}  {'50% fail @':>10s}")
    for key, label, unit in COVARIATES:
        cov = (analysis.get("covariates") or {}).get(key) or {}
        fail_m = cov.get("failed_mean", float("nan"))
        live_m = cov.get("survived_mean", float("nan"))
        thr = cov.get("threshold_50")
        unit_s = f" {unit}" if unit else ""
        fail_s = f"{fail_m:8.2f}" if math.isfinite(fail_m) else f"{'—':>8s}"
        live_s = f"{live_m:8.2f}" if math.isfinite(live_m) else f"{'—':>8s}"
        thr_s = f"{thr:.2f}{unit_s}" if isinstance(thr, (int, float)) and math.isfinite(thr) else "—"
        lines.append(f"  {label:<28s}  {fail_s}  {live_s}  {thr_s:>10s}")
    lines.append("  50% fail @ is the binned survival crossing; — means no clean crossing.")
    return "\n".join(lines)
