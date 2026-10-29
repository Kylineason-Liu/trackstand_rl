# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/eval/eval_csv_plots.py (imports/paths rewritten for trackstand_rl locomotive).
"""Plot eval_report_*.csv files — one graph set per CSV.

Used automatically after the logger writes a CSV, and as a standalone analyzer
that does not need Isaac Sim:

    python scripts/analyze_eval_csv.py path/to/eval_report_*.csv
    python scripts/analyze_eval_csv.py path/to/eval/   # every CSV under the tree
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

try:
    from .eval_failure import (
        COVARIATES,
        analyze_survival_thresholds,
        episodes_from_csv,
        format_failure_report,
    )
except ImportError:  # standalone script on sys.path
    from eval_failure import (  # type: ignore
        COVARIATES,
        analyze_survival_thresholds,
        episodes_from_csv,
        format_failure_report,
    )

try:
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


COLORS = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "purple": "#8172B3",
    "grey": "#8C8C8C",
}

_BOX_KEYS = (
    ("spawn_vx_ms", "Spawn vx (m/s)"),
    ("spawn_yaw_deg", "Spawn yaw (°)"),
    ("spawn_roll_deg", "Spawn roll (°)"),
    ("spawn_cmd_vx_ms", "vx* spawn (m/s)"),
    ("spawn_cmd_heading_deg", "heading* (°)"),
    ("episode_time_s", "Episode time (s)"),
    ("max_lean_deg", "Max lean (°)"),
    ("mean_cmd_vx_ms", "Mean vx* (m/s)"),
)


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def _attr(ep: Any, name: str, default: float = float("nan")) -> float:
    value = getattr(ep, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def plot_eval_csv(csv_path: str | os.PathLike[str], out_dir: str | None = None) -> dict[str, str]:
    """Generate the graph set that belongs to one eval CSV.

    Writes ``<stem>_*.png`` next to the CSV (or in ``out_dir``). Returns
    ``{plot_name: path}``.
    """
    if not _HAS_MPL:
        print("[EvalCSV] matplotlib not installed — skipping graphs.")
        return {}

    csv_path = str(Path(csv_path).expanduser().resolve())
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    episodes = episodes_from_csv(csv_path)
    if not episodes:
        print(f"[EvalCSV] {csv_path} has no rows — skipping graphs.")
        return {}

    analysis = analyze_survival_thresholds(episodes)
    stem = Path(csv_path).stem
    dest = Path(out_dir).expanduser().resolve() if out_dir else Path(csv_path).parent
    dest.mkdir(parents=True, exist_ok=True)
    title = stem
    paths: dict[str, str] = {}

    paths["survival_thresholds"] = _plot_survival_thresholds(
        analysis, dest / f"{stem}_survival_thresholds", title
    )
    paths["fail_times"] = _plot_fail_times(episodes, analysis, dest / f"{stem}_fail_times", title)
    paths["crash_vs_survive"] = _plot_crash_vs_survive(
        episodes, dest / f"{stem}_crash_vs_survive", title
    )
    paths["term_reasons"] = _plot_term_reasons(analysis, dest / f"{stem}_term_reasons", title)
    paths["spawn_scatter"] = _plot_spawn_scatter(episodes, dest / f"{stem}_spawn_scatter", title)
    paths["metrics_overview"] = _plot_metrics_overview(
        episodes, dest / f"{stem}_metrics_overview", title
    )
    paths["end_xy"] = _plot_end_xy(episodes, dest / f"{stem}_end_xy", title)

    report_path = dest / f"{stem}_thresholds.txt"
    report_path.write_text(f"{title}\n{csv_path}\n\n{format_failure_report(analysis)}\n")
    paths["thresholds_txt"] = str(report_path)

    print(f"[EvalCSV] {len(paths)} graphs for {Path(csv_path).name} → {dest}/")
    return paths


def plot_eval_csvs(paths: list[str], out_dir: str | None = None) -> dict[str, dict[str, str]]:
    """Plot every CSV path. Returns ``{csv_path: {plot_name: png_path}}``."""
    results: dict[str, dict[str, str]] = {}
    for csv_path in paths:
        results[csv_path] = plot_eval_csv(csv_path, out_dir=out_dir)
    return results


def find_eval_csvs(root: str | os.PathLike[str]) -> list[str]:
    """``eval_report*.csv`` under ``root`` (file or directory)."""
    root = Path(root).expanduser().resolve()
    if root.is_file():
        return [str(root)] if root.suffix.lower() == ".csv" else []
    return sorted(str(p) for p in root.rglob("eval_report*.csv"))


def _save(fig: Any, base: Path) -> str:
    fig.tight_layout()
    png = f"{base}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def _plot_survival_thresholds(analysis: dict[str, Any], base: Path, title: str) -> str:
    keys = (
        "spawn_vx_ms",
        "spawn_yaw_abs_deg",
        "spawn_cmd_vx_ms",
        "spawn_cmd_heading_abs_deg",
        "spawn_roll_deg",
        "spawn_terrain_level",
    )
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 11.0), dpi=150)
    for ax, key in zip(axes.ravel(), keys):
        label, unit = next(((lab, u) for k, lab, u in COVARIATES if k == key), (key, ""))
        cov = (analysis.get("covariates") or {}).get(key) or {}
        bins = cov.get("bins") or []
        if not bins:
            ax.set_title(label)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylim(0, 1)
            continue
        centers = [0.5 * (b["lo"] + b["hi"]) for b in bins]
        surv = [b["survival"] for b in bins]
        ns = [b["n"] for b in bins]
        ax.bar(range(len(bins)), surv, color=COLORS["blue"], alpha=0.85)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(range(len(bins)))
        tick_unit = f" {unit}" if unit else ""
        ax.set_xticklabels([f"{c:.2g}{tick_unit}\nn={n}" for c, n in zip(centers, ns)], fontsize=7)
        ax.axhline(0.5, color=COLORS["red"], linestyle=":", linewidth=0.9)
        thr = cov.get("threshold_50")
        head = label
        if isinstance(thr, (int, float)) and math.isfinite(thr):
            head = f"{label}  —  50% @ {thr:.2g}{tick_unit}"
        ax.set_title(head, fontsize=10)
        ax.set_ylabel("Survival")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Survival vs spawn / command\n{title}", fontsize=12)
    return _save(fig, base)


def _plot_fail_times(episodes: list[Any], analysis: dict[str, Any], base: Path, title: str) -> str:
    crashed = [e for e in episodes if getattr(e, "complete", True) and not getattr(e, "survived", False)]
    times = _finite([_attr(e, "episode_time_s") for e in crashed])
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    if times:
        ax.hist(times, bins=min(20, max(5, len(times) // 3)), color=COLORS["orange"], edgecolor="white")
    ft = analysis.get("fail_time_s") or {}
    if math.isfinite(ft.get("median", float("nan"))):
        ax.axvline(ft["median"], color=COLORS["red"], linestyle="--", label=f"median {ft['median']:.2f}s")
        ax.legend(framealpha=0.9)
    ax.set_xlabel("Time to crash (s)")
    ax.set_ylabel("Episodes")
    ax.set_title(f"When the bike fails\n{title}")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, base)


def _plot_crash_vs_survive(episodes: list[Any], base: Path, title: str) -> str:
    crashed = [e for e in episodes if getattr(e, "complete", True) and not getattr(e, "survived", False)]
    survived = [e for e in episodes if getattr(e, "complete", True) and getattr(e, "survived", False)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=150)
    for ax, (key, label) in zip(axes.ravel(), _BOX_KEYS):
        fail = _finite([_attr(e, key) for e in crashed])
        live = _finite([_attr(e, key) for e in survived])
        data = []
        names = []
        if fail:
            data.append(fail)
            names.append(f"crash\nn={len(fail)}")
        if live:
            data.append(live)
            names.append(f"survive\nn={len(live)}")
        if not data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        try:
            bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
        except TypeError:
            bp = ax.boxplot(data, labels=names, patch_artist=True)
        colors = []
        if fail:
            colors.append(COLORS["red"])
        if live:
            colors.append(COLORS["green"])
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Crash vs survive\n{title}", fontsize=12)
    return _save(fig, base)


def _plot_term_reasons(analysis: dict[str, Any], base: Path, title: str) -> str:
    counts = analysis.get("term_counts") or {}
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    if not counts:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        names = list(counts.keys())
        vals = [counts[k] for k in names]
        colors = [COLORS["green"] if n == "time_out" else COLORS["red"] for n in names]
        ax.bar(names, vals, color=colors, alpha=0.85)
        ax.set_ylabel("Episodes")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_title(f"Termination reason\n{title}")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, base)


def _plot_spawn_scatter(episodes: list[Any], base: Path, title: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    pairs = (
        ("spawn_vx_ms", "Spawn vx (m/s)", axes[0]),
        ("spawn_yaw_deg", "Spawn yaw (°)", axes[1]),
    )
    for key, xlabel, ax in pairs:
        xs_c, ys_c, xs_s, ys_s = [], [], [], []
        for ep in episodes:
            if not getattr(ep, "complete", True):
                continue
            x = _attr(ep, key)
            y = _attr(ep, "episode_time_s")
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if getattr(ep, "survived", False):
                xs_s.append(x)
                ys_s.append(y)
            else:
                xs_c.append(x)
                ys_c.append(y)
        if xs_s:
            ax.scatter(xs_s, ys_s, c=COLORS["green"], s=18, alpha=0.7, label="survive")
        if xs_c:
            ax.scatter(xs_c, ys_c, c=COLORS["red"], s=22, alpha=0.8, label="crash")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Episode time (s)")
        ax.legend(framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Spawn vs time-to-end\n{title}", fontsize=12)
    return _save(fig, base)


_HIST_KEYS = (
    ("max_lean_deg", "Max lean (°)"),
    ("vx_rmse_ms", "vx RMSE (m/s)"),
    ("heading_rmse_deg", "Heading RMSE (°)"),
    ("mean_drift_m", "Mean drift (m)"),
    ("steer_clip_sat_ratio", "Steer clip sat."),
    ("drive_usage_ratio", "Drive usage"),
)


def _plot_metrics_overview(episodes: list[Any], base: Path, title: str) -> str:
    complete = [e for e in episodes if getattr(e, "complete", True)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=150)
    for ax, (key, label) in zip(axes.ravel(), _HIST_KEYS):
        fail = _finite([
            _attr(e, key) for e in complete if not getattr(e, "survived", False)
        ])
        live = _finite([
            _attr(e, key) for e in complete if getattr(e, "survived", False)
        ])
        if fail:
            ax.hist(fail, bins=12, color=COLORS["red"], alpha=0.55, label="crash")
        if live:
            ax.hist(live, bins=12, color=COLORS["green"], alpha=0.55, label="survive")
        if not fail and not live:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.legend(fontsize=7, framealpha=0.9)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Episodes")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Episode metrics\n{title}", fontsize=12)
    return _save(fig, base)


def _plot_end_xy(episodes: list[Any], base: Path, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 7), dpi=150)
    xs_c, ys_c, xs_s, ys_s = [], [], [], []
    for ep in episodes:
        if not getattr(ep, "complete", True):
            continue
        x = _attr(ep, "end_pos_x_m")
        y = _attr(ep, "end_pos_y_m")
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if getattr(ep, "survived", False):
            xs_s.append(x)
            ys_s.append(y)
        else:
            xs_c.append(x)
            ys_c.append(y)
    if xs_s:
        ax.scatter(xs_s, ys_s, c=COLORS["green"], s=22, alpha=0.75, label="survive")
    if xs_c:
        ax.scatter(xs_c, ys_c, c=COLORS["red"], s=28, alpha=0.85, label="crash", zorder=3)
    ax.set_xlabel("End x (m)")
    ax.set_ylabel("End y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Where episodes ended\n{title}")
    return _save(fig, base)
