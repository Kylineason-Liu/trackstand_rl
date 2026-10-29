# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/eval/eval_csv_plotly.py (imports/paths rewritten for trackstand_rl locomotive).
"""Interactive Plotly HTML from saved ``eval_report_*.csv`` (no Isaac Sim needed).

Mirrors ``eval_csv_plots.py`` so prior matplotlib-only eval runs can be
re-exported as zoomable / pannable HTML.

    python scripts/eval/export_eval_plotly.py path/to/eval/
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
    )
    from .eval_plotly import _config, _HAS_PLOTLY, _write_fig, _write_html_cdn
except ImportError:  # standalone (scripts/eval/export_eval_plotly.py)
    from eval_failure import (  # type: ignore
        COVARIATES,
        analyze_survival_thresholds,
        episodes_from_csv,
    )
    from eval_plotly import _config, _HAS_PLOTLY, _write_fig, _write_html_cdn  # type: ignore

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]

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

_HIST_KEYS = (
    ("max_lean_deg", "Max lean (°)"),
    ("vx_rmse_ms", "vx RMSE (m/s)"),
    ("heading_rmse_deg", "Heading RMSE (°)"),
    ("mean_drift_m", "Mean drift (m)"),
    ("steer_clip_sat_ratio", "Steer clip sat."),
    ("drive_usage_ratio", "Drive usage"),
)


def find_eval_csvs(root: str | os.PathLike[str]) -> list[str]:
    """``eval_report*.csv`` under ``root`` (file or directory)."""
    root_p = Path(root).expanduser().resolve()
    if root_p.is_file():
        return [str(root_p)] if root_p.suffix.lower() == ".csv" else []
    return sorted(str(p) for p in root_p.rglob("eval_report*.csv"))


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def _attr(ep: Any, name: str, default: float = float("nan")) -> float:
    value = getattr(ep, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _layout(title: str, **extra: Any) -> dict[str, Any]:
    layout: dict[str, Any] = {
        "title": {"text": title},
        "template": "plotly_white",
        "dragmode": "zoom",
        "hovermode": "closest",
        "margin": {"l": 60, "r": 30, "t": 70, "b": 60},
    }
    layout.update(extra)
    return layout


def plot_eval_csv_plotly(
    csv_path: str | os.PathLike[str],
    out_dir: str | None = None,
) -> dict[str, str]:
    """Write interactive HTML for one eval CSV. Returns ``{name: path}``."""
    csv_path = str(Path(csv_path).expanduser().resolve())
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    episodes = episodes_from_csv(csv_path)
    if not episodes:
        print(f"[EvalCSVPlotly] {csv_path} has no rows — skipping.")
        return {}

    analysis = analyze_survival_thresholds(episodes)
    stem = Path(csv_path).stem
    dest = Path(out_dir).expanduser().resolve() if out_dir else Path(csv_path).parent / "plotly"
    dest.mkdir(parents=True, exist_ok=True)
    title = stem
    paths: dict[str, str] = {}

    writers = (
        ("survival_thresholds", lambda p: _plot_survival_thresholds(analysis, p, title)),
        ("fail_times", lambda p: _plot_fail_times(episodes, analysis, p, title)),
        ("crash_vs_survive", lambda p: _plot_crash_vs_survive(episodes, p, title)),
        ("term_reasons", lambda p: _plot_term_reasons(analysis, p, title)),
        ("spawn_scatter", lambda p: _plot_spawn_scatter(episodes, p, title)),
        ("metrics_overview", lambda p: _plot_metrics_overview(episodes, p, title)),
        ("end_xy", lambda p: _plot_end_xy(episodes, p, title)),
    )
    for name, fn in writers:
        path = str(dest / f"{stem}_{name}.html")
        try:
            paths[name] = fn(path)
        except Exception as exc:  # pragma: no cover
            print(f"[EvalCSVPlotly] WARN: {name} failed: {exc}")

    dash = _write_dashboard(dest, stem, title, paths)
    if dash:
        paths["dashboard"] = dash

    print(f"[EvalCSVPlotly] {len(paths)} HTML for {Path(csv_path).name} → {dest}/")
    return paths


def export_eval_dir_plotly(root: str | os.PathLike[str]) -> dict[str, dict[str, str]]:
    """Find every ``eval_report*.csv`` under ``root`` and write Plotly HTML."""
    root = Path(root).expanduser().resolve()
    csvs = find_eval_csvs(root)
    if not csvs:
        print(f"[EvalCSVPlotly] No eval_report*.csv under {root}")
        return {}

    results: dict[str, dict[str, str]] = {}
    for csv_path in csvs:
        # Prefer mode/plots/plotly when CSV lives in mode/reports/
        csv_p = Path(csv_path)
        if csv_p.parent.name == "reports" and csv_p.parent.parent.is_dir():
            out = csv_p.parent.parent / "plots" / "plotly"
        else:
            out = csv_p.parent / "plotly"
        results[csv_path] = plot_eval_csv_plotly(csv_path, out_dir=str(out))

    index = _write_suite_index(root, results)
    if index:
        print(f"[EvalCSVPlotly] Suite index → {index}")
    return results


def _write_suite_index(root: Path, results: dict[str, dict[str, str]]) -> str | None:
    if not results:
        return None
    rows: list[str] = []
    for csv_path, paths in sorted(results.items()):
        mode = Path(csv_path).parent.parent.name
        if Path(csv_path).parent.name != "reports":
            mode = Path(csv_path).parent.name
        dash = paths.get("dashboard", "")
        n = len([k for k in paths if k != "dashboard"])
        link = f'<a href="{os.path.relpath(dash, root)}">{mode}</a>' if dash else mode
        rows.append(f"<li>{link} — {n} charts</li>")
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        "<title>CycleRL eval — Plotly suite</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;background:#fafafa}"
        "a{color:#1a5fb4}</style></head><body>"
        f"<h1>CycleRL eval — interactive Plotly</h1>"
        f"<p>Source: <code>{root}</code></p><ul>"
        + "\n".join(rows)
        + "</ul>"
        "<p>scroll = zoom · drag = box-zoom · double-click = reset</p>"
        "</body></html>"
    )
    path = root / "plotly_index.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


def _write_dashboard(dest: Path, stem: str, title: str, paths: dict[str, str]) -> str | None:
    if not paths:
        return None
    items = []
    for name, p in sorted(paths.items()):
        rel = os.path.relpath(p, dest)
        items.append(f'<li><a href="{rel}">{name}</a></li>')
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        f"<title>{title} — Plotly</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;background:#fafafa}"
        "a{color:#1a5fb4}</style></head><body>"
        f"<h1>{title}</h1><ul>{''.join(items)}</ul>"
        "<p>scroll = zoom · drag = box-zoom · double-click = reset</p>"
        "</body></html>"
    )
    path = dest / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


def _plot_survival_thresholds(analysis: dict[str, Any], path: str, title: str) -> str:
    keys = (
        "spawn_vx_ms",
        "spawn_yaw_abs_deg",
        "spawn_cmd_vx_ms",
        "spawn_cmd_heading_abs_deg",
        "spawn_roll_deg",
        "spawn_terrain_level",
    )
    heads: list[str] = []
    for key in keys:
        label, unit = next(((lab, u) for k, lab, u in COVARIATES if k == key), (key, ""))
        cov = (analysis.get("covariates") or {}).get(key) or {}
        thr = cov.get("threshold_50")
        tick_unit = f" {unit}" if unit else ""
        head = label
        if isinstance(thr, (int, float)) and math.isfinite(thr):
            head = f"{label} — 50% @ {thr:.2g}{tick_unit}"
        heads.append(head)

    if _HAS_PLOTLY and make_subplots is not None:
        fig = make_subplots(rows=3, cols=2, subplot_titles=heads)
        for i, key in enumerate(keys):
            row, col = divmod(i, 2)
            row += 1
            col += 1
            label, unit = next(((lab, u) for k, lab, u in COVARIATES if k == key), (key, ""))
            cov = (analysis.get("covariates") or {}).get(key) or {}
            bins = cov.get("bins") or []
            if not bins:
                fig.add_annotation(
                    text="no data",
                    xref=f"x{i+1} domain" if i else "x domain",
                    yref=f"y{i+1} domain" if i else "y domain",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
                continue
            centers = [0.5 * (b["lo"] + b["hi"]) for b in bins]
            surv = [b["survival"] for b in bins]
            ns = [b["n"] for b in bins]
            tick_unit = f" {unit}" if unit else ""
            labels = [f"{c:.2g}{tick_unit}<br>n={n}" for c, n in zip(centers, ns)]
            fig.add_trace(
                go.Bar(x=labels, y=surv, marker_color=COLORS["blue"], name=label, showlegend=False),
                row=row,
                col=col,
            )
            fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS["red"], row=row, col=col)
            fig.update_yaxes(range=[0, 1.05], title_text="Survival", row=row, col=col)
        fig.update_layout(
            **_layout(f"Survival vs spawn / command<br>{title}"),
            height=900,
        )
        return _write_fig(fig, path)

    # CDN fallback: first covariate only as a simple bar
    key = keys[0]
    cov = (analysis.get("covariates") or {}).get(key) or {}
    bins = cov.get("bins") or []
    labels = [f"{0.5*(b['lo']+b['hi']):.2g}" for b in bins]
    traces = [{"type": "bar", "x": labels, "y": [b["survival"] for b in bins], "name": key}]
    return _write_html_cdn(path, traces, _layout(f"Survival — {title}"))


def _plot_fail_times(episodes: list[Any], analysis: dict[str, Any], path: str, title: str) -> str:
    crashed = [e for e in episodes if getattr(e, "complete", True) and not getattr(e, "survived", False)]
    times = _finite([_attr(e, "episode_time_s") for e in crashed])
    if _HAS_PLOTLY and go is not None:
        fig = go.Figure()
        if times:
            fig.add_trace(go.Histogram(x=times, nbinsx=min(20, max(5, len(times) // 3)),
                                       marker_color=COLORS["orange"], name="crash time"))
        ft = analysis.get("fail_time_s") or {}
        if math.isfinite(ft.get("median", float("nan"))):
            fig.add_vline(x=ft["median"], line_dash="dash", line_color=COLORS["red"],
                          annotation_text=f"median {ft['median']:.2f}s")
        fig.update_layout(**_layout(f"When the bike fails<br>{title}"),
                          xaxis_title="Time to crash (s)", yaxis_title="Episodes")
        return _write_fig(fig, path)
    traces = [{"type": "histogram", "x": times, "name": "crash time",
               "marker": {"color": COLORS["orange"]}}]
    return _write_html_cdn(path, traces, _layout(f"Fail times — {title}",
                                                 xaxis={"title": "Time to crash (s)"},
                                                 yaxis={"title": "Episodes"}))


def _plot_crash_vs_survive(episodes: list[Any], path: str, title: str) -> str:
    crashed = [e for e in episodes if getattr(e, "complete", True) and not getattr(e, "survived", False)]
    survived = [e for e in episodes if getattr(e, "complete", True) and getattr(e, "survived", False)]
    if _HAS_PLOTLY and make_subplots is not None:
        fig = make_subplots(rows=2, cols=4, subplot_titles=[lab for _, lab in _BOX_KEYS])
        for i, (key, _label) in enumerate(_BOX_KEYS):
            row, col = divmod(i, 4)
            row += 1
            col += 1
            fail = _finite([_attr(e, key) for e in crashed])
            live = _finite([_attr(e, key) for e in survived])
            if fail:
                fig.add_trace(
                    go.Box(y=fail, name="crash", marker_color=COLORS["red"], showlegend=(i == 0)),
                    row=row, col=col,
                )
            if live:
                fig.add_trace(
                    go.Box(y=live, name="survive", marker_color=COLORS["green"], showlegend=(i == 0)),
                    row=row, col=col,
                )
        fig.update_layout(**_layout(f"Crash vs survive<br>{title}"), height=700)
        return _write_fig(fig, path)
    # CDN: first metric only
    key, label = _BOX_KEYS[0]
    fail = _finite([_attr(e, key) for e in crashed])
    live = _finite([_attr(e, key) for e in survived])
    traces = [
        {"type": "box", "y": fail, "name": "crash", "marker": {"color": COLORS["red"]}},
        {"type": "box", "y": live, "name": "survive", "marker": {"color": COLORS["green"]}},
    ]
    return _write_html_cdn(path, traces, _layout(f"{label} — {title}"))


def _plot_term_reasons(analysis: dict[str, Any], path: str, title: str) -> str:
    counts = analysis.get("term_counts") or {}
    names = list(counts.keys())
    vals = [counts[k] for k in names]
    colors = [COLORS["green"] if n == "time_out" else COLORS["red"] for n in names]
    if _HAS_PLOTLY and go is not None:
        fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors))
        fig.update_layout(**_layout(f"Termination reason<br>{title}"),
                          yaxis_title="Episodes")
        return _write_fig(fig, path)
    traces = [{"type": "bar", "x": names, "y": vals, "marker": {"color": colors}}]
    return _write_html_cdn(path, traces, _layout(f"Term reasons — {title}",
                                                 yaxis={"title": "Episodes"}))


def _plot_spawn_scatter(episodes: list[Any], path: str, title: str) -> str:
    if _HAS_PLOTLY and make_subplots is not None:
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Spawn vx", "Spawn yaw"])
        pairs = (("spawn_vx_ms", "Spawn vx (m/s)", 1), ("spawn_yaw_deg", "Spawn yaw (°)", 2))
        for key, xlabel, col in pairs:
            xs_c, ys_c, xs_s, ys_s = [], [], [], []
            for ep in episodes:
                if not getattr(ep, "complete", True):
                    continue
                x = _attr(ep, key)
                y = _attr(ep, "episode_time_s")
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                if getattr(ep, "survived", False):
                    xs_s.append(x); ys_s.append(y)
                else:
                    xs_c.append(x); ys_c.append(y)
            if xs_s:
                fig.add_trace(
                    go.Scatter(x=xs_s, y=ys_s, mode="markers", name="survive",
                               marker=dict(color=COLORS["green"], size=7),
                               showlegend=(col == 1)),
                    row=1, col=col,
                )
            if xs_c:
                fig.add_trace(
                    go.Scatter(x=xs_c, y=ys_c, mode="markers", name="crash",
                               marker=dict(color=COLORS["red"], size=8),
                               showlegend=(col == 1)),
                    row=1, col=col,
                )
            fig.update_xaxes(title_text=xlabel, row=1, col=col)
            fig.update_yaxes(title_text="Episode time (s)", row=1, col=col)
        fig.update_layout(**_layout(f"Spawn vs time-to-end<br>{title}"), height=480)
        return _write_fig(fig, path)
    return _write_html_cdn(path, [], _layout(f"Spawn scatter — {title}"))


def _plot_metrics_overview(episodes: list[Any], path: str, title: str) -> str:
    complete = [e for e in episodes if getattr(e, "complete", True)]
    if _HAS_PLOTLY and make_subplots is not None:
        fig = make_subplots(rows=2, cols=3, subplot_titles=[lab for _, lab in _HIST_KEYS])
        for i, (key, _label) in enumerate(_HIST_KEYS):
            row, col = divmod(i, 3)
            row += 1
            col += 1
            fail = _finite([_attr(e, key) for e in complete if not getattr(e, "survived", False)])
            live = _finite([_attr(e, key) for e in complete if getattr(e, "survived", False)])
            if fail:
                fig.add_trace(
                    go.Histogram(x=fail, name="crash", marker_color=COLORS["red"],
                                 opacity=0.55, showlegend=(i == 0)),
                    row=row, col=col,
                )
            if live:
                fig.add_trace(
                    go.Histogram(x=live, name="survive", marker_color=COLORS["green"],
                                 opacity=0.55, showlegend=(i == 0)),
                    row=row, col=col,
                )
        fig.update_layout(**_layout(f"Episode metrics<br>{title}"),
                          barmode="overlay", height=700)
        return _write_fig(fig, path)
    return _write_html_cdn(path, [], _layout(f"Metrics — {title}"))


def _plot_end_xy(episodes: list[Any], path: str, title: str) -> str:
    xs_c, ys_c, xs_s, ys_s = [], [], [], []
    for ep in episodes:
        if not getattr(ep, "complete", True):
            continue
        x = _attr(ep, "end_pos_x_m")
        y = _attr(ep, "end_pos_y_m")
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if getattr(ep, "survived", False):
            xs_s.append(x); ys_s.append(y)
        else:
            xs_c.append(x); ys_c.append(y)
    if _HAS_PLOTLY and go is not None:
        fig = go.Figure()
        if xs_s:
            fig.add_trace(go.Scatter(x=xs_s, y=ys_s, mode="markers", name="survive",
                                     marker=dict(color=COLORS["green"], size=8)))
        if xs_c:
            fig.add_trace(go.Scatter(x=xs_c, y=ys_c, mode="markers", name="crash",
                                     marker=dict(color=COLORS["red"], size=9)))
        fig.update_layout(**_layout(f"Where episodes ended<br>{title}"),
                          xaxis_title="End x (m)", yaxis_title="End y (m)",
                          yaxis_scaleanchor="x")
        return _write_fig(fig, path)
    traces = []
    if xs_s:
        traces.append({"type": "scatter", "mode": "markers", "x": xs_s, "y": ys_s,
                       "name": "survive", "marker": {"color": COLORS["green"]}})
    if xs_c:
        traces.append({"type": "scatter", "mode": "markers", "x": xs_c, "y": ys_c,
                       "name": "crash", "marker": {"color": COLORS["red"]}})
    return _write_html_cdn(path, traces, _layout(f"End XY — {title}",
                                                 xaxis={"title": "End x (m)"},
                                                 yaxis={"title": "End y (m)", "scaleanchor": "x"}))
