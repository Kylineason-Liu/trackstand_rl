# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/eval/eval_plotly.py (imports/paths rewritten for trackstand_rl locomotive).
"""Interactive Plotly HTML plots for CycleRL eval runs.

Writes zoomable / pannable / rescalable HTML next to the matplotlib PNGs.
Uses Plotly.py when available; otherwise embeds Plotly.js from CDN so Isaac
Sim's Python (often without plotly) still produces interactive output.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]
    _HAS_PLOTLY = False

# Offline-friendly CDN (cached by the browser after first open).
_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _finite(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=float)


def _hist_arrays(hist: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {k: _finite(hist.get(k, [])) for k in hist}


def _config() -> dict[str, Any]:
    return {
        "responsive": True,
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        "toImageButtonOptions": {"format": "png", "filename": "cyclerl_eval", "scale": 2},
    }


def _time_layout(title: str, y_title: str, *, rangeslider: bool = True) -> dict[str, Any]:
    xaxis: dict[str, Any] = {
        "title": "Time (s)",
        "rangeslider": {"visible": bool(rangeslider)},
        "autorange": True,
    }
    return {
        "title": {"text": title},
        "xaxis": xaxis,
        "yaxis": {"title": y_title, "autorange": True, "fixedrange": False},
        "hovermode": "x unified",
        "dragmode": "zoom",
        "template": "plotly_white",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
    }


def _write_html_cdn(
    path: str,
    traces: list[dict[str, Any]],
    layout: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> str:
    """Write a self-contained interactive HTML file (Plotly.js via CDN)."""
    cfg = config or _config()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{layout.get("title", {}).get("text", "CycleRL eval")}</title>
  <script src="{_PLOTLY_CDN}"></script>
  <style>
    html, body {{ margin: 0; padding: 0; height: 100%; background: #fafafa; font-family: system-ui, sans-serif; }}
    #chart {{ width: 100%; height: 100vh; min-height: 520px; }}
    .hint {{ position: fixed; right: 12px; bottom: 8px; font-size: 11px; color: #666; z-index: 5;
             background: rgba(255,255,255,0.85); padding: 4px 8px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div id="chart"></div>
  <div class="hint">scroll = zoom · drag = box-zoom · double-click = reset · mode bar = pan / autoscale</div>
  <script>
    const data = {json.dumps(traces)};
    const layout = {json.dumps(layout)};
    const config = {json.dumps(cfg)};
    Plotly.newPlot("chart", data, layout, config);
  </script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _write_fig(fig: Any, path: str) -> str:
    if _HAS_PLOTLY:
        fig.update_layout(dragmode="zoom", hovermode=fig.layout.hovermode or "x unified")
        fig.write_html(
            path,
            include_plotlyjs="cdn",
            full_html=True,
            config=_config(),
        )
        return path
    raise RuntimeError("plotly not installed")


def _scatter(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    *,
    color: str | None = None,
    dash: str | None = None,
    mode: str = "lines",
) -> dict[str, Any]:
    line: dict[str, Any] = {}
    if color:
        line["color"] = color
    if dash:
        line["dash"] = dash
    return {
        "type": "scatter",
        "mode": mode,
        "name": name,
        "x": x.tolist(),
        "y": y.tolist(),
        "line": line,
    }


class BicycleEvalPlotly:
    """Interactive Plotly HTML plots from a ``BicycleMetricsLogger``."""

    COLORS = {
        "blue": "#4C72B0",
        "orange": "#DD8452",
        "green": "#55A868",
        "red": "#C44E52",
        "purple": "#8172B3",
        "grey": "#8C8C8C",
    }

    def __init__(self, logger: Any, save_dir: str = "./eval_plots/plotly") -> None:
        self.logger = logger
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _save(self, name: str, traces: list[dict[str, Any]], layout: dict[str, Any]) -> str:
        path = os.path.join(self.save_dir, f"{name}.html")
        if _HAS_PLOTLY:
            fig = go.Figure()
            for tr in traces:
                fig.add_trace(go.Scatter(
                    x=tr["x"],
                    y=tr["y"],
                    name=tr.get("name"),
                    mode=tr.get("mode", "lines"),
                    line=tr.get("line") or {},
                ))
            fig.update_layout(**layout)
            return _write_fig(fig, path)
        return _write_html_cdn(path, traces, layout)

    def plot_velocity_profile(self) -> Optional[str]:
        hist = _hist_arrays(self.logger.history)
        t, vx, cmd = hist.get("time_s"), hist.get("velocity_ms"), hist.get("target_velocity_ms")
        if t is None or len(t) == 0:
            return None
        traces = [
            _scatter(t, vx, "actual vx", color=self.COLORS["blue"]),
            _scatter(t, cmd, "command vx*", color=self.COLORS["orange"], dash="dash"),
        ]
        return self._save(
            "velocity_profile",
            traces,
            _time_layout("Velocity profile", "vx (m/s)"),
        )

    def plot_posture_stability(self) -> Optional[str]:
        hist = _hist_arrays(self.logger.history)
        t = hist.get("time_s")
        if t is None or len(t) == 0:
            return None
        roll = hist.get("roll_deg", np.array([]))
        pitch = hist.get("pitch_deg", np.array([]))
        traces = [
            _scatter(t, roll, "roll φ (deg)", color=self.COLORS["blue"]),
            _scatter(t, pitch, "pitch θ (deg)", color=self.COLORS["orange"]),
        ]
        # Reference bands as shapes in layout.
        layout = _time_layout("Posture stability", "Angle (deg)")
        layout["shapes"] = [
            {
                "type": "rect",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "y0": -3,
                "y1": 3,
                "fillcolor": "rgba(85,168,104,0.12)",
                "line": {"width": 0},
                "layer": "below",
            },
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "y0": 45,
                "y1": 45,
                "line": {"color": self.COLORS["red"], "dash": "dash", "width": 1},
            },
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "y0": -45,
                "y1": -45,
                "line": {"color": self.COLORS["red"], "dash": "dash", "width": 1},
            },
        ]
        return self._save("posture_stability", traces, layout)

    def plot_command_tracking(self) -> Optional[str]:
        hist = _hist_arrays(self.logger.history)
        t = hist.get("time_s")
        if t is None or len(t) == 0:
            return None
        path = os.path.join(self.save_dir, "command_tracking.html")
        if _HAS_PLOTLY:
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                subplot_titles=("Forward speed", "Yaw rate", "Heading error"),
                vertical_spacing=0.06,
            )
            fig.add_trace(
                go.Scatter(x=t, y=hist["velocity_ms"], name="vx", line={"color": self.COLORS["blue"]}),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=hist["target_velocity_ms"],
                    name="vx*",
                    line={"color": self.COLORS["orange"], "dash": "dash"},
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=t, y=hist["wz"], name="ωz", line={"color": self.COLORS["blue"]}),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=hist["cmd_wz"],
                    name="ωz*",
                    line={"color": self.COLORS["orange"], "dash": "dash"},
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=hist["heading_err_deg"],
                    name="heading err",
                    line={"color": self.COLORS["purple"]},
                ),
                row=3,
                col=1,
            )
            fig.update_layout(
                title="Command tracking",
                template="plotly_white",
                hovermode="x unified",
                dragmode="zoom",
                height=900,
                legend={"orientation": "h", "y": 1.08},
            )
            fig.update_xaxes(title_text="Time (s)", row=3, col=1, rangeslider={"visible": True})
            fig.update_yaxes(title_text="m/s", row=1, col=1)
            fig.update_yaxes(title_text="rad/s", row=2, col=1)
            fig.update_yaxes(title_text="deg", row=3, col=1)
            return _write_fig(fig, path)

        # CDN fallback: three stacked traces on one chart with dual meaning via names.
        traces = [
            _scatter(t, hist["velocity_ms"], "vx (m/s)", color=self.COLORS["blue"]),
            _scatter(t, hist["target_velocity_ms"], "vx*", color=self.COLORS["orange"], dash="dash"),
            _scatter(t, hist["wz"], "ωz (rad/s)", color=self.COLORS["green"]),
            _scatter(t, hist["cmd_wz"], "ωz*", color=self.COLORS["grey"], dash="dash"),
            _scatter(t, hist["heading_err_deg"], "heading err (deg)", color=self.COLORS["purple"]),
        ]
        return _write_html_cdn(path, traces, _time_layout("Command tracking", "value"))

    def plot_action_torque(self) -> Optional[str]:
        hist = _hist_arrays(self.logger.history)
        t = hist.get("time_s")
        if t is None or len(t) == 0:
            return None
        path = os.path.join(self.save_dir, "action_torque.html")
        if _HAS_PLOTLY:
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                subplot_titles=("Policy actions", "Joint torques", "Action jitter"),
                vertical_spacing=0.06,
            )
            fig.add_trace(
                go.Scatter(x=t, y=hist["steer_action"], name="steer a0", line={"color": self.COLORS["blue"]}),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=t, y=hist["drive_action"], name="drive a1", line={"color": self.COLORS["orange"]}),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=t, y=hist["steer_torque"], name="steer τ", line={"color": self.COLORS["blue"]}),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=t, y=hist["drive_torque"], name="drive τ", line={"color": self.COLORS["orange"]}),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=t, y=hist["action_jitter"], name="|Δa|", line={"color": self.COLORS["red"]}),
                row=3,
                col=1,
            )
            fig.update_layout(
                title="Action / torque",
                template="plotly_white",
                hovermode="x unified",
                dragmode="zoom",
                height=900,
                legend={"orientation": "h", "y": 1.08},
            )
            fig.update_xaxes(title_text="Time (s)", row=3, col=1, rangeslider={"visible": True})
            return _write_fig(fig, path)

        traces = [
            _scatter(t, hist["steer_action"], "steer a0", color=self.COLORS["blue"]),
            _scatter(t, hist["drive_action"], "drive a1", color=self.COLORS["orange"]),
            _scatter(t, hist["steer_torque"], "steer τ", color=self.COLORS["green"]),
            _scatter(t, hist["drive_torque"], "drive τ", color=self.COLORS["purple"]),
            _scatter(t, hist["action_jitter"], "|Δa|", color=self.COLORS["red"]),
        ]
        return _write_html_cdn(path, traces, _time_layout("Action / torque", "value"))

    def plot_2d_trajectory(self) -> Optional[str]:
        hist = _hist_arrays(self.logger.history)
        xs = hist.get("pos_x", np.array([]))
        ys = hist.get("pos_y", np.array([]))
        t = hist.get("time_s", np.arange(len(xs), dtype=float))
        if len(xs) == 0:
            return None
        path = os.path.join(self.save_dir, "trajectory_2d.html")
        max_drift = float(np.max(np.sqrt(xs**2 + ys**2))) if len(xs) else 0.0
        if _HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name="path",
                    marker={"size": 4, "color": t, "colorscale": "Viridis", "colorbar": {"title": "t (s)"}},
                    line={"width": 2},
                    text=[f"t={ti:.2f}s" for ti in t],
                    hovertemplate="x=%{x:.3f} m<br>y=%{y:.3f} m<br>%{text}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(x=[0], y=[0], mode="markers", name="spawn", marker={"symbol": "x", "size": 12, "color": self.COLORS["red"]})
            )
            if max_drift > 1e-6:
                theta = np.linspace(0, 2 * math.pi, 128)
                fig.add_trace(
                    go.Scatter(
                        x=max_drift * np.cos(theta),
                        y=max_drift * np.sin(theta),
                        mode="lines",
                        name=f"max drift {max_drift:.3f} m",
                        line={"dash": "dash", "color": self.COLORS["grey"]},
                    )
                )
            fig.update_layout(
                title="2D trajectory — top-down drift map",
                xaxis={"title": "X (m)", "scaleanchor": "y", "scaleratio": 1, "autorange": True},
                yaxis={"title": "Y (m)", "autorange": True},
                template="plotly_white",
                dragmode="zoom",
                hovermode="closest",
            )
            return _write_fig(fig, path)

        traces = [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "path",
                "x": xs.tolist(),
                "y": ys.tolist(),
                "marker": {"size": 4, "color": t.tolist(), "colorscale": "Viridis", "showscale": True},
            },
            {
                "type": "scatter",
                "mode": "markers",
                "name": "spawn",
                "x": [0.0],
                "y": [0.0],
                "marker": {"symbol": "x", "size": 12, "color": self.COLORS["red"]},
            },
        ]
        layout = {
            "title": {"text": "2D trajectory — top-down drift map"},
            "xaxis": {"title": "X (m)", "scaleanchor": "y", "scaleratio": 1, "autorange": True},
            "yaxis": {"title": "Y (m)", "autorange": True},
            "hovermode": "closest",
            "dragmode": "zoom",
            "template": "plotly_white",
        }
        return _write_html_cdn(path, traces, layout)

    def plot_dashboard(self) -> Optional[str]:
        """Single HTML page with all time-series panels (best for zoom/pan)."""
        hist = _hist_arrays(self.logger.history)
        t = hist.get("time_s")
        if t is None or len(t) == 0:
            return None
        path = os.path.join(self.save_dir, "dashboard.html")
        if not _HAS_PLOTLY:
            # Fallback: write an index that links the individual pages.
            links = []
            for name in (
                "velocity_profile",
                "posture_stability",
                "command_tracking",
                "action_torque",
                "trajectory_2d",
            ):
                links.append(f'<li><a href="{name}.html">{name}</a></li>')
            html = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
                "<title>CycleRL eval — interactive plots</title></head><body>"
                "<h1>CycleRL eval — interactive plots</h1><ul>"
                + "".join(links)
                + "</ul><p>Open any link: scroll to zoom, drag to box-zoom, double-click to reset.</p>"
                "</body></html>"
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            return path

        fig = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Velocity",
                "Roll / pitch",
                "Yaw rate",
                "Actions",
                "Action jitter",
            ),
            vertical_spacing=0.04,
            row_heights=[0.2, 0.2, 0.2, 0.2, 0.2],
        )
        C = self.COLORS
        fig.add_trace(go.Scatter(x=t, y=hist["velocity_ms"], name="vx", line={"color": C["blue"]}), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=t, y=hist["target_velocity_ms"], name="vx*", line={"color": C["orange"], "dash": "dash"}),
            row=1,
            col=1,
        )
        fig.add_trace(go.Scatter(x=t, y=hist["roll_deg"], name="roll", line={"color": C["blue"]}), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=hist["pitch_deg"], name="pitch", line={"color": C["orange"]}), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=hist["wz"], name="ωz", line={"color": C["blue"]}), row=3, col=1)
        fig.add_trace(
            go.Scatter(x=t, y=hist["cmd_wz"], name="ωz*", line={"color": C["orange"], "dash": "dash"}),
            row=3,
            col=1,
        )
        fig.add_trace(go.Scatter(x=t, y=hist["steer_action"], name="a0 steer", line={"color": C["blue"]}), row=4, col=1)
        fig.add_trace(go.Scatter(x=t, y=hist["drive_action"], name="a1 drive", line={"color": C["orange"]}), row=4, col=1)
        fig.add_trace(go.Scatter(x=t, y=hist["action_jitter"], name="|Δa|", line={"color": C["red"]}), row=5, col=1)
        fig.update_layout(
            title="CycleRL eval dashboard (scroll zoom · drag box-zoom · double-click reset)",
            template="plotly_white",
            hovermode="x unified",
            dragmode="zoom",
            height=1200,
            legend={"orientation": "h", "y": 1.02},
        )
        fig.update_xaxes(title_text="Time (s)", row=5, col=1, rangeslider={"visible": True})
        return _write_fig(fig, path)

    def generate_all(self) -> dict[str, str]:
        """Write all interactive HTML plots. Returns name → path."""
        out: dict[str, str] = {}
        printers = (
            ("velocity_profile", self.plot_velocity_profile),
            ("posture_stability", self.plot_posture_stability),
            ("command_tracking", self.plot_command_tracking),
            ("action_torque", self.plot_action_torque),
            ("trajectory_2d", self.plot_2d_trajectory),
            ("dashboard", self.plot_dashboard),
        )
        for name, fn in printers:
            try:
                path = fn()
            except Exception as exc:  # pragma: no cover
                print(f"[EvalPlotly] WARN: {name} failed: {exc}")
                continue
            if path:
                out[f"plotly_{name}"] = path
        if out:
            backend = "plotly.py" if _HAS_PLOTLY else "plotly.js CDN"
            print(f"[EvalPlotly] Interactive HTML ({backend}) → {self.save_dir}/")
            dash = out.get("plotly_dashboard")
            if dash:
                print(f"[EvalPlotly] Open dashboard: {dash}")
        return out


def generate_plotly_plots(logger: Any, save_dir: str) -> dict[str, str]:
    """Public helper used by ``BicycleEvalPlotter`` / play_with_metrics."""
    plot_dir = os.path.join(save_dir, "plotly")
    return BicycleEvalPlotly(logger, save_dir=plot_dir).generate_all()
