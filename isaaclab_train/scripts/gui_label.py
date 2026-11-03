"""On-screen policy label for Isaac Lab GUI play: a text box in the top-left corner of the Kit viewport.

Used by every play script (rsl_rl/play.py, rsl_rl/play_eval.py, skrl/play_locomotive_metrics.py) so the
window says which policy is running. Does nothing when headless (no viewport window).

    from scripts.gui_label import add_label_arg, show_policy_label, checkpoint_label
    add_label_arg(parser)                                   # before parse_args
    ...
    show_policy_label(args_cli.label or checkpoint_label(path), [f"task {task}", ...])
"""

from __future__ import annotations

import os

_KEEP: dict = {}  # holds the omni.ui widgets; they disappear when garbage-collected


def add_label_arg(parser) -> None:
    parser.add_argument(
        "--label", type=str, default=None,
        help="Title shown in the viewport (default: the checkpoint's run directory).",
    )


def checkpoint_label(path: str | None) -> str:
    """'<run dir>/<file>' from a checkpoint path, skipping a 'checkpoints' directory."""
    if not path:
        return "no checkpoint"
    parts = [p for p in os.path.abspath(path).split(os.sep) if p and p != "checkpoints"]
    return "/".join(parts[-2:])


def checkpoint_experiment(path: str | None) -> str:
    """'<experiment>/<run dir>/<file>' (the line under the title)."""
    if not path:
        return ""
    parts = [p for p in os.path.abspath(path).split(os.sep) if p and p != "checkpoints"]
    return "/".join(parts[-3:])


def show_policy_label(title: str, lines: list[str], name: str = "policy_label"):
    """Draw `title` (large) and `lines` (smaller) over the active viewport. Returns the frame or None."""
    try:
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport_window
    except ImportError:
        return None
    try:
        window = get_active_viewport_window()
    except Exception:  # noqa: BLE001 - no viewport extension loaded (headless)
        window = None
    if window is None:
        return None
    print(f"[INFO] viewport label: {title} | " + " | ".join(lines))
    frame = window.get_frame(name)
    with frame:
        with ui.VStack():
            ui.Spacer(height=44)  # clear the viewport toolbar
            with ui.HStack(height=0):
                ui.Spacer(width=12)
                with ui.ZStack(width=0, height=0):
                    ui.Rectangle(style={"background_color": 0xC0101010, "border_radius": 6})
                    with ui.HStack(width=0, height=0):
                        ui.Spacer(width=12)
                        with ui.VStack(width=0, height=0, spacing=3):
                            ui.Spacer(height=8)
                            ui.Label(title, width=0, style={"font_size": 26, "color": 0xFFFFFFFF})
                            for line in lines:
                                ui.Label(line, width=0, style={"font_size": 17, "color": 0xFFD8D8D8})
                            ui.Spacer(height=8)
                        ui.Spacer(width=12)
                ui.Spacer()
            ui.Spacer()
    _KEEP[name] = frame
    return frame
