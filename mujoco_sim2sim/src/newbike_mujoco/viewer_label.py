"""On-screen policy label for MuJoCo passive-viewer play (top-left text overlay via viewer.set_texts).

Used by scripts/run_deploy_viewer.py and scripts/locomotive/play_act_refine_lo.py so each window says
which policy is running. Two columns: field names on the left, values on the right.
"""

from __future__ import annotations

import os

import mujoco


def policy_title(path, label: str | None = None) -> str:
    """`label` if given, else '<parent dir>/<file>' of the policy path (skipping 'checkpoints')."""
    if label:
        return label
    if not path:
        return "no policy"
    parts = [p for p in os.path.abspath(str(path)).split(os.sep) if p and p != "checkpoints"]
    return "/".join(parts[-2:])


def short_path(path, depth: int = 3) -> str:
    if not path:
        return "-"
    parts = [p for p in os.path.abspath(str(path)).split(os.sep) if p]
    return "/".join(parts[-depth:])


def label_overlay(title: str, rows: list[tuple[str, str]], live: str = ""):
    """One set_texts tuple: 'Policy <title>' then one line per (field, value), then an optional live line."""
    keys = ["Policy"] + [k for k, _ in rows] + (["Live"] if live else [])
    vals = [title] + [v for _, v in rows] + ([live] if live else [])
    return (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(keys), "\n".join(vals))


def show_label(viewer, overlay) -> None:
    """Set the overlay; silently skipped on MuJoCo builds without Handle.set_texts."""
    setter = getattr(viewer, "set_texts", None)
    if setter is not None:
        setter(overlay)
