# Ported from ~/bike_test/CycleRL/scripts/skrl/play_camera.py (imports/paths rewritten for trackstand_rl locomotive).
"""Viewport camera helpers for CycleRL play scripts."""

from __future__ import annotations

import argparse


def add_play_camera_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for tracking a chosen env spawn in the Kit viewport."""
    group = parser.add_argument_group("camera")
    group.add_argument(
        "--camera_env",
        type=int,
        default=None,
        help=(
            "Environment index (0-based) for the viewport camera to track. "
            "When set, the camera follows that env's robot as it moves."
        ),
    )
    group.add_argument(
        "--camera_mode",
        type=str,
        default="robot",
        choices=["env", "robot", "body"],
        help=(
            "Tracking mode when --camera_env is set. "
            "'env': fixed offset from env spawn origin; "
            "'robot': follow robot articulation root (default); "
            "'body': follow a robot body (see --camera_body)."
        ),
    )
    group.add_argument(
        "--camera_asset",
        type=str,
        default="robot",
        help="Scene asset name for robot/body tracking (default: robot).",
    )
    group.add_argument(
        "--camera_body",
        type=str,
        default="base_link",
        help="Body name when --camera_mode=body (default: base_link).",
    )
    group.add_argument(
        "--no_camera_ui",
        action="store_true",
        default=False,
        help="Disable the CycleRL Camera dock window in the Kit viewport.",
    )


def attach_play_camera_ui(env, args=None):
    """Open the CycleRL Camera panel when running with the Kit viewport (--viz)."""
    if args is not None and getattr(args, "no_camera_ui", False):
        return None

    base = resolve_unwrapped_env(env)
    if getattr(base, "viewport_camera_controller", None) is None:
        return None

    tracker = None
    try:
        from play_camera_tracker import SmoothPlayCameraTracker

        tracker = SmoothPlayCameraTracker(
            base.viewport_camera_controller, base, smooth_alpha=0.45
        )
        tracker.install()
    except Exception as exc:
        print(f"[WARN] Smooth camera tracker unavailable: {exc}")

    try:
        from play_camera_ui import PlayCameraWindow
    except ImportError as exc:
        print(f"[WARN] CycleRL Camera UI unavailable: {exc}")
        return tracker

    try:
        window = PlayCameraWindow(base, tracker=tracker)
        print("[INFO] CycleRL Camera window opened (pick env index + follow mode).")
        return window
    except Exception as exc:
        print(f"[WARN] Failed to open CycleRL Camera UI: {exc}")
        return tracker


def apply_play_camera_cfg(env_cfg, args, *, num_envs: int | None = None) -> None:
    """Configure ``env_cfg.viewer`` so the Kit camera tracks one env spawn."""
    camera_env = getattr(args, "camera_env", None)
    if camera_env is None:
        return

    n = num_envs if num_envs is not None else env_cfg.scene.num_envs
    if camera_env < 0 or camera_env >= n:
        raise ValueError(
            f"--camera_env {camera_env} out of range; num_envs={n} (valid: 0..{n - 1})."
        )

    viewer = env_cfg.viewer
    viewer.env_index = camera_env
    mode = getattr(args, "camera_mode", "robot")
    asset = getattr(args, "camera_asset", "robot")

    if mode == "env":
        viewer.origin_type = "env"
        viewer.asset_name = None
        viewer.body_name = None
    elif mode == "robot":
        viewer.origin_type = "asset_root"
        viewer.asset_name = asset
        viewer.body_name = None
    else:  # body
        viewer.origin_type = "asset_body"
        viewer.asset_name = asset
        viewer.body_name = getattr(args, "camera_body", "base_link")

    print(
        f"[INFO] Viewport camera tracking env {camera_env} "
        f"(mode={mode}, asset={asset!r}, eye={viewer.eye}, lookat={viewer.lookat})"
    )


def resolve_unwrapped_env(env):
    """Walk gym wrappers to the underlying Isaac Lab environment."""
    unwrapped = env
    while hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env
    return unwrapped


def set_play_camera_env(env, env_index: int) -> None:
    """Switch the tracked env index at runtime (e.g. after env creation)."""
    base = resolve_unwrapped_env(env)
    vcc = getattr(base, "viewport_camera_controller", None)
    if vcc is None:
        print("[WARN] viewport_camera_controller unavailable (headless?).")
        return
    vcc.set_view_env_index(env_index)
    if vcc.cfg.origin_type == "asset_root" and vcc.cfg.asset_name:
        vcc.update_view_to_asset_root(vcc.cfg.asset_name)
    elif vcc.cfg.origin_type == "asset_body" and vcc.cfg.asset_name and vcc.cfg.body_name:
        vcc.update_view_to_asset_body(vcc.cfg.asset_name, vcc.cfg.body_name)
    print(f"[INFO] Camera now tracking env {env_index}.")
