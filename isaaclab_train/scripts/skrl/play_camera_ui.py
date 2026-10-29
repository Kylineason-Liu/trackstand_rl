# Ported from ~/bike_test/CycleRL/scripts/skrl/play_camera_ui.py (imports/paths rewritten for trackstand_rl locomotive).
"""Kit viewport UI for picking which env spawn the play camera follows."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from play_camera_tracker import SmoothPlayCameraTracker


class PlayCameraWindow:
    """Small dockable window to pick env index and camera follow mode during play."""

    WINDOW_NAME = "CycleRL Camera"
    _FOLLOW_ITEMS = ("Robot", "Env Spawn", "Base Link")
    _DEFAULT_ASSET = "robot"
    _DEFAULT_BODY = "base_link"

    def __init__(self, env: ManagerBasedEnv, tracker: SmoothPlayCameraTracker | None = None):
        self._env = env
        self._vcc = env.viewport_camera_controller
        self._tracker = tracker
        if self._vcc is None:
            raise RuntimeError("Viewport camera controller is not available (headless mode?).")

        import omni.ui
        import isaacsim.gui.components.ui_utils as ui_utils

        self._ui_utils = ui_utils

        self._window = omni.ui.Window(
            self.WINDOW_NAME,
            width=360,
            height=280,
            visible=True,
            dock_preference=omni.ui.DockPreference.LEFT_BOTTOM,
        )
        asyncio.ensure_future(self._dock_window(self._window.title))

        follow_default = self._follow_mode_to_label(
            self._vcc.cfg.origin_type,
            self._vcc.cfg.body_name,
        )
        default_smooth = 0.40 if self._tracker is not None else 0.0
        with self._window.frame:
            with omni.ui.VStack(spacing=8, height=0):
                omni.ui.Label(
                    "Pick env + follow mode. Scroll / middle-mouse in the viewport to zoom.",
                    word_wrap=True,
                    height=0,
                )
                self._env_index = ui_utils.int_builder(
                    label="Environment Index",
                    type="button",
                    default_val=self._vcc.cfg.env_index + 1,
                    min=1,
                    max=self._env.num_envs,
                    tooltip="1-based env index (same ordering as env_0, env_1, … in the scene).",
                )
                self._env_index.add_value_changed_fn(self._on_env_index_changed)
                self._follow_mode = ui_utils.dropdown_builder(
                    label="Follow Mode",
                    type="dropdown",
                    default_val=self._FOLLOW_ITEMS.index(follow_default),
                    items=list(self._FOLLOW_ITEMS),
                    tooltip="Robot follows the bike as it drives; Env Spawn stays at the tile origin.",
                    on_clicked_fn=self._on_follow_mode_changed,
                )
                if self._tracker is not None:
                    self._smooth_slider = ui_utils.float_builder(
                        label="Smoothing",
                        type="drag",
                        default_val=default_smooth,
                        min=0.0,
                        max=0.98,
                        step=0.02,
                        tooltip="Higher = less jitter, slower follow (0 = raw / jittery).",
                    )
                    self._smooth_slider.add_value_changed_fn(self._on_smooth_changed)
                    # Apply default immediately (slider may not fire on create).
                    try:
                        self._on_smooth_changed(self._smooth_slider)
                    except Exception:
                        self._tracker.smooth_alpha = max(0.02, 1.0 - default_smooth * 0.96)
                    self._distance_slider = ui_utils.float_builder(
                        label="Distance scale",
                        type="drag",
                        default_val=1.0,
                        min=0.15,
                        max=3.0,
                        step=0.05,
                        tooltip="Scale default camera distance (also use viewport scroll to zoom).",
                    )
                    self._distance_slider.add_value_changed_fn(self._on_distance_changed)
                    ui_utils.btn_builder(
                        label="Reset View",
                        type="button",
                        text="Reset distance",
                        tooltip="Restore default camera distance to the tracked target.",
                        on_clicked_fn=self._on_reset_view,
                    )

        # World origin is useless for play — default to robot tracking on env 0.
        if self._vcc.cfg.origin_type == "world":
            self._apply_follow_mode("Robot")

    async def _dock_window(self, window_title: str) -> None:
        """Dock beside the Property panel (same pattern as Isaac Lab env window)."""
        import omni.ui

        for _ in range(5):
            if omni.ui.Workspace.get_window(window_title):
                break
            await self._env.sim.app.next_update_async()

        custom_window = omni.ui.Workspace.get_window(window_title)
        property_window = omni.ui.Workspace.get_window("Property")
        if custom_window and property_window:
            # Isaac Lab env windows use DockPosition.SAME (not DockPreference).
            dock_pos = getattr(omni.ui, "DockPosition", None)
            if dock_pos is not None and hasattr(dock_pos, "SAME"):
                custom_window.dock_in(property_window, dock_pos.SAME, 1.0)
            else:
                # Fallback: leave the window floating if DockPosition is unavailable.
                pass
            custom_window.focus()

    def _follow_mode_to_label(self, origin_type: str, body_name: str | None) -> str:
        if origin_type == "env":
            return "Env Spawn"
        if origin_type == "asset_body":
            return "Base Link"
        return "Robot"

    def _on_env_index_changed(self, model) -> None:
        env_index = model.as_int - 1
        self._vcc.set_view_env_index(env_index)
        if self._tracker is not None:
            self._tracker.reset_smooth_state()
        print(f"[CycleRL Camera] Tracking env {env_index}.")

    def _on_follow_mode_changed(self, label: str) -> None:
        self._apply_follow_mode(label)
        print(f"[CycleRL Camera] Follow mode: {label} (env {self._vcc.cfg.env_index}).")

    def _on_smooth_changed(self, model) -> None:
        if self._tracker is None:
            return
        smoothing = float(model.get_value_as_float())
        # Map UI smoothing (0=raw, 0.98=very smooth) to EMA alpha on raw pose.
        self._tracker.smooth_alpha = max(0.02, 1.0 - smoothing * 0.96)

    def _on_distance_changed(self, model) -> None:
        if self._tracker is None:
            return
        self._tracker.distance_scale = max(0.05, float(model.get_value_as_float()))

    def _on_reset_view(self) -> None:
        if self._tracker is not None:
            self._tracker.reset_view_distance()

    def _apply_follow_mode(self, label: str) -> None:
        if self._tracker is not None:
            self._tracker.reset_smooth_state()
        if label == "Env Spawn":
            self._vcc.update_view_to_env()
        elif label == "Base Link":
            if self._tracker is not None:
                self._vcc.cfg.asset_name = self._DEFAULT_ASSET
                self._vcc.cfg.body_name = self._DEFAULT_BODY
                self._vcc.cfg.origin_type = "asset_body"
            else:
                self._vcc.update_view_to_asset_body(self._DEFAULT_ASSET, self._DEFAULT_BODY)
        else:
            if self._tracker is not None:
                self._vcc.cfg.asset_name = self._DEFAULT_ASSET
                self._vcc.cfg.body_name = None
                self._vcc.cfg.origin_type = "asset_root"
            else:
                self._vcc.update_view_to_asset_root(self._DEFAULT_ASSET)

    def _refresh_follow_target(self) -> None:
        # With smooth tracker active, cfg updates are enough — avoid reset_camera calls.
        if self._tracker is not None:
            return
        origin = self._vcc.cfg.origin_type
        if origin == "asset_root" and self._vcc.cfg.asset_name:
            self._vcc.update_view_to_asset_root(self._vcc.cfg.asset_name)
        elif origin == "asset_body" and self._vcc.cfg.asset_name and self._vcc.cfg.body_name:
            self._vcc.update_view_to_asset_body(self._vcc.cfg.asset_name, self._vcc.cfg.body_name)
        elif origin == "env":
            self._vcc.update_view_to_env()
