# Ported from ~/bike_test/CycleRL/scripts/skrl/play_camera_tracker.py (imports/paths rewritten for trackstand_rl locomotive).
"""Smoothed viewport camera tracking that preserves user zoom / orbit."""

from __future__ import annotations

import weakref

import numpy as np
import torch


class SmoothPlayCameraTracker:
    """Replace Isaac Lab's per-frame camera reset with smoothed, zoom-friendly tracking.

    The stock :class:`~isaaclab.envs.ui.viewport_camera_controller.ViewportCameraController`
    re-applies fixed eye/lookat offsets every post-update tick. That causes jitter (raw
    root pose) and prevents viewport scroll-zoom (user orbit is overwritten immediately).

    This tracker instead:
      1. Low-pass filters the follow target in world space.
      2. Reads the *current* viewport eye/target and preserves ``eye - target``.
      3. Translates both by the smoothed target delta only.
    """

    def __init__(
        self,
        vcc,
        env,
        *,
        smooth_alpha: float = 0.12,
        distance_scale: float = 1.0,
    ):
        self._vcc = vcc
        self._env = env
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.02, 1.0))
        self.distance_scale = max(0.05, float(distance_scale))
        self._smooth_origin: np.ndarray | None = None
        self._prev_smooth_origin: np.ndarray | None = None
        self._lookat_offset = np.array(vcc.default_cam_lookat, dtype=float)
        self._default_eye_offset = np.array(vcc.default_cam_eye - vcc.default_cam_lookat, dtype=float)
        self._handle = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        handle = getattr(self._vcc, "_viewport_camera_update_handle", None)
        if handle is not None:
            handle.unsubscribe()
            self._vcc._viewport_camera_update_handle = None

        import omni.kit.app

        app_interface = omni.kit.app.get_app_interface()
        stream = app_interface.get_post_update_event_stream()
        self._handle = stream.create_subscription_to_pop(
            lambda event, obj=weakref.proxy(self): obj._on_post_update(event)
        )
        self._installed = True

    def uninstall(self) -> None:
        if self._handle is not None:
            self._handle.unsubscribe()
            self._handle = None
        self._installed = False

    def reset_smooth_state(self) -> None:
        """Snap smoothing state (e.g. after switching env index)."""
        self._smooth_origin = None
        self._prev_smooth_origin = None

    def reset_view_distance(self) -> None:
        """Restore default eye distance while keeping the current look target."""
        eye, target = self._read_viewport_camera()
        if eye is None or target is None:
            return
        direction = self._default_eye_offset
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 1.0, 0.5], dtype=float)
            norm = np.linalg.norm(direction)
        direction = direction / norm
        distance = np.linalg.norm(self._default_eye_offset) * self.distance_scale
        new_eye = target + direction * distance
        self._write_viewport_camera(new_eye, target)

    def _on_post_update(self, _event) -> None:
        origin_type = self._vcc.cfg.origin_type
        if origin_type not in ("asset_root", "asset_body"):
            return

        raw_origin = self._get_raw_origin()
        if raw_origin is None:
            return

        if self._smooth_origin is None:
            self._smooth_origin = raw_origin.copy()
        else:
            a = self.smooth_alpha
            self._smooth_origin = a * raw_origin + (1.0 - a) * self._smooth_origin

        eye, target = self._read_viewport_camera()

        if self._prev_smooth_origin is None or eye is None or target is None:
            eye_offset = self._default_eye_offset * self.distance_scale
            new_target = self._smooth_origin + self._lookat_offset
            new_eye = new_target + eye_offset
        else:
            delta = self._smooth_origin - self._prev_smooth_origin
            new_target = target + delta
            new_eye = eye + delta

        self._prev_smooth_origin = self._smooth_origin.copy()
        self._write_viewport_camera(new_eye, new_target)
        self._vcc.viewer_origin = torch.as_tensor(self._smooth_origin, device=self._env.device)

    def _get_raw_origin(self) -> np.ndarray | None:
        cfg = self._vcc.cfg
        env_idx = cfg.env_index
        try:
            if cfg.origin_type == "asset_root" and cfg.asset_name:
                pos = self._env.scene[cfg.asset_name].data.root_pos_w.torch[env_idx]
            elif cfg.origin_type == "asset_body" and cfg.asset_name and cfg.body_name:
                asset = self._env.scene[cfg.asset_name]
                body_id, _ = asset.find_bodies(cfg.body_name)
                pos = asset.data.body_pos_w.torch[env_idx, body_id].squeeze(0)
            else:
                return None
            return pos.detach().cpu().numpy().astype(float)
        except Exception:
            return None

    def _read_viewport_camera(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            from omni.kit.viewport.utility import get_active_viewport
            from omni.kit.viewport.utility.camera_state import ViewportCameraState

            viewport = get_active_viewport()
            if viewport is None:
                return None, None
            cam_path = viewport.camera_path or self._vcc.cfg.cam_prim_path
            state = ViewportCameraState(cam_path, viewport)
            eye = np.array(state.position_world, dtype=float)
            target = np.array(state.target_world, dtype=float)
            return eye, target
        except Exception:
            return None, None

    def _write_viewport_camera(self, eye: np.ndarray, target: np.ndarray) -> None:
        try:
            from omni.kit.viewport.utility import get_active_viewport
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
            from pxr import Gf

            viewport = get_active_viewport()
            if viewport is None:
                return
            cam_path = viewport.camera_path or self._vcc.cfg.cam_prim_path
            state = ViewportCameraState(cam_path, viewport)
            state.set_position_world(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])), False)
            state.set_target_world(Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])), True)
        except Exception:
            # Fallback to Isaac Lab helper (no scroll preservation, but better than nothing).
            eye_t = (float(eye[0]), float(eye[1]), float(eye[2]))
            target_t = (float(target[0]), float(target[1]), float(target[2]))
            self._env.sim.set_camera_view(eye=eye_t, target=target_t)
