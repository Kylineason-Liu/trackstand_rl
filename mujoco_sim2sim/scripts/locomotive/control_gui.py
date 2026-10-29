# Ported from ~/bike_test/CycleRL/scripts/mujoco_paper/control_gui.py (imports/paths rewritten for trackstand_rl locomotive).
"""Tkinter box-joystick panel for live vx* / heading commands."""

from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from dataclasses import dataclass


@dataclass
class CommandState:
    """Shared with the sim thread (write from GUI, read each policy step).

    ``turn`` is the value the sim should use. Positive turn is a left yaw.
    The mouse writes ``mouse_turn``. Arrow keys write ``key_turn`` and win
    until ``key_turn_until``.
    """

    vx_cmd: float = 2.0
    mouse_turn: float = 0.0
    key_turn: float = 0.0
    key_turn_until: float = 0.0
    knob_dirty: bool = False
    drive_remote: bool = False
    reset_requested: bool = False
    quit_requested: bool = False
    paused: bool = False

    @property
    def turn(self) -> float:
        if time.time() < self.key_turn_until and abs(self.key_turn) > 1e-6:
            return self.key_turn
        return self.mouse_turn


class JoystickPanel:
    """Small always-on-top window: drag pad for speed (Y) and turn (X)."""

    def __init__(
        self,
        state: CommandState,
        vx_min: float = 0.8,
        vx_max: float = 3.0,
        size: int = 220,
    ):
        self.state = state
        self.vx_min = float(vx_min)
        self.vx_max = float(vx_max)
        self.size = int(size)
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._knob = None
        self._status: tk.StringVar | None = None
        self._cx = self.size // 2
        self._cy = self.size // 2
        self._radius = self.size // 2 - 18
        self._knob_r = 18
        self._nx = 0.0
        self._ny = self._vx_to_ny(state.vx_cmd)
        self._dragging = False
        self._thread: threading.Thread | None = None

    def _vx_to_ny(self, vx: float) -> float:
        # up (−ny) = faster
        span = max(self.vx_max - self.vx_min, 1e-6)
        mid = 0.5 * (self.vx_min + self.vx_max)
        return float(max(-1.0, min(1.0, -(vx - mid) / (0.5 * span))))

    def _ny_to_vx(self, ny: float) -> float:
        mid = 0.5 * (self.vx_min + self.vx_max)
        half = 0.5 * (self.vx_max - self.vx_min)
        return float(mid + (-ny) * half)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="joystick-gui", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        root = tk.Tk()
        self._root = root
        root.title("CycleRL joystick")
        root.resizable(False, False)
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass

        self._status = tk.StringVar(value="")
        frm = tk.Frame(root, padx=10, pady=8)
        frm.pack()
        tk.Label(
            frm,
            text="Drag, or arrows / WASD\n← left    right →    ↑ faster",
            font=("sans-serif", 10),
        ).pack()
        self._canvas = tk.Canvas(
            frm, width=self.size, height=self.size, bg="#1e1e1e", highlightthickness=0
        )
        self._canvas.pack(pady=6)
        tk.Label(frm, textvariable=self._status, font=("monospace", 11)).pack()
        btn_row = tk.Frame(frm)
        btn_row.pack(pady=4)
        tk.Button(btn_row, text="Reset (R)", width=10, command=self._request_reset).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(btn_row, text="Pause", width=8, command=self._toggle_pause).pack(
            side=tk.LEFT, padx=4
        )
        self._drive_btn = tk.Button(btn_row, text="", width=14, command=self._toggle_drive)
        self._drive_btn.pack(side=tk.LEFT, padx=4)
        self._refresh_drive_btn()
        tk.Label(
            frm,
            text=f"vx* ∈ [{self.vx_min:.1f}, {self.vx_max:.1f}] m/s\n"
            "Y holds speed; X springs back\n"
            "Drive: stick sends vx* to the motor",
            font=("sans-serif", 9),
            fg="#444",
        ).pack()

        self._draw_base()
        self._set_knob_from_n(self._nx, self._ny, write_state=True)

        c = self._canvas
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release)
        root.bind("<KeyPress>", self._on_key_press)
        root.bind("<KeyRelease>", self._on_key_release)
        self._keys: set[str] = set()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(50, self._tick_status)
        root.mainloop()

    def _draw_base(self) -> None:
        assert self._canvas is not None
        c = self._canvas
        c.delete("all")
        r = self._radius
        c.create_oval(
            self._cx - r,
            self._cy - r,
            self._cx + r,
            self._cy + r,
            outline="#666",
            width=2,
            fill="#2a2a2a",
        )
        c.create_line(self._cx, self._cy - r, self._cx, self._cy + r, fill="#444")
        c.create_line(self._cx - r, self._cy, self._cx + r, self._cy, fill="#444")
        c.create_text(self._cx, 14, text="FAST", fill="#8ec07c", font=("sans-serif", 9))
        c.create_text(self._cx, self.size - 12, text="SLOW", fill="#d3869b", font=("sans-serif", 9))
        c.create_text(12, self._cy, text="L", fill="#83a598", font=("sans-serif", 9))
        c.create_text(self.size - 12, self._cy, text="R", fill="#83a598", font=("sans-serif", 9))
        self._knob = c.create_oval(0, 0, 0, 0, fill="#83a598", outline="#eee", width=2)

    def _set_knob_from_n(self, nx: float, ny: float, write_state: bool) -> None:
        self._nx = float(max(-1.0, min(1.0, nx)))
        self._ny = float(max(-1.0, min(1.0, ny)))
        # clamp to circle
        mag = math.hypot(self._nx, self._ny)
        if mag > 1.0:
            self._nx /= mag
            self._ny /= mag
        x = self._cx + self._nx * self._radius
        y = self._cy + self._ny * self._radius
        assert self._canvas is not None and self._knob is not None
        kr = self._knob_r
        self._canvas.coords(self._knob, x - kr, y - kr, x + kr, y + kr)
        if write_state:
            self.state.vx_cmd = self._ny_to_vx(self._ny)
            # Pad left (negative nx) is a left turn, which is positive yaw.
            self.state.mouse_turn = -self._nx

    def _event_to_n(self, event) -> tuple[float, float]:
        dx = event.x - self._cx
        dy = event.y - self._cy
        return dx / self._radius, dy / self._radius

    def _on_press(self, event) -> None:
        self._dragging = True
        nx, ny = self._event_to_n(event)
        self._set_knob_from_n(nx, ny, write_state=True)

    def _on_drag(self, event) -> None:
        if not self._dragging:
            return
        nx, ny = self._event_to_n(event)
        self._set_knob_from_n(nx, ny, write_state=True)

    def _on_release(self, _event) -> None:
        self._dragging = False
        # Spring X (turn) back to center; keep Y (speed).
        self._set_knob_from_n(0.0, self._ny, write_state=True)

    def _on_key_press(self, event) -> None:
        key = event.keysym
        if key in ("Up", "w", "W", "Down", "s", "S"):
            step = 0.1 if key in ("Up", "w", "W") else -0.1
            self.state.vx_cmd = float(min(self.vx_max, max(self.vx_min, self.state.vx_cmd + step)))
            self.state.knob_dirty = True
        self._keys.add(key)
        self._apply_key_turn()

    def _on_key_release(self, event) -> None:
        self._keys.discard(event.keysym)
        self._apply_key_turn()

    def _apply_key_turn(self) -> None:
        turn = 0.0
        if self._keys & {"Left", "a", "A"}:
            turn += 1.0
        if self._keys & {"Right", "d", "D"}:
            turn -= 1.0
        self.state.key_turn = float(max(-1.0, min(1.0, turn)))
        # Stay live while held. A release clears the keys and drops the deadline.
        self.state.key_turn_until = time.time() + 0.5 if abs(turn) > 0.0 else 0.0
        self.state.knob_dirty = True

    def _request_reset(self) -> None:
        self.state.reset_requested = True

    def _toggle_pause(self) -> None:
        self.state.paused = not self.state.paused

    def _toggle_drive(self) -> None:
        self.state.drive_remote = not self.state.drive_remote
        self._refresh_drive_btn()

    def _refresh_drive_btn(self) -> None:
        if self._drive_btn is None:
            return
        if self.state.drive_remote:
            self._drive_btn.configure(text="Drive: stick")
        else:
            self._drive_btn.configure(text="Drive: policy")

    def _on_close(self) -> None:
        self.state.quit_requested = True
        if self._root is not None:
            self._root.destroy()

    def _tick_status(self) -> None:
        if self._root is None or self._status is None:
            return
        if self.state.knob_dirty and not self._dragging:
            self.state.knob_dirty = False
            self._set_knob_from_n(-self.state.turn, self._vx_to_ny(self.state.vx_cmd), write_state=False)
        pause = "  PAUSE" if self.state.paused else ""
        drive = "stick" if self.state.drive_remote else "policy"
        self._status.set(
            f"vx*={self.state.vx_cmd:4.2f} m/s   turn={self.state.turn:+.2f}  drv={drive}{pause}"
        )
        self._root.after(50, self._tick_status)
