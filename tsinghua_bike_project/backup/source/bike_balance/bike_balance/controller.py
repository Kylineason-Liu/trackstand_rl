"""Nominal continuous-time state-feedback controllers for the 3-state bicycle.

Supports three gain methods:
- ``place_multiple_poles``   — all closed-loop poles at the same location wc
- ``place_distinct_poles``   — arbitrary distinct pole vector
- ``lqr``                    — continuous-time LQR with Qc, Rc

Control law:   u = u_eq + K (x_eq - x)

where (x_eq, u_eq) is the steady-state equilibrium consistent with the
requested controlled-output reference (steering angle by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles


@dataclass(frozen=True)
class EquilibriumPoint:
    """Dynamically consistent nominal steady-state operating point."""
    state: np.ndarray
    control: np.ndarray
    output: np.ndarray
    residual_norm: float
    is_unique: bool


class NominalBikeController:
    """Full-state SISO feedback controller (pole-placement or LQR).

    Parameters
    ----------
    sys:
        Object with attributes ``A``, ``Bu``, ``Cm``, ``Co``, ``dt``.
        ``Cm`` must be square and invertible (full-state measurement).
    method:
        ``"place_multiple_poles"``, ``"place_distinct_poles"`` or ``"lqr"``.
    u_min, u_max:
        Scalar or per-input command saturation limits.
    equilibrium_tolerance:
        Tolerance for steady-state residual checks.

    Keyword arguments (per method)
    ------------------------------
    place_multiple_poles:
        ``wc`` — scalar (all poles placed here)
    place_distinct_poles:
        ``poles_ctr`` — list/array of length *nx*
    lqr:
        ``Qc`` — (nx, nx) state cost
        ``Rc`` — (nu, nu) control cost
    """

    def __init__(
        self,
        sys: Any,
        method: str = "place_multiple_poles",
        *,
        u_min: float | None = None,
        u_max: float | None = None,
        equilibrium_tolerance: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        self.method = method
        self.gain_kwargs = dict(kwargs)
        self.equilibrium_tolerance = float(equilibrium_tolerance)
        self._set_matrices(sys)
        self.u_min = self._scalar_limit(u_min, -np.inf)
        self.u_max = self._scalar_limit(u_max, np.inf)
        if self.u_min > self.u_max:
            raise ValueError("u_min must not be greater than u_max.")
        self.update_gain(method, **kwargs)
        self.reset()

    # ------------------------------------------------------------------
    # Matrix helpers
    # ------------------------------------------------------------------
    def _set_matrices(self, sys: Any) -> None:
        A = np.asarray(sys.A, dtype=float)
        B = np.asarray(sys.Bu, dtype=float)
        Cm = np.asarray(sys.Cm, dtype=float)
        Co = np.asarray(sys.Co, dtype=float)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square.")
        self.nx = A.shape[0]
        if B.ndim != 2 or B.shape[0] != self.nx:
            raise ValueError("Bu shape mismatch.")
        self.nu = B.shape[1]
        if Cm.shape != (self.nx, self.nx):
            raise ValueError("Cm must be square and full-rank (full-state measurement).")
        self.no = Co.shape[0]

        self.A = A.copy()
        self.Bu = B.copy()
        self.Cm = Cm.copy()
        self.Co = Co.copy()
        self.dt = float(getattr(sys, "dt", 0.0))

    @staticmethod
    def _scalar_limit(value: float | None, default: float) -> float:
        return float(default) if value is None else float(value)

    @staticmethod
    def _controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.hstack(
            [np.linalg.matrix_power(A, k) @ B for k in range(A.shape[0])]
        )

    # ------------------------------------------------------------------
    # Ackermann for SISO (supports repeated poles)
    # ------------------------------------------------------------------
    @classmethod
    def _ackermann(cls, A: np.ndarray, B: np.ndarray, poles: np.ndarray) -> np.ndarray:
        n = A.shape[0]
        C = cls._controllability_matrix(A, B)
        if np.linalg.matrix_rank(C) != n:
            raise ValueError("System is not controllable.")

        coeffs = np.poly(poles)
        phi = np.linalg.matrix_power(A, n)
        for i in range(1, n + 1):
            phi += coeffs[i] * np.linalg.matrix_power(A, n - i)

        e_n = np.zeros((1, n))
        e_n[0, -1] = 1.0
        return np.linalg.solve(C.T, e_n.T).T @ phi

    # ------------------------------------------------------------------
    # Gain computation
    # ------------------------------------------------------------------
    def update_gain(
        self, method: str | None = None, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recompute the feedback gain K from the current A, Bu matrices."""
        if method is not None:
            self.method = method
        if kwargs:
            self.gain_kwargs = dict(kwargs)
        opts = self.gain_kwargs

        C = self._controllability_matrix(self.A, self.Bu)
        if np.linalg.matrix_rank(C) != self.nx:
            raise ValueError("The nominal system is not controllable at this speed.")

        if self.method == "place_multiple_poles":
            wc = opts.get("wc")
            if wc is None:
                raise ValueError("'wc' is required for place_multiple_poles.")
            poles = np.full(self.nx, complex(wc))
            self.K = np.real_if_close(self._ackermann(self.A, self.Bu, poles)).astype(float)

        elif self.method == "place_distinct_poles":
            poles = np.asarray(opts.get("poles_ctr", ()), dtype=complex)
            if poles.size != self.nx:
                raise ValueError(f"Need exactly {self.nx} poles.")
            self.K = np.real_if_close(
                place_poles(self.A, self.Bu, poles).gain_matrix
            ).astype(float)

        elif self.method == "lqr":
            Q = np.asarray(opts.get("Qc"), dtype=float)
            R = np.asarray(opts.get("Rc"), dtype=float)
            if Q.shape != (self.nx, self.nx) or R.shape != (self.nu, self.nu):
                raise ValueError("Qc/Rc dimensions mismatch.")
            P = solve_continuous_are(self.A, self.Bu, Q, R)
            self.K = np.linalg.solve(R, self.Bu.T @ P)
            poles = np.linalg.eigvals(self.A - self.Bu @ self.K)
        else:
            raise ValueError(f"Unknown method '{self.method}'.")

        self.closed_loop_poles = np.linalg.eigvals(self.A - self.Bu @ self.K)
        return self.K.copy(), self.closed_loop_poles.copy()

    # CamelCase alias for tutorial compatibility
    updateSysAndGain = update_gain

    # ------------------------------------------------------------------
    # Equilibrium
    # ------------------------------------------------------------------
    def solve_equilibrium(self, reference: np.ndarray | float) -> EquilibriumPoint:
        """Solve  A x_eq + Bu u_eq = 0,  Co x_eq = reference."""
        ref = np.atleast_1d(np.asarray(reference, dtype=float)).reshape(self.no, 1)
        M = np.block([
            [self.A, self.Bu],
            [self.Co, np.zeros((self.no, self.nu))],
        ])
        rhs = np.vstack((np.zeros((self.nx, 1)), ref))
        rank = np.linalg.matrix_rank(M)
        if M.shape[0] == M.shape[1] and rank == self.nx + self.nu:
            sol = np.linalg.solve(M, rhs)
        else:
            sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        residual = np.linalg.norm(M @ sol - rhs)
        return EquilibriumPoint(
            state=sol[: self.nx].copy(),
            control=sol[self.nx:].copy(),
            output=(self.Co @ sol[: self.nx]).copy(),
            residual_norm=float(residual),
            is_unique=(rank == self.nx + self.nu),
        )

    # ------------------------------------------------------------------
    # Control step
    # ------------------------------------------------------------------
    def step(
        self,
        reference: np.ndarray | float,
        measured_state: np.ndarray,
        t: float | None = None,
    ) -> np.ndarray:
        """Compute  u = u_eq + K (x_eq - x), saturated to [u_min, u_max].

        ``reference`` is the controlled-output reference (steering angle).
        ``measured_state`` is the 3×1 full state vector.
        """
        del t
        ref = np.atleast_1d(np.asarray(reference, dtype=float)).reshape(self.no, 1)
        x = np.atleast_1d(np.asarray(measured_state, dtype=float)).reshape(self.nx, 1)

        eq = self.solve_equilibrium(ref)
        x_eq, u_eq = eq.state, eq.control

        unsaturated = u_eq + self.K @ (x_eq - x)
        control = np.clip(unsaturated, self.u_min, self.u_max)

        self.last_state = x
        self.last_x_eq = x_eq
        self.last_u_eq = u_eq
        self.last_control = control
        return control.copy()

    def reset(self) -> None:
        """Clear diagnostic state (controller has no dynamic observer)."""
        self.last_state = np.zeros((self.nx, 1))
        self.last_x_eq = np.zeros((self.nx, 1))
        self.last_u_eq = np.zeros((self.nu, 1))
        self.last_control = np.zeros((self.nu, 1))
