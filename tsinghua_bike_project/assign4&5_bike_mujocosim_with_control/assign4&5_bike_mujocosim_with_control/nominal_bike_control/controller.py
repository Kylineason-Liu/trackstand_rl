"""Nominal continuous-time state-feedback controllers.

The controller deliberately contains no disturbance model or disturbance
observer.  It uses the measured state directly, computes the nominal
equilibrium for the requested output, and applies

    u = u_equilibrium + K (x_equilibrium - x).

The compatibility methods ``updateSysAndGain`` and ``stepAndGetControl`` make
the controller a drop-in replacement for ``GesobcController`` in the example
notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles


ArrayLike = np.ndarray | Iterable[float] | float


@dataclass(frozen=True)
class _Matrices:
    A: np.ndarray
    B: np.ndarray
    Cm: np.ndarray
    Co: np.ndarray
    dt: float


@dataclass(frozen=True)
class EquilibriumPoint:
    """A dynamically consistent nominal steady-state operating point."""

    state: np.ndarray
    control: np.ndarray
    output: np.ndarray
    residual_norm: float
    is_unique: bool


class NominalStateFeedbackController:
    """Full-state feedback controller without disturbance observation.

    Parameters
    ----------
    sys:
        Object exposing ``A``, ``Bu``, ``Cm``, ``Co`` and optionally ``dt``.
        ``Cm`` must provide an invertible full-state measurement map.
    method:
        ``"place_multiple_poles"``, ``"place_distinct_poles"`` or ``"lqr"``.
    u_min, u_max:
        Optional scalar or per-input command limits.

    Other keyword arguments follow the existing toolbox convention: ``wc``
    for repeated-pole placement, ``poles_ctr`` for distinct poles, and
    ``Qc``/``Rc`` for LQR.
    """

    def __init__(
        self,
        sys: Any,
        method: str = "place_multiple_poles",
        *,
        u_min: ArrayLike | None = None,
        u_max: ArrayLike | None = None,
        equilibrium_tolerance: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        self.method = method
        self.gain_kwargs = dict(kwargs)
        self.equilibrium_tolerance = float(equilibrium_tolerance)
        self._set_matrices(sys)
        self.u_min = self._as_limit(u_min, "u_min")
        self.u_max = self._as_limit(u_max, "u_max")
        if np.any(self.u_min > self.u_max):
            raise ValueError("u_min must not be greater than u_max.")
        self.updateGain(method, **kwargs)
        self.resetState()

    def _set_matrices(self, sys: Any) -> None:
        A = np.asarray(sys.A, dtype=float)
        B = np.asarray(sys.Bu, dtype=float)
        Cm = np.asarray(sys.Cm, dtype=float)
        Co = np.asarray(sys.Co, dtype=float)
        dt = float(getattr(sys, "dt", 0.0))

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix.")
        nx = A.shape[0]
        if B.ndim != 2 or B.shape[0] != nx:
            raise ValueError("Bu must have shape (number of states, number of inputs).")
        if Cm.shape != (nx, nx) or np.linalg.matrix_rank(Cm) != nx:
            raise ValueError(
                "A disturbance-observer-free controller needs all states. "
                "Cm must therefore be a square, invertible full-state measurement map."
            )
        if Co.ndim != 2 or Co.shape[1] != nx:
            raise ValueError("Co must have one column per state.")

        self.matrices = _Matrices(A.copy(), B.copy(), Cm.copy(), Co.copy(), dt)
        self.A, self.Bu, self.Cm, self.Co, self.dt = (
            self.matrices.A,
            self.matrices.B,
            self.matrices.Cm,
            self.matrices.Co,
            self.matrices.dt,
        )
        self.nx, self.nu, self.no = nx, B.shape[1], Co.shape[0]

    def _as_limit(self, value: ArrayLike | None, name: str) -> np.ndarray:
        if value is None:
            fill = -np.inf if name == "u_min" else np.inf
            return np.full((self.nu, 1), fill)
        array = np.asarray(value, dtype=float)
        if array.size == 1:
            return np.full((self.nu, 1), float(array.reshape(-1)[0]))
        try:
            return array.reshape(self.nu, 1)
        except ValueError as exc:
            raise ValueError(f"{name} must be scalar or contain {self.nu} values.") from exc

    @staticmethod
    def _controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.hstack(
            [np.linalg.matrix_power(A, power) @ B for power in range(A.shape[0])]
        )

    @classmethod
    def _ackermann_siso(
        cls, A: np.ndarray, B: np.ndarray, poles: np.ndarray
    ) -> np.ndarray:
        """Place SISO poles, including repeated poles, using Ackermann's formula."""
        n = A.shape[0]
        controllability = cls._controllability_matrix(A, B)
        if np.linalg.matrix_rank(controllability) != n:
            raise ValueError("The system is not controllable.")

        coefficients = np.poly(poles)
        phi_A = np.linalg.matrix_power(A, n)
        for index in range(1, n + 1):
            phi_A += coefficients[index] * np.linalg.matrix_power(A, n - index)

        last_row = np.zeros((1, n))
        last_row[0, -1] = 1.0
        # last_row @ inv(controllability), written as a solve for stability.
        return np.linalg.solve(controllability.T, last_row.T).T @ phi_A

    def updateGain(self, method: str | None = None, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Recompute the feedback gain for the current system matrices."""
        if method is not None:
            self.method = method
        if kwargs:
            self.gain_kwargs = dict(kwargs)
        options = self.gain_kwargs

        controllability = self._controllability_matrix(self.A, self.Bu)
        if np.linalg.matrix_rank(controllability) != self.nx:
            raise ValueError("The nominal system is not controllable.")

        if self.method == "place_multiple_poles":
            wc = options.get("wc")
            if wc is None:
                raise ValueError("'wc' is required for place_multiple_poles.")
            desired_poles = np.full(self.nx, complex(wc))
            if self.nu != 1:
                raise ValueError("Repeated-pole placement is implemented for SISO systems only.")
            self.K = np.real_if_close(
                self._ackermann_siso(self.A, self.Bu, desired_poles)
            ).astype(float)

        elif self.method == "place_distinct_poles":
            desired_poles = np.asarray(options.get("poles_ctr", ()), dtype=complex)
            if desired_poles.size != self.nx:
                raise ValueError(f"'poles_ctr' must contain exactly {self.nx} poles.")
            self.K = np.real_if_close(
                place_poles(self.A, self.Bu, desired_poles).gain_matrix
            ).astype(float)

        elif self.method == "lqr":
            Q = options.get("Qc")
            R = options.get("Rc")
            if Q is None or R is None:
                raise ValueError("'Qc' and 'Rc' are required for lqr.")
            Q = np.asarray(Q, dtype=float)
            R = np.asarray(R, dtype=float)
            if Q.shape != (self.nx, self.nx) or R.shape != (self.nu, self.nu):
                raise ValueError("Qc and Rc have incompatible dimensions.")
            P = solve_continuous_are(self.A, self.Bu, Q, R)
            self.K = np.linalg.solve(R, self.Bu.T @ P)
            desired_poles = np.linalg.eigvals(self.A - self.Bu @ self.K)

        else:
            raise ValueError(
                "Unknown method. Use 'place_multiple_poles', "
                "'place_distinct_poles' or 'lqr'."
            )

        self.poles = np.asarray(desired_poles)
        self.closed_loop_poles = np.linalg.eigvals(self.A - self.Bu @ self.K)
        return self.K.copy(), self.closed_loop_poles.copy()

    def updateSysAndGain(
        self, sys: Any, method: str | None = None, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update a speed-scheduled model and recompute its gain."""
        old_nu = self.nu
        old_min, old_max = self.u_min.copy(), self.u_max.copy()
        self._set_matrices(sys)
        if self.nu != old_nu:
            raise ValueError("The number of control inputs cannot change during scheduling.")
        self.u_min, self.u_max = old_min, old_max
        return self.updateGain(method, **kwargs)

    def solve_equilibrium(self, reference: ArrayLike) -> EquilibriumPoint:
        """Solve and validate the steady state for a controlled-output reference.

        ``reference`` is an output reference with dimension ``no``; it is not
        a full state vector.  The returned state and feedforward control satisfy

            A @ x_eq + Bu @ u_eq = 0
            Co @ x_eq = reference.

        This distinction prevents a steering reference such as
        ``[delta_ref, 0, 0]`` from being treated as a valid steady state when
        the bicycle must lean to sustain that steering angle.
        """
        reference = np.asarray(reference, dtype=float).reshape(self.no, 1)
        matrix = np.block(
            [
                [self.A, self.Bu],
                [self.Co, np.zeros((self.no, self.nu))],
            ]
        )
        rhs = np.vstack((np.zeros((self.nx, 1)), reference))
        rank = np.linalg.matrix_rank(matrix)
        unknown_count = self.nx + self.nu
        if matrix.shape[0] == matrix.shape[1] and rank == unknown_count:
            solution = np.linalg.solve(matrix, rhs)
        else:
            solution, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        residual = np.linalg.norm(matrix @ solution - rhs)
        scale = max(1.0, np.linalg.norm(rhs))
        if residual > self.equilibrium_tolerance * scale:
            raise ValueError(
                "The requested controlled-output reference has no dynamically "
                "consistent nominal steady-state equilibrium."
            )
        state = solution[: self.nx]
        control = solution[self.nx :]
        return EquilibriumPoint(
            state=state.copy(),
            control=control.copy(),
            output=(self.Co @ state).copy(),
            residual_norm=float(residual),
            is_unique=rank == unknown_count,
        )

    def nominal_equilibrium(self, reference: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility wrapper returning ``(x_eq, u_eq)``."""
        equilibrium = self.solve_equilibrium(reference)
        return equilibrium.state, equilibrium.control

    def validate_state_reference(
        self,
        state_reference: ArrayLike,
        control_reference: ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate a full-state steady-state reference.

        If ``control_reference`` is omitted, the method finds the least-squares
        steady input and accepts it only when the dynamics residual is below
        ``equilibrium_tolerance``.  An inconsistent reference such as
        ``[delta_ref, 0, 0]`` is therefore rejected explicitly.
        """
        state = np.asarray(state_reference, dtype=float).reshape(self.nx, 1)
        if control_reference is None:
            control, _, _, _ = np.linalg.lstsq(
                self.Bu, -self.A @ state, rcond=None
            )
        else:
            control = np.asarray(control_reference, dtype=float).reshape(self.nu, 1)

        residual = self.A @ state + self.Bu @ control
        scale = max(
            1.0,
            np.linalg.norm(self.A @ state),
            np.linalg.norm(self.Bu @ control),
        )
        if np.linalg.norm(residual) > self.equilibrium_tolerance * scale:
            raise ValueError(
                "The full-state reference is not a steady state of the nominal "
                "dynamics. Use solve_equilibrium(output_reference) to obtain a "
                "dynamically consistent state and feedforward control."
            )
        return state.copy(), control.copy()

    def measured_state(self, measured_output: ArrayLike) -> np.ndarray:
        """Recover the state from the full-state measurement map Cm."""
        measured_output = np.asarray(measured_output, dtype=float).reshape(self.nx, 1)
        return np.linalg.solve(self.Cm, measured_output)

    def stepAndGetControl(
        self, yo_ref: ArrayLike, ym_cur: ArrayLike, t: float | None = None
    ) -> np.ndarray:
        """Return the nominal state-feedback command; ``t`` is API-compatible only."""
        del t
        state = self.measured_state(ym_cur)
        equilibrium = self.solve_equilibrium(yo_ref)
        x_equilibrium = equilibrium.state
        u_equilibrium = equilibrium.control
        unsaturated = u_equilibrium + self.K @ (x_equilibrium - state)
        control = np.clip(unsaturated, self.u_min, self.u_max)

        self.last_state = state
        self.last_x_equilibrium = x_equilibrium
        self.last_u_equilibrium = u_equilibrium
        self.last_unsaturated_control = unsaturated
        self.last_control = control
        return control.copy()

    def stepWithStateReference(
        self,
        state_reference: ArrayLike,
        measured_output: ArrayLike,
        t: float | None = None,
        control_reference: ArrayLike | None = None,
    ) -> np.ndarray:
        """Control toward an explicitly supplied, validated steady state."""
        del t
        state = self.measured_state(measured_output)
        x_equilibrium, u_equilibrium = self.validate_state_reference(
            state_reference, control_reference
        )
        unsaturated = u_equilibrium + self.K @ (x_equilibrium - state)
        control = np.clip(unsaturated, self.u_min, self.u_max)

        self.last_state = state
        self.last_x_equilibrium = x_equilibrium
        self.last_u_equilibrium = u_equilibrium
        self.last_unsaturated_control = unsaturated
        self.last_control = control
        return control.copy()

    def step(
        self, reference: ArrayLike, measured_state: ArrayLike, t: float | None = None
    ) -> np.ndarray:
        """Snake-case-friendly alias of :meth:`stepAndGetControl`."""
        return self.stepAndGetControl(reference, measured_state, t)

    def resetState(self, init_state: ArrayLike | None = None) -> None:
        """Reset diagnostic values; the controller has no dynamic observer state."""
        if init_state is None:
            state = np.zeros((self.nx, 1))
        else:
            state = np.asarray(init_state, dtype=float).reshape(self.nx, 1)
        self.last_state = state
        self.last_x_equilibrium = np.zeros((self.nx, 1))
        self.last_u_equilibrium = np.zeros((self.nu, 1))
        self.last_unsaturated_control = np.zeros((self.nu, 1))
        self.last_control = np.zeros((self.nu, 1))

    # Conventional aliases for new code while retaining notebook compatibility.
    update_system_and_gain = updateSysAndGain
    update_gain = updateGain
    step_with_state_reference = stepWithStateReference
    reset = resetState


NominalBikeController = NominalStateFeedbackController
