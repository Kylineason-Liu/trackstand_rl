#!/usr/bin/env python3
"""
Solve the exact closed-loop poles of a state-feedback gain for the 3-state bicycle.

One job: take a gain K (or the LQR weights that produce one) plus a forward
speed, and print the three exact poles.  Nothing else -- no approximation, no
parameter files written, no node touched.  Read the poles off the output and
put them wherever you need them.

Why poles rather than K
-----------------------
A gain is only valid at the speed it was designed for: a1 grows with v and a2
with v^2, so the same K lands on different poles at every speed.  Poles are the
speed-independent statement of design intent, which is what makes them the right
thing to carry over into gain scheduling.

Method
------
With x = [steer_angle, roll_angle, roll_rate] and u = steer_rate:

    A = [[0, 0, 0], [0, 0, 1], [a2, a4, 0]]        Bu = [[1], [0], [a1]]

    a1 = m a h v cos(lambda) / (b I)
    a2 = (m v^2 h - m a c g) cos(lambda) / (b I)
    a4 = m g h / I

the characteristic polynomial of (A - Bu K) is, exactly:

    s^3 + (k1 + a1 k3) s^2 + (a1 k2 + a2 k3 - a4) s + (a2 k2 - a4 k1)

(the k1 k3 cross terms cancel in the s coefficient).  The poles are its roots,
so this is a closed-form solve -- np.roots, no search.  Every run cross-checks
the roots against eigvals(A - Bu K) and inverts them back to K, so a silent
formula error cannot slip through.

The triple is often one real pole plus a complex conjugate pair; that is
reported exactly as it comes out.

Under the poles comes a table of K over 1.0-2.7 m/s with those poles held
fixed -- what the node's gain scheduling will actually feed the control law
at every speed, since a1 grows with v and a2 with v^2.  The gain you typed
in is one row of it.  See _print_gain_sweep and the _SWEEP_* constants.

Usage
-----
    # gain given directly (speed defaults to 1.0 m/s)
    python3 k_to_poles.py --K 7.5824 27.8073 3.3528

    # derive the gain from LQR weights first, then solve its poles
    python3 k_to_poles.py --Q 4 100 2 --R 10

    # other speed, and bike parameters read from the params file in use
    python3 k_to_poles.py --K 7.6 27.8 3.4 --speed 1.5
    python3 k_to_poles.py --Q 4 100 2 --R 10 --params-file ../config/params.yaml

Bike parameters default to the declare_parameter() values in main_balance.py.
Those differ from config/params.yaml (mass 2.44 vs 2.2370, rear_to_com 0.1225
vs 0.1350, trail 0.0135 vs 0.0100), which moves the poles -- so pass
--params-file whenever you launch with one.

Runnable as a script or as ``python3 -m new_balance.k_to_poles``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Bike parameters
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BikeParams:
    """Defaults mirror declare_parameter() in main_balance.py."""

    mass: float = 2.44                   # bike_mass                 [kg]
    wheelbase: float = 0.2900            # bike_wheelbase            [m]
    rear_to_com: float = 0.1225          # bike_rear_to_com          [m]
    com_height: float = 0.1400           # bike_com_height           [m]
    trail: float = 0.0135                # bike_trail                [m]
    steer_axis_angle: float = 0.1745     # bike_steering_axis_angle  [rad]
    roll_inertia: float | None = 0.0438  # bike_roll_inertia         [kg m^2]
    gravity: float = 9.81

    @property
    def effective_roll_inertia(self) -> float:
        """Same fallback as BicycleParameters: 4/3 m h^2 when unset."""
        if self.roll_inertia is not None and self.roll_inertia >= 0.0:
            return float(self.roll_inertia)
        return 4.0 / 3.0 * self.mass * self.com_height**2


# ══════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════


def model_coeffs(p: BikeParams, speed: float,
                 min_speed: float = 1.0) -> tuple[float, float, float]:
    """Return (a1, a2, a4) at ``speed``, clamped like the node's scheduling."""
    v = max(abs(float(speed)), abs(float(min_speed)))
    inertia = p.effective_roll_inertia
    cos_lambda = math.cos(p.steer_axis_angle)
    denom = p.wheelbase * inertia

    a1 = p.mass * p.rear_to_com * p.com_height * v * cos_lambda / denom
    a2 = ((p.mass * v**2 * p.com_height
           - p.mass * p.rear_to_com * p.trail * p.gravity) * cos_lambda / denom)
    a4 = p.mass * p.gravity * p.com_height / inertia
    return a1, a2, a4


def state_matrices(p: BikeParams, speed: float,
                   min_speed: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    a1, a2, a4 = model_coeffs(p, speed, min_speed)
    A = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [a2, a4, 0.0]])
    Bu = np.array([[1.0], [0.0], [a1]])
    return A, Bu


# ══════════════════════════════════════════════════════════════════════
# K -> poles, and back again for verification
# ══════════════════════════════════════════════════════════════════════


def poles_from_gain(K, p: BikeParams, speed: float,
                    min_speed: float = 1.0) -> np.ndarray:
    """K -> the three exact closed-loop poles, as roots of the analytic cubic."""
    a1, a2, a4 = model_coeffs(p, speed, min_speed)
    k1, k2, k3 = (float(k) for k in np.ravel(K))
    coeffs = [1.0,
              k1 + a1 * k3,
              a1 * k2 + a2 * k3 - a4,
              a2 * k2 - a4 * k1]
    return np.sort_complex(np.roots(coeffs))


def gain_from_poles(poles, p: BikeParams, speed: float,
                    min_speed: float = 1.0) -> np.ndarray:
    """Poles -> K, the 3x3 linear solve on the matched coefficients."""
    poles = np.asarray(poles, dtype=complex).ravel()
    if poles.size != 3:
        raise ValueError("exactly 3 poles are required")

    alpha = np.poly(poles)                    # [1, alpha2, alpha1, alpha0]
    if np.max(np.abs(alpha.imag)) > 1e-9:
        raise ValueError("the pole set is not closed under conjugation")
    _, alpha2, alpha1, alpha0 = alpha.real

    a1, a2, a4 = model_coeffs(p, speed, min_speed)
    M = np.array([[1.0, 0.0, a1],
                  [0.0, a1, a2],
                  [-a4, a2, 0.0]])
    return np.linalg.solve(M, np.array([alpha2, alpha1 + a4, alpha0]))


def lqr_gain(p: BikeParams, speed: float, Q_diag, R_diag,
             min_speed: float = 1.0) -> np.ndarray:
    """Continuous-time LQR gain, matching the node's lqr branch."""
    from scipy.linalg import solve_continuous_are

    A, Bu = state_matrices(p, speed, min_speed)
    Q = np.diag(np.asarray(Q_diag, dtype=float))
    R = np.diag(np.asarray(R_diag, dtype=float))
    P = solve_continuous_are(A, Bu, Q, R)
    return np.linalg.solve(R, Bu.T @ P).ravel()


# ══════════════════════════════════════════════════════════════════════
# Params file (optional) and CLI
# ══════════════════════════════════════════════════════════════════════

_YAML_KEYS = {
    "bike_mass": "mass",
    "bike_wheelbase": "wheelbase",
    "bike_rear_to_com": "rear_to_com",
    "bike_com_height": "com_height",
    "bike_trail": "trail",
    "bike_steering_axis_angle": "steer_axis_angle",
    "bike_roll_inertia": "roll_inertia",
}


def params_from_yaml(path: str, base: BikeParams) -> tuple[BikeParams, float]:
    """Overlay bike_* and min_scheduling_speed from a ROS params file."""
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}

    node: dict = {}
    for key, value in doc.items():
        if isinstance(value, dict) and "ros__parameters" in value:
            if key.endswith("balance_node"):
                node = value["ros__parameters"] or {}
                break

    updates = {field: float(node[key]) for key, field in _YAML_KEYS.items()
               if key in node}
    if updates.get("roll_inertia", 1.0) < 0.0:
        updates["roll_inertia"] = None
    params = replace(base, **updates) if updates else base
    return params, float(node.get("min_scheduling_speed", 1.0))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Solve the exact closed-loop poles of a gain K "
                    "for the 3-state bicycle model.")
    src = ap.add_argument_group("the gain (give --K, or --Q with --R)")
    src.add_argument("--K", nargs=3, type=float, metavar=("K1", "K2", "K3"),
                     help="gain [k1 k2 k3] at --speed")
    src.add_argument("--Q", nargs=3, type=float, metavar=("Q1", "Q2", "Q3"),
                     help="LQR state weights; the resulting gain is solved")
    src.add_argument("--R", nargs=1, type=float, metavar="R1",
                     help="LQR control weight")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="speed the gain belongs to, m/s (default: 1.0)")
    ap.add_argument("--min-speed", type=float, default=1.0,
                    help="min_scheduling_speed, m/s (default: 1.0)")
    ap.add_argument("--params-file",
                    help="params.yaml to read bike_* values from")
    return ap


def _format_pole(z: complex) -> str:
    if abs(z.imag) > 1e-9:
        sign = "+" if z.imag >= 0 else "-"
        return f"{z.real:+.6f} {sign} {abs(z.imag):.6f}j"
    return f"{z.real:+.6f}"


# Speed sweep printed under the poles: the operating range of the bike.
_SWEEP_MIN, _SWEEP_MAX, _SWEEP_STEP = 1.0, 2.7, 0.1


def _controllability_det(p: BikeParams, speed: float, min_speed: float) -> float:
    """det[Bu, A Bu, A^2 Bu].  Works out to a1^2 a4 - a2^2 for this model."""
    a1, a2, a4 = model_coeffs(p, speed, min_speed)
    return a1 * a1 * a4 - a2 * a2


def _singular_speeds(p: BikeParams, lo: float, hi: float,
                     min_speed: float, samples: int = 4001) -> list[float]:
    """Speeds in [lo, hi] where the model loses controllability.

    Found by scanning for a sign change in the controllability determinant
    and bisecting, rather than by solving the quadratic in closed form, so
    it stays correct if the model or the bike parameters ever change.
    """
    grid = np.linspace(lo, hi, samples)
    det = [_controllability_det(p, float(s), min_speed) for s in grid]
    found = []
    for i in range(samples - 1):
        if det[i] == 0.0:
            found.append(float(grid[i]))
        elif det[i] * det[i + 1] < 0.0:
            a, b = float(grid[i]), float(grid[i + 1])
            fa = det[i]
            for _ in range(80):
                mid = 0.5 * (a + b)
                fm = _controllability_det(p, mid, min_speed)
                if fa * fm <= 0.0:
                    b = mid
                else:
                    a, fa = mid, fm
            found.append(0.5 * (a + b))
    return found


def _print_gain_sweep(poles, p: BikeParams, min_speed: float,
                      design_speed: float) -> None:
    """Print K over the speed range for a fixed pole set.

    This is exactly what the node's gain scheduling does: poles_ctr is held
    fixed while place_distinct_poles re-solves K every tick from the
    rescheduled A, Bu.  a1 grows with v and a2 with v^2, so one pole set
    asks for a very different -- and not necessarily monotonic -- gain at
    each speed.  The row the gain was designed at is only one line of this
    table; the bike rides the whole column.
    """
    count = int(round((_SWEEP_MAX - _SWEEP_MIN) / _SWEEP_STEP)) + 1
    speeds = _SWEEP_MIN + _SWEEP_STEP * np.arange(count)

    print(f"  K vs speed, these poles held fixed"
          f"   (min_scheduling_speed {min_speed:g} m/s)")
    print(f"      {'v [m/s]':>7}   {'K_delta':>9} {'K_phi':>9} {'K_phidot':>9}")
    for speed in speeds:
        gain = gain_from_poles(poles, p, float(speed), min_speed)
        mark = ("  <- design point"
                if abs(speed - design_speed) < _SWEEP_STEP / 2.0 else "")
        print(f"      {speed:7.1f}   "
              f"{gain[0]:9.4f} {gain[1]:9.4f} {gain[2]:9.4f}{mark}")

    # A zero of the controllability determinant is a pole of the gain
    # schedule.  The grid above steps straight over it, so the printed rows
    # understate the excursion by orders of magnitude -- say so explicitly.
    for speed in _singular_speeds(p, _SWEEP_MIN, _SWEEP_MAX, min_speed):
        gain = gain_from_poles(poles, p, speed, min_speed)
        print()
        print(f"  !! UNCONTROLLABLE at v = {speed:.4f} m/s: "
              f"det[Bu, A Bu, A^2 Bu] = 0 there.")
        print(f"     No finite gain places these poles at that speed -- K runs "
              f"to infinity")
        print(f"     and flips sign across it (here {gain[0]:.4g}, {gain[1]:.4g}, "
              f"{gain[2]:.4g}).  The grid")
        print(f"     above steps over the pole, so those rows look far tamer "
              f"than it is.")
        print(f"     place_poles does not warn: matrix_rank still reports 3.  "
              f"LQR is immune")
        print(f"     -- solve_continuous_are only needs stabilizability, not "
              f"controllability.")


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    params = BikeParams()
    min_speed = args.min_speed
    if args.params_file:
        params, yaml_min = params_from_yaml(args.params_file, params)
        if args.min_speed == 1.0:
            min_speed = yaml_min

    if args.K is not None:
        K = np.asarray(args.K, dtype=float)
        origin = "given"
    elif args.Q is not None and args.R is not None:
        K = lqr_gain(params, args.speed, args.Q, args.R, min_speed)
        origin = f"LQR Q={args.Q} R={args.R}"
    else:
        ap.error("give --K, or --Q together with --R")

    v = args.speed
    a1, a2, a4 = model_coeffs(params, v, min_speed)
    poles = poles_from_gain(K, params, v, min_speed)

    # Guard the analytic cubic both ways before reporting anything.
    A, Bu = state_matrices(params, v, min_speed)
    eig_err = float(np.max(np.abs(
        poles - np.sort_complex(np.linalg.eigvals(A - Bu @ np.atleast_2d(K))))))
    trip_err = float(np.max(np.abs(
        gain_from_poles(poles, params, v, min_speed) - np.ravel(K))))
    if max(eig_err, trip_err) > 1e-6:
        raise AssertionError(
            f"verification failed: eig {eig_err:.2e}, round trip {trip_err:.2e}")

    print()
    print("=" * 70)
    print(f"  K at v = {v} m/s  ({origin})")
    print("=" * 70)
    print(f"  K     = [{K[0]:.4f}, {K[1]:.4f}, {K[2]:.4f}]"
          "     x = [steer_angle, roll_angle, roll_rate]")
    print(f"  model = a1 {a1:.4f}   a2 {a2:.4f}   a4 {a4:.4f}")
    print()
    print("  poles")
    for pole in poles:
        print(f"      {_format_pole(complex(pole))}")
    print()
    listed = ", ".join(
        f"{z.real:.6f}" if abs(z.imag) <= 1e-9 else f"{z.real:.6f}{z.imag:+.6f}j"
        for z in (complex(p) for p in poles))
    print(f"  list  [{listed}]")
    worst = float(np.max(poles.real))
    print(f"  maxRe {worst:+.6f}"
          f"   {'UNSTABLE at this speed' if worst > 0 else '(stable)'}")
    print(f"  check cubic vs eigvals {eig_err:.1e}   "
          f"poles -> K round trip {trip_err:.1e}")
    print()
    _print_gain_sweep(poles, params, min_speed, v)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
