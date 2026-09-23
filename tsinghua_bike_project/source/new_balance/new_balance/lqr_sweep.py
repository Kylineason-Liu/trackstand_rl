#!/usr/bin/env python3
"""
Tabulate the LQR gain, and the closed-loop poles it lands on, across speed.

One job: take the LQR weights Q and R plus a speed range, and print what
``main_balance`` will actually feed the control law at every speed in that
range.  Nothing is written and no node is touched.

Why the poles are in the table too
----------------------------------
LQR fixes the *cost*, not the closed-loop dynamics.  Q and R stay put while
both K and the resulting poles move with speed, because a1 grows with v and
a2 with v^2.  That is the whole difference from pole placement, which fixes
the poles and lets K go wherever it must -- to infinity, at a speed where
the model is uncontrollable.  See k_to_poles.py for that direction.

So a gain tuned at 1 m/s does not describe the bike at 2.7 m/s, and the
pole columns are how you see by how much.  Watch the pair of real poles
near -sqrt(a4) collide and split into a conjugate pair as speed rises.

Where the weights come from
---------------------------
``main_balance`` reads lqr_Q_diag / lqr_R_diag from a ROS parameter, whose
value is the declare_parameter() default unless a params file overrides it.
Those two differ, so the tool can read either, and always prints which one
it used:

    --Q / --R          given explicitly, highest priority
    --source-file      declare_parameter() defaults in main_balance.py
    --params-file      a ROS params yaml (also supplies bike_*)

Usage
-----
    # the declare_parameter() defaults compiled into the node
    python3 lqr_sweep.py --source-file main_balance.py

    # what you get when launching with --params-file
    python3 lqr_sweep.py --params-file ../config/params.yaml

    # weights given by hand, bike_* from the params file
    python3 lqr_sweep.py --Q 4 100 2 --R 10 --params-file ../config/params.yaml

    # a different speed range
    python3 lqr_sweep.py --Q 4 100 2 --R 10 --speeds 0.5 3.0 0.25

Runnable as a script or as ``python3 -m new_balance.lqr_sweep``.
"""

from __future__ import annotations

import argparse
import ast

import numpy as np

try:  # installed as part of the package
    from new_balance.k_to_poles import (BikeParams, lqr_gain, model_coeffs,
                                        params_from_yaml, state_matrices)
except ImportError:  # run as a plain script from this directory
    from k_to_poles import (BikeParams, lqr_gain, model_coeffs,
                            params_from_yaml, state_matrices)


_SOURCE_KEYS = {
    "bike_mass": "mass",
    "bike_wheelbase": "wheelbase",
    "bike_rear_to_com": "rear_to_com",
    "bike_com_height": "com_height",
    "bike_trail": "trail",
    "bike_steering_axis_angle": "steer_axis_angle",
    "bike_roll_inertia": "roll_inertia",
}


def declare_parameter_defaults(path: str) -> dict:
    """Every ``self.declare_parameter("name", <literal>)`` in a node source.

    Parsed rather than imported: main_balance pulls in rclpy and python-can,
    which are not importable off the robot, and importing it would build a
    ROS node as a side effect.
    """
    tree = ast.parse(open(path, "r", encoding="utf-8").read())
    out: dict = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "declare_parameter"
                and len(node.args) == 2):
            try:
                out[ast.literal_eval(node.args[0])] = ast.literal_eval(node.args[1])
            except ValueError:
                pass  # a computed default (e.g. rc_max_rpm); not needed here
    return out


def params_from_source(path: str, base: BikeParams) -> tuple[BikeParams, float, dict]:
    """Overlay bike_* from a node source file; also return its raw defaults."""
    from dataclasses import replace

    defaults = declare_parameter_defaults(path)
    updates = {field: float(defaults[key])
               for key, field in _SOURCE_KEYS.items() if key in defaults}
    if updates.get("roll_inertia", 1.0) < 0.0:
        updates["roll_inertia"] = None
    params = replace(base, **updates) if updates else base
    return params, float(defaults.get("min_scheduling_speed", 1.0)), defaults


def lqr_weights_from_yaml(path: str) -> tuple[list | None, list | None]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    for key, value in doc.items():
        if (isinstance(value, dict) and "ros__parameters" in value
                and key.endswith("balance_node")):
            node = value["ros__parameters"] or {}
            return node.get("lqr_Q_diag"), node.get("lqr_R_diag")
    return None, None


_POLE_W = 15


def _format_pole(z: complex) -> str:
    """Fixed-width, so the column does not shift when a pair goes real."""
    z = complex(z)
    text = (f"{z.real:.3f}" if abs(z.imag) < 1e-9
            else f"{z.real:.2f}{z.imag:+.2f}j")
    return f"{text:>{_POLE_W}}"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Tabulate the LQR gain and its closed-loop poles "
                    "over a speed range.")
    src = ap.add_argument_group("the weights (one of these)")
    src.add_argument("--Q", nargs=3, type=float, metavar=("Q1", "Q2", "Q3"),
                     help="LQR state weights diag(Q_delta, Q_phi, Q_phidot)")
    src.add_argument("--R", nargs=1, type=float, metavar="R1",
                     help="LQR control weight")
    src.add_argument("--source-file",
                     help="node source to read declare_parameter() defaults from")
    src.add_argument("--params-file",
                     help="ROS params yaml to read lqr_* and bike_* from")
    ap.add_argument("--speeds", nargs=3, type=float, default=[1.0, 2.7, 0.1],
                    metavar=("MIN", "MAX", "STEP"),
                    help="speed range to sweep (default: 1.0 2.7 0.1)")
    ap.add_argument("--min-speed", type=float, default=None,
                    help="min_scheduling_speed; below it the schedule clamps")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    params, min_speed = BikeParams(), 1.0
    Q = R = None
    origin = "built-in defaults"

    if args.source_file:
        params, min_speed, defaults = params_from_source(args.source_file, params)
        Q, R = defaults.get("lqr_Q_diag"), defaults.get("lqr_R_diag")
        origin = f"declare_parameter() defaults in {args.source_file}"
    if args.params_file:
        params, min_speed = params_from_yaml(args.params_file, params)
        yq, yr = lqr_weights_from_yaml(args.params_file)
        if yq is not None:
            Q, R = yq, yr
            origin = f"lqr_* in {args.params_file}"
    if args.Q is not None and args.R is not None:
        Q, R = args.Q, args.R
        origin = "given on the command line"
    if Q is None or R is None:
        ap.error("give --Q with --R, or --source-file, or --params-file")
    if args.min_speed is not None:
        min_speed = args.min_speed

    lo, hi, step = args.speeds
    count = max(int(round((hi - lo) / step)) + 1, 1)
    speeds = lo + step * np.arange(count)

    print()
    print("=" * 78)
    print(f"  LQR gain vs speed   ({origin})")
    print("=" * 78)
    print(f"  Q = diag({[float(q) for q in Q]})")
    print(f"  R = diag({[float(r) for r in R]})")
    print(f"  bike: mass {params.mass} wheelbase {params.wheelbase} "
          f"rear_to_com {params.rear_to_com} com_height {params.com_height}")
    print(f"        trail {params.trail} steer_axis {params.steer_axis_angle} "
          f"roll_inertia {params.roll_inertia}")
    print(f"  min_scheduling_speed {min_speed:g} m/s   "
          f"sweep {lo:g}-{hi:g} step {step:g}")
    print()
    print(f"  {'v [m/s]':>7}   {'K_delta':>9} {'K_phi':>9} {'K_phidot':>9}   "
          f"{'closed-loop poles':^{3 * _POLE_W}}  {'maxRe':>8}")
    print("  " + "-" * (7 + 3 + 30 + 3 + 3 * _POLE_W + 2 + 8))

    worst, worst_v, failures = -np.inf, None, []
    for speed in speeds:
        speed = float(speed)
        try:
            gain = lqr_gain(params, speed, Q, R, min_speed)
            A, Bu = state_matrices(params, speed, min_speed)
            poles = np.sort_complex(np.linalg.eigvals(A - Bu @ np.atleast_2d(gain)))
        except Exception as exc:  # non-stabilizable: the ARE has no finite solution
            failures.append(speed)
            print(f"  {speed:7.2f}   *** {type(exc).__name__}: {exc}")
            continue
        max_re = float(poles.real.max())
        if max_re > worst:
            worst, worst_v = max_re, speed
        print(f"  {speed:7.2f}   {gain[0]:9.4f} {gain[1]:9.4f} {gain[2]:9.4f}   "
              f"{_format_pole(poles[0])}{_format_pole(poles[1])}"
              f"{_format_pole(poles[2])}  {max_re:8.3f}")

    print()
    if failures:
        print(f"  !! no finite ARE solution at {len(failures)} speed(s): "
              f"{', '.join(f'{s:g}' for s in failures)}")
        print(f"     the uncontrollable mode there is unstable, so the plant "
              f"is not stabilizable.")
    if worst_v is not None:
        verdict = "all stable" if worst < 0 else "UNSTABLE"
        print(f"  worst closed-loop maxRe over the sweep: {worst:+.4f} "
              f"at v = {worst_v:g} m/s   -> {verdict}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
