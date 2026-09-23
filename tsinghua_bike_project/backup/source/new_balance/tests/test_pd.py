"""Tests for the inner PD controller of the balance node.

The controller module carries two names: new_main_balance.py in this
repository (where main_balance.py is the archived previous version), and
main_balance.py once deployed to the Pi.  This module locates whichever is
present — see _CANDIDATES below.

It imports rclpy / can / scipy, which are not importable off the robot, so
the PD class is lifted out by AST and exec'd on its own — it is pure Python
with no dependencies, so this exercises the shipped source rather than a copy
that could drift from it.

The behaviour under test is derivative-on-measurement: the outer loop
refreshes the setpoint at outer_hz while the inner PD runs at inner_hz, so
differentiating the *error* turned every setpoint change into a one-tick
current spike unrelated to anything the plant did.

    python -m unittest discover -s tests        # from source/new_balance/
"""
import ast
import unittest
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "new_balance"

# The controller module is named new_main_balance.py in this repository, but
# is deployed to the Pi as main_balance.py.  Prefer the former where both
# exist, so the repo copy tests the current source rather than the archive.
_CANDIDATES = ("new_main_balance.py", "main_balance.py")
_PATH = next((p for p in (_PKG / name for name in _CANDIDATES) if p.exists()), None)
if _PATH is None:
    raise FileNotFoundError(
        f"no controller module found in {_PKG} (looked for {_CANDIDATES})")

_cls = next((n for n in ast.walk(ast.parse(_PATH.read_text(encoding="utf-8")))
             if isinstance(n, ast.ClassDef) and n.name == "PD"), None)
if _cls is None:
    raise AssertionError(f"no PD class in {_PATH.name}")
_ns: dict = {}
exec(compile(ast.Module(body=[_cls], type_ignores=[]), "<PD>", "exec"), _ns)
PD = _ns["PD"]

KP, KD, DT = 0.0125, 0.0001, 0.004          # params.yaml defaults, inner_hz=250
CLAMP = 1.5


def make():
    return PD(kp=KP, kd=KD, output_min=-CLAMP, output_max=CLAMP)


def primed():
    """A PD past its post-reset tick, so the derivative is live."""
    pd = make()
    pd.step(0.0, 0.0, DT)
    return pd


class TestNoDerivativeKick(unittest.TestCase):

    def test_setpoint_step_produces_no_derivative(self):
        """Plant has not moved, so the output must be pure proportional."""
        pd = primed()
        step = 0.5
        out = pd.step(step, 0.0, DT)
        self.assertAlmostEqual(out, KP * step, places=12)

        # Scale of what derivative-on-error would have injected on this tick.
        # The ratio is kd/(kp·dt) regardless of step size — it grows as the
        # inner loop runs faster: 4× at inner_hz=500, 2× at 250.
        kick_over_p = (KD * step / DT) / (KP * step)
        self.assertAlmostEqual(kick_over_p, KD / (KP * DT), places=12)
        self.assertGreater(kick_over_p, 1.0,
                           "the avoided kick should exceed the P term")

    def test_repeated_setpoint_steps_never_spike(self):
        pd = primed()
        for setpoint in (0.5, 1.5, -2.0, 3.0, 0.0):
            self.assertAlmostEqual(pd.step(setpoint, 0.0, DT),
                                   KP * setpoint, places=12)

    def test_derivative_opposes_plant_motion(self):
        """Constant setpoint, measurement rising → D must damp, not assist."""
        pd = primed()
        pd.step(0.0, 0.0, DT)
        out = pd.step(0.0, 0.1, DT)
        d = -KD * 0.1 / DT
        self.assertLess(d, 0.0)
        self.assertAlmostEqual(out, KP * -0.1 + d, places=12)

    def test_matches_derivative_on_error_when_setpoint_is_constant(self):
        """Equivalence claimed in PD.step's docstring, checked numerically.

        This is what lets the existing steady-state gain tuning carry over:
        only the transient at a setpoint change differs.
        """
        def derivative_on_error(errors):
            prev, out = 0.0, []
            for i, e in enumerate(errors):
                d = 0.0 if i == 0 else KD * (e - prev) / DT
                prev = e
                out.append(KP * e + d)
            return out

        setpoint = 1.0
        measurements = [0.0, 0.05, 0.11, 0.2, 0.31, 0.4]

        pd = make()
        got = [pd.step(setpoint, m, DT) for m in measurements]
        want = derivative_on_error([setpoint - m for m in measurements])
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=12)


class TestReset(unittest.TestCase):
    """reset() runs at arming, fault latch and fault recovery — the three
    moments where a spurious current impulse is least acceptable."""

    def test_first_step_of_a_new_controller_has_no_derivative(self):
        self.assertAlmostEqual(make().step(0.0, 4.0, DT), KP * -4.0, places=12)

    def test_first_step_after_reset_has_no_derivative(self):
        pd = primed()
        pd.step(0.0, 4.0, DT)
        pd.step(0.0, 5.0, DT)
        pd.reset()
        self.assertAlmostEqual(pd.step(0.0, 9.0, DT), KP * -9.0, places=12)

    def test_derivative_resumes_on_the_following_tick(self):
        pd = make()
        pd.step(0.0, 1.0, DT)                   # primes at meas = 1.0
        self.assertAlmostEqual(pd.step(0.0, 1.2, DT),
                               KP * -1.2 + (-KD * 0.2 / DT), places=12)

    def test_no_derivative_on_error_state_remains(self):
        self.assertFalse(hasattr(make(), "_prev_error"))


class TestOutputHandling(unittest.TestCase):

    def test_output_is_clamped_both_ways(self):
        pd = primed()
        self.assertEqual(pd.step(1e6, 0.0, DT), CLAMP)
        self.assertEqual(pd.step(-1e6, 0.0, DT), -CLAMP)
        self.assertEqual(pd.last_output, -CLAMP)

    def test_zero_dt_does_not_divide(self):
        pd = primed()
        self.assertAlmostEqual(pd.step(1.0, 0.5, 0.0), KP * 0.5, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
