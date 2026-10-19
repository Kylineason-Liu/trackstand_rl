"""Pure-survival configuration/CLI tests without launching Isaac Sim."""

import argparse
import ast
import contextlib
import copy
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluation_options import add_evaluation_args, configure_pure_survival
from scripts.rsl_rl.cli_args import add_rsl_rl_args


class TestPureSurvival(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(
            terminations=SimpleNamespace(
                time_out=SimpleNamespace(time_out=True, params={}),
                fallen=SimpleNamespace(time_out=False, params={"limit_angle": 0.5236}),
                yaw_deviation=object(),
                drifted=object(),
                wheel_net=object(),
                diverged=object(),
                unused=None,
            ),
            rewards={"survival": 5.0, "drift_gauss": 2.0},
            domain_randomization=True,
            episode_length_s=30.0,
        )

    def configure(self, enabled=True):
        with contextlib.redirect_stdout(io.StringIO()):
            configure_pure_survival(self.cfg, enabled)

    def test_cli_is_opt_in_and_independent_of_dr(self):
        parser = argparse.ArgumentParser()
        add_rsl_rl_args(parser)
        add_evaluation_args(parser)
        for args, pure, dr in (([], False, False), (["--dr"], False, True),
                               (["--pure_survival"], True, False),
                               (["--dr", "--pure_survival"], True, True)):
            with self.subTest(args=args):
                parsed = parser.parse_args(args)
                self.assertEqual(parsed.pure_survival, pure)
                self.assertEqual(parsed.dr, dr)

    def test_default_preserves_every_term_and_emits_nothing(self):
        before = vars(self.cfg.terminations).copy()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            configure_pure_survival(self.cfg, False)
        self.assertEqual(vars(self.cfg.terminations), before)
        self.assertEqual(output.getvalue(), "")

    def test_keeps_only_time_out_and_fallen_with_parameters_and_identity(self):
        timeout, fallen = self.cfg.terminations.time_out, self.cfg.terminations.fallen
        self.configure()
        terms = vars(self.cfg.terminations)
        self.assertEqual({name for name, term in terms.items() if term is not None}, {"time_out", "fallen"})
        self.assertIs(terms["time_out"], timeout)
        self.assertIs(terms["fallen"], fallen)
        self.assertTrue(timeout.time_out)
        self.assertFalse(fallen.time_out)
        self.assertEqual(fallen.params, {"limit_angle": 0.5236})

    def test_unrelated_configuration_is_unchanged(self):
        before = copy.deepcopy({k: v for k, v in vars(self.cfg).items() if k != "terminations"})
        self.configure()
        self.assertEqual({k: v for k, v in vars(self.cfg).items() if k != "terminations"}, before)

    def test_additional_termination_is_also_disabled(self):
        self.cfg.terminations.other_failure = object()
        self.configure()
        self.assertIsNone(self.cfg.terminations.other_failure)

    def test_dictionary_configuration_and_repeated_calls(self):
        self.cfg.terminations = vars(self.cfg.terminations).copy()
        self.configure()
        self.configure()
        self.assertEqual({n for n, v in self.cfg.terminations.items() if v is not None}, {"time_out", "fallen"})

    def test_missing_required_term_rejected_before_mutation(self):
        for missing in ("time_out", "fallen"):
            with self.subTest(missing=missing):
                original = getattr(self.cfg.terminations, missing)
                setattr(self.cfg.terminations, missing, None)
                before = vars(self.cfg.terminations).copy()
                with self.assertRaisesRegex(ValueError, "requires configured time_out and fallen"):
                    self.configure()
                self.assertEqual(vars(self.cfg.terminations), before)
                setattr(self.cfg.terminations, missing, original)

    def test_unsupported_task_is_allowed_only_without_flag(self):
        configure_pure_survival(SimpleNamespace(), False)
        with self.assertRaises(ValueError):
            configure_pure_survival(SimpleNamespace(), True)

    def test_warning_explicitly_mentions_disabled_divergence_check(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            configure_pure_survival(self.cfg, True)
        self.assertIn("[WARN] pure_survival disables the diverged", output.getvalue())

    def test_evaluator_wires_flag_before_parse_and_override_before_environment(self):
        script = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/play_eval.py"
        tree = ast.parse(script.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        register = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "add_evaluation_args")
        parse = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "parse_known_args")
        configure = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "configure_pure_survival")
        make = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "make")
        self.assertLess(register.lineno, parse.lineno)
        self.assertLess(configure.lineno, make.lineno)
        self.assertEqual(ast.unparse(configure.args[1]), "args_cli.pure_survival")


if __name__ == "__main__":
    unittest.main()
