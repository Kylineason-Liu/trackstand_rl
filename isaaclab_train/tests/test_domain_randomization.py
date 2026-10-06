"""CLI unit tests plus real task-config tests when run in the Isaac Lab Python."""

import argparse
import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.domain_randomization import configure_domain_randomization
from scripts.rsl_rl.cli_args import add_rsl_rl_args

DR_TERMS = (
    "randomize_mass",
    "randomize_com",
    "randomize_material",
    "randomize_actuator_gains",
    "reset_imu_turnon_residuals",
)


class TestDomainRandomizationCLI(unittest.TestCase):
    def test_flag_is_opt_in(self):
        parser = argparse.ArgumentParser()
        add_rsl_rl_args(parser)
        self.assertFalse(parser.parse_args([]).dr)
        self.assertTrue(parser.parse_args(["--dr"]).dr)

    def test_removed_flag_is_rejected(self):
        parser = argparse.ArgumentParser()
        add_rsl_rl_args(parser)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parser.parse_args(["--no_dr"])
        self.assertEqual(error.exception.code, 2)

    def test_common_switch_calls_task_configuration(self):
        setter = Mock()
        cfg = SimpleNamespace(set_domain_randomization=setter)
        with contextlib.redirect_stdout(io.StringIO()):
            configure_domain_randomization(cfg, False)
            configure_domain_randomization(cfg, True)
        self.assertEqual([call.args[0] for call in setter.call_args_list], [False, True])

    def test_opt_in_cannot_silently_fail_on_unsupported_task(self):
        cfg = SimpleNamespace()
        configure_domain_randomization(cfg, False)
        with self.assertRaisesRegex(ValueError, "requires a task configuration"):
            configure_domain_randomization(cfg, True)


@unittest.skipUnless(importlib.util.find_spec("isaaclab"), "Run with Isaac Lab Python for task-config tests")
class TestTrackstandDomainRandomization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from isaaclab.app import AppLauncher

        cls.app = AppLauncher(headless=True).app
        cls.addClassCleanup(cls.app.close)
        module = importlib.import_module(
            "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand.trackstand_bike.trackstand_rl_env_cfg"
        )

        cls.cfg_type = module.TrackstandRLEnvCfg
        cls.full_events_type = module.TrackstandRLEventsCfg

    def assert_nominal(self, cfg):
        self.assertFalse(cfg.domain_randomization)
        for name in DR_TERMS:
            self.assertIsNone(getattr(cfg.events, name), name)
        self.assertEqual(cfg.events.initial_state.params["pose_range"]["roll"], (0.0, 0.0))
        self.assertEqual(cfg.events.initial_state.params["velocity_range"]["roll"], (0.0, 0.0))

    def assert_full_dr(self, cfg):
        self.assertTrue(cfg.domain_randomization)
        expected = self.full_events_type().to_dict()
        actual = cfg.events.to_dict()
        for name in DR_TERMS + ("initial_state",):
            self.assertEqual(actual[name], expected[name], name)

    def test_default_configuration_has_no_dr(self):
        cfg = self.cfg_type()
        self.assert_nominal(cfg)
        self.assertTrue(cfg.observations["policy"].enable_corruption)
        self.assertEqual(cfg.events.randomize_ground_friction.params["static_friction_range"], (1.0, 1.0))

    def test_constructor_can_opt_in(self):
        self.assert_full_dr(self.cfg_type(domain_randomization=True))

    def test_switch_is_reversible_and_idempotent_before_env_construction(self):
        cfg = self.cfg_type()
        for enabled in (True, True, False, False, True):
            configure_domain_randomization(cfg, enabled)
            if enabled:
                self.assert_full_dr(cfg)
            else:
                self.assert_nominal(cfg)

    def test_switch_keeps_unrelated_configuration_identical(self):
        cfg = self.cfg_type()
        before = cfg.to_dict()
        configure_domain_randomization(cfg, True)
        after = cfg.to_dict()
        for key in before.keys() - {"events", "domain_randomization"}:
            self.assertEqual(before[key], after[key], key)

    def test_configurations_do_not_share_mutable_dr_events(self):
        first = self.cfg_type(domain_randomization=True)
        second = self.cfg_type(domain_randomization=True)
        first.events.randomize_mass.params["mass_distribution_params"] = (0.99, 1.01)
        self.assertEqual(second.events.randomize_mass.params["mass_distribution_params"], (0.8, 1.2))
        self.assert_nominal(self.cfg_type())

    def test_serialized_configuration_records_the_effective_setting(self):
        cfg = self.cfg_type()
        self.assertFalse(cfg.to_dict()["domain_randomization"])
        configure_domain_randomization(cfg, True)
        self.assertTrue(cfg.to_dict()["domain_randomization"])


if __name__ == "__main__":
    unittest.main()
