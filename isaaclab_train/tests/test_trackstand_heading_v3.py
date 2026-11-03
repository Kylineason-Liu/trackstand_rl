"""V3 = v1 with yaw_deviation termination replaced by the yaw_penalty heading penalty."""

import importlib
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TASK_PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"


@unittest.skipUnless(importlib.util.find_spec("isaaclab"), "Run with Isaac Lab Python")
class TestHeadingRewardV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gymnasium as gym
        import torch

        importlib.import_module(TASK_PACKAGE)
        cls.gym, cls.torch = gym, torch
        cls.rewards = importlib.import_module(f"{TASK_PACKAGE}.mdp.stationary.rewards")

    def spec(self, version):
        return self.gym.spec(f"Trackstand-Balance-v{version}")

    def cfg(self, version):
        return self.spec(version).kwargs["env_cfg_entry_point"]()

    @staticmethod
    def terms(group):
        return {name: value for name, value in vars(group).items() if not name.startswith("_")}

    def test_only_yaw_termination_is_removed(self):
        v1, v3 = self.terms(self.cfg(1).terminations), self.terms(self.cfg(3).terminations)
        self.assertIsNotNone(v1["yaw_deviation"])
        self.assertIsNone(v3["yaw_deviation"])
        self.assertEqual(set(v1), set(v3))
        for name in ("time_out", "fallen", "diverged", "drifted", "wheel_net"):
            with self.subTest(term=name):
                self.assertIs(v3[name].func, v1[name].func)
                self.assertEqual(v3[name].params, v1[name].params)

    def test_v1_rewards_kept_and_yaw_penalty_added(self):
        v1, v3 = self.terms(self.cfg(1).rewards), self.terms(self.cfg(3).rewards)
        self.assertEqual(set(v3) - set(v1), {"yaw_penalty"})
        for name in v1:
            with self.subTest(term=name):
                self.assertIs(v3[name].func, v1[name].func)
                self.assertEqual(v3[name].weight, v1[name].weight)
                self.assertEqual(v3[name].params, v1[name].params)
        yaw = v3["yaw_penalty"]
        self.assertIs(yaw.func, self.rewards.balance_yaw_excess_penalty)
        self.assertEqual(yaw.weight, 2.0)
        self.assertEqual(yaw.params["band"], 0.1)
        self.assertAlmostEqual(yaw.params["tail_slope"], 1.0 / (math.pi - 0.1))

    def test_contract_plant_and_env_class_match_v1(self):
        v1, v3 = self.cfg(1), self.cfg(3)
        self.assertEqual(v3.actions.to_dict(), v1.actions.to_dict())
        self.assertEqual(v3.observations["policy"].to_dict(), v1.observations["policy"].to_dict())
        self.assertEqual(v3.scene.robot.to_dict(), v1.scene.robot.to_dict())
        self.assertEqual(v3.events.to_dict(), v1.events.to_dict())
        timing = ("decimation", "episode_length_s")
        self.assertEqual(v3.sim.dt, v1.sim.dt)
        self.assertEqual([getattr(v3, k) for k in timing], [getattr(v1, k) for k in timing])
        self.assertEqual(self.spec(3).entry_point, self.spec(1).entry_point)

    def test_runner_matches_v1_except_log_dir(self):
        def runner(version):
            module, name = self.spec(version).kwargs["rsl_rl_cfg_entry_point"].split(":")
            return getattr(importlib.import_module(module), name)().to_dict()

        v1, v3 = runner(1), runner(3)
        self.assertEqual(v3.pop("experiment_name"), "trackstand_rl_v3_rsl")
        v1.pop("experiment_name")
        self.assertEqual(v3, v1)

    def test_yaw_penalty_values_gradient_and_wrap(self):
        torch = self.torch
        term = self.cfg(3).rewards.yaw_penalty
        rel = torch.tensor([0.0, 0.1, -math.radians(30), math.radians(90), math.pi - 1e-6])
        for spawn in (0.0, 3.0, -3.0):  # 3.0 rad spawns make the heading wrap through +-pi.
            with self.subTest(spawn=spawn):
                yaw = spawn + rel  # Quaternion built from the unwrapped angle; the term must wrap.
                quat = torch.stack((torch.zeros(5), torch.zeros(5), (yaw / 2).sin(), (yaw / 2).cos()), dim=-1)
                env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
                    root_quat_w=SimpleNamespace(torch=quat)))}, spawn_yaw=torch.full((5,), spawn))
                value = term.func(env, **term.params)
                expected = -term.params["tail_slope"] * (rel.abs() - 0.1).clamp(min=0.0)
                torch.testing.assert_close(value, expected, atol=1e-5, rtol=0)
                # Exactly the audited band term minus 1.
                band = self.rewards.balance_yaw_band(env, **term.params)
                torch.testing.assert_close(value, band - 1.0, atol=1e-6, rtol=0)
        # Zero inside the band (no alive bonus); -0.139 at the old 30 deg wall; -1 at a full reversal.
        self.assertEqual(float(expected[0]), 0.0)
        self.assertEqual(float(expected[1]), 0.0)
        self.assertAlmostEqual(float(expected[2]), -0.1393, places=3)
        self.assertAlmostEqual(float(expected[4]), -1.0, places=5)
        # Worst weighted step stays below survival: ending early never pays.
        self.assertLess(term.weight, self.cfg(3).rewards.survival.weight)


if __name__ == "__main__":
    unittest.main()
