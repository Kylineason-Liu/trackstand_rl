"""Headless regression tests; no GUI or Isaac Sim required."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newbike_mujoco.env import BikeBalanceEnv
from newbike_mujoco.policy import DeployedPolicy
from newbike_mujoco.viewer_envs import ViewerEnvGroup

SPEC = importlib.util.spec_from_file_location("deploy_viewer", ROOT / "scripts/run_deploy_viewer.py")
VIEWER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIEWER)

XML = """<mujoco><option timestep="0.0025" gravity="0 0 0"/>
<worldbody><geom name="floor" type="plane" size="10 10 .1" friction=".1 0 0"/>
<body name="base_link"><freejoint name="base_free"/><geom type="sphere" size=".1" mass="1"/>
<body pos=".2 0 0"><joint name="steering_joint"/><geom type="sphere" size=".05" mass=".1"/></body>
<body pos="-.2 0 0"><joint name="rearwheel_joint"/><geom type="sphere" size=".05" mass=".1"/></body>
<body pos=".3 0 0"><joint name="frontwheel_joint"/><geom type="sphere" size=".05" mass=".1"/></body>
</body></worldbody><actuator>
<general name="steering_act" joint="steering_joint" gainprm="10" biastype="affine"
 biasprm="0 -10 -.2" forcelimited="true" forcerange="-5.6 5.6"/>
<general name="drive_act" joint="rearwheel_joint" biastype="affine" biasprm="0 0 -.05"
 forcelimited="true" forcerange="-6.28 6.28"/>
</actuator></mujoco>"""


def first_env(dr=False):
    return BikeBalanceEnv(scene_xml=XML, seed=42, enable_dr=dr)


class MultiViewerTests(unittest.TestCase):
    def test_single_environment_matches_original(self):
        original = first_env()
        group = ViewerEnvGroup(first_env(), 1, 42)
        np.testing.assert_array_equal(original.reset(), group.reset()[0])
        for _ in range(10):
            obs, done = original.step(np.array([.1, .02]), observe_terminal=False)
            batch, finished, _ = group.step(np.array([[.1, .02]]))
            np.testing.assert_array_equal(obs, batch[0])
            self.assertEqual(done, finished[0])

    def test_64_independent_models_and_data(self):
        group = ViewerEnvGroup(first_env(True), 64, 42)
        obs = group.reset()
        self.assertEqual(obs.shape, (64, 11))
        self.assertEqual(len({id(e.model) for e in group.envs}), 64)
        self.assertEqual(len({id(e.data) for e in group.envs}), 64)
        self.assertFalse(np.shares_memory(group.envs[0].model.body_mass, group.envs[1].model.body_mass))
        self.assertFalse(np.array_equal(group.envs[0].dr["mass_scale"], group.envs[1].dr["mass_scale"]))
        mass = group.envs[1].model.body_mass.copy()
        position = group.envs[1].data.qpos.copy()
        group.envs[0].reset()
        np.testing.assert_array_equal(mass, group.envs[1].model.body_mass)
        np.testing.assert_array_equal(position, group.envs[1].data.qpos)

    def test_repeatable_seed_streams(self):
        one = ViewerEnvGroup(first_env(True), 4, 42)
        two = ViewerEnvGroup(first_env(True), 4, 42)
        np.testing.assert_array_equal(one.reset(), two.reset())
        for _ in range(3):
            for a, b in zip(one.step(np.zeros((4, 2))), two.step(np.zeros((4, 2)))):
                np.testing.assert_array_equal(a, b)

    def test_reset_one_does_not_reset_others(self):
        group = ViewerEnvGroup(first_env(), 2, 42)
        group.reset()
        group.envs[0].MAX_STEPS = 1
        _, done, torque = group.step(np.zeros((2, 2)))
        np.testing.assert_array_equal(done, [True, False])
        self.assertEqual(torque.shape, (2, 2))
        group.envs[0].reset()
        self.assertEqual(group.envs[0].stats.length, 0)
        self.assertEqual(group.envs[1].stats.length, 1)

    def test_grid_is_render_only_and_hides_duplicate_floor(self):
        group = ViewerEnvGroup(first_env(), 4, 42)
        group.reset()
        poses = [env.data.qpos.copy() for env in group.envs]
        scene = mujoco.MjvScene(group.envs[0].model, maxgeom=100)
        group.update_scene(scene, mujoco.MjvOption())
        self.assertEqual(scene.ngeom, 12)  # 3 additional bikes x 4 moving geoms
        for geom in scene.geoms[:scene.ngeom]:
            self.assertEqual(geom.objid, -1)  # render copies are not pickable physics bodies
            self.assertEqual(geom.objtype, mujoco.mjtObj.mjOBJ_UNKNOWN)
        np.testing.assert_allclose(scene.geoms[0].pos, group.envs[1].data.geom_xpos[1] + group.offsets[1])
        group.update_scene(scene, mujoco.MjvOption())
        self.assertEqual(scene.ngeom, 12)  # no accumulation between frames
        for expected, env in zip(poses, group.envs):
            np.testing.assert_array_equal(expected, env.data.qpos)

    def test_invalid_counts_actions_and_render_capacity(self):
        with self.assertRaises(ValueError):
            ViewerEnvGroup(first_env(), 0, 42)
        group = ViewerEnvGroup(first_env(), 4, 42)
        with self.assertRaises(ValueError):
            group.step(np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            group.update_scene(mujoco.MjvScene(group.envs[0].model, maxgeom=1), mujoco.MjvOption())

    def test_batched_policy_matches_single_actions(self):
        from newbike_mujoco.contract import DEFAULT_CHECKPOINT
        path = DEFAULT_CHECKPOINT
        if not path.exists():
            self.skipTest("v1 model75300 is not installed")
        policy = DeployedPolicy(pt_path=str(path))
        obs = ViewerEnvGroup(first_env(True), 64, 42).reset()
        expected = np.stack([policy.act(row) for row in obs])
        np.testing.assert_allclose(policy.act_batch(obs), expected, atol=2e-6, rtol=1e-5)

    def test_fixed_batch_one_policy(self):
        class Session:
            def get_inputs(self):
                return [type("Input", (), {"shape": [1, 11]})()]
            def run(self, outputs, inputs):
                data = inputs["observation"]
                self.assert_single(data)
                return [data[:, :2]]
            @staticmethod
            def assert_single(data):
                if data.shape != (1, 11):
                    raise ValueError("fixed batch-one export")
        policy = DeployedPolicy.__new__(DeployedPolicy)
        policy.backend = "onnxruntime"
        policy._ort_sess = Session()
        policy._input_name = "observation"
        obs = np.linspace(-1, 1, 33, dtype=np.float32).reshape(3, 11)
        np.testing.assert_array_equal(policy.act_batch(obs), obs[:, :2])

    def test_torch_batch(self):
        import torch
        policy = DeployedPolicy.__new__(DeployedPolicy)
        policy.backend = "torch"
        policy._torch_net = torch.nn.Linear(11, 4)
        obs = np.zeros((4, 11), dtype=np.float32)
        np.testing.assert_allclose(policy.act_batch(obs), np.stack([policy.act(row) for row in obs]), atol=2e-7)

    def test_torque_export_shape_and_reset_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            base = str(Path(directory) / "torque")
            VIEWER._plot_multi_torque_traces(np.zeros((3, 4, 2)), [(2, 3)], 6.28, 5.6, base, 42)
            self.assertTrue(Path(base + "_torque.png").is_file())
            with np.load(base + "_torque.npz") as saved:
                self.assertEqual(saved["drive"].shape, (3, 4))
                np.testing.assert_array_equal(saved["env_ids"], [0, 1, 2, 3])
                np.testing.assert_array_equal(saved["seeds"], [42, 43, 44, 45])
                np.testing.assert_array_equal(saved["reset_events"], [[2, 3]])

    def test_default_64_and_cli_main_headlessly(self):
        class Handle:
            def __init__(self):
                self.cam = mujoco.MjvCamera()
                self.opt = mujoco.MjvOption()
                self.user_scn = mujoco.MjvScene(first_env().model, maxgeom=1000)
                self.frames = 0
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def lock(self): return contextlib.nullcontext()
            def sync(self): self.frames += 1
            def is_running(self): return self.frames < 2
        handle = Handle()
        with patch.object(sys, "argv", ["viewer", "--task", "zero_agent"]), \
             patch.object(VIEWER, "build_env_from_mjcf", return_value=(first_env(), True)), \
             patch.object(VIEWER.mujoco.viewer, "launch_passive", return_value=handle), \
             patch.object(VIEWER.time, "sleep"), contextlib.redirect_stdout(io.StringIO()) as output:
            VIEWER.main()
        self.assertEqual(handle.frames, 2)
        self.assertEqual(handle.user_scn.ngeom, 63 * 4)
        self.assertIn("environments: 64", output.getvalue())

    def test_episode_length_histogram_boundaries(self):
        seconds = [0, 1, 2.98, 3, 5, 6, 9, 12, 15, 18, 21, 24, 27, 29, 30]
        records = [(round(value / BikeBalanceEnv.CTRL_DT), None, None) for value in seconds]
        with tempfile.TemporaryDirectory() as directory:
            base = str(Path(directory) / "evaluation")
            VIEWER._plot_episode_length_distribution(records, base, "model_91850.onnx", False)
            csv_path = Path(base + "_epi_len_histogram.csv")
            counts = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            np.testing.assert_array_equal(counts[:, 0], np.arange(0, 30, 3))
            np.testing.assert_array_equal(counts[:, 1], np.arange(3, 31, 3))
            np.testing.assert_array_equal(counts[:, 2], [3, 2, 1, 1, 1, 1, 1, 1, 1, 3])
            self.assertEqual(counts[:, 2].sum(), len(records))
            self.assertGreater(csv_path.with_suffix(".png").stat().st_size, 0)

    def test_episode_length_histogram_draws_ten_bars(self):
        with tempfile.TemporaryDirectory() as directory:
            fig, ax = VIEWER.plt.subplots()
            with patch.object(VIEWER.plt, "subplots", return_value=(fig, ax)):
                VIEWER._plot_episode_length_distribution([(250, None, None), (1500, None, None)],
                                                         str(Path(directory) / "evaluation"), "model_91850.onnx", False)
            self.assertEqual(len(ax.patches), 10)
            self.assertEqual(ax.get_xlim(), (0, 30))
            np.testing.assert_array_equal([bar.get_height() for bar in ax.patches],
                                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
            self.assertEqual(ax.get_xlabel(), "Episode length (s)")
            self.assertEqual(ax.get_ylabel(), "Number of completed episodes")

    def test_episode_length_histogram_without_completed_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = str(Path(directory) / "evaluation")
            VIEWER._plot_episode_length_distribution([], base, "model_91850.onnx", False)
            csv_path = Path(base + "_epi_len_histogram.csv")
            counts = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            self.assertEqual(counts.shape, (10, 3))
            np.testing.assert_array_equal(counts[:, 2], np.zeros(10))

    def test_episode_length_histogram_model_and_dr_labels(self):
        for filename, label in (
            ("trackstand_rl_rsl_2026-09-22_19-45-06_model_91850.onnx", "model_91850"),
            ("model_41050.pt", "model_41050"),
            ("custom_actor.onnx", "custom_actor.onnx"),
        ):
            for enable_dr in (False, True):
                with self.subTest(filename=filename, dr=enable_dr), tempfile.TemporaryDirectory() as directory:
                    fig, ax = VIEWER.plt.subplots()
                    with patch.object(VIEWER.plt, "subplots", return_value=(fig, ax)):
                        VIEWER._plot_episode_length_distribution(
                            [], str(Path(directory) / "evaluation"), filename, enable_dr,
                        )
                    self.assertIn(label, ax.get_title())
                    self.assertIn(f"Domain randomization: {'ON' if enable_dr else 'OFF'}", ax.get_title())

    def test_evaluation_concise_filenames_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(VIEWER.time, "strftime", return_value="202610151649") as timestamp:
            policy_path = "/policies/trackstand_rl_rsl_2026-09-17_19-46-37_model_41050.onnx"
            first_base = VIEWER._evaluation_output_base(directory, policy_path, False)
            VIEWER._plot_episode_length_distribution(
                [(250, None, None)], first_base, policy_path, False,
            )
            second_base = VIEWER._evaluation_output_base(directory, "model_41050.pt", False)
            VIEWER._plot_episode_length_distribution([], second_base, "model_41050.pt", False)
            dr_base = VIEWER._evaluation_output_base(directory, "model_91850.onnx", True)
            VIEWER._plot_episode_length_distribution([], dr_base, "model_91850.onnx", True)
            expected = ["202610151649_41050_no_dr", "202610151649_41050_no_dr_2", "202610151649_91850_dr"]
            self.assertEqual({path.name for path in Path(directory).iterdir()},
                             {stem + "_epi_len_histogram" + ext for stem in expected for ext in (".png", ".csv")})
            original = np.loadtxt(first_base + "_epi_len_histogram.csv", delimiter=",", skiprows=1)
            self.assertEqual(original[:, 2].sum(), 1)
            timestamp.assert_called_with("%Y%m%d%H%M")

    def test_any_existing_report_reserves_the_whole_prefix(self):
        for ending in ("_epi_len_histogram.png", "_epi_len_histogram.csv", "_torque.png", "_torque.npz", "_summary.log"):
            with self.subTest(ending=ending), tempfile.TemporaryDirectory() as directory, \
                 patch.object(VIEWER.time, "strftime", return_value="202610151649"):
                first = VIEWER._evaluation_output_base(directory, "model_41050.pt", False)
                Path(first + ending).touch()
                second = VIEWER._evaluation_output_base(directory, "model_41050.pt", False)
                self.assertEqual(second, first + "_2")

    def test_histogram_counts_every_printed_episode_and_excludes_unfinished(self):
        class Handle:
            def __init__(self, env):
                self.cam = mujoco.MjvCamera()
                self.opt = mujoco.MjvOption()
                self.user_scn = mujoco.MjvScene(env.model, maxgeom=1000)
                self.frames = 0
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def lock(self): return contextlib.nullcontext()
            def sync(self): self.frames += 1
            def is_running(self): return self.frames < 3

        for num_envs in (1, 64):
            with self.subTest(num_envs=num_envs), tempfile.TemporaryDirectory() as directory:
                env = first_env()
                group = ViewerEnvGroup(env, num_envs, 42)
                handle = Handle(env)

                def scripted_step(actions):
                    # Two completions for env 0; also a simultaneous timeout
                    # for the last env in the multi-bike case. Others stay active.
                    done = np.zeros(num_envs, dtype=bool)
                    for instance in group.envs:
                        instance.stats.length = 1
                    if handle.frames < 2:
                        done[0] = True
                        env.stats.length = (250, 1450)[handle.frames]
                        env.stats.death = "fallen"
                        if handle.frames == 1 and num_envs > 1:
                            done[-1] = True
                            group.envs[-1].stats.length = 1500
                            group.envs[-1].stats.timeout = True
                    return np.zeros((num_envs, 11)), done, np.zeros((num_envs, 2))

                argv = ["viewer", "--mjcf", "unused.xml", "--num_envs", str(num_envs),
                        "--onnx", "model_91850.onnx", "--pt", "model_41050.pt"]
                if num_envs > 1:
                    argv.append("--dr")
                with patch.object(sys, "argv", argv), \
                     patch.object(VIEWER, "ROOT", directory), \
                     patch.object(VIEWER, "build_env_from_mjcf", return_value=(env, True)), \
                     patch.object(VIEWER, "ViewerEnvGroup", return_value=group), \
                     patch.object(VIEWER, "DeployedPolicy") as policy, \
                     patch.object(group, "step", side_effect=scripted_step), \
                     patch.object(VIEWER.mujoco.viewer, "launch_passive", return_value=handle), \
                     patch.object(VIEWER.time, "sleep"), \
                     patch.object(VIEWER, "_plot_episode_length_distribution",
                                  wraps=VIEWER._plot_episode_length_distribution) as plot_histogram, \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    policy.return_value.act_batch.return_value = np.zeros((num_envs, 2))
                    policy.return_value.backend = "onnxruntime" if num_envs == 1 else "torch"
                    VIEWER.main()

                expected_path = "model_91850.onnx" if num_envs == 1 else "model_41050.pt"
                self.assertEqual(plot_histogram.call_args.args[2:], (expected_path, num_envs > 1))
                output_dir = Path(directory) / "logs"
                csv_path, = output_dir.glob("*_epi_len_histogram.csv")
                prefix = csv_path.name.removesuffix("_epi_len_histogram.csv")
                self.assertEqual(plot_histogram.call_args.args[1], str(output_dir / prefix))
                self.assertEqual({path.name for path in output_dir.iterdir()},
                                 {prefix + ending for ending in ("_epi_len_histogram.png", "_epi_len_histogram.csv",
                                                                "_torque.png", "_torque.npz", "_summary.log", "_summary.json")})
                with np.load(output_dir / (prefix + "_torque.npz")) as saved:
                    self.assertEqual(saved["drive"].shape, (3,) if num_envs == 1 else (3, num_envs))
                counts = np.loadtxt(csv_path, delimiter=",", skiprows=1)[:, 2]
                expected = np.zeros(10)
                expected[1] = 1
                expected[9] = 1 + int(num_envs > 1)
                np.testing.assert_array_equal(counts, expected)
                printed = [line for line in output.getvalue().splitlines() if " steps (" in line]
                self.assertEqual(int(counts.sum()), len(printed))


if __name__ == "__main__":
    unittest.main()
