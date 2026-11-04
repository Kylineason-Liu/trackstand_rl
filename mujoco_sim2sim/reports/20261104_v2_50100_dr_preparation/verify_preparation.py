"""CPU-only export/contract check; a short rollout is not a survival evaluation."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch
import yaml

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
MUJOCO = ROOT / "mujoco_sim2sim"
sys.path.insert(0, str(MUJOCO / "src"))
from newbike_mujoco_v2 import DeployedPolicy, build_env_from_mjcf
from newbike_mujoco_v2 import contract as c
from newbike_mujoco_v2.viewer_envs import ViewerEnvGroup

RUN = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_v2_rsl/2026-11-02_22-40-01"
CHECKPOINT = RUN / "model_50100.pt"
ONNX = MUJOCO / "policies/trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    env = yaml.load((RUN / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    agent = yaml.safe_load((RUN / "params/agent.yaml").read_text())
    assert env["domain_randomization"] == "true"
    assert agent["actor"]["activation"] == "elu"
    assert agent["actor"]["obs_normalization"] is False
    assert agent["actor"]["distribution_cfg"]["class_name"].endswith(":BetaDistribution")
    assert float(env["sim"]["dt"]) == c.PHYSICS_DT
    assert int(env["decimation"]) == c.DECIMATION
    obs = env["observations"]["policy"]["imu"]
    assert obs["func"].endswith(":balance_imu_state_v1")
    assert obs["clip"] == "null"
    assert tuple(map(float, obs["noise"]["std"])) == c.OBS_NOISE_STD
    steering = env["actions"]["steering"]
    assert steering["class_type"].endswith(":SlewLimitedJointPositionAction")
    assert float(steering["scale"]) == c.STEER_SCALE
    assert tuple(map(float, steering["clip"]["steering_joint"])) == (-c.STEER_POSITION_LIMIT, c.STEER_POSITION_LIMIT)
    assert float(steering["max_target_velocity_rad_s"]) == c.STEER_SLEW_RATE
    for name, expected in (
        ("steering", (10., .2, 5.6, 23.9, c.STEER_FRICTION, c.STEER_VISCOUS_FRICTION)),
        ("drive", (0., .3, 6.28, 61.6, .106, 0.)),
    ):
        actuator = env["scene"]["robot"]["actuators"][name]
        assert actuator["class_type"].endswith(":IdealPDActuator")
        actual = tuple(float(actuator[key]) for key in (
            "stiffness", "damping", "effort_limit", "velocity_limit", "friction", "viscous_friction"))
        assert actual == expected
        assert float(actuator["dynamic_friction"]) == expected[4]
    assert tuple(env["terminations"]) == c.TERMINATIONS
    events = env["events"]
    assert events["randomize_actuator_gains"] == "null"
    assert tuple(map(float, events["randomize_mass"]["params"]["mass_distribution_params"])) == c.DR_MASS
    for axis in "xyz":
        assert tuple(map(float, events["randomize_com"]["params"]["com_range"][axis])) == (-c.DR_COM, c.DR_COM)
    for kind in ("static", "dynamic"):
        assert tuple(map(float, events["randomize_material"]["params"][kind + "_friction_range"])) == c.DR_FRICTION
    initial = events["initial_state"]["params"]
    assert tuple(map(float, initial["pose_range"]["roll"])) == c.DR_INIT_ROLL
    assert tuple(map(float, initial["velocity_range"]["roll"])) == c.DR_INIT_ROLL_RATE

    pt = DeployedPolicy(pt_path=CHECKPOINT)
    exported = DeployedPolicy(onnx_path=ONNX, config_path=RUN / "params/env.yaml")
    assert exported.backend == "onnxruntime"
    rng = np.random.default_rng(42)
    inputs, expected_outputs, actual_outputs = [], [], []
    max_error = 0.

    def compare(x):
        nonlocal max_error
        expected = pt.act_batch(x)
        actual = exported.act_batch(x)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-5)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        inputs.append(x.copy())
        expected_outputs.append(expected)
        actual_outputs.append(actual)
        return actual

    # Include a natural upright observation and all supported test batch sizes.
    compare(np.array([[0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=np.float32))
    for batch in (1, 4, 64, 257):
        for scale in (.1, 1., 10., 100.):
            compare((rng.normal(size=(batch, 11)) * scale).astype(np.float32))
    first, _ = build_env_from_mjcf(c.DEFAULT_MJCF, seed=42, enable_dr=True)
    group = ViewerEnvGroup(first, 4, 42)
    observations = group.reset()
    assert all(item.enable_dr and item.dr is not None for item in group.envs)
    resets = 0
    for _ in range(50):
        actions = compare(observations)
        observations, done, _ = group.step(actions)
        for index in np.flatnonzero(done):
            observations[index] = group.envs[index].reset()
            resets += 1

    sidecar = dict(
        task=c.TASK, drive_action="velocity", observations=list(c.OBSERVATIONS),
        drive_velocity_scale_rad_s=c.DRIVE_SCALE, drive_velocity_p_nm_s_per_rad=c.DRIVE_KD,
        drive_integral_gain_nm_per_rad=c.DRIVE_INTEGRAL_GAIN, drive_integral_rate_hz=c.DRIVE_INTEGRAL_RATE_HZ,
        drive_effort_limit_nm=c.DRIVE_EFFORT_LIMIT, policy_sha256=sha(ONNX),
        source_checkpoint=str(CHECKPOINT.relative_to(ROOT)), source_checkpoint_sha256=sha(CHECKPOINT),
        source_iteration=50100, source_config=str((RUN / "params/env.yaml").relative_to(ROOT)),
        source_config_sha256=sha(RUN / "params/env.yaml"), deterministic_action="Beta mean: 2*alpha/(alpha+beta)-1",
        scales_baked_into_onnx=False, training_domain_randomization=True,
    )
    ONNX.with_suffix(".contract.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    automatic = DeployedPolicy(onnx_path=ONNX)
    assert automatic.contract_metadata["path"] == str(ONNX.with_suffix(".contract.json"))
    np.testing.assert_array_equal(automatic.act_batch(inputs[0]), actual_outputs[0])
    np.savez_compressed(REPORT / "inference_comparison.npz", observations=np.concatenate(inputs),
                        pytorch_actions=np.concatenate(expected_outputs), onnx_actions=np.concatenate(actual_outputs))
    before = json.loads((REPORT / "protected_sha256_before.json").read_text())
    after = {name: sha(ROOT / name) for name in before}
    assert before == after, "An existing input was modified"
    (REPORT / "protected_sha256_after.json").write_text(json.dumps(after, indent=2) + "\n")
    result = dict(
        passed=True, task=c.TASK, checkpoint=str(CHECKPOINT), checkpoint_sha256=sha(CHECKPOINT),
        onnx=str(ONNX), onnx_sha256=sha(ONNX), source_config=sidecar["source_config"],
        training_dr=True, inference="deterministic bounded Beta means, no observation normalization",
        input_shape=exported._ort_sess.get_inputs()[0].shape,
        output_shape=exported._ort_sess.get_outputs()[0].shape,
        onnx_providers=exported._ort_sess.get_providers(),
        comparisons=sum(len(x) for x in inputs), max_absolute_action_error=max_error,
        tolerance=1e-5, batch_sizes=[1, 4, 64, 257],
        rollout_smoke=dict(environments=4, policy_steps_per_env=50, environment_transitions=200,
                           physics_steps=4800, domain_randomization=True, seed=42, automatic_resets=resets,
                           purpose="inference/controller integration only; not survival evaluation"),
        timing=dict(policy_hz=50, physics_pd_hz=1200, rear_integral_hz=200),
        protected_existing_files=len(before),
        defaults_changed=False, real_gui_opened=False,
        versions=dict(torch=torch.__version__, numpy=np.__version__, onnxruntime=ort.__version__),
    )
    (REPORT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
