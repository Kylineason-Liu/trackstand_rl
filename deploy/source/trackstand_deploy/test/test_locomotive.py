"""Locomotive (CycleRL riding policy) contract, models, C1 drive backend and controller."""

from collections import deque
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from trackstand_deploy.can_protocol import (
    b3_current_limit_payload, c1_speed_payload, parse_b3_current_limit, parse_c1_speed, parse_current_feedback,
)
from trackstand_deploy.contracts import rpy_degrees_to_quaternion_xyzw
from trackstand_deploy.crsf import normalize_axis
from trackstand_deploy.locomotive_contracts import (
    DRIVE_VELOCITY_LIMIT_RAD_S, MANIFEST_FILENAME, OBSERVATION_NAMES, CommandShaper, LocomotiveMode,
    LocomotiveStateMachine, build_observation, load_model_spec, normalized_stick, scale_policy_action, sha256_file,
)
from trackstand_deploy.mit_protocol import MitRanges


PACKAGE = Path(__file__).parents[1]
MODELS = PACKAGE / "models"
MANIFEST = MODELS / MANIFEST_FILENAME
EXPECTED = (
    "locomotive_1900_drive03_2026-10-19_16-30-17.onnx",
    "locomotive_1900_2026-10-19_19-04-32.onnx",
    "locomotive_1900_drive03_2026-10-20_03-18-29.onnx",
    "locomotive_deploy_10deg_2026-10-20_04-32-20.onnx",
    "locomotive_act_refine_lo_2026-10-19_15-30-18.onnx",
)


def manifest():
    return json.loads(MANIFEST.read_text())


# ---- models / manifest ------------------------------------------------------------------
def test_manifest_lists_top5_in_rank_order_with_matching_embedded_files():
    data = manifest()
    assert [entry["onnx"] for entry in data["models"]] == list(EXPECTED)
    assert [entry["rank"] for entry in data["models"]] == [1, 2, 3, 4, 5]
    assert data["default_model"] == EXPECTED[0]
    for entry in data["models"]:
        assert entry["weights_embedded"] and not (MODELS / (entry["onnx"] + ".data")).exists()
        assert sha256_file(str(MODELS / entry["onnx"])) == entry["onnx_sha256"]
        assert entry["contract"]["observation_names"] == list(OBSERVATION_NAMES)
        assert entry["network"] == {"hidden_layers": [512, 256, 128, 64], "activation": "elu"}
        assert entry["observation_preprocessor"]["clip"] == 5.0


def test_per_model_contracts_keep_the_trained_scales():
    specs = {name: load_model_spec(str(MANIFEST), name) for name in EXPECTED}
    for name, spec in specs.items():
        assert spec.drive_scale_rad_s == pytest.approx(3.0 / 0.07)
        assert spec.drive_clip_rad_s == 45.0 == DRIVE_VELOCITY_LIMIT_RAD_S
        assert spec.wheel_radius_m == pytest.approx(0.07)
        assert spec.yaw_rate_cmd_max_rad_s == 0.5
        assert spec.initial_speed_min_m_s == 1.0
        assert spec.lean_termination_rad == pytest.approx(math.radians(45))
        steer = math.radians(10 if "deploy_10deg" in name else 20)
        assert spec.steering_scale_rad == pytest.approx(steer) and spec.steering_clip_rad == pytest.approx(steer)
        assert spec.vel_cmd_min_m_s == (0.3 if "act_refine_lo" in name else 0.5) and spec.vel_cmd_max_m_s == 2.5


def test_unknown_or_tampered_manifest_entries_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_model_spec(str(MANIFEST), "missing.onnx")
    data = manifest()
    data["models"][0]["contract"]["observation_names"].reverse()
    path = tmp_path / MANIFEST_FILENAME
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_model_spec(str(path), EXPECTED[0])


@pytest.mark.parametrize("name", EXPECTED)
def test_onnx_reproduces_skrl_reference_actions(name):
    pytest.importorskip("onnxruntime")
    from trackstand_deploy.locomotive_policy import LocomotiveOnnxPolicy

    spec = load_model_spec(str(MANIFEST), name)
    policy = LocomotiveOnnxPolicy(str(MODELS / name), spec)  # Constructor checks every probe.
    action = policy.act(np.asarray(spec.reference_observations[1], dtype=np.float32))
    assert action == pytest.approx(np.asarray(spec.reference_raw_actions[1]), abs=1e-5)
    with pytest.raises(ValueError):
        policy.act(np.zeros(11, dtype=np.float32))


def test_controller_and_overlay_defaults_name_the_rank1_model():
    data = manifest()["models"][0]
    source = (PACKAGE / "trackstand_deploy" / "locomotive_controller.py").read_text()
    overlay = (PACKAGE / "config" / "locomotive.yaml").read_text()
    for text in (source, overlay):
        assert data["onnx"] in text and data["onnx_sha256"] in text


# ---- observation / action contract ------------------------------------------------------
def test_observation_matches_isaac_signs_and_order():
    roll = math.radians(10)
    observation = build_observation(
        rpy_degrees_to_quaternion_xyzw(np.array([10.0, 0.0, 0.0])), np.array([0.1, -0.2, 0.3]),
        steering_position_rad=2 * math.pi + 0.1, rear_wheel_velocity_rad_s=20.0,
        vel_cmd_m_s=1.2, yaw_rate_cmd_rad_s=-0.25, wheel_radius_m=0.07)
    assert observation.dtype == np.float32 and observation.shape == (8,)
    # projected_gravity_b[1] = -sin(roll); steer wrapped to [-pi, pi]; speed = r * w_rear.
    assert observation == pytest.approx([-math.sin(roll), 0.1, -0.2, 0.3, 1.4, 0.1, 1.2, -0.25], abs=1e-6)


def test_non_finite_observation_fails_closed():
    with pytest.raises(ValueError):
        build_observation(np.array([0, 0, 0, 1.0]), np.array([math.nan, 0, 0]), 0, 0, 1, 0, 0.07)


@pytest.mark.parametrize("name,steer_clip", [(EXPECTED[0], math.radians(20)), (EXPECTED[3], math.radians(10))])
def test_action_scaling_then_clipping_per_model(name, steer_clip):
    spec = load_model_spec(str(MANIFEST), name)
    steering, rear = scale_policy_action(np.array([0.5, 0.5], dtype=np.float32), spec)
    assert steering == pytest.approx(0.5 * steer_clip) and rear == pytest.approx(0.5 * 3 / 0.07)
    steering, rear = scale_policy_action(np.array([-3.0, 3.0], dtype=np.float32), spec)
    assert steering == pytest.approx(-steer_clip) and rear == 45.0
    with pytest.raises(ValueError):
        scale_policy_action(np.array([math.inf, 0.0]), spec)


def test_command_shaper_maps_sticks_into_trained_ranges_and_slews():
    shaper = CommandShaper(0.5, 1.5, 0.5, accel_m_s2=1.0, deadband=0.05)
    assert shaper.target_speed(-1.0) == 0.5 and shaper.target_speed(1.0) == 1.5
    assert shaper.target_speed(0.0) == pytest.approx(1.0)
    shaper.reset(0.5)
    vel, yaw = shaper.update(1.0, 1.0)
    assert vel == pytest.approx(0.52) and yaw == pytest.approx(-0.5)  # Stick right -> right turn.
    for _ in range(100):
        vel, yaw = shaper.update(1.0, 0.03)
    assert vel == pytest.approx(1.5) and yaw == 0.0
    assert normalized_stick(0.525, 0.05) == pytest.approx(0.5)


def test_axis_normalization_uses_both_half_ranges():
    channels = [992] * 16
    channels[2], channels[3] = 1811, 172
    assert normalize_axis(channels, 172, 992, 1811, 3) == 1.0
    assert normalize_axis(channels, 172, 992, 1811, 4) == -1.0
    assert normalize_axis(channels, 172, 992, 1811, 4, reversed_axis=True) == 1.0
    assert normalize_axis(channels, 172, 992, 1811, 1) == 0.0


def test_state_machine_launch_ride_and_fault_latch():
    machine = LocomotiveStateMachine(wake_speed_m_s=1.0, min_riding_speed_m_s=0.3)
    assert machine.set_rc(True, ready=False) == LocomotiveMode.DOWN
    assert machine.update(True, [], 0.0) == LocomotiveMode.LAUNCH and machine.enabled and not machine.riding
    assert machine.update(True, [], 0.99) == LocomotiveMode.LAUNCH
    assert machine.update(True, [], 1.0) == LocomotiveMode.RIDING and machine.riding
    assert machine.update(True, [], 0.29) == LocomotiveMode.FAULT_LATCHED
    assert machine.fault_reasons == ("speed_below_riding_envelope",)
    assert machine.update(True, [], 2.0) == LocomotiveMode.FAULT_LATCHED  # Only RC down clears.
    assert machine.set_rc(False, True) == LocomotiveMode.DOWN and machine.fault_reasons == ()
    machine.set_rc(True, True)
    assert machine.update(False, ["imu_timeout"], 0.0) == LocomotiveMode.FAULT_LATCHED
    with pytest.raises(ValueError):
        LocomotiveStateMachine(wake_speed_m_s=0.3, min_riding_speed_m_s=0.3)


# ---- CAN encoding -------------------------------------------------------------------------
def _load_drive_speed_sweep():
    path = PACKAGE.parents[1] / "experiments" / "drive_speed_sweep.py"
    spec = importlib.util.spec_from_file_location("_drive_speed_sweep_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their defining module by name.
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


@pytest.mark.parametrize("speed", [0.0, 1.0, -8.0, 21.42857, 45.0])
def test_c1_encoding_matches_the_proven_speed_sweep(speed):
    sweep = _load_drive_speed_sweep()
    reference, _, _ = sweep.speed_payload(SimpleNamespace(command_sign=-1.0, gear_ratio=1.0), speed)
    assert c1_speed_payload(speed, 1.0, -1.0) == reference
    assert parse_c1_speed(reference, 1.0, -1.0) == pytest.approx(speed, abs=2 * math.pi / 6000)


def test_b3_ceiling_rounds_down_and_rejects_invalid_values():
    assert b3_current_limit_payload() == b"\xb3"
    payload = b3_current_limit_payload(6.28 / 0.3)
    assert struct.unpack("<I", payload[1:])[0] == 20933 and parse_b3_current_limit(payload) == pytest.approx(20.933)
    for bad in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError):
            b3_current_limit_payload(bad)


# ---- C1 velocity backend -----------------------------------------------------------------
@pytest.fixture
def velocity_node(node_module):
    node_module("motor_mit_io", canonical=True)
    module = node_module("motor_mit_velocity_io")
    clock = SimpleNamespace(value=100.0)
    module.time = SimpleNamespace(monotonic=lambda: clock.value, time=lambda: 123456.0)
    base = sys.modules["trackstand_deploy.motor_mit_io"]
    base.time = module.time
    node = object.__new__(module.MotorMitVelocityIo)
    node._initialize_command_state()
    node._steering = base.ChannelState("steering", 0x102, 2, 1, -1, -1)
    node._drive = base.ChannelState("drive", 0x101, 1, 1, -1, -1)
    for motor in (node._steering, node._drive):
        motor.zero_count = 0
        motor.torque_constant_nm_per_a = 0.78 if motor is node._steering else 0.3
        motor.status_sample_monotonic = motor.sample_monotonic = 99.99
        motor.position_sample_monotonic = motor.velocity_sample_monotonic = 99.99
    node._dry_run = node._hard_fault = False
    node._feedback_timeout_s, node._command_timeout_s = 0.05, 0.08
    node._steering_limit_rad, node._steering_fault_clear_rad = math.pi / 2, math.radians(5)
    node._requested_enable, node._command_monotonic = True, 99.99
    node._steering_target_rad = node._requested_steering_target_rad = 0.05
    node._drive_target_nm = node._requested_drive_target_nm = 10.0  # rad/s in this backend.
    node._last_control_monotonic = 99.995
    node._mit_ranges, node._mit_zero_rad, node._mit_feedback = MitRanges(3, 45, 18), 0.002, None
    node._mit_encoded_command, node._mit_send_interval_ms = None, math.nan
    node._mit_config_ready, node._mit_kp, node._mit_kd, node._mit_f1_at = True, 10, .2, 99.99
    node._mit_last_sent_at = node._mit_first_sent_at = node._next_mit_at = node._slew_last_at = 0.0
    node._mit_sent_this_enable = False
    node._disable_wait, node._disable_f1_wait, node._disable_confirmed = {}, {}, True
    node._disable_deadline_at = node._next_disable_retry_at = 0
    node._last_disable_errors, node._pending, node._poll_jobs = (), None, deque()
    node._drive_b3_limit_a = 20.933
    node._tx_bus = Mock()
    node._tx_bus.receive.return_value = None
    for name in ("control", "event", "fault", "status", "state"):
        setattr(node, f"_{name}_publisher", Mock())
    node._control_dimension = SimpleNamespace()
    node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0)))
    node.get_logger = Mock(return_value=Mock())
    node.context = SimpleNamespace(ok=lambda: True)
    return node, module, clock


def control_record(node, module):
    return dict(zip(module.VELOCITY_CONTROL_FIELDS, node._control_publisher.publish.call_args.args[0].data))


def test_velocity_backend_uses_its_own_topic_and_speed_bounds(velocity_node):
    node, module, _ = velocity_node
    assert module.MotorMitVelocityIo._command_topic == "/locomotive/motor_command"
    assert sys.modules["trackstand_deploy.motor_mit_io"].MotorMitIo._command_topic == \
        "/trackstand/motor_command"
    node._command_callback(sys.modules["std_msgs.msg"].Float64MultiArray(data=[1.0, 0.1, 60.0]))
    assert node._drive_target_nm == 45.0 and node._requested_drive_target_nm == 60.0


def test_velocity_backend_sends_c1_not_c0_and_keeps_mit_steering(velocity_node):
    node, module, _ = velocity_node
    node._control_tick()
    sends = [call.args for call in node._tx_bus.send.call_args_list]
    assert (0x101, c1_speed_payload(10.0, 1.0, -1.0)) in sends
    assert not any(can_id == 0x101 and data[0] == 0xC0 for can_id, data in sends)
    assert any(can_id == 0x502 for can_id, _ in sends)  # MIT steering frame unchanged.
    data = control_record(node, module)
    assert data["drive_velocity_target_rad_s"] == 10.0
    assert data["drive_c1_encoded_velocity_rad_s"] == pytest.approx(10.0, abs=1e-3)
    assert math.isnan(data["drive_final_torque_nm"]) and data["drive_b3_current_limit_a"] == 20.933


def test_velocity_backend_disable_still_zeroes_current_then_cf(velocity_node):
    node, module, _ = velocity_node
    node._control_tick()
    node._tx_bus.reset_mock()
    node._command_callback(sys.modules["std_msgs.msg"].Float64MultiArray(data=[0.0, 0.0, 0.0]))
    node._control_tick()
    sends = [call.args for call in node._tx_bus.send.call_args_list]
    drive = [data for can_id, data in sends if can_id == 0x101]
    assert drive[0][0] == 0xC0 and parse_current_feedback(drive[0]) == 0.0 and drive[1] == b"\xcf"
    assert not any(data[0] == 0xC1 for data in drive)


def test_missing_b3_configuration_blocks_drive_motion(velocity_node):
    node, _, _ = velocity_node
    node._drive_b3_limit_a = math.nan
    node._control_tick()
    assert node._fault_latched and node._fault_reason == "drive_can_send_failed"
    assert not any(call.args[1][0] == 0xC1 for call in node._tx_bus.send.call_args_list)


@pytest.mark.parametrize("original,expected", [(82.498, 20.933), (10.0, 10.0)])
def test_startup_lowers_b3_to_trained_torque_cap_but_never_raises(velocity_node, monkeypatch, original, expected):
    node, module, _ = velocity_node
    base = module.MotorMitIo
    monkeypatch.setattr(base, "_startup", lambda self: None)
    state = {"milliamps": round(original * 1000)}
    writes = []

    def query(motor, opcode, length, timeout, payload=None):
        assert motor is node._drive and opcode == 0xB3
        if payload is not None and len(payload) == 5:
            writes.append(parse_b3_current_limit(payload))
            state["milliamps"] = struct.unpack("<I", payload[1:])[0] - 1  # One-count firmware quantization.
        return bytes([0xB3]) + struct.pack("<I", state["milliamps"])

    node._query = query
    node._startup()
    assert writes == [pytest.approx(expected)]
    assert node._drive_b3_limit_a == pytest.approx(expected - 0.001, abs=1e-6)


def test_startup_rejects_unconfirmed_b3_and_dry_run_only_reads(velocity_node, monkeypatch):
    node, module, _ = velocity_node
    monkeypatch.setattr(module.MotorMitIo, "_startup", lambda self: None)
    node._query = lambda motor, opcode, length, timeout, payload=None: b3_current_limit_payload(82.498)
    with pytest.raises(RuntimeError, match="B3 ceiling not confirmed"):
        node._startup()
    node._dry_run = True
    calls = []
    node._query = lambda motor, opcode, length, timeout, payload=None: (
        calls.append(payload), b3_current_limit_payload(82.498))[1]
    node._startup()
    assert calls == [None] and node._drive_b3_limit_a == pytest.approx(82.498)


# ---- controller -------------------------------------------------------------------------
@pytest.fixture
def controller(node_module):
    module = node_module("locomotive_controller")
    clock = SimpleNamespace(value=50.0)
    module.time = SimpleNamespace(monotonic=lambda: clock.value, time=lambda: 1000.0)
    node = object.__new__(module.LocomotiveController)
    spec = load_model_spec(str(MANIFEST), EXPECTED[0])
    node._spec = spec
    node._policy = Mock()
    node._policy.act.return_value = np.array([0.5, 0.7], dtype=np.float32)
    node._imu_timeout_s = node._motor_timeout_s = 0.06
    node._rc_timeout_s = 0.15
    node._max_roll_rad, node._max_steering_start_rad = 0.5236, math.radians(5)
    node._launch_accel_m_s2, node._use_policy_drive = 0.5, True
    node._shaper = CommandShaper(spec.vel_cmd_min_m_s, 1.5, 0.5, 1.0)
    node._state_machine = LocomotiveStateMachine(1.0, 0.3)
    node._world_from_bike = np.array([0.0, 0.0, 0.0, 1.0])
    node._body_rate_rad_s = np.zeros(3)
    node._motor_state = np.zeros(4)
    node._sticks = (1.0, 0.0)
    node._imu_monotonic = node._motor_monotonic = node._rc_monotonic = node._sticks_monotonic = clock.value
    node._motor_fault = False
    node._launch_speed_m_s = 0.0
    node._last_kickstand_command = None
    node._last_command = (0.0, 0.0, 0.0)
    node._episode_index = node._policy_sequence = 0
    node._last_policy_monotonic = 0.0
    node._last_event_state = None
    node._model_sha256 = spec.onnx_sha256
    node._timing_dimension = SimpleNamespace()
    for name in ("motor", "kickstand", "observation", "action", "status", "timing", "event"):
        setattr(node, f"_{name}_publisher", Mock())
    node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0)))
    node.get_logger = Mock(return_value=Mock())
    return node, module, clock


def commands(node):
    return [call.args[0].data for call in node._motor_publisher.publish.call_args_list]


def kickstands(node):
    return [call.args[0].data for call in node._kickstand_publisher.publish.call_args_list]


def test_controller_launches_with_stands_down_then_hands_over_to_policy(controller):
    node, module, clock = controller
    node._rc_callback(module.Bool(data=True))
    assert node._state_machine.mode == LocomotiveMode.LAUNCH
    node._policy_tick()
    enable, steer, rear = commands(node)[-1]
    assert enable == 1.0 and steer == 0.0 and rear == pytest.approx(0.5 * 0.02 / 0.07)
    assert kickstands(node) == ["down"] and not node._policy.act.called
    node._motor_state = np.array([0.0, 0.0, 0.0, 1.05 / 0.07])  # Measured speed past wake.
    clock.value += 0.02
    node._imu_monotonic = node._motor_monotonic = node._rc_monotonic = node._sticks_monotonic = clock.value
    node._policy_tick()
    assert node._state_machine.mode == LocomotiveMode.RIDING and kickstands(node) == ["down", "up"]
    observation = node._policy.act.call_args.args[0]
    assert observation[4] == pytest.approx(1.05) and observation[6] == pytest.approx(1.07)
    enable, steer, rear = commands(node)[-1]
    assert enable == 1.0 and steer == pytest.approx(0.5 * math.radians(20)) and rear == pytest.approx(0.7 * 3 / 0.07)


def test_controller_fault_latches_motors_off_and_rc_down_lowers_stands(controller):
    node, module, clock = controller
    node._state_machine.set_rc(True, True)
    node._state_machine.mode = LocomotiveMode.RIDING
    node._motor_state = np.array([0.0, 0.0, 0.0, 1.5 / 0.07])
    node._policy.act.side_effect = RuntimeError("boom")
    node._policy_tick()
    assert node._state_machine.mode == LocomotiveMode.FAULT_LATCHED
    assert commands(node)[-1] == [0.0, 0.0, 0.0]
    node._rc_callback(module.Bool(data=False))
    assert node._state_machine.mode == LocomotiveMode.DOWN and kickstands(node)[-1] == "down"


def test_controller_requires_sticks_before_launch(controller):
    node, module, _ = controller
    node._sticks = None
    node._rc_callback(module.Bool(data=True))
    node._policy_tick()
    assert node._state_machine.mode == LocomotiveMode.DOWN and commands(node)[-1] == [0.0, 0.0, 0.0]
    event = json.loads(node._event_publisher.publish.call_args.args[0].data)
    assert event["readiness_reasons"] == ["rc_sticks_not_received"]
