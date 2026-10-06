import json
import math
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np

from trackstand_deploy.can_protocol import parse_current_feedback
from trackstand_deploy.telemetry import MOTOR_CONTROL_FIELDS


@pytest.fixture
def control_node(node_module):
    module = node_module("motor_io")
    module.time = SimpleNamespace(monotonic=lambda: 100.0, time=lambda: 123456.0)
    node = object.__new__(module.MotorIo)
    node._steering = module.ChannelState("steering", 0x102, 0x002, 1.0, -1.0, -1.0)
    node._drive = module.ChannelState("drive", 0x101, 0x001, 1.0, 1.0, -1.0)
    for motor in (node._steering, node._drive):
        motor.torque_constant_nm_per_a = 1.25
        motor.sample_monotonic = motor.velocity_sample_monotonic = 99.990
        motor.position_sample_monotonic = 99.989
    node._dry_run = False
    node._control_rate_hz = 200.0
    node._feedback_timeout_s = 0.05
    node._command_timeout_s = 0.08
    node._steering_limit_rad = math.pi / 2
    node._steering_fault_clear_rad = math.radians(5)
    node._requested_enable = True
    node._command_monotonic = 99.990
    node._requested_steering_target_rad = node._steering_target_rad = 0.05
    node._slewed_steering_target_rad = 0.0
    node._requested_drive_target_nm = node._drive_target_nm = 0.0
    node._outputs_active = False
    node._fault_latched = False
    node._fault_reason = None
    node._control_sequence = 0
    node._last_control_monotonic = 99.995
    node._last_event_state = None
    node._command_lock = threading.Lock()
    node._tx_bus = Mock()
    node._tx_bus.recv.return_value = None
    node._control_publisher = Mock()
    node._event_publisher = Mock()
    node._fault_publisher = Mock()
    node._control_dimension = SimpleNamespace()
    node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=123456000000000)))
    node.get_logger = Mock(return_value=Mock())
    return node


def record(node):
    data = node._control_publisher.publish.call_args.args[0].data
    assert len(data) == len(MOTOR_CONTROL_FIELDS)
    return dict(zip(MOTOR_CONTROL_FIELDS, data))


def test_normal_controller_terms_units_and_native_command_quantization(control_node):
    node = control_node
    node._steering.velocity_rad_s = 0.25
    node._drive.velocity_rad_s = 1.0
    node._control_tick()
    data = record(node)
    assert data["steering_p_torque_nm"] == pytest.approx(0.5)
    assert data["steering_d_torque_nm"] == pytest.approx(-0.05)
    assert data["steering_final_torque_nm"] == pytest.approx(0.45)
    assert data["drive_final_torque_nm"] == pytest.approx(-0.05)
    assert data["steering_encoded_current_a"] == pytest.approx(-0.36)
    assert data["steering_send_success"] == data["drive_send_success"] == 1.0
    assert data["cycle_interval_ms"] == pytest.approx(5.0)
    assert data["steering_position_age_ms"] == pytest.approx(11.0)
    sent = node._tx_bus.send.call_args_list
    assert sent[0].args[0].arbitration_id == 0x102
    assert parse_current_feedback(bytes(sent[0].args[0].data)) == pytest.approx(-0.36)
    assert node._fault_publisher.publish.call_args.args[0].data is False


def test_slew_saturation_and_requested_target_are_distinguished(control_node):
    node = control_node
    node._requested_steering_target_rad = 2.0
    node._steering_target_rad = math.pi / 2
    node._requested_drive_target_nm = 12.56
    node._drive_target_nm = 6.28
    node._steering.position_rad = -1.0
    node._drive.velocity_rad_s = -1.0
    node._control_tick()
    data = record(node)
    assert data["slewed_steering_target_rad"] == pytest.approx(23.9 / 200)
    for name in ("steering_slew_limited", "steering_torque_saturated", "drive_torque_saturated",
                 "steering_target_clipped", "drive_target_clipped"):
        assert data[name] == 1.0
    assert data["steering_final_torque_nm"] == pytest.approx(5.6)
    assert data["drive_final_torque_nm"] == pytest.approx(6.28)


def test_dry_run_never_reports_a_real_can_send(control_node):
    control_node._dry_run = True
    control_node._control_tick()
    data = record(control_node)
    assert data["gate_enabled"] == 1.0
    assert data["steering_send_attempted"] == data["steering_send_success"] == 0.0
    assert math.isfinite(data["steering_encoded_current_a"])
    control_node._tx_bus.send.assert_not_called()


def test_command_watchdog_disables_and_records_reason_without_fault(control_node):
    node = control_node
    node._outputs_active = True
    node._command_monotonic = 99.0
    node._control_tick()
    data = record(node)
    assert data["gate_enabled"] == 0.0 and data["disable_attempted"] == 1.0
    assert data["steering_disable_send_success"] == data["drive_disable_send_success"] == 1.0
    assert node._fault_publisher.publish.call_args.args[0].data is False
    event = json.loads(node._event_publisher.publish.call_args.args[0].data)
    assert event["disable_reasons"] == ["command_timeout"]
    assert [bytes(call.args[0].data)[0] for call in node._tx_bus.send.call_args_list] == [0xC0, 0xCF, 0xC0, 0xCF]


def test_stale_encoder_feedback_disables_and_reports_fault(control_node):
    control_node._steering.sample_monotonic = 99.0
    control_node._control_tick()
    assert record(control_node)["gate_enabled"] == 0.0
    assert control_node._fault_publisher.publish.call_args.args[0].data is True
    assert "steering_feedback_missing_or_stale" in json.loads(
        control_node._event_publisher.publish.call_args.args[0].data
    )["disable_reasons"]


@pytest.mark.parametrize("fail_send", [1, 2])
def test_can_failure_disables_in_same_tick_and_never_reports_healthy(control_node, fail_send):
    calls = []

    def send(frame):
        calls.append((frame.arbitration_id, bytes(frame.data)))
        if len(calls) == fail_send:
            raise RuntimeError("CAN injected failure")

    node = control_node
    node._tx_bus.send.side_effect = send
    node._control_tick()
    data = record(node)
    assert node._fault_latched
    assert data["gate_enabled"] == 0.0 and data["disable_attempted"] == 1.0
    assert node._fault_publisher.publish.call_args.args[0].data is True
    assert [payload[0] for _, payload in calls[-4:]] == [0xC0, 0xCF, 0xC0, 0xCF]
    if fail_send == 1:
        assert data["drive_send_attempted"] == 0.0
        assert math.isnan(data["drive_encoded_current_a"])


def test_disable_failure_still_attempts_other_frames_and_retries(control_node):
    node = control_node
    node._outputs_active = True
    node._requested_enable = False
    calls = []

    def send(frame):
        calls.append(frame)
        if len(calls) == 1:
            raise RuntimeError("zero-current send failed")

    node._tx_bus.send.side_effect = send
    node._control_tick()
    assert len(calls) == 4  # CF still attempted after zero-current failure.
    assert record(node)["steering_zero_send_success"] == 0.0
    assert node._outputs_active  # Retry pending, not confirmed physical enable.
    node._control_tick()
    assert len(calls) == 8 and not node._outputs_active


def test_transition_event_is_not_repeated_on_steady_ticks(control_node):
    node = control_node
    node._control_tick()
    node._control_tick()
    assert node._control_publisher.publish.call_count == 2
    assert node._event_publisher.publish.call_count == 1


def test_idle_ticks_do_not_fabricate_zero_current_commands(control_node):
    control_node._requested_enable = False
    control_node._control_tick()
    data = record(control_node)
    assert math.isnan(data["steering_encoded_current_a"])
    assert data["disable_attempted"] == 0.0


def test_normal_control_matches_previous_laws_for_500_seeded_targets(control_node):
    node = control_node
    random = np.random.default_rng(42)
    for _ in range(500):
        node._requested_steering_target_rad = float(random.uniform(-5, 5))
        node._steering_target_rad = float(np.clip(node._requested_steering_target_rad, -math.pi / 2, math.pi / 2))
        node._requested_drive_target_nm = float(random.uniform(-20, 20))
        node._drive_target_nm = float(np.clip(node._requested_drive_target_nm, -6.28, 6.28))
        node._steering.position_rad = float(random.uniform(-1, 1))
        node._steering.velocity_rad_s = float(random.uniform(-10, 10))
        node._drive.velocity_rad_s = float(random.uniform(-20, 20))
        expected_slew = node._slewed_steering_target_rad + float(np.clip(
            node._steering_target_rad - node._slewed_steering_target_rad, -23.9 / 200, 23.9 / 200,
        ))
        expected_steer = float(np.clip(10 * (expected_slew - node._steering.position_rad)
                                      - 0.2 * node._steering.velocity_rad_s, -5.6, 5.6))
        expected_drive = float(np.clip(node._drive_target_nm - 0.05 * node._drive.velocity_rad_s, -6.28, 6.28))
        node._control_tick()
        data = record(node)
        assert data["steering_final_torque_nm"] == expected_steer
        assert data["drive_final_torque_nm"] == expected_drive
        assert abs(data["steering_encoded_torque_nm"] - expected_steer) <= 0.001 * 1.25 / 2 + 1e-12
