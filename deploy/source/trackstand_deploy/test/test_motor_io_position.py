"""C2 protocol, unit, safety, and regression checks; no ROS or hardware I/O."""

import json
import math
import struct
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from trackstand_deploy.can_protocol import RADIANS_PER_COUNT, parse_current_feedback
from trackstand_deploy.telemetry import MotorControlPublishGate


@pytest.fixture
def position_node(node_module, monkeypatch):
    module = node_module("motor_io_position")
    clock = SimpleNamespace(value=100.0)
    module.time = SimpleNamespace(monotonic=lambda: clock.value, time=lambda: 123456.0)
    monkeypatch.setattr(sys.modules[module.MotorIo.__module__], "time", SimpleNamespace(
        monotonic=lambda: module.time.monotonic(), time=lambda: module.time.time(),
    ))
    node = object.__new__(module.MotorIoPosition)
    node._test_clock = clock
    node._steering = module.ChannelState("steering", 0x102, 0x002, 1.0, -1.0, -1.0)
    node._drive = module.ChannelState("drive", 0x101, 0x001, 1.0, 1.0, -1.0)
    for motor, constant in ((node._steering, 0.78), (node._drive, 0.3)):
        motor.torque_constant_nm_per_a = constant
        motor.zero_count = 1234
        motor.multi_turn_count = 1234
        motor.sample_monotonic = motor.velocity_sample_monotonic = 99.990
        motor.position_sample_monotonic = 99.989
    node._position_command_sign = node._steering.encoder_sign
    node._position_limits_configured = True
    node._position_speed_limit_raw, node._position_current_limit_raw = module.position_mode_limits(1.0, 0.78)
    node._position_limits_retry_at = 0.0
    node._dry_run = False
    node._can_interface = "can0"
    node._control_rate_hz = 200.0
    node._feedback_timeout_s = 0.05
    node._command_timeout_s = 0.08
    node._steering_limit_rad = math.pi / 2
    node._steering_fault_clear_rad = math.radians(5)
    node._requested_enable = True
    node._command_monotonic = 99.990
    node._requested_steering_target_rad = node._steering_target_rad = 0.05
    node._slewed_steering_target_rad = 0.0
    node._requested_drive_target_nm = node._drive_target_nm = 0.05
    node._outputs_active = False
    node._fault_latched = False
    node._fault_reason = None
    node._command_lock = threading.Lock()
    node._control_sequence = 0
    node._last_control_monotonic = 99.995
    node._last_event_state = None
    node._control_publish_gate = MotorControlPublishGate()
    node._tx_bus = Mock()
    node._tx_bus.recv.return_value = None
    node._control_publisher = Mock()
    node._event_publisher = Mock()
    node._fault_publisher = Mock()
    node._status_publisher = Mock()
    node._control_dimension = SimpleNamespace()
    node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: SimpleNamespace(
        nanoseconds=123456000000000, to_msg=lambda: "receipt_stamp",
    )))
    node.get_logger = Mock(return_value=Mock())
    return module, node


def telemetry(module, node):
    values = node._control_publisher.publish.call_args.args[0].data
    assert len(values) == len(module.MOTOR_CONTROL_FIELDS)
    return dict(zip(module.MOTOR_CONTROL_FIELDS, values))


def frames(node):
    return [(call.args[0].arbitration_id, bytes(call.args[0].data)) for call in node._tx_bus.send.call_args_list]


def set_position(node, angle_rad):
    motor = node._steering
    delta_count = round(motor.encoder_sign * angle_rad * motor.gear_ratio / RADIANS_PER_COUNT)
    motor.multi_turn_count = motor.zero_count + delta_count
    motor.position_rad = motor.encoder_sign * delta_count * RADIANS_PER_COUNT / motor.gear_ratio


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("gear", [1.0, 2.5])
@pytest.mark.parametrize("angle", [-math.pi / 2, 0.0, math.pi / 2])
@pytest.mark.parametrize("zero", [-100000, 1234, 100000])
def test_c2_signed_counts_units_and_quantization(position_node, sign, gear, angle, zero):
    module, _ = position_node
    payload = module.absolute_position_payload(angle, zero, gear, sign, math.pi / 2)
    assert len(payload) == 5 and payload[0] == 0xC2
    raw = struct.unpack("<i", payload[1:5])[0]
    assert raw == zero + round(sign * angle * gear / RADIANS_PER_COUNT)
    assert abs(sign * (raw - zero) * RADIANS_PER_COUNT / gear - angle) <= RADIANS_PER_COUNT / gear / 2


@pytest.mark.parametrize("angle,gear,sign", [
    (math.nan, 1.0, 1.0), (math.inf, 1.0, 1.0), (0.1, 0.0, 1.0),
    (0.1, math.inf, 1.0), (0.1, 1.0, 0.0),
])
def test_invalid_c2_inputs_rejected(position_node, angle, gear, sign):
    module, _ = position_node
    with pytest.raises(ValueError):
        module.absolute_position_payload(angle, 1234, gear, sign, math.pi / 2)


@pytest.mark.parametrize("zero", [None, True, 1.5, -2**31 - 1, 2**31])
def test_invalid_startup_zero_rejected(position_node, zero):
    module, _ = position_node
    with pytest.raises(ValueError):
        module.absolute_position_payload(0.0, zero, 1.0, -1.0, math.pi / 2)


@pytest.mark.parametrize("limit", [0.0, -1.0, math.nan, math.inf])
def test_invalid_c2_angle_limit_rejected(position_node, limit):
    module, _ = position_node
    with pytest.raises(ValueError):
        module.absolute_position_payload(0.0, 1234, 1.0, -1.0, limit)


def test_c2_int32_boundaries_and_offset_overflow(position_node):
    module, _ = position_node
    for raw in (-2**31, 2**31 - 1):
        payload = module.absolute_position_payload(0.0, raw, 1.0, 1.0, math.pi / 2)
        assert struct.unpack("<i", payload[1:])[0] == raw
    for raw, angle in ((2**31 - 1, RADIANS_PER_COUNT), (-2**31, -RADIANS_PER_COUNT)):
        with pytest.raises(OverflowError):
            module.absolute_position_payload(angle, raw, 1.0, 1.0, math.pi / 2)


def test_c2_overflowing_unit_conversion_rejected(position_node):
    module, _ = position_node
    with pytest.raises(OverflowError):
        module.absolute_position_payload(1.0, 0, 1e308, -1.0, math.pi / 2)


@pytest.mark.parametrize("gear", [1.0, 2.0, 8.0])
@pytest.mark.parametrize("constant", [0.3, 0.78, 1.25])
def test_firmware_limits_round_down_and_preserve_physical_caps(position_node, gear, constant):
    module, _ = position_node
    speed_raw, current_raw = module.position_mode_limits(gear, constant)
    assert speed_raw * 0.01 * (2 * math.pi / 60) / gear <= 23.9
    assert current_raw * 0.001 * constant <= 5.6
    assert (speed_raw + 1) * 0.01 * (2 * math.pi / 60) / gear > 23.9
    assert (current_raw + 1) * 0.001 * constant > 5.6


def test_normal_c2_replaces_only_steering_current_command(position_node):
    module, node = position_node
    node._drive.velocity_rad_s = 0.4
    node._control_tick()
    sent = frames(node)
    assert [(address, payload[0]) for address, payload in sent] == [(0x102, 0xC2), (0x101, 0xC0), (0x102, 0xA1)]
    assert struct.unpack("<i", sent[0][1][1:])[0] == node._steering.zero_count + round(-0.05 / RADIANS_PER_COUNT)
    expected_drive_nm = 0.05
    assert parse_current_feedback(sent[1][1]) == pytest.approx(round(-expected_drive_nm / 0.3 * 1000) * 0.001)
    data = telemetry(module, node)
    assert data["gate_enabled"] == 1.0
    assert data["drive_final_torque_nm"] == pytest.approx(expected_drive_nm)
    assert data["steering_send_success"] == data["drive_send_success"] == 1.0
    assert data["steering_current_query_send_success"] == 1.0
    for name in ("steering_p_torque_nm", "steering_d_torque_nm", "steering_final_torque_nm",
                 "steering_current_command_a", "steering_encoded_current_a", "steering_encoded_torque_nm"):
        assert math.isnan(data[name])


def test_held_c2_target_is_identical_regardless_of_feedback_position(position_node):
    module, node = position_node
    node._control_tick()
    first_target = telemetry(module, node)["steering_c2_target_count"]
    set_position(node, 0.025)
    node._control_tick()
    second_target = telemetry(module, node)["steering_c2_target_count"]
    assert second_target == first_target
    set_position(node, 0.05)
    node._control_tick()
    assert telemetry(module, node)["steering_c2_target_count"] == first_target
    assert len({payload for _, payload in frames(node) if payload[0] == 0xC2}) == 1
    assert node._requested_steering_target_rad == 0.05


def test_startup_encoder_zero_is_added_to_absolute_command(position_node):
    module, node = position_node
    set_position(node, -0.04)
    node._control_tick()
    original_target = telemetry(module, node)["steering_c2_target_count"]
    node._steering.zero_count += 100000
    node._steering.multi_turn_count += 100000
    node._test_clock.value += 0.02
    node._control_tick()
    data = telemetry(module, node)
    assert data["steering_c2_zero_count"] == node._steering.zero_count
    assert data["steering_c2_target_count"] == original_target + 100000
    assert abs(data["steering_c2_encoded_target_rad"] - 0.05) <= RADIANS_PER_COUNT / 2


@pytest.mark.parametrize("target", [-0.3, 0.0, 0.3])
def test_native_positive_turns_left_and_policy_positive_turns_right(position_node, target):
    module, node = position_node
    node._steering_target_rad = node._slewed_steering_target_rad = target
    node._control_tick()
    data = telemetry(module, node)
    assert data["steering_c2_target_offset_count"] == round(-target / RADIANS_PER_COUNT)
    assert data["steering_c2_target_count"] == node._steering.zero_count + round(-target / RADIANS_PER_COUNT)
    assert data["steering_position_command_sign"] == -1.0


def test_startup_zero_is_retained_across_disable_and_reenable(position_node):
    module, node = position_node
    node._control_tick()
    zero = node._steering.zero_count
    target = telemetry(module, node)["steering_c2_target_count"]
    node._requested_enable = False
    set_position(node, 0.02)
    node._control_tick()
    assert node._steering.zero_count == zero
    node._requested_enable = True
    node._control_tick()
    assert node._steering.zero_count == zero
    assert telemetry(module, node)["steering_c2_target_count"] == target


def test_absolute_target_overflow_latches_and_disables_both_motors(position_node):
    module, node = position_node
    node._steering.zero_count = node._steering.multi_turn_count = 2**31 - 1
    node._steering_target_rad = -0.05
    node._control_tick()
    assert node._fault_latched and node._fault_reason == "steering_c2_send_failed"
    data = telemetry(module, node)
    assert data["gate_enabled"] == data["steering_send_attempted"] == data["drive_send_attempted"] == 0.0
    assert [payload[0] for _, payload in frames(node)] == [0xC0, 0xCF, 0xC0, 0xCF]


def test_status_identifies_c2_and_reports_native_startup_zero(position_node):
    _, node = position_node
    node._publish_status()
    status = json.loads(node._status_publisher.publish.call_args.args[0].data)
    assert status["steering_control_mode"] == "C2_absolute_position"
    assert status["steering_position_command_sign"] == -1.0
    assert status["steering_c2_zero_count"] == node._steering.zero_count
    assert "c3_current_reference_verified" not in status


def test_v1_position_clip_is_held_without_a_second_motor_rate_slew(position_node):
    module, node = position_node
    node._command_callback(SimpleNamespace(data=[1.0, 2.0, 10.0]))
    node._control_tick()
    data = telemetry(module, node)
    assert data["bounded_steering_target_rad"] == 1.57
    assert data["bounded_drive_target_nm"] == 6.28
    assert data["slewed_steering_target_rad"] == pytest.approx(1.57)
    assert data["steering_slew_limited"] == 0.0
    assert data["drive_final_torque_nm"] == 6.28


@pytest.mark.parametrize("gear", [1.1, 1.3, 2.7])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_c2_quantized_target_never_exceeds_angle_limit(position_node, gear, direction):
    module, node = position_node
    node._steering.gear_ratio = gear
    node._steering_target_rad = direction * math.pi / 2
    node._slewed_steering_target_rad = direction * math.pi / 2
    node._control_tick()
    data = telemetry(module, node)
    assert abs(data["steering_c2_encoded_target_rad"]) <= math.pi / 2
    assert abs(abs(data["steering_c2_encoded_target_rad"]) - math.pi / 2) < RADIANS_PER_COUNT / gear


def test_dry_run_calculates_c2_but_sends_no_frames(position_node):
    module, node = position_node
    node._dry_run = True
    node._position_limits_configured = False
    node._control_tick()
    data = telemetry(module, node)
    assert data["gate_enabled"] == 1.0
    assert math.isfinite(data["steering_c2_target_count"])
    assert data["steering_position_limits_configured"] == 0.0
    assert data["steering_send_attempted"] == data["steering_send_success"] == 0.0
    assert data["steering_current_query_send_attempted"] == data["steering_current_query_send_success"] == 0.0
    node._tx_bus.send.assert_not_called()


def test_c2_no_longer_requires_a_c3_reference_assertion(position_node):
    module, node = position_node
    assert not hasattr(node, "_c3_current_reference_verified")
    node._control_tick()
    assert telemetry(module, node)["gate_enabled"] == 1.0
    assert frames(node)[0][1][0] == 0xC2


@pytest.mark.parametrize("case", ["command_timeout", "steering_stale", "drive_stale", "disabled", "missing_b0"])
def test_existing_gates_disable_both_motors(position_node, case):
    module, node = position_node
    node._outputs_active = True
    if case == "command_timeout":
        node._command_monotonic = 99.0
    elif case == "steering_stale":
        node._steering.sample_monotonic = 99.0
    elif case == "drive_stale":
        node._drive.sample_monotonic = 99.0
    elif case == "disabled":
        node._requested_enable = False
    else:
        node._drive.torque_constant_nm_per_a = None
    node._control_tick()
    assert telemetry(module, node)["gate_enabled"] == 0.0
    assert [(address, payload[0]) for address, payload in frames(node)] == [
        (0x102, 0xC0), (0x102, 0xCF), (0x101, 0xC0), (0x101, 0xCF),
    ]


@pytest.mark.parametrize("motor,field,value,reason", [
    ("_steering", "position_rad", 2.0, "steering_position_limit"),
    ("_steering", "velocity_rad_s", 27.0, "steering_speed_limit"),
    ("_drive", "velocity_rad_s", 70.0, "drive_speed_limit"),
])
def test_existing_angle_and_speed_faults_remain(position_node, motor, field, value, reason):
    module, node = position_node
    setattr(getattr(node, motor), field, value)
    node._control_tick()
    assert node._fault_latched and node._fault_reason == reason
    assert telemetry(module, node)["gate_enabled"] == 0.0
    node._tx_bus.send.assert_not_called()


def test_c2_safety_guard_rejects_wrapped_extra_revolution(position_node):
    module, node = position_node
    node._steering.multi_turn_count += 16384
    node._control_tick()
    assert node._fault_latched
    assert telemetry(module, node)["steering_send_attempted"] == 0.0
    assert all(payload[0] != 0xC2 for _, payload in frames(node))


@pytest.mark.parametrize("failed_frame", [1, 2, 3])
def test_c2_rear_or_a1_send_error_disables_both_in_same_tick(position_node, failed_frame):
    module, node = position_node
    submitted = []

    def send(frame):
        submitted.append(bytes(frame.data))
        if len(submitted) == failed_frame:
            raise RuntimeError("injected CAN error")

    node._tx_bus.send.side_effect = send
    node._control_tick()
    data = telemetry(module, node)
    assert node._fault_latched and data["gate_enabled"] == 0.0
    assert node._fault_publisher.publish.call_args.args[0].data is True
    assert [payload[0] for payload in submitted[-4:]] == [0xC0, 0xCF, 0xC0, 0xCF]
    if failed_frame == 1:
        assert data["drive_send_attempted"] == 0.0
    elif failed_frame == 3:
        assert data["steering_send_success"] == data["drive_send_success"] == 1.0
        assert data["steering_current_query_send_success"] == 0.0


def mock_limit_queries(module, fail_at=None, wrong_echo=False):
    submitted = []
    configured = {}

    def query(bus, request, feedback_id, timeout_s):
        assert feedback_id == 0x002 and request.arbitration_id == 0x102
        assert timeout_s == 0.008
        payload = bytes(request.data)
        submitted.append(payload)
        command = payload[0]
        if len(payload) == 5:
            configured[command] = struct.unpack("<I", payload[1:])[0]
        if len(submitted) == fail_at:
            return None
        return bytes([0xA1 if wrong_echo else command]) + struct.pack("<I", configured[command])

    module.query = query
    return submitted


def test_c2_limits_are_set_and_read_back_before_motion(position_node):
    module, node = position_node
    node._position_limits_configured = False
    submitted = mock_limit_queries(module)
    node._control_tick()
    assert [(payload[0], len(payload)) for payload in submitted] == [(0xB2, 5), (0xB2, 1), (0xB3, 5), (0xB3, 1)]
    assert node._position_limits_configured
    assert frames(node)[0][1][0] == 0xC2
    assert telemetry(module, node)["gate_enabled"] == 1.0


@pytest.mark.parametrize("failed_reply", [1, 2, 3, 4])
def test_unconfirmed_limits_gate_off_and_retry_only_after_one_second(position_node, failed_reply):
    module, node = position_node
    node._position_limits_configured = False
    submitted = mock_limit_queries(module, fail_at=failed_reply)
    node._control_tick()
    assert not node._position_limits_configured
    assert telemetry(module, node)["gate_enabled"] == 0.0
    assert node._fault_publisher.publish.call_args.args[0].data is True
    assert len(submitted) == failed_reply
    node._control_tick()
    assert len(submitted) == failed_reply
    node._tx_bus.send.assert_not_called()


def test_wrong_limit_echo_is_not_confirmed(position_node):
    module, node = position_node
    node._position_limits_configured = False
    mock_limit_queries(module, wrong_echo=True)
    node._control_tick()
    assert not node._position_limits_configured
    assert telemetry(module, node)["gate_enabled"] == 0.0


def test_initial_limit_setup_rechecks_command_age_and_feedback_age(position_node):
    module, node = position_node
    node._position_limits_configured = False
    clock = SimpleNamespace(now=100.0)
    module.time = SimpleNamespace(monotonic=lambda: clock.now)
    submitted = mock_limit_queries(module)
    original_query = module.query

    def delayed_query(*args):
        clock.now += 0.025
        return original_query(*args)

    module.query = delayed_query
    node._control_tick()
    data = telemetry(module, node)
    assert len(submitted) == 4 and node._position_limits_configured
    assert data["command_ready"] == data["gate_enabled"] == 0.0
    assert data["command_age_ms"] == pytest.approx(110.0)
    assert data["gate_monotonic_time_s"] == pytest.approx(100.1)
    node._tx_bus.send.assert_not_called()


def test_safety_position_feedback_expiring_before_send_disables(position_node):
    module, node = position_node
    node._steering.position_sample_monotonic = 99.94
    node._control_tick()
    assert node._fault_latched
    assert telemetry(module, node)["gate_enabled"] == 0.0
    assert all(payload[0] != 0xC2 for _, payload in frames(node))


def test_zero_count_absent_gates_off_without_absolute_command(position_node):
    module, node = position_node
    node._steering.zero_count = None
    node._control_tick()
    assert telemetry(module, node)["gate_enabled"] == 0.0
    node._tx_bus.send.assert_not_called()


def test_a1_measured_torque_is_preserved_but_c2_reply_is_not_torque(position_node):
    _, node = position_node
    common = dict(arbitration_id=0x002, is_extended_id=False, is_remote_frame=False, is_error_frame=False)
    node._update_torque_feedback(SimpleNamespace(data=b"\xc2" + bytes(6), **common))
    assert math.isnan(node._steering.torque_nm) and node._steering.torque_sample_monotonic == 0.0
    node._update_torque_feedback(SimpleNamespace(data=b"\xa1" + struct.pack("<i", -100), **common))
    assert node._steering.torque_nm == pytest.approx(0.078)
    assert node._steering.torque_sample_monotonic == 100.0


def test_nonsteering_methods_are_shared_not_copied(position_node):
    module, _ = position_node
    for name in ("_command_callback", "_send_torque", "_disable_outputs", "_publish_state",
                 "_receive_torque_feedback", "_update_torque_feedback", "_drain_tx", "destroy_node",
                 "_publish_control_telemetry"):
        assert getattr(module.MotorIoPosition, name) is getattr(module.MotorIo, name), name


@pytest.mark.parametrize("dry_run", [True, False])
def test_constructor_registers_correct_parameters_topics_and_timers(node_module, dry_run):
    class NodeStub:
        def __init__(self, name):
            assert name == "trackstand_motor_io"
            self.params = {"dry_run": dry_run}
            self.publishers = {}
            self.timers = []

        def declare_parameter(self, name, value, descriptor=None):
            assert descriptor.read_only
            self.params.setdefault(name, value)

        def get_parameter(self, name):
            return SimpleNamespace(value=self.params[name])

        def create_publisher(self, message_type, topic, qos):
            self.publishers[topic] = Mock()
            return self.publishers[topic]

        def create_subscription(self, *args):
            return Mock()

        def create_timer(self, interval, callback):
            self.timers.append((interval, callback))
            return Mock()

        def get_logger(self):
            return Mock()

        def destroy_node(self):
            pass

    sys.modules["rclpy.node"].Node = NodeStub
    module = node_module("motor_io_position")
    tx_bus, torque_bus = Mock(), Mock()
    module.can.interface = SimpleNamespace(Bus=Mock(side_effect=[tx_bus, torque_bus]))
    module.threading = SimpleNamespace(Thread=Mock, Lock=threading.Lock)
    sys.modules[module.MotorIo.__module__].threading = module.threading
    node = module.MotorIoPosition()
    assert node._position_command_sign == node._steering.encoder_sign == -1.0
    assert "c3_current_reference_verified" not in node.params
    assert node._control_dimension.label.startswith("motor_control_position_v2:")
    assert node._control_dimension.label.split(":", 1)[1].split(",") == list(module.MOTOR_CONTROL_FIELDS)
    assert [interval for interval, _ in node.timers] == [0.005, 0.005, 0.5]
    assert node._steering.host_id == 0x102 and node._drive.host_id == 0x101
    assert node._command_timeout_s == 0.08 and node._feedback_timeout_s == 0.05
    assert "/trackstand/motor_state" in node.publishers
    assert "/trackstand/motor_torque" not in node.publishers
    if dry_run:
        tx_bus.send.assert_not_called()
    else:
        assert [bytes(call.args[0].data)[0] for call in tx_bus.send.call_args_list] == [0xC0, 0xCF, 0xC0, 0xCF]
    node.destroy_node()
    torque_bus.shutdown.assert_called_once()
    tx_bus.shutdown.assert_called_once()


def test_rear_law_matches_v1_pure_feedforward_for_500_seeded_inputs(position_node):
    module, node = position_node
    rng = np.random.default_rng(42)
    for index in range(500):
        node._test_clock.value = 100.0 + index * 0.02
        node._command_monotonic = node._test_clock.value
        for motor in (node._steering, node._drive):
            motor.sample_monotonic = motor.position_sample_monotonic = node._test_clock.value
        node._drive_target_nm = float(rng.uniform(-6.28, 6.28))
        node._drive.velocity_rad_s = float(rng.uniform(-20.0, 20.0))
        expected = float(np.clip(node._drive_target_nm, -6.28, 6.28))
        node._control_tick()
        assert telemetry(module, node)["drive_final_torque_nm"] == expected


@pytest.mark.parametrize("dry_run", [False, True])
def test_fifty_hz_telemetry_keeps_two_hundred_hz_c2_c0_and_faults(position_node, dry_run):
    module, node = position_node
    node._dry_run = dry_run
    for index in range(200):
        node._test_clock.value = 100 + index * .005
        node._command_monotonic = node._test_clock.value
        for motor in (node._steering, node._drive):
            motor.sample_monotonic = motor.position_sample_monotonic = node._test_clock.value
        node._control_tick()
    assert node._control_publisher.publish.call_count == 50
    assert node._fault_publisher.publish.call_count == node._control_sequence == 200
    sent = frames(node)
    assert sum(payload[0] == 0xC2 for _, payload in sent) == (0 if dry_run else 200)
    assert sum(payload[0] == 0xC0 for _, payload in sent) == (0 if dry_run else 200)
    assert telemetry(module, node)["cycle_interval_ms"] == pytest.approx(5.0)


@pytest.mark.parametrize("cause", ["rc_down", "stale_feedback", "can_failure"])
def test_unsampled_c2_tick_still_disables_and_reports_faults_and_events(position_node, cause):
    _, node = position_node
    node._control_tick()
    node._tx_bus.reset_mock()
    node._test_clock.value += .005
    if cause == "rc_down":
        node._requested_enable = False
    elif cause == "stale_feedback":
        node._steering.sample_monotonic = 99.0
    else:
        node._tx_bus.send.side_effect = [RuntimeError("injected failure"), None, None, None, None]
    node._control_tick()
    assert node._control_publisher.publish.call_count == 1
    assert node._fault_publisher.publish.call_count == 2
    assert node._fault_publisher.publish.call_args.args[0].data is (cause != "rc_down")
    assert [payload[0] for _, payload in frames(node)[-4:]] == [0xC0, 0xCF, 0xC0, 0xCF]
    event = json.loads(node._event_publisher.publish.call_args.args[0].data)
    assert event["sequence"] == 2 and not event["gate_enabled"]


def test_shutdown_is_idempotent_after_ros_signal(position_node, monkeypatch):
    module, _ = position_node
    executors = ModuleType("rclpy.executors")
    executors.ExternalShutdownException = type("ExternalShutdownException", (Exception,), {})
    monkeypatch.setitem(sys.modules, "rclpy.executors", executors)
    ros = sys.modules["rclpy"]
    ros.init = Mock()
    ros.spin = Mock(side_effect=executors.ExternalShutdownException())
    ros.try_shutdown = Mock()
    fake_node = Mock()
    module.MotorIoPosition = Mock(return_value=fake_node)
    module.main()
    fake_node.destroy_node.assert_called_once()
    ros.try_shutdown.assert_called_once()
