"""Deterministic MIT protocol, deployment gate and feedback tests; no CAN hardware."""

from collections import deque
import json
import math
import struct
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trackstand_deploy.can_protocol import RADIANS_PER_COUNT, parse_current_feedback
from trackstand_deploy.mit_protocol import (
    MitRanges, f0_with_position, mit_motion_payload, parse_f0, parse_f1, parse_mit_motion,
)


def feedback(ranges, position=0.0, velocity=0.0, torque=0.0, status=0):
    def enc(value, maximum, bits):
        return round((value + maximum) * ((1 << bits) - 1) / (2 * maximum))
    p, v, t = enc(position, ranges.position_rad, 16), enc(velocity, ranges.velocity_rad_s, 12), enc(torque, ranges.torque_nm, 12)
    return bytes((0xF1, p >> 8, p & 255, v >> 4, ((v & 15) << 4) | (t >> 8), t & 255, status))


def test_mit_wire_format_agrees_with_proven_experiment():
    # round-to-nearest as in steering_sine_mit_io, with the actual experiment ranges.
    payload = mit_motion_payload(1.0, MitRanges(3, 45, 18), 10, 0.2)
    assert payload.hex() == "aaaa8000520a4800"
    assert ((payload[3] & 15) << 8 | payload[4]) == 82
    assert (payload[5] << 4 | payload[6] >> 4) == 164
    decoded = parse_mit_motion(payload, MitRanges(3, 45, 18))
    assert decoded.position_rad == pytest.approx(1.0)
    assert decoded.velocity_rad_s == pytest.approx(45 / 4095)
    assert decoded.feedforward_torque_nm == pytest.approx(18 / 4095)
    zero_gains = mit_motion_payload(0, MitRanges(3, 45, 18), 0, 0)
    assert zero_gains[3] & 15 == zero_gains[4] == zero_gains[5] == zero_gains[6] >> 4 == 0


def test_f0_uses_motor_settings_and_f1_decodes_native_units():
    raw = bytes([0xF0]) + struct.pack("<HHH", 955, 4500, 1800)
    changed = f0_with_position(raw, 3.0)
    assert changed == bytes([0xF0]) + struct.pack("<HHH", 30, 4500, 1800)
    ranges = parse_f0(changed)
    state = parse_f1(feedback(ranges, 0.25, -2.0, 1.0, 3), ranges)
    assert state.position_rad == pytest.approx(0.25, abs=3 / 65535)
    assert state.velocity_rad_s == pytest.approx(-2, abs=45 / 4095)
    assert state.torque_nm == pytest.approx(1, abs=18 / 4095)
    assert state.mode_active and state.system_fault


@pytest.mark.parametrize("payload", [b"", b"\xf0", b"\xf0" + b"\x00" * 6, b"\xf1" + b"\x01" * 6])
def test_bad_f0_rejected(payload):
    with pytest.raises(ValueError):
        parse_f0(payload)


@pytest.mark.parametrize("position,kp,kd", [(4, 10, .2), (math.nan, 10, .2), (0, -1, .2), (0, 501, .2), (0, 10, 6)])
def test_bad_motion_rejected(position, kp, kd):
    with pytest.raises(ValueError):
        mit_motion_payload(position, MitRanges(3, 45, 18), kp, kd)


@pytest.fixture
def mit_node(node_module, monkeypatch):
    module = node_module("motor_mit_io")
    clock = SimpleNamespace(value=100.0)
    module.time = SimpleNamespace(monotonic=lambda: clock.value, time=lambda: 123456.0)
    # Inherited methods use their own module globals; share this deterministic clock.
    monkeypatch.setattr(__import__("trackstand_deploy.motor_io", fromlist=["time"]), "time", module.time)
    node = object.__new__(module.MotorMitIo)
    node._steering = module.ChannelState("steering", 0x102, 2, 1, -1, -1)
    node._drive = module.ChannelState("drive", 0x101, 1, 1, 1, -1)
    for motor in (node._steering, node._drive):
        motor.zero_count = 0
        motor.torque_constant_nm_per_a = 0.78 if motor is node._steering else 0.3
        motor.sample_monotonic = motor.position_sample_monotonic = motor.velocity_sample_monotonic = 99.99
    node._dry_run = False
    node._control_rate_hz = node._poll_rate_hz = 200
    node._feedback_timeout_s = 0.05
    node._command_timeout_s = 0.08
    node._steering_limit_rad = math.pi / 2
    node._steering_fault_clear_rad = math.radians(5)
    node._requested_enable = True
    node._command_monotonic = 99.99
    node._requested_steering_target_rad = node._steering_target_rad = 0.05
    node._slewed_steering_target_rad = 0
    node._requested_drive_target_nm = node._drive_target_nm = 0
    node._outputs_active = node._fault_latched = node._hard_fault = False
    node._fault_reason = None
    node._control_sequence = 0
    node._last_control_monotonic = 99.995
    node._last_event_state = None
    node._command_lock = threading.Lock()
    node._mit_ranges = MitRanges(3, 45, 18)
    node._mit_zero_rad = 0.002
    node._mit_feedback = None
    node._mit_encoded_command = None
    node._mit_send_interval_ms = math.nan
    node._mit_config_ready = True
    node._mit_kp, node._mit_kd = 10, .2
    node._mit_f1_at = 99.99
    node._mit_last_sent_at = node._mit_first_sent_at = node._next_mit_at = 0
    node._mit_sent_this_enable = False
    node._disable_wait = {}
    node._disable_confirmed = True
    node._disable_deadline_at = node._next_disable_retry_at = 0
    node._last_disable_errors = ()
    node._pending = None
    node._poll_jobs = deque()
    node._next_poll_at = node._next_b0_at = node._next_f0_at = 0
    node._last_steering_current_query_at = 0
    node._running = True
    node._tx_bus = Mock()
    node._tx_bus.receive.return_value = None
    for name in ("control", "event", "fault", "status", "state", "torque"):
        setattr(node, f"_{name}_publisher", Mock())
    node._control_dimension = SimpleNamespace()
    node._last_published_sample_times = (0, 0)
    node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: SimpleNamespace(
        nanoseconds=123456000000000, to_msg=lambda: "receipt_stamp")))
    node.get_logger = Mock(return_value=Mock())
    return node, module, clock


def record(node, module):
    return dict(zip(module.MOTOR_CONTROL_FIELDS, node._control_publisher.publish.call_args.args[0].data))


def frame(module, motor, payload, stamp=None):
    return module.CanReply(motor.feedback_id, payload, False, 0,
                           module.time.monotonic() if stamp is None else stamp)


def confirm_disable(node, module):
    for motor in (node._steering, node._drive):
        node._consume(frame(module, motor, bytes([0xCF]) + bytes(7)), module.time.monotonic())


def test_mit_50hz_drive_200hz_and_exact_motor_signs(mit_node):
    node, module, clock = mit_node
    node._drive.velocity_rad_s = 1.0
    for i in range(9):
        clock.value = 100.0 + i * 0.005
        for motor in (node._steering, node._drive):
            motor.sample_monotonic = clock.value - .001
        node._control_tick()
    sends = node._tx_bus.send.call_args_list
    mit = [call for call in sends if call.args[0] == 0x502]
    drive = [call for call in sends if call.args[0] == 0x101]
    assert len(mit) == 3 and len(drive) == 9
    expected = mit_motion_payload(0.002 - 0.05, node._mit_ranges, 10, .2)
    assert mit[0].args[1] == expected
    assert parse_current_feedback(drive[0].args[1]) == pytest.approx(.167)
    data = record(node, module)
    assert data["drive_final_torque_nm"] == pytest.approx(-.05)
    assert math.isnan(data["steering_final_torque_nm"])
    assert math.isnan(data["steering_current_command_a"])
    assert node._mit_zero_rad == .002


def test_delayed_control_never_bursts_catchup_mit(mit_node):
    node, _, clock = mit_node
    node._control_tick()
    clock.value += .031
    node._control_tick()
    node._control_tick()
    assert sum(call.args[0] == 0x502 for call in node._tx_bus.send.call_args_list) == 2


def test_mit_deadline_dispatch_does_not_add_a_rear_command(mit_node):
    node, _, clock = mit_node
    node._control_tick()
    clock.value += .0201
    node._tx_bus.reset_mock()
    node._control_tick(send_drive=False)
    assert [call.args[0] for call in node._tx_bus.send.call_args_list] == [0x502]


def test_mit_slew_uses_50hz_target_interval(mit_node):
    node, module, _ = mit_node
    node._steering_target_rad = node._requested_steering_target_rad = 1.57
    node._control_tick()
    assert record(node, module)["slewed_steering_target_rad"] == pytest.approx(23.9 / 50)
    assert record(node, module)["steering_slew_limited"] == 1


@pytest.mark.parametrize("target_sign", [-1, 1])
def test_encoded_position_rounds_inside_ninety_degree_limit(mit_node, target_sign):
    node, _, _ = mit_node
    result, _ = node._send_mit(target_sign * node._steering_limit_rad)
    assert result.success
    joint_target = node._steering.encoder_sign * (node._mit_encoded_command.position_rad - node._mit_zero_rad)
    assert abs(joint_target) <= node._steering_limit_rad
    assert abs(joint_target - target_sign * node._steering_limit_rad) <= 6 / 65535


def test_dry_run_never_sends_even_if_enabled(mit_node):
    node, module, _ = mit_node
    node._dry_run = True
    node._control_tick()
    node._disable_outputs(force=True)
    node._tx_bus.send.assert_not_called()
    assert record(node, module)["gate_enabled"] == 1
    assert record(node, module)["steering_send_success"] == 0


def test_incomplete_mit_preflight_gates_both_motors(mit_node):
    node, module, _ = mit_node
    node._mit_config_ready = False
    node._control_tick()
    node._tx_bus.send.assert_not_called()
    assert record(node, module)["gate_enabled"] == 0
    event = json.loads(node._event_publisher.publish.call_args.args[0].data)
    assert "mit_preflight_incomplete" in event["disable_reasons"]


def test_rc_down_uses_cf_steering_and_zero_cf_drive_preserves_zero(mit_node):
    node, module, _ = mit_node
    node._control_tick()
    node._tx_bus.reset_mock()
    node._requested_enable = False
    node._control_tick()
    assert [(call.args[0], call.args[1][0]) for call in node._tx_bus.send.call_args_list] == [
        (0x102, 0xCF), (0x101, 0xC0), (0x101, 0xCF),
    ]
    assert node._mit_zero_rad == .002 and node._steering.zero_count == 0
    assert node._outputs_active and node._disable_wait
    confirm_disable(node, module)
    assert not node._outputs_active and not node._mit_sent_this_enable


def test_f1_watchdog_is_not_refreshed_by_sending_commands(mit_node):
    node, module, clock = mit_node
    node._control_tick()
    initial_f1 = node._mit_f1_at
    for seconds in (.02, .04, .055):
        clock.value = 100 + seconds
        for motor in (node._steering, node._drive):
            motor.sample_monotonic = clock.value - .001
        node._command_monotonic = clock.value - .001
        node._control_tick()
    assert node._mit_f1_at == initial_f1
    assert node._fault_latched and node._hard_fault
    assert node._fault_reason == "steering_mit_feedback_timeout"
    confirm_disable(node, module)
    assert not node._outputs_active


@pytest.mark.parametrize("status,reason", [(3, "steering_mit_system_fault"), (0, "steering_mit_unexpected_exit")])
def test_mit_status_fault_disables_both_on_next_tick(mit_node, status, reason):
    node, module, _ = mit_node
    node._control_tick()
    node._consume(frame(module, node._steering, feedback(node._mit_ranges, status=status)), 100.001)
    node._control_tick()
    assert node._fault_reason == reason and node._hard_fault
    confirm_disable(node, module)
    assert not node._outputs_active


def test_faulted_inactive_f1_prevents_first_enable(mit_node):
    node, module, _ = mit_node
    node._consume(frame(module, node._steering, feedback(node._mit_ranges, status=2)), 100)
    node._control_tick()
    node._tx_bus.send.assert_not_called()
    assert node._hard_fault


def test_can_failure_never_sends_active_drive_after_failed_steering(mit_node):
    node, _, _ = mit_node
    node._tx_bus.send.side_effect = [OSError("bus off"), None, None, None]
    node._control_tick()
    assert [call.args[0] for call in node._tx_bus.send.call_args_list] == [0x502, 0x102, 0x101, 0x101]
    assert [call.args[1][0] for call in node._tx_bus.send.call_args_list[1:]] == [0xCF, 0xC0, 0xCF]
    assert node._fault_latched


def test_disable_attempts_all_motors_and_retries_failed_send(mit_node):
    node, module, clock = mit_node
    node._outputs_active = True
    node._tx_bus.send.side_effect = [OSError("bad steer"), None, None]
    result = node._disable_outputs()
    assert len(result.errors) == 1 and node._outputs_active
    node._tx_bus.send.side_effect = None
    clock.value += .021
    assert not node._disable_outputs().errors
    assert node._outputs_active
    confirm_disable(node, module)
    assert not node._outputs_active


def test_wrapped_observation_unwrapped_safety_and_fresh_pair(mit_node):
    node, module, _ = mit_node
    motor = node._steering
    a3 = bytes([0xA3]) + struct.pack("<Hi", 0, 16384)
    node._pending = (motor, 0xA3, 99.999, 100.002, 7)
    node._consume(frame(module, motor, a3), 100.0)
    assert motor.sample_monotonic == 99.99
    assert motor.position_sample_monotonic == 99.99  # Paired state still refers to the older sample.
    node._pending = (motor, 0xA2, 100, 100.002, 5)
    node._consume(frame(module, motor, bytes([0xA2]) + struct.pack("<i", 0), stamp=100.001), 100.001)
    assert motor.position_rad == pytest.approx(0)
    assert motor._unwrapped_rad == pytest.approx(-2 * math.pi)
    assert motor.sample_monotonic == 100
    assert motor.position_sample_monotonic == 100 and motor.velocity_sample_monotonic == 100.001
    node._control_tick()
    assert node._fault_reason == "steering_position_limit"


def test_actual_current_torque_has_receipt_stamp_and_no_clipping(mit_node):
    node, module, _ = mit_node
    motor = node._steering
    node._consume(frame(module, motor, bytes([0xA1]) + struct.pack("<i", 20000)), 100.01)
    sample = node._torque_publisher.publish.call_args.args[0]
    assert sample.header.stamp == "receipt_stamp"
    assert sample.name == ["steering_joint"]
    assert sample.effort == pytest.approx([-15.6])


def test_changed_f0_or_b1_and_other_can_writer_rejected(mit_node):
    node, module, _ = mit_node
    for bad in (module.CanReply(0x102, b"\xa3", True, 0, 100),
                frame(module, node._steering, b"\xb1\x00\x00"),
                frame(module, node._steering, bytes([0xF0]) + struct.pack("<HHH", 40, 4500, 1800))):
        with pytest.raises(RuntimeError):
            node._consume(bad, 100)


def test_status_unknown_values_are_json_null(mit_node):
    node, _, _ = mit_node
    node._mit_f1_at = 0
    node._publish_status()
    raw = node._status_publisher.publish.call_args.args[0].data
    assert "NaN" not in raw
    status = json.loads(raw)
    assert status["steering_torque_age_ms"] is None
    assert status["mit_f1_age_ms"] is None


def test_normal_fault_clears_only_disabled_fresh_and_inside_five_degrees(mit_node):
    node, _, _ = mit_node
    node._fault("steering_position_limit")
    node._requested_enable = False
    node._steering.position_rad = .1
    node._control_tick()
    assert node._fault_latched
    node._steering.position_rad = 0
    node._steering.sample_monotonic = 99
    node._control_tick()
    assert node._fault_latched
    node._steering.sample_monotonic = 99.99
    node._control_tick()
    assert not node._fault_latched


def test_hard_fault_reason_survives_later_limits_and_disable_errors(mit_node):
    node, _, _ = mit_node
    node._fault("steering_origin_changed", hard=True)
    node._fault("steering_position_limit")
    node._requested_enable = False
    node._control_tick()
    assert node._fault_reason == "steering_origin_changed" and node._fault_latched


def test_powered_startup_sets_origin_once_preserves_velocity_torque_ranges(mit_node):
    node, module, _ = mit_node
    node._mit_config_ready = False
    node._requested_enable = False
    ranges = MitRanges(3, 45, 18)
    f0 = bytes([0xF0]) + struct.pack("<HHH", 30, 4500, 1800)
    answers = {0xCF: bytes([0xCF]) + bytes(7), 0xB1: b"\xb1\x00\x00", 0xF0: f0,
               0xA3: b"\xa3" + bytes(6), 0xF1: feedback(ranges),
               0xB0: b"\xb0\x00" + struct.pack("<f", .78) + b"\x00"}
    node._query = Mock(side_effect=lambda motor, opcode, *args: answers[opcode])
    node._startup()
    calls = node._query.call_args_list
    assert sum(call.args[1] == 0xB1 for call in calls) == 1
    write = next(call for call in calls if call.args[1] == 0xF0 and len(call.args) == 5)
    assert write.args[4] == f0
    assert node._mit_config_ready


def test_dry_run_startup_only_reads(mit_node):
    node, _, _ = mit_node
    node._dry_run = True
    node._mit_config_ready = False
    f0 = bytes([0xF0]) + struct.pack("<HHH", 30, 4500, 1800)
    answers = {0xF0: f0, 0xA3: b"\xa3" + bytes(6), 0xF1: feedback(node._mit_ranges),
               0xB0: b"\xb0\x00" + struct.pack("<f", .78) + b"\x00"}
    node._query = Mock(side_effect=lambda motor, opcode, *args: answers[opcode])
    node._startup()
    node._tx_bus.send.assert_not_called()
    assert all(call.args[1] in (0xA3, 0xB0, 0xF0, 0xF1) and len(call.args) == 4
               for call in node._query.call_args_list)


def test_poll_timeout_does_not_publish_an_old_pair_as_fresh(mit_node):
    node, _, _ = mit_node
    motor = node._steering
    motor._pending_position = (1, 99)
    node._pending = (motor, 0xA2, 99.99, 99.999, 5)
    node._schedule_polls(100)
    assert motor.velocity_query_failures == 1
    assert motor._pending_position is None
    assert motor.sample_monotonic == 99.99


def test_late_encoder_reply_cannot_refresh_a_pair(mit_node):
    node, module, _ = mit_node
    motor = node._steering
    node._pending = (motor, 0xA3, 99.99, 99.992, 7)
    node._consume(frame(module, motor, bytes([0xA3]) + struct.pack("<Hi", 0, 500)), 100.0)
    assert motor.sample_monotonic == motor.position_sample_monotonic == 99.99


def test_protocol_safety_constants_and_package_entrypoint():
    from pathlib import Path
    package = Path(__file__).parents[1]
    assert "motor_mit_io = trackstand_deploy.motor_mit_io:main" in (package / "setup.py").read_text()
    assert RADIANS_PER_COUNT == pytest.approx(2 * math.pi / 16384)


@pytest.mark.parametrize("dry_run", [False, True])
def test_single_owner_loop_startup_polling_and_command_timing(mit_node, monkeypatch, dry_run):
    node, module, clock = mit_node
    node._mit_config_ready = False
    node._requested_enable = False
    node._drive_target_nm = node._requested_drive_target_nm = .1
    node._dry_run = dry_run
    node._send_mit = Mock(wraps=node._send_mit)
    target_times = []
    send_mit = node._send_mit
    def record_target(target):
        target_times.append(clock.value)
        return send_mit(target)
    node._send_mit = record_target
    sent = []
    closed = []

    class Bus:
        def __init__(self, *args):
            self.replies = deque()
            self.f0 = bytes([0xF0]) + struct.pack("<HHH", 955, 4500, 1800)

        def send(self, can_id, payload):
            sent.append((clock.value, can_id, payload))
            mid = can_id & 255
            if can_id & 0x400:
                self.replies.append(module.CanReply(mid, feedback(parse_f0(self.f0), status=1), False, 0, clock.value))
                return
            op = payload[0]
            if op == 0xF0:
                if len(payload) == 7:
                    self.f0 = payload
                answer = self.f0
            else:
                answer = {
                    0xCF: bytes([0xCF]) + bytes(7), 0xB1: b"\xb1\x00\x00",
                    0xA3: b"\xa3" + bytes(6), 0xA2: b"\xa2" + bytes(4),
                    0xA1: b"\xa1" + bytes(4), 0xC0: payload,
                    0xB0: b"\xb0\x00" + struct.pack("<f", .78 if mid == 2 else .3) + b"\x00",
                    0xF1: feedback(parse_f0(self.f0)),
                }[op]
            self.replies.append(module.CanReply(mid, answer, False, 0, clock.value))

        def receive(self, timeout):
            clock.value += .00001 if self.replies else max(.00001, timeout)
            if node._mit_config_ready:
                node._requested_enable = True
                node._command_monotonic = clock.value
            if clock.value >= 100.12:
                node._running = False
            return self.replies.popleft() if self.replies else None

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module, "SocketCan", Bus)
    node._can_interface = "can0"
    node._run_io()
    assert closed == [True] and not node._hard_fault
    assert len(target_times) >= 5
    assert all(.020 - 1e-9 <= b - a <= .0201 for a, b in zip(target_times, target_times[1:]))
    assert node._disable_confirmed and not node._disable_wait
    if dry_run:
        assert all(not (can_id & 0x400) and len(data) == 1 and data[0] in (0xF0, 0xF1, 0xA3, 0xA2, 0xA1, 0xB0)
                   for _, can_id, data in sent)
        return
    assert sum(can_id == 0x102 and data[0] == 0xB1 for _, can_id, data in sent) == 1
    mit_times = [t for t, can_id, data in sent if can_id == 0x502]
    drive_times = [t for t, can_id, data in sent if can_id == 0x101 and data[0] == 0xC0
                   and parse_current_feedback(data) != 0]
    assert len(mit_times) >= 5 and len(drive_times) >= 20
    assert all(.020 - 1e-9 <= b - a <= .0201 for a, b in zip(mit_times, mit_times[1:]))
    assert all(.0049 <= b - a <= .0051 for a, b in zip(drive_times, drive_times[1:]))
    assert node._steering.position_query_failures == node._drive.velocity_query_failures == 0


@pytest.mark.parametrize("problem", ["truncated", "overflow", "bad_metadata", "extended", "remote", "error"])
def test_raw_socket_rejects_incomplete_or_nonstandard_data(mit_node, monkeypatch, problem):
    _, module, _ = mit_node
    raw = object.__new__(module.SocketCan)
    raw.socket = Mock()
    raw._last_dropped = 0
    flags, ancillary, can_id = 0, [], 2
    if problem == "truncated":
        flags = module.socket.MSG_TRUNC
    elif problem in ("overflow", "bad_metadata"):
        value = struct.pack("=I", 1) if problem == "overflow" else b"\x01"
        ancillary = [(module.socket.SOL_SOCKET, module.SO_RXQ_OVFL, value)]
    else:
        can_id |= {"extended": 0x80000000, "remote": 0x40000000, "error": 0x20000000}[problem]
    raw.socket.recvmsg.return_value = (module.CAN_FRAME.pack(can_id, 7, b"\xf1" + bytes(6)), ancillary, flags, None)
    monkeypatch.setattr(module.select, "select", lambda *args: ([raw.socket], [], []))
    with pytest.raises(RuntimeError):
        raw.receive(.01)


def test_raw_socket_short_send_is_failure(mit_node):
    _, module, _ = mit_node
    raw = object.__new__(module.SocketCan)
    raw.socket = Mock()
    raw.socket.send.return_value = 15
    with pytest.raises(OSError):
        raw.send(0x502, bytes(8))


def test_constructor_registers_existing_ros_contract_and_no_can_pollers(mit_node, monkeypatch):
    _, module, _ = mit_node
    publishers = []

    class FakeNode:
        def __init__(self, name):
            self.node_name = name
            values = {}
            self.declare_parameter = lambda name, value: values.setdefault(name, value)
            self.get_parameter = lambda name: SimpleNamespace(value=values[name])
            self.create_publisher = lambda kind, topic, qos: publishers.append(topic) or Mock()
            self.create_subscription = Mock(return_value=Mock())
            self.create_timer = Mock(return_value=Mock())
            self.get_logger = Mock(return_value=Mock())

    fake_thread = Mock()
    monkeypatch.setattr(module, "Node", FakeNode)
    monkeypatch.setattr(module, "threading", SimpleNamespace(Lock=threading.Lock, Thread=Mock(return_value=fake_thread)))
    node = module.MotorMitIo()
    assert node.node_name == "trackstand_motor_io"
    assert node._steering.feedback_id == 2 and node._drive.feedback_id == 1
    assert node._steering.encoder_sign == -1 and node._drive.encoder_sign == 1
    assert node._control_rate_hz == node._poll_rate_hz == 200
    assert node._mit_kp == 10 and node._mit_kd == .2
    assert set(publishers) == {f"/trackstand/{name}" for name in (
        "motor_state", "motor_torque", "motor_control", "motor_event", "motor_fault", "motor_status")}
    assert node.create_subscription.call_args.args[1] == "/trackstand/motor_command"
    fake_thread.start.assert_called_once()
    assert node._tx_bus is None  # The worker owns socket creation and closing.


def test_main_handles_ros_external_shutdown_and_uses_idempotent_shutdown(mit_node, monkeypatch):
    import sys
    from types import ModuleType
    _, module, _ = mit_node
    executor = ModuleType("rclpy.executors")
    executor.ExternalShutdownException = type("ExternalShutdownException", (Exception,), {})
    monkeypatch.setitem(sys.modules, "rclpy.executors", executor)
    node = Mock()
    monkeypatch.setattr(module, "MotorMitIo", Mock(return_value=node))
    monkeypatch.setattr(module, "rclpy", SimpleNamespace(
        init=Mock(), spin=Mock(side_effect=executor.ExternalShutdownException()), try_shutdown=Mock()))
    module.main()
    node.destroy_node.assert_called_once()
    module.rclpy.try_shutdown.assert_called_once()


@pytest.mark.parametrize("motor_name", ["steering", "drive"])
@pytest.mark.parametrize("opcode", [0xAE, 0xCF])
@pytest.mark.parametrize("fault_code", [1, 2, 4, 8, 32, 64, 128])
def test_either_driver_fault_stops_both_motors(mit_node, motor_name, opcode, fault_code):
    node, module, _ = mit_node
    node._control_tick()
    node._tx_bus.reset_mock()
    motor = getattr(node, f"_{motor_name}")
    node._consume(frame(module, motor, bytes([opcode]) + bytes(5) + bytes([2, fault_code])), 100)
    node._control_tick()
    assert node._fault_reason == f"{motor_name}_driver_fault" and node._hard_fault
    assert [(call.args[0], call.args[1][0]) for call in node._tx_bus.send.call_args_list] == [
        (0x102, 0xCF), (0x101, 0xC0), (0x101, 0xCF)]
    assert parse_current_feedback(node._tx_bus.send.call_args_list[1].args[1]) == 0
    confirm_disable(node, module)
    node._requested_enable = False
    node._control_tick()
    assert node._fault_latched  # RC down must not erase a driver hardware fault.


def test_disable_remains_unconfirmed_until_both_cf_acknowledgements(mit_node):
    node, module, clock = mit_node
    node._control_tick()
    node._requested_enable = False
    node._control_tick()
    assert node._outputs_active and not node._disable_confirmed
    assert set(node._disable_wait) == {"steering", "drive"}
    node._consume(frame(module, node._steering, bytes([0xCF]) + bytes(7)), clock.value)
    assert set(node._disable_wait) == {"drive"} and node._outputs_active
    node._requested_enable = True
    node._tx_bus.reset_mock()
    clock.value += .005
    node._control_tick()
    node._tx_bus.send.assert_not_called()
    assert node._fault_publisher.publish.call_args.args[0].data
    node._consume(frame(module, node._drive, bytes([0xCF]) + bytes(7)), clock.value)
    assert not node._outputs_active and node._disable_confirmed
    node._control_tick()
    assert not node._fault_latched and node._mit_sent_this_enable


def test_missing_disable_acknowledgement_latches_restart_fault_and_retries(mit_node):
    node, module, clock = mit_node
    node._control_tick()
    node._requested_enable = False
    node._control_tick()
    clock.value += module.DISABLE_TIMEOUT_S + .001
    node._requested_enable = True
    node._command_monotonic = clock.value
    for motor in (node._steering, node._drive):
        motor.sample_monotonic = clock.value
    node._tx_bus.reset_mock()
    node._control_tick()
    assert node._hard_fault and node._fault_reason == "disable_ack_timeout"
    assert all(call.args[0] != 0x502 for call in node._tx_bus.send.call_args_list)
    assert len(node._tx_bus.send.call_args_list) == 3
    confirm_disable(node, module)
    node._requested_enable = False
    node._control_tick()
    assert node._fault_latched  # A late confirmation cannot erase the timeout.


def test_waiting_for_successful_cf_send_does_not_flood_duplicate_commands(mit_node):
    node, _, clock = mit_node
    node._outputs_active = True
    node._disable_outputs()
    for step in range(1, 20):
        clock.value = 100 + step * .005
        node._disable_outputs()
    assert node._tx_bus.send.call_count == 3


def test_failed_cf_send_retries_only_unconfirmed_failed_motor(mit_node):
    node, module, clock = mit_node
    node._outputs_active = True
    node._tx_bus.send.side_effect = [OSError("steering send failed"), None, None]
    node._disable_outputs()
    node._consume(frame(module, node._drive, bytes([0xCF]) + bytes(7)), clock.value)
    node._tx_bus.send.side_effect = None
    node._tx_bus.reset_mock()
    clock.value += module.DISABLE_RETRY_S + .001
    node._disable_outputs()
    assert [call.args[0] for call in node._tx_bus.send.call_args_list] == [0x102]
    assert node._outputs_active
    confirm_disable(node, module)
    assert node._disable_confirmed and not node._outputs_active


def test_old_cf_acknowledgement_cannot_confirm_a_new_disable(mit_node):
    node, module, clock = mit_node
    node._outputs_active = True
    node._disable_outputs()
    node._consume(frame(module, node._steering, bytes([0xCF]) + bytes(7), stamp=99.9), clock.value)
    assert "steering" in node._disable_wait and not node._disable_confirmed


def test_late_cf_acknowledgement_cannot_hide_deadline_expiry(mit_node):
    node, module, clock = mit_node
    node._outputs_active = True
    node._disable_outputs()
    clock.value += module.DISABLE_TIMEOUT_S + .001
    confirm_disable(node, module)  # Arrival occurs late, before the next safety tick.
    assert node._disable_confirmed and not node._outputs_active
    assert node._hard_fault and node._fault_reason == "disable_ack_timeout"


def test_cf_acknowledgement_with_active_mode_fails_closed(mit_node):
    node, module, clock = mit_node
    node._outputs_active = True
    node._disable_outputs()
    node._consume(frame(module, node._steering, bytes([0xCF]) + bytes(5) + bytes([2, 0])), clock.value)
    assert node._hard_fault and node._fault_reason == "steering_disable_not_confirmed"
    assert not node._disable_confirmed and node._disable_wait["steering"] is None


def test_old_inactive_feedback_does_not_fault_a_new_enable(mit_node):
    node, module, clock = mit_node
    node._control_tick()
    node._requested_enable = False
    node._control_tick()
    confirm_disable(node, module)
    clock.value += .02
    node._requested_enable = True
    node._control_tick()
    node._consume(frame(module, node._steering, bytes([0xCF]) + bytes(7), stamp=100.001), clock.value)
    node._consume(frame(module, node._steering, feedback(node._mit_ranges), stamp=100.001), clock.value)
    assert not node._hard_fault and node._mit_sent_this_enable


def test_new_episode_slew_does_not_use_time_spent_disabled(mit_node):
    node, module, clock = mit_node
    node._control_tick()
    node._requested_enable = False
    node._control_tick()
    confirm_disable(node, module)
    clock.value += 1
    for motor in (node._steering, node._drive):
        motor.sample_monotonic = clock.value
    node._command_monotonic = clock.value
    node._requested_enable = True
    node._steering_target_rad = node._requested_steering_target_rad = 1.57
    node._control_tick()
    assert record(node, module)["slewed_steering_target_rad"] == pytest.approx(23.9 / 50)


@pytest.mark.parametrize("motor_name,speed,reason", [
    ("steering", 30, "steering_speed_limit"), ("drive", 70, "drive_speed_limit")])
def test_present_speed_fault_cannot_clear_while_rc_is_down(mit_node, motor_name, speed, reason):
    node, _, clock = mit_node
    node._requested_enable = False
    motor = getattr(node, f"_{motor_name}")
    motor.velocity_rad_s = speed
    node._control_tick()
    assert node._fault_latched and node._fault_reason == reason
    assert node._fault_publisher.publish.call_args.args[0].data
    motor.velocity_rad_s = 0
    clock.value += .005
    node._control_tick()
    assert not node._fault_latched


@pytest.mark.parametrize("opcode,payload", [
    (0xA3, bytes([0xA3]) + struct.pack("<Hi", 0, 10000)),
    (0xA2, bytes([0xA2]) + struct.pack("<i", 10000)),
])
def test_kernel_stamp_before_query_send_cannot_refresh_feedback(mit_node, opcode, payload):
    node, module, clock = mit_node
    motor = node._steering
    node._pending = (motor, opcode, 100, 100.002, len(payload))
    motor._pending_position = (1.0, 100)
    node._consume(frame(module, motor, payload, stamp=99.99), clock.value + .001)
    assert motor.zero_count == 0 and motor.sample_monotonic == 99.99
    assert node._pending is not None


def test_kernel_stamp_preserves_age_of_f1_and_cannot_refresh_watchdog(mit_node):
    node, module, clock = mit_node
    node._control_tick()
    clock.value += .055
    for motor in (node._steering, node._drive):
        motor.sample_monotonic = clock.value
    node._command_monotonic = clock.value
    node._consume(frame(module, node._steering, feedback(node._mit_ranges, status=1), stamp=99.98), clock.value)
    assert node._mit_f1_at == 99.99
    node._control_tick()
    assert node._hard_fault and node._fault_reason == "steering_mit_feedback_timeout"


def test_startup_uses_verified_a3_zero_even_if_prior_zero_was_wrong(mit_node):
    node, module, _ = mit_node
    node._mit_config_ready = False
    node._requested_enable = False
    f0 = bytes([0xF0]) + struct.pack("<HHH", 30, 4500, 1800)
    answers = {0xCF: bytes([0xCF]) + bytes(7), 0xB1: b"\xb1\x00\x00", 0xF0: f0,
               0xA3: b"\xa3" + bytes(6), 0xF1: feedback(node._mit_ranges),
               0xB0: b"\xb0\x00" + struct.pack("<f", .78) + b"\x00"}
    def query(motor, opcode, *args):
        if opcode == 0xF0:
            node._steering.zero_count = 10000
        return answers[opcode]
    node._query = Mock(side_effect=query)
    node._startup()
    assert node._steering.zero_count == 0 and node._steering._unwrapped_rad == 0


def test_startup_query_drains_old_encoder_reply_before_recording_zero(mit_node):
    node, module, clock = mit_node
    node._mit_config_ready = False
    node._steering.zero_count = None
    old = frame(module, node._steering, bytes([0xA3]) + struct.pack("<Hi", 0, 10000), stamp=99.9)
    verified = bytes([0xA3]) + bytes(6)
    replies = deque([old])
    def receive(timeout):
        clock.value += .00001
        return replies.popleft() if replies else None
    def send(can_id, payload):
        replies.append(frame(module, node._steering, verified, stamp=clock.value + .00001))
    node._tx_bus.receive.side_effect = receive
    node._tx_bus.send.side_effect = send
    assert node._query(node._steering, 0xA3, 7, .25) == verified
    assert node._steering.zero_count == 0 and node._pending is None


def raw_socket(module, monkeypatch, timestamp_payload=None):
    raw = object.__new__(module.SocketCan)
    raw.socket = Mock()
    raw._last_dropped = 0
    raw._wall_offset_s = None
    raw._clock_uncertainty_s = 0
    ancillary = [] if timestamp_payload is None else [
        (module.socket.SOL_SOCKET, module.SO_TIMESTAMPNS_NEW, timestamp_payload)]
    raw.socket.recvmsg.return_value = (module.CAN_FRAME.pack(2, 7, b"\xf1" + bytes(6)), ancillary, 0, None)
    monkeypatch.setattr(module.select, "select", lambda *args: ([raw.socket], [], []))
    return raw


def test_raw_socket_uses_kernel_arrival_not_dequeue_time(mit_node, monkeypatch):
    _, module, _ = mit_node
    raw = raw_socket(module, monkeypatch, module.KERNEL_TIMESPEC.pack(123455, 900_000_000))
    reply = raw.receive(0)
    assert reply.received_monotonic == pytest.approx(99.9)
    assert reply.can_id == 2 and reply.data == b"\xf1" + bytes(6)
    assert raw.socket.recvmsg.call_args.args[1] == (
        module.socket.CMSG_SPACE(4) + module.socket.CMSG_SPACE(16))


@pytest.mark.parametrize("timestamp", [None, b"bad", struct.pack("=qq", -1, 0), struct.pack("=qq", 123456, 10**9)])
def test_raw_socket_rejects_missing_or_invalid_timestamp(mit_node, monkeypatch, timestamp):
    _, module, _ = mit_node
    with pytest.raises(RuntimeError):
        raw_socket(module, monkeypatch, timestamp).receive(0)


def test_raw_socket_fails_closed_on_system_clock_step(mit_node, monkeypatch):
    _, module, clock = mit_node
    raw = raw_socket(module, monkeypatch, module.KERNEL_TIMESPEC.pack(123456, 0))
    raw.receive(0)
    clock.value += .1  # Wall clock remains fixed: a 100 ms change in its offset.
    with pytest.raises(RuntimeError, match="system clock changed"):
        raw.receive(0)


def test_raw_socket_rejects_future_timestamp(mit_node, monkeypatch):
    _, module, _ = mit_node
    raw = raw_socket(module, monkeypatch, module.KERNEL_TIMESPEC.pack(123456, 100_000_000))
    with pytest.raises(RuntimeError, match="future"):
        raw.receive(0)


def test_raw_socket_rejects_uncertain_clock_sampling(mit_node, monkeypatch):
    _, module, _ = mit_node
    raw = raw_socket(module, monkeypatch, module.KERNEL_TIMESPEC.pack(123456, 0))
    ticks = iter([100, 100.002])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="clock sampling delayed"):
        raw.receive(0)


def test_bounded_receive_drain_fails_closed_on_persistent_backlog(mit_node):
    node, module, _ = mit_node
    node._tx_bus.receive.return_value = frame(module, node._steering, bytes([0xA1]) + bytes(4))
    with pytest.raises(RuntimeError, match="backlog"):
        node._drain_received()
    assert node._tx_bus.receive.call_count == module.MAX_RX_BATCH


def test_socket_closes_even_if_shutdown_receive_fails(mit_node, monkeypatch):
    node, module, _ = mit_node
    bus = Mock()
    bus.receive.side_effect = OSError("receiver failed")
    monkeypatch.setattr(module, "SocketCan", Mock(return_value=bus))
    node._can_interface = "can0"
    node._startup = Mock()
    node._run_io()
    bus.close.assert_called_once()
    assert node._tx_bus is None and node._hard_fault
    assert [(call.args[0], call.args[1][0]) for call in bus.send.call_args_list] == [
        (0x102, 0xCF), (0x101, 0xC0), (0x101, 0xCF)]


def test_first_hard_fault_reason_survives_disable_timeout(mit_node):
    node, _, _ = mit_node
    node._fault("drive_driver_fault", hard=True)
    node._fault("disable_ack_timeout", hard=True)
    assert node._fault_reason == "drive_driver_fault"


def test_real_linux_socket_preserves_kernel_queue_age(node_module):
    import socket
    import time
    module = node_module("motor_mit_io")
    try:
        sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    except PermissionError:
        pytest.skip("sandbox does not permit local socket pairs")
    try:
        receiver.setsockopt(socket.SOL_SOCKET, module.SO_TIMESTAMPNS_NEW, 1)
        raw = object.__new__(module.SocketCan)
        raw.socket = receiver
        raw._last_dropped, raw._wall_offset_s, raw._clock_uncertainty_s = 0, None, 0
        before_send = time.monotonic()
        sender.send(module.CAN_FRAME.pack(2, 7, b"\xf1" + bytes(6)))
        after_send = time.monotonic()
        time.sleep(.02)
        reply = raw.receive(.1)
        assert before_send - .001 <= reply.received_monotonic <= after_send + .001
        assert time.monotonic() - reply.received_monotonic >= .019
    except PermissionError:
        pytest.skip("sandbox does not permit kernel socket timestamp options")
    finally:
        sender.close()
        receiver.close()
