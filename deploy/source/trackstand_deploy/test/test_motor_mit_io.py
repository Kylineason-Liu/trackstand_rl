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
from trackstand_deploy.telemetry import MotorControlPublishGate


def test_mit_import_and_ros_handling_are_independent_of_legacy_nodes(node_module, monkeypatch):
    import builtins

    real_import = builtins.__import__
    forbidden = {"motor_io", "motor_io_position", "motor_position_io", "can"}

    def independent_import(name, *args, **kwargs):
        if name.split(".")[-1] in forbidden or name.startswith("can."):
            raise ImportError(f"MIT must not import legacy motor nodes or python-can: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", independent_import)
    module = node_module("motor_mit_io")
    assert module.MotorMitIo.__bases__ == (module.Node,)
    assert module.ChannelState.__module__ == module.__name__
    for name in (
        "_initialize_command_state", "_create_interfaces", "_command_callback", "_snapshot",
        "_publish_state", "_publish_control_telemetry", "_event_details",
    ):
        assert getattr(module.MotorMitIo, name).__module__ == module.__name__
    node = object.__new__(module.MotorMitIo)
    node._initialize_command_state()
    assert not node._requested_enable and not node._outputs_active and not node._fault_latched
    node.create_publisher = Mock()
    node.create_subscription = Mock()
    node._create_interfaces()
    node._steering_limit_rad = math.pi / 2
    node._command_callback(module.Float64MultiArray(data=[1.0, 0.1, -0.2]))
    assert node._requested_enable and node._steering_target_rad == 0.1 and node._drive_target_nm == -0.2
    motor = module.ChannelState("steering", 0x102, 2, 1, -1, -1)
    assert len(node._snapshot(motor)) == 10
    assert motor.zero_count is None and motor.torque_constant_nm_per_a is None


def test_mit_owns_parameter_defaults_and_three_frequency_divisions(node_module):
    module = node_module("motor_mit_io")
    expected = {
        "can_interface": "can0", "steering_host_id": "102", "steering_feedback_id": "002",
        "drive_host_id": "101", "drive_feedback_id": "001", "steering_gear_ratio": 1.0,
        "drive_gear_ratio": 1.0, "steering_encoder_sign": -1.0, "drive_encoder_sign": -1.0,
        "steering_torque_sign": -1.0, "drive_torque_sign": -1.0,
        "command_rate_hz": 50.0, "state_rate_hz": 100.0, "actuation_rate_hz": 200.0,
        "command_timeout_s": 0.08, "feedback_timeout_s": 0.05,
        "steering_limit_rad": math.pi / 2, "steering_fault_clear_rad": math.radians(5),
        "dry_run": False, "mit_kp_nm_per_motor_rad": 10.0, "mit_kd_nm_s_per_motor_rad": 0.2,
    }
    assert list(module.MotorMitIo._parameter_defaults.items()) == list(expected.items())


def feedback(ranges, position=0.0, velocity=0.0, torque=0.0, status=0):
    def enc(value, maximum, bits):
        return round((value + maximum) * ((1 << bits) - 1) / (2 * maximum))
    p, v, t = (enc(position, ranges.position_rad, 16), enc(velocity, ranges.velocity_rad_s, 12),
               enc(torque, ranges.torque_nm, 12))
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
def mit_node(node_module):
    module = node_module("motor_mit_io")
    clock = SimpleNamespace(value=100.0)
    module.time = SimpleNamespace(monotonic=lambda: clock.value, time=lambda: 123456.0)
    node = object.__new__(module.MotorMitIo)
    node._steering = module.ChannelState("steering", 0x102, 2, 1, -1, -1)
    node._drive = module.ChannelState("drive", 0x101, 1, 1, 1, -1)
    for motor in (node._steering, node._drive):
        motor.zero_count = 0
        motor.torque_constant_nm_per_a = 0.78 if motor is node._steering else 0.3
        motor.status_sample_monotonic = 99.99
        motor.sample_monotonic = motor.position_sample_monotonic = motor.velocity_sample_monotonic = 99.99
    node._dry_run = False
    node._command_rate_hz, node._poll_rate_hz, node._control_rate_hz = 50, 100, 200
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
    node._control_publish_gate = MotorControlPublishGate(module.MIT_RATE_HZ)
    node._feedback_gate_fault = False
    node._command_lock = threading.Lock()
    node._disable_generation = 0
    node._active_disable_generation = 0
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
    node._slew_last_at = 0.0
    node._disable_wait = {}
    node._disable_f1_wait = {}  # Match the current CF/F1 confirmation state.
    node._disable_confirmed = True
    node._disable_deadline_at = node._next_disable_retry_at = 0
    node._last_disable_errors = ()
    node._pending = None
    node._poll_jobs = deque()
    node._next_poll_at = node._next_b0_at = node._next_f0_at = 0
    node._running = True
    node._tx_bus = Mock()
    node._tx_bus.receive.return_value = None
    for name in ("control", "event", "fault", "status", "state"):
        setattr(node, f"_{name}_publisher", Mock())
    node._state_dimension = module.MultiArrayDimension(
        label=module.MOTOR_STATE_LABEL, size=len(module.MOTOR_STATE_FIELDS), stride=len(module.MOTOR_STATE_FIELDS),
    )
    node._control_dimension = SimpleNamespace()
    node._last_published_sample_times = (0, 0)
    node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: SimpleNamespace(
        nanoseconds=123456000000000, to_msg=lambda: "receipt_stamp")))
    node.get_logger = Mock(return_value=Mock())
    node.context = SimpleNamespace(ok=lambda: True)
    return node, module, clock


def record(node, module):
    return dict(zip(module.MOTOR_CONTROL_FIELDS, node._control_publisher.publish.call_args.args[0].data))


def frame(module, motor, payload, stamp=None):
    return module.CanReply(motor.feedback_id, payload, False, 0,
                           module.time.monotonic() if stamp is None else stamp)


def confirm_disable(node, module):
    for motor in (node._steering, node._drive):
        node._consume(frame(module, motor, bytes([0xCF]) + bytes(7)), module.time.monotonic())


@pytest.mark.parametrize("enable", [0.0, 0.5, 0.500001, 1.0])
@pytest.mark.parametrize("target,drive", [(0.1, -0.2), (2.0, -8.0), (-2.0, 8.0)])
def test_independent_callback_preserves_requests_and_bounds_motor_targets(mit_node, enable, target, drive):
    node, module, clock = mit_node
    node._command_callback(module.Float64MultiArray(data=[enable, target, drive]))
    assert node._requested_enable == (enable > 0.5)
    assert node._requested_steering_target_rad == target and node._requested_drive_target_nm == drive
    assert node._steering_target_rad == max(-1.57, min(1.57, target))
    assert node._drive_target_nm == max(-6.28, min(6.28, drive))
    assert node._command_monotonic == clock.value


@pytest.mark.parametrize("data", [[], [1.0, 0.0], [1.0, 0.0, 0.0, 0.0]] + [
    [bad if index == field else 0.0 for index in range(3)]
    for field in range(3) for bad in (math.nan, math.inf, -math.inf)
])
def test_independent_callback_rejects_invalid_commands_without_refreshing_watchdog(mit_node, data):
    node, module, _ = mit_node
    names = ("_requested_enable", "_requested_steering_target_rad", "_requested_drive_target_nm",
             "_steering_target_rad", "_drive_target_nm")
    before = tuple(getattr(node, name) for name in names)
    node._command_callback(module.Float64MultiArray(data=data))
    assert tuple(getattr(node, name) for name in names) == before
    assert node._command_monotonic == 0 and node._fault_latched
    node._control_tick()
    assert node._fault_latched and node._fault_reason == "invalid_motor_command"
    assert not any(call.args[0] == 0x502 for call in node._tx_bus.send.call_args_list)


@pytest.mark.parametrize("name", ["steering", "drive"])
def test_lost_driver_status_disables_even_with_fresh_encoders_and_commands(mit_node, name):
    node, module, clock = mit_node
    node._control_tick()
    clock.value += .051
    node._command_monotonic = node._mit_f1_at = clock.value
    for motor in (node._steering, node._drive):
        motor.sample_monotonic = motor.status_sample_monotonic = clock.value
    getattr(node, "_" + name).status_sample_monotonic = clock.value - .051
    node._tx_bus.reset_mock()
    node._control_tick()
    assert node._fault_latched and node._fault_reason == "active_feedback_unavailable"
    assert record(node, module)["gate_enabled"] == 0
    assert not any(call.args[0] == 0x502 for call in node._tx_bus.send.call_args_list)
    assert node._disable_wait
    event = json.loads(node._event_publisher.publish.call_args.args[0].data)
    assert name + "_driver_status_missing_or_stale" in event["disable_reasons"]


def test_driver_status_not_received_prevents_first_motion_without_startup_fault_latch(mit_node):
    node, module, _ = mit_node
    node._drive.status_sample_monotonic = 0
    node._control_tick()
    assert record(node, module)["gate_enabled"] == 0
    assert not node._fault_latched and not node._outputs_active
    node._tx_bus.send.assert_not_called()


@pytest.mark.parametrize("steering_at,drive_at,previous,expected", [
    (0.0, 99.99, (0.0, 0.0), False), (99.99, 0.0, (0.0, 0.0), False),
    (99.9, 99.99, (0.0, 0.0), False), (99.99, 99.9, (0.0, 0.0), False),
    (99.99, 99.99, (99.99, 99.99), False), (100.0, 99.99, (99.99, 99.99), False),
    (99.99, 100.0, (99.99, 99.99), False), (100.0, 100.0, (99.99, 99.99), True),
])
def test_independent_state_publisher_requires_two_fresh_new_pairs(mit_node, steering_at, drive_at, previous, expected):
    node, _, _ = mit_node
    node._steering.sample_monotonic, node._drive.sample_monotonic = steering_at, drive_at
    node._last_published_sample_times = previous
    node._steering.position_rad, node._drive.position_rad = 0.1, -0.2
    node._steering.velocity_rad_s, node._drive.velocity_rad_s = 0.3, -0.4
    node._publish_state()
    assert node._state_publisher.publish.call_count == int(expected)
    if expected:
        message = node._state_publisher.publish.call_args.args[0]
        assert message.data[:5] == [0.1, -0.2, 0.3, -0.4, 123456.0]
        assert len(message.data) == 7 and all(math.isnan(value) for value in message.data[5:])
        assert node._last_published_sample_times == (steering_at, drive_at)
    else:
        assert node._last_published_sample_times == previous


def test_mit_and_drive_200hz_and_exact_motor_signs(mit_node):
    node, module, clock = mit_node
    node._drive.velocity_rad_s = 1.0
    for i in range(9):
        clock.value = 100.0 + i * 0.005
        for motor in (node._steering, node._drive):
            motor.sample_monotonic = motor.status_sample_monotonic = clock.value - .001
        node._control_tick()
    sends = node._tx_bus.send.call_args_list
    mit = [call for call in sends if call.args[0] == 0x502]
    drive = [call for call in sends if call.args[0] == 0x101]
    assert len(mit) == len(drive) == 9
    expected = mit_motion_payload(0.002 - 0.05, node._mit_ranges, 10, .2)
    assert mit[0].args[1] == expected
    assert parse_current_feedback(drive[0].args[1]) == 0.0
    data = record(node, module)
    assert data["drive_final_torque_nm"] == 0.0
    assert math.isnan(data["steering_final_torque_nm"])
    assert math.isnan(data["steering_current_command_a"])
    assert node._mit_zero_rad == .002


@pytest.mark.parametrize("dry_run", [False, True])
def test_three_rate_divisions_keep_policy_state_and_actuation_independent(mit_node, dry_run):
    node, module, clock = mit_node
    node._dry_run = dry_run
    for index in range(200):
        clock.value = 100 + index * .005
        if index % 4 == 0:
            node._command_callback(module.Float64MultiArray(data=[1.0, .05, 0.0]))
        node._mit_f1_at = clock.value
        for motor in (node._steering, node._drive):
            motor.sample_monotonic = motor.status_sample_monotonic = clock.value
        node._control_tick()
        if index % 2 == 0:
            node._publish_feedback()
    assert node._control_publisher.publish.call_count == node._control_sequence == 200
    assert node._fault_publisher.publish.call_count == node._status_publisher.publish.call_count == 100
    assert node._state_publisher.publish.call_count == 100
    sent = node._tx_bus.send.call_args_list
    assert sum(call.args[0] == 0x502 for call in sent) == (0 if dry_run else 200)
    assert sum(call.args[0] == 0x101 for call in sent) == (0 if dry_run else 200)
    assert record(node, module)["cycle_interval_ms"] == pytest.approx(5.0)


def test_v1_targets_are_conditioned_once_and_held_for_four_actual_mit_sends(mit_node):
    import numpy as np
    from trackstand_deploy.contracts import SteeringTargetLimiter, scale_policy_action

    node, module, clock = mit_node
    limiter = SteeringTargetLimiter()
    previous = 0.
    for step, raw in enumerate(([1., -.5], [-1., 1.], [1., -1.], [0., 0.])):
        steering, rear = scale_policy_action(np.array(raw))
        held = limiter.update(steering)
        assert abs(held - previous) <= .478 + 1e-12
        node._tx_bus.reset_mock()
        for repeat in range(4):
            clock.value = 100 + step * .02 + repeat * .005
            if repeat == 0:
                node._command_callback(module.Float64MultiArray(data=[1., held, rear]))
            node._mit_f1_at = clock.value
            for motor in (node._steering, node._drive):
                motor.sample_monotonic = motor.status_sample_monotonic = clock.value
            node._control_tick()
        commands = [call.args[1] for call in node._tx_bus.send.call_args_list if call.args[0] == 0x502]
        assert len(commands) == 4 and all(payload == commands[0] for payload in commands)
        encoded = module.parse_mit_motion(commands[0], node._mit_ranges)
        assert node._steering.encoder_sign * (encoded.position_rad - node._mit_zero_rad) == pytest.approx(
            held, abs=3 / 65535,
        )
        drive = [call.args[1] for call in node._tx_bus.send.call_args_list if call.args[0] == 0x101]
        assert len(drive) == 4
        assert parse_current_feedback(drive[0]) * .3 * node._drive.torque_sign == pytest.approx(rear, abs=.00015)
        previous = held


@pytest.mark.parametrize("cause", ["rc_down", "stale_feedback", "can_failure"])
def test_between_feedback_ticks_control_still_disables_and_reports_events(mit_node, cause):
    node, _, clock = mit_node
    node._control_tick()
    node._publish_feedback()
    node._tx_bus.reset_mock()
    clock.value += .005
    if cause == "rc_down":
        node._requested_enable = False
    elif cause == "stale_feedback":
        node._steering.sample_monotonic = 99.0
    else:
        node._tx_bus.send.side_effect = [OSError("injected failure"), None, None, None]
    node._control_tick()
    assert node._control_publisher.publish.call_count == 2
    assert node._fault_publisher.publish.call_count == 1  # No off-phase fault publication.
    clock.value += .005
    node._publish_feedback()
    assert node._fault_publisher.publish.call_count == 2
    assert node._fault_publisher.publish.call_args.args[0].data is (cause != "rc_down")
    assert [(call.args[0], call.args[1][0]) for call in node._tx_bus.send.call_args_list[-3:]] == [
        (0x102, 0xCF), (0x101, 0xC0), (0x101, 0xCF),
    ]
    event = json.loads(node._event_publisher.publish.call_args.args[0].data)
    assert event["sequence"] == 2 and not event["gate_enabled"]


@pytest.mark.parametrize("cause", ["steering", "drive", "command", "constant"])
def test_brief_active_watchdog_failure_latches_until_safe_rc_down(mit_node, cause):
    node, module, clock = mit_node
    node._control_tick()
    node._publish_feedback()
    clock.value += .005
    if cause == "command":
        node._command_monotonic = clock.value - .081
    elif cause == "constant":
        node._drive.torque_constant_nm_per_a = None
    else:
        getattr(node, f"_{cause}").sample_monotonic = clock.value - .051
    node._control_tick()
    assert node._fault_latched and node._disable_wait
    clock.value += .001
    confirm_disable(node, module)
    for motor in (node._steering, node._drive):
        motor.sample_monotonic = motor.status_sample_monotonic = clock.value
    node._drive.torque_constant_nm_per_a = .3
    node._command_callback(module.Float64MultiArray(data=[1.0, .05, 0.0]))
    node._tx_bus.reset_mock()
    clock.value += .004
    node._control_tick()
    node._publish_feedback()
    node._tx_bus.send.assert_not_called()
    assert [c.args[0].data for c in node._fault_publisher.publish.call_args_list] == [False, True]
    assert node._fault_latched and not node._hard_fault
    node._command_callback(module.Float64MultiArray(data=[0.0, 0.0, 0.0]))
    node._control_tick()
    assert not node._fault_latched
    node._command_callback(module.Float64MultiArray(data=[1.0, .05, 0.0]))
    node._control_tick()
    assert node._mit_sent_this_enable


@pytest.mark.parametrize("cause", ["steering", "drive", "constant", "command"])
def test_startup_incomplete_inputs_gate_without_latching(mit_node, cause):
    node, _, clock = mit_node
    if cause == "constant":
        node._drive.torque_constant_nm_per_a = None
    elif cause == "command":
        node._command_monotonic = 0.0
    else:
        getattr(node, f"_{cause}").sample_monotonic = 0.0
    node._control_tick()
    node._tx_bus.send.assert_not_called()
    assert not node._fault_latched
    node._drive.torque_constant_nm_per_a = .3
    node._command_monotonic = clock.value
    for motor in (node._steering, node._drive):
        motor.sample_monotonic = motor.status_sample_monotonic = clock.value
    node._control_tick()
    assert node._mit_sent_this_enable


@pytest.mark.parametrize("pause_at", ["diagnostic_clock", "mit_encoding", "between_motors", "drive_encoding"])
@pytest.mark.parametrize("cause", ["stale", "disable", "disable_then_enable"])
def test_pre_send_revalidation_rejects_stale_or_withdrawn_motion(mit_node, monkeypatch, pause_at, cause):
    node, module, clock = mit_node
    node._drive_target_nm = node._requested_drive_target_nm = .1

    def pause():
        clock.value += .09
        if cause != "stale":
            node._command_callback(module.Float64MultiArray(data=[0.0, 0.0, 0.0]))
            if cause == "disable_then_enable":
                node._command_callback(module.Float64MultiArray(data=[1.0, .05, .1]))
                for motor in (node._steering, node._drive):
                    motor.sample_monotonic = motor.status_sample_monotonic = clock.value

    if pause_at == "diagnostic_clock":
        def ros_now():
            pause()
            return SimpleNamespace(nanoseconds=123456000000000)
        node.get_clock = Mock(return_value=SimpleNamespace(now=ros_now))
    elif pause_at in ("mit_encoding", "drive_encoding"):
        name = "mit_motion_payload" if pause_at == "mit_encoding" else "q_current_payload"
        original = getattr(module, name)
        def encode(*args):
            payload = original(*args)
            # Do not inject delay into zero-current disable commands.
            if pause_at == "mit_encoding" or args[0] != 0.0:
                pause()
            return payload
        monkeypatch.setattr(module, name, encode)
    else:
        node._tx_bus.send.side_effect = lambda can_id, payload: pause() if can_id == 0x502 else None
    node._control_tick()
    sent = node._tx_bus.send.call_args_list
    assert not any(c.args[0] == 0x101 and c.args[1][0] == 0xC0
                   and parse_current_feedback(c.args[1]) != 0 for c in sent)
    assert sum(c.args[0] == 0x502 for c in sent) == int(pause_at in ("between_motors", "drive_encoding"))
    assert record(node, module)["gate_enabled"] == 0
    assert node._fault_latched is (cause != "disable" or pause_at == "drive_encoding")
    if cause == "disable" and pause_at == "drive_encoding":
        assert node._fault_reason == "steering_mit_feedback_timeout"
    if pause_at in ("between_motors", "drive_encoding"):
        assert node._disable_wait  # A submitted MIT packet requires confirmed CF disable.


@pytest.mark.parametrize("dry_run", [False, True])
def test_direct_target_under_jitter_does_not_fake_wire_feedback(mit_node, dry_run):
    node, module, clock = mit_node
    node._dry_run = dry_run
    node._requested_steering_target_rad = node._steering_target_rad = 1.0
    node._control_tick()
    clock.value += .008
    node._control_tick()
    assert record(node, module)["slewed_steering_target_rad"] == pytest.approx(1.0)
    assert record(node, module)["steering_slew_limited"] == 0
    if dry_run:
        assert node._mit_last_sent_at == 0 and not node._mit_sent_this_enable
        assert math.isnan(record(node, module)["mit_actual_send_interval_ms"])
        node._tx_bus.send.assert_not_called()


def test_worker_failure_publishes_terminal_event_without_another_control_tick(mit_node, monkeypatch):
    node, module, _ = mit_node
    node._control_tick()
    before = node._event_publisher.publish.call_count
    bus = Mock()
    bus.receive.side_effect = OSError("injected receive failure")
    monkeypatch.setattr(module, "SocketCan", Mock(return_value=bus))
    node._startup = Mock()
    node._can_interface = "can0"
    node._run_io()
    event = json.loads(node._event_publisher.publish.call_args.args[0].data)
    assert node._event_publisher.publish.call_count > before
    assert not event["gate_enabled"] and event["fault_latched"]
    assert "mit_io_stopped" in event["disable_reasons"]
    assert event["fault_reason"].startswith("mit_io_error:OSError:")
    bus.close.assert_called_once()
    assert not node._running and node._tx_bus is None


def test_disable_enable_between_ticks_cannot_skip_motor_disable(mit_node):
    node, module, clock = mit_node
    node._control_tick()
    node._command_callback(module.Float64MultiArray(data=[0.0, 0.0, 0.0]))
    node._command_callback(module.Float64MultiArray(data=[1.0, .05, .1]))
    node._tx_bus.reset_mock()
    clock.value += .005
    node._control_tick()
    assert node._fault_latched and node._disable_wait
    assert all(c.args[0] != 0x502 for c in node._tx_bus.send.call_args_list)
    assert node._fault_reason == "enable_withdrawn_during_control"


@pytest.mark.parametrize("cause", ["stale", "disable", "disable_then_enable"])
def test_pause_during_guard_snapshot_cannot_restore_stale_authorization(mit_node, cause):
    node, module, clock = mit_node
    original = node._snapshot
    calls = 0
    def snapshot(motor):
        nonlocal calls
        calls += 1
        if calls == 3:  # First encoder snapshot in the final steering guard.
            clock.value += .09
            if cause != "stale":
                node._command_callback(module.Float64MultiArray(data=[0.0, 0.0, 0.0]))
                if cause == "disable_then_enable":
                    node._command_callback(module.Float64MultiArray(data=[1.0, .05, .1]))
                    for channel in (node._steering, node._drive):
                        channel.sample_monotonic = channel.status_sample_monotonic = clock.value
        return original(motor)
    node._snapshot = snapshot
    node._control_tick()
    assert not any(c.args[0] == 0x502 for c in node._tx_bus.send.call_args_list)
    assert record(node, module)["gate_enabled"] == 0


def test_close_failure_does_not_skip_terminal_event(mit_node, monkeypatch):
    node, module, _ = mit_node
    bus = Mock()
    bus.close.side_effect = OSError("close failed")
    monkeypatch.setattr(module, "SocketCan", Mock(return_value=bus))
    node._startup = Mock(side_effect=RuntimeError("injected preflight failure"))
    node._disable_outputs = Mock()
    node._can_interface = "can0"
    node._run_io()
    assert node._tx_bus is None and node._hard_fault
    event = json.loads(node._event_publisher.publish.call_args.args[0].data)
    assert "mit_io_stopped" in event["disable_reasons"] and event["fault_latched"]


def test_disable_confirmation_events_include_each_motor(mit_node):
    node, module, clock = mit_node
    node._control_tick()
    node._command_callback(module.Float64MultiArray(data=[0.0, 0.0, 0.0]))
    node._control_tick()
    node._control_tick()  # Stabilize pending-disable reasons before acknowledging one motor.
    before = node._event_publisher.publish.call_count
    node._consume(frame(module, node._steering, bytes([0xCF]) + bytes(7)), clock.value)
    clock.value += .005
    node._control_tick()
    assert node._event_publisher.publish.call_count == before + 1
    event = json.loads(node._event_publisher.publish.call_args.args[0].data)
    assert event["steering_disable_confirmed"] and not event["drive_disable_confirmed"]


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


def test_mit_direct_target_is_not_slewed_over_200hz_ticks(mit_node):
    node, module, _ = mit_node
    node._steering_target_rad = node._requested_steering_target_rad = 1.57
    node._control_tick()
    assert record(node, module)["slewed_steering_target_rad"] == pytest.approx(1.57)
    assert record(node, module)["steering_slew_limited"] == 0


@pytest.mark.parametrize("target", [-2.8, -math.pi / 2, 0.0, math.pi / 2, 2.8])
def test_encoded_position_does_not_apply_commented_travel_clamp(mit_node, target):
    node, module, _ = mit_node
    result, native = node._send_mit(target)
    assert result.success
    joint_target = node._steering.encoder_sign * (node._mit_encoded_command.position_rad - node._mit_zero_rad)
    assert joint_target == pytest.approx(target, abs=3 / 65535)
    assert node._tx_bus.send.call_args.args == (
        0x502, module.mit_motion_payload(native, node._mit_ranges, node._mit_kp, node._mit_kd),
    )


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_v0_oversized_action_is_rejected_and_direct_topic_requests_are_bounded(mit_node, sign):
    # An actual recorded startup-sized request: no real CAN interface is opened.
    import numpy as np
    from trackstand_deploy.contracts import scale_policy_action

    node, module, _ = mit_node
    raw_action = np.array([sign * 1.599545, sign * 16.720318], dtype=np.float32)
    with pytest.raises(ValueError, match="within"):
        scale_policy_action(raw_action)
    steering, rear = float(raw_action[0]) * 1.57, float(raw_action[1]) * 6.28
    node._drive.velocity_rad_s = sign * 60.0
    node._command_callback(module.Float64MultiArray(data=[1.0, steering, rear]))
    node._control_tick()
    data = record(node, module)
    assert abs(steering) > math.pi / 2 and abs(rear) > 6.28
    assert data["bounded_steering_target_rad"] == sign * 1.57
    assert data["bounded_drive_target_nm"] == sign * 6.28
    assert data["slewed_steering_target_rad"] == sign * 1.57
    assert data["steering_target_clipped"] == data["drive_target_clipped"] == 1
    assert data["steering_slew_limited"] == 0
    assert data["drive_torque_saturated"] == 0
    assert data["drive_final_torque_nm"] == sign * 6.28
    assert data["drive_damping_torque_nm"] == 0
    motion = {call.args[0]: call.args[1] for call in node._tx_bus.send.call_args_list}
    encoded = module.parse_mit_motion(motion[0x502], node._mit_ranges)
    assert node._steering.encoder_sign * (encoded.position_rad - node._mit_zero_rad) == pytest.approx(
        sign * 1.57, abs=3 / 65535,
    )
    encoded_rear_nm = module.parse_current_feedback(motion[0x101]) * 0.3 * node._drive.torque_sign
    assert encoded_rear_nm == pytest.approx(sign * 6.28, abs=0.00015)


@pytest.mark.parametrize("target", [-3.2, 3.2])
def test_direct_encoder_still_rejects_out_of_range_native_mit_position(mit_node, target):
    node, module, _ = mit_node
    node._send_mit(target)
    node._control_tick()
    assert node._fault_latched and node._fault_reason == "steering_mit_send_failed"
    assert not any(call.args[0] == 0x502 for call in node._tx_bus.send.call_args_list)
    assert not any(call.args[0] == 0x101 and call.args[1][0] == 0xC0
                   and parse_current_feedback(call.args[1]) != 0 for call in node._tx_bus.send.call_args_list)


def test_direct_current_encoder_overflow_still_faults_and_disables(mit_node):
    node, module, _ = mit_node
    node._outputs_active = True
    node._send_torque(node._drive, 1e9, .3)
    node._control_tick()
    assert node._fault_latched and node._fault_reason == "drive_can_send_failed"
    assert node._disable_wait
    assert not any(call.args[0] == 0x101 and call.args[1][0] == 0xC0
                   and parse_current_feedback(call.args[1]) != 0 for call in node._tx_bus.send.call_args_list)


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
            motor.sample_monotonic = motor.status_sample_monotonic = clock.value - .001
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


def test_continuous_a3_state_keeps_extra_turn_visible_and_prevents_fault_clear(mit_node):
    node, module, _ = mit_node
    motor = node._steering
    a3 = bytes([0xA3]) + struct.pack("<Hi", 0, 16384)
    node._pending = (motor, 0xA3, 99.999, 100.002, 7)
    node._consume(frame(module, motor, a3), 100.0)
    assert motor.sample_monotonic == 99.99
    assert motor.position_sample_monotonic == 99.99  # Paired state still refers to the older sample.
    node._pending = (motor, 0xA2, 100, 100.002, 5)
    node._consume(frame(module, motor, bytes([0xA2]) + struct.pack("<i", 0), stamp=100.001), 100.001)
    assert motor.position_rad == pytest.approx(-2 * math.pi)
    assert motor._unwrapped_rad == pytest.approx(-2 * math.pi)
    assert motor.sample_monotonic == 100
    assert motor.position_sample_monotonic == 100 and motor.velocity_sample_monotonic == 100.001
    node._requested_enable = False
    node._fault("injected_normal_fault")
    node._control_tick()
    assert node._fault_latched and node._fault_reason == "injected_normal_fault"


def test_actual_current_torque_keeps_kernel_receipt_time_and_no_clipping(mit_node):
    node, module, _ = mit_node
    motor = node._steering
    node._consume(frame(module, motor, bytes([0xA1]) + struct.pack("<i", 20000), stamp=99.995), 100.01)
    assert motor.torque_nm == pytest.approx(-15.6)
    assert motor.torque_sample_monotonic == 99.995  # Arrival time, not dequeue/publication time.


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
    assert status["policy_command_rate_hz"] == 50
    assert status["state_query_rate_hz"] == status["state_status_fault_rate_hz"] == 100
    assert status["mit_command_rate_hz"] == status["motor_control_rate_hz"] == status["drive_command_rate_hz"] == 200


def test_fault_and_status_publish_even_without_fresh_state_or_running_worker(mit_node):
    node, _, _ = mit_node
    node._steering.sample_monotonic = node._drive.sample_monotonic = 0
    node._fault("mit_io_error", hard=True)
    node._publish_feedback()
    node._state_publisher.publish.assert_not_called()
    assert node._fault_publisher.publish.call_args.args[0].data
    assert json.loads(node._status_publisher.publish.call_args.args[0].data)["mit_restart_required"]


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
               0xB0: b"\xb0\x00" + struct.pack("<f", .78) + b"\x00", 0xAE: bytes([0xAE]) + bytes(7)}
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
               0xB0: b"\xb0\x00" + struct.pack("<f", .78) + b"\x00", 0xAE: bytes([0xAE]) + bytes(7)}
    node._query = Mock(side_effect=lambda motor, opcode, *args: answers[opcode])
    node._startup()
    node._tx_bus.send.assert_not_called()
    assert all(call.args[1] in (0xA3, 0xB0, 0xF0, 0xF1, 0xAE) and len(call.args) == 4
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


def test_all_live_state_queries_use_100hz_not_mit_rate(mit_node):
    node, module, clock = mit_node
    node._next_b0_at = node._next_f0_at = math.inf
    queries = []
    for index in range(200):
        clock.value = 100 + index / module.MIT_RATE_HZ
        node._schedule_polls(clock.value)
        # Finish this round without advancing its phase. CAN packets are
        # serialized; 100 Hz is per query type, not the total packet rate.
        while node._pending is not None:
            queries.append((clock.value, node._pending[0].name, node._pending[1]))
            node._pending = None
            if not node._poll_jobs:
                break
            node._schedule_polls(clock.value)
    for motor in ("steering", "drive"):
        for opcode in (0xA3, 0xA2, 0xA1, 0xAE):
            times = [stamp for stamp, name, op in queries if name == motor and op == opcode]
            assert len(times) == 100
            assert all(b - a == pytest.approx(.01) for a, b in zip(times, times[1:]))


def test_poll_phase_skips_missed_rounds_without_bursts(mit_node):
    node, _, clock = mit_node
    node._next_b0_at = node._next_f0_at = math.inf
    for stamp in (100, 100.031):
        clock.value = stamp
        node._schedule_polls(stamp)
        node._pending = None
        node._poll_jobs.clear()
    assert node._next_poll_at == pytest.approx(100.04)
    node._schedule_polls(clock.value)
    assert node._pending is None and not node._poll_jobs


def test_repeated_mit_sends_cannot_refresh_50hz_ros_command_watchdog(mit_node):
    node, module, clock = mit_node
    node._command_callback(module.Float64MultiArray(data=[1.0, .05, 0.0]))
    for index in range(18):
        clock.value = 100 + index * .005
        for motor in (node._steering, node._drive):
            motor.sample_monotonic = motor.status_sample_monotonic = clock.value
        node._mit_f1_at = clock.value
        node._control_tick()
    assert node._command_monotonic == 100
    assert node._disable_wait and not record(node, module)["gate_enabled"]
    assert "command_timeout" in json.loads(node._event_publisher.publish.call_args.args[0].data)["disable_reasons"]


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
@pytest.mark.parametrize("send_work_s", [0.0, .00002])
def test_single_owner_loop_startup_polling_and_command_timing(mit_node, monkeypatch, dry_run, send_work_s):
    node, module, clock = mit_node
    node._mit_config_ready = False
    node._requested_enable = False
    node._drive_target_nm = node._requested_drive_target_nm = .1
    node._dry_run = dry_run
    node._send_mit = Mock(wraps=node._send_mit)
    target_times = []
    send_mit = node._send_mit
    def record_target(target, **kwargs):
        target_times.append(clock.value)
        return send_mit(target, **kwargs)
    node._send_mit = record_target
    sent = []
    closed = []
    runtime_queries = []
    policy_times = []
    feedback_times = []
    runtime_started = []
    original_startup = node._startup

    def startup():
        original_startup()
        runtime_started.append(True)

    node._startup = startup

    class Bus:
        def __init__(self, *args):
            self.replies = deque()
            self.f0 = bytes([0xF0]) + struct.pack("<HHH", 955, 4500, 1800)
            self.next_policy = self.next_feedback = 100.0

        def send(self, can_id, payload):
            sent.append((clock.value, can_id, payload))
            clock.value += send_work_s
            mid = can_id & 255
            if can_id & 0x400:
                self.replies.append(module.CanReply(mid, feedback(parse_f0(self.f0), status=1), False, 0, clock.value))
                return
            op = payload[0]
            if runtime_started and op in (0xA1, 0xA2, 0xA3, 0xAE):
                runtime_queries.append((clock.value, mid, op))
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
                    0xAE: bytes([0xAE]) + bytes(5) + bytes([
                        (5 if mid == 2 else 2) if node._mit_sent_this_enable else 0, 0]),
                }[op]
            self.replies.append(module.CanReply(mid, answer, False, 0, clock.value))

        def receive(self, timeout):
            clock.value += .00001 if self.replies else max(.00001, timeout)
            if node._mit_config_ready:
                if clock.value >= self.next_policy:
                    node._command_callback(module.Float64MultiArray(data=[1.0, .05, .1]))
                    policy_times.append(clock.value)
                    self.next_policy += 1.0 / module.COMMAND_RATE_HZ
                if clock.value >= self.next_feedback:
                    node._publish_feedback()
                    feedback_times.append(clock.value)
                    self.next_feedback += module.STATE_DT_S
            if clock.value >= 100.12:
                node._running = False
            return self.replies.popleft() if self.replies else None

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module, "SocketCan", Bus)
    node._can_interface = "can0"
    node._run_io()
    assert closed == [True] and not node._hard_fault
    assert len(target_times) >= 20
    assert all(.005 - 1e-9 <= b - a <= .0051 for a, b in zip(target_times, target_times[1:]))
    assert len(policy_times) == 7 and len(feedback_times) == 13
    assert node._fault_publisher.publish.call_count == node._status_publisher.publish.call_count == 13
    assert node._control_publisher.publish.call_count == node._control_sequence
    for motor_id in (1, 2):
        for opcode in (0xA3, 0xA2, 0xA1, 0xAE):
            times = [t for t, mid, op in runtime_queries if mid == motor_id and op == opcode]
            assert 11 <= len(times) <= 13
            assert all(.0097 <= b - a <= .0103 for a, b in zip(times, times[1:]))
    assert node._disable_confirmed and not node._disable_wait
    if dry_run:
        assert all(not (can_id & 0x400) and len(data) == 1 and data[0] in (0xF0, 0xF1, 0xA3, 0xA2, 0xA1, 0xB0, 0xAE)
                   for _, can_id, data in sent)
        return
    assert sum(can_id == 0x102 and data[0] == 0xB1 for _, can_id, data in sent) == 1
    mit_times = [t for t, can_id, data in sent if can_id == 0x502]
    drive_times = [t for t, can_id, data in sent if can_id == 0x101 and data[0] == 0xC0
                   and parse_current_feedback(data) != 0]
    assert len(mit_times) >= 20 and len(drive_times) >= 20
    assert all(.005 - 1e-9 <= b - a <= .0051 for a, b in zip(mit_times, mit_times[1:]))
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


@pytest.fixture
def mit_constructor(node_module, monkeypatch):
    module = node_module("motor_mit_io")
    publishers = []
    overrides = {}

    class FakeNode:
        def __init__(self, name):
            self.node_name = name
            values = {}
            self.parameter_descriptors = {}
            def declare(name, value, descriptor):
                self.parameter_descriptors[name] = descriptor
                return values.setdefault(name, overrides.get(name, value))
            self.declare_parameter = declare
            self.get_parameter = lambda name: SimpleNamespace(value=values[name])
            self.create_publisher = lambda kind, topic, qos: publishers.append(topic) or Mock()
            self.create_subscription = Mock(return_value=Mock())
            self.create_timer = Mock(return_value=Mock())
            self.get_logger = Mock(return_value=Mock())

    fake_thread = Mock()
    monkeypatch.setattr(module, "Node", FakeNode)
    monkeypatch.setattr(module, "threading", SimpleNamespace(
        Lock=threading.Lock, Thread=Mock(return_value=fake_thread)))

    def construct(**parameters):
        overrides.clear()
        overrides.update(parameters)
        return module.MotorMitIo()

    return construct, publishers, fake_thread


def test_constructor_registers_existing_ros_contract_and_no_can_pollers(mit_constructor):
    construct, publishers, fake_thread = mit_constructor
    node = construct()
    assert node.node_name == "trackstand_motor_io"
    assert node._steering.feedback_id == 2 and node._drive.feedback_id == 1
    assert node._steering.encoder_sign == -1 and node._drive.encoder_sign == -1
    assert (node._command_rate_hz, node._poll_rate_hz, node._control_rate_hz) == (50, 100, 200)
    assert node.create_timer.call_count == 1
    assert node.create_timer.call_args.args == (.01, node._publish_feedback)
    assert node._mit_kp == 10 and node._mit_kd == .2
    assert all(d.read_only for d in node.parameter_descriptors.values())
    assert set(publishers) == {f"/trackstand/{name}" for name in (
        "motor_state", "motor_control", "motor_event", "motor_fault", "motor_status")}
    assert node.create_subscription.call_args.args[1] == "/trackstand/motor_command"
    fake_thread.start.assert_called_once()
    assert node._tx_bus is None  # The worker owns socket creation and closing.


@pytest.mark.parametrize("parameter,value", [
    ("command_rate_hz", 100.0), ("state_rate_hz", 200.0), ("actuation_rate_hz", 50.0),
    ("command_rate_hz", math.nan), ("state_rate_hz", 0.0), ("actuation_rate_hz", math.inf),
    ("command_timeout_s", .02), ("feedback_timeout_s", .01),
])
def test_constructor_rejects_profile_mismatches_and_too_short_watchdogs(mit_constructor, parameter, value):
    construct, _, fake_thread = mit_constructor
    with pytest.raises(ValueError):
        construct(**{parameter: value})
    fake_thread.start.assert_not_called()


def test_legacy_shared_yaml_rate_keys_cannot_override_mit_divisions(mit_constructor):
    construct, _, _ = mit_constructor
    node = construct(poll_rate_hz=200.0, control_rate_hz=50.0)
    assert (node._command_rate_hz, node._poll_rate_hz, node._control_rate_hz) == (50, 100, 200)


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
    node._publish_feedback()
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
        motor.sample_monotonic = motor.status_sample_monotonic = clock.value
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
    # A retained selected mode requires subsequent inactive F1 verification.
    assert "steering" in node._disable_f1_wait and not node._disable_confirmed
    node._consume(frame(module, node._steering, feedback(node._mit_ranges, status=1)), clock.value)
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


def test_new_episode_submits_direct_target_without_disabled_time_slew(mit_node):
    node, module, clock = mit_node
    node._control_tick()
    node._requested_enable = False
    node._control_tick()
    confirm_disable(node, module)
    clock.value += 1
    for motor in (node._steering, node._drive):
        motor.sample_monotonic = motor.status_sample_monotonic = clock.value
    node._command_monotonic = clock.value
    node._requested_enable = True
    node._steering_target_rad = node._requested_steering_target_rad = 1.57
    node._control_tick()
    assert record(node, module)["slewed_steering_target_rad"] == pytest.approx(1.57)
    assert record(node, module)["steering_slew_limited"] == 0


@pytest.mark.parametrize("motor_name,speed", [("steering", 30), ("drive", 70)])
def test_mit_speed_feedback_is_not_replaced_by_a_simulator_speed_constraint(mit_node, motor_name, speed):
    node, _, clock = mit_node
    node._requested_enable = False
    motor = getattr(node, f"_{motor_name}")
    motor.velocity_rad_s = speed
    node._control_tick()
    assert not node._fault_latched and motor.velocity_rad_s == speed
    node._publish_feedback()
    assert not node._fault_publisher.publish.call_args.args[0].data
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
        motor.sample_monotonic = motor.status_sample_monotonic = clock.value
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
               0xB0: b"\xb0\x00" + struct.pack("<f", .78) + b"\x00", 0xAE: bytes([0xAE]) + bytes(7)}
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
