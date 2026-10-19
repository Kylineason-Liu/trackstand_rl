import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from trackstand_deploy.contracts import BalanceMode, BalanceStateMachine
from trackstand_deploy.telemetry import POLICY_TIMING_FIELDS


@pytest.fixture
def policy_node(node_module):
    module = node_module("policy_controller")
    module.time = SimpleNamespace(monotonic=lambda: 100.0)
    node = object.__new__(module.PolicyController)
    node._world_from_bike = np.array([0.0, 0.0, 0.0, 1.0])
    node._body_rate_rad_s = np.zeros(3)
    node._motor_state = np.zeros(4)
    node._imu_monotonic = node._motor_monotonic = node._rc_monotonic = 99.99
    node._imu_timeout_s = node._motor_timeout_s = 0.06
    node._rc_timeout_s = 0.15
    node._motor_fault = False
    node._max_roll_rad = math.radians(30)
    node._max_steering_target_rad = math.pi / 2
    node._wheel_radius_m = 0.07
    node._state_machine = BalanceStateMachine(BalanceMode.BALANCING, True)
    node._policy_sequence = 0
    node._last_policy_monotonic = 99.98
    node._last_policy_event_state = None
    node._policy_fault_reasons = []
    node._timing_dimension = SimpleNamespace()
    node._timing_publisher = Mock()
    node._policy_event_publisher = Mock()
    node._observation_publisher = Mock()
    node._action_publisher = Mock()
    node._publish_motor_command = Mock()
    node._command_kickstands = Mock()
    node._policy = SimpleNamespace(act=Mock(return_value=np.array([2.0, -2.0])))
    node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=123456000000000)))
    node.get_logger = Mock(return_value=Mock())
    return node


def timing(node):
    message = node._timing_publisher.publish.call_args.args[0]
    assert len(message.data) == len(POLICY_TIMING_FIELDS)
    return dict(zip(POLICY_TIMING_FIELDS, message.data))


def test_normal_timing_does_not_change_action_scaling(policy_node):
    node = policy_node
    node._policy_tick()
    node._publish_motor_command.assert_called_once_with(True, 3.14, -12.56)
    data = timing(node)
    assert data["cycle_interval_ms"] == pytest.approx(20.0)
    assert data["imu_age_ms"] == pytest.approx(10.0)
    assert data["inference_success"] == 1.0 and math.isfinite(data["inference_ms"])
    assert node._action_publisher.publish.call_args.args[0].data == [2.0, -2.0]


def test_disabled_tick_has_timing_but_no_fabricated_inference(policy_node):
    node = policy_node
    node._state_machine = BalanceStateMachine()
    node._policy_tick()
    node._publish_motor_command.assert_called_once_with(False, 0.0, 0.0)
    data = timing(node)
    assert math.isnan(data["inference_ms"]) and data["inference_success"] == 0.0
    node._policy.act.assert_not_called()


def test_inference_failure_latches_and_still_publishes_timing_and_reason(policy_node):
    node = policy_node
    node._policy.act.side_effect = RuntimeError("injected inference failure")
    node._policy_tick()
    assert node._state_machine.mode == BalanceMode.FAULT_LATCHED
    node._publish_motor_command.assert_called_once_with(False, 0.0, 0.0)
    assert timing(node)["inference_success"] == 0.0
    assert math.isfinite(timing(node)["inference_ms"])
    event = json.loads(node._policy_event_publisher.publish.call_args.args[0].data)
    assert event["latched_fault_reasons"] == ["policy_inference_error"]


@pytest.mark.parametrize("attribute,reason", [
    ("_imu_monotonic", "imu_timeout"), ("_motor_monotonic", "motor_state_timeout"), ("_rc_monotonic", "rc_timeout"),
])
def test_readiness_timeout_reason_and_latch(policy_node, attribute, reason):
    setattr(policy_node, attribute, 99.0)
    policy_node._policy_tick()
    assert policy_node._state_machine.mode == BalanceMode.FAULT_LATCHED
    event = json.loads(policy_node._policy_event_publisher.publish.call_args.args[0].data)
    assert reason in event["latched_fault_reasons"]


def test_roll_safety_diagnostics_match_existing_gravity_condition(policy_node):
    roll = math.radians(31)
    policy_node._world_from_bike = np.array([math.sin(roll / 2), 0.0, 0.0, math.cos(roll / 2)])
    assert policy_node._readiness_reasons(100.0) == ["roll_or_inversion_safety"]
    assert not policy_node._ready(100.0)


def test_events_are_transition_only(policy_node):
    policy_node._policy_tick()
    policy_node._policy_tick()
    assert policy_node._timing_publisher.publish.call_count == 2
    assert policy_node._policy_event_publisher.publish.call_count == 1
