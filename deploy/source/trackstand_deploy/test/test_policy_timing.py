import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from trackstand_deploy.contracts import BalanceMode, BalanceStateMachine, SteeringTargetLimiter
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
    node._max_steering_target_rad = 1.57
    node._max_steering_start_rad = math.radians(5)
    node._steering_limiter = SteeringTargetLimiter()
    node._rear_episode_zero_rad = None
    node._episode_index = 0
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
    node._policy = SimpleNamespace(act=Mock(return_value=np.array([1.0, -1.0])))
    node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=123456000000000)))
    node.get_logger = Mock(return_value=Mock())
    return node


def timing(node):
    message = node._timing_publisher.publish.call_args.args[0]
    assert len(message.data) == len(POLICY_TIMING_FIELDS)
    return dict(zip(POLICY_TIMING_FIELDS, message.data))


def test_normal_timing_preserves_v1_scale_and_50hz_slew(policy_node):
    node = policy_node
    node._policy_tick()
    node._publish_motor_command.assert_called_once_with(True, 0.478, -6.28)
    data = timing(node)
    assert data["cycle_interval_ms"] == pytest.approx(20.0)
    assert data["imu_age_ms"] == pytest.approx(10.0)
    assert data["inference_success"] == 1.0 and math.isfinite(data["inference_ms"])
    assert node._action_publisher.publish.call_args.args[0].data == [1.0, -1.0]


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


def test_episode_rezero_only_rebases_rear_and_resets_steering_limiter(policy_node):
    node = policy_node
    node._motor_state = np.array([.04, 17.0, .1, -.2])
    node._policy_tick()
    assert node._episode_index == 1 and node._rear_episode_zero_rad == 17.0
    first = node._observation_publisher.publish.call_args.args[0].data
    assert first[5:9] == pytest.approx([.04, 0, .1, -.2])
    assert node._steering_limiter.target_rad == pytest.approx(.518)
    node._motor_state[1] += 3.5
    node._policy_tick()
    assert node._observation_publisher.publish.call_args.args[0].data[6] == pytest.approx(3.5)
    node._rc_callback(SimpleNamespace(data=False))
    assert node._rear_episode_zero_rad is None
    node._motor_state[1] = 27.
    node._rc_callback(SimpleNamespace(data=True))
    node._policy_tick()
    assert node._episode_index == 2 and node._rear_episode_zero_rad == 27.
    assert node._observation_publisher.publish.call_args.args[0].data[6] == 0
    assert node._steering_limiter.target_rad == pytest.approx(.518)


def test_new_episode_waits_for_centered_steering_without_relabeling_it(policy_node):
    node = policy_node
    node._state_machine = BalanceStateMachine()
    node._motor_state[0] = .2
    node._rc_callback(SimpleNamespace(data=True))
    node._policy_tick()
    assert not node._state_machine.enabled and node._rear_episode_zero_rad is None
    assert node._readiness_reasons(100.) == ["steering_not_centered_for_start"]
    node._motor_state[0] = .02
    node._policy_tick()
    assert node._state_machine.enabled and node._episode_index == 1
    assert node._observation_publisher.publish.call_args.args[0].data[5] == pytest.approx(.02)


def test_out_of_contract_actor_disables_instead_of_silently_clipping(policy_node):
    node = policy_node
    node._policy.act.return_value = np.array([2., 0])
    node._policy_tick()
    assert node._state_machine.mode == BalanceMode.FAULT_LATCHED
    node._publish_motor_command.assert_called_once_with(False, 0., 0.)
    node._action_publisher.publish.assert_not_called()


def test_slow_inference_cannot_refresh_expired_inputs(policy_node):
    node = policy_node

    def delayed(observation):
        node._imu_monotonic = 99.
        return np.zeros(2)

    node._policy.act.side_effect = delayed
    node._policy_tick()
    assert node._state_machine.mode == BalanceMode.FAULT_LATCHED
    node._publish_motor_command.assert_called_once_with(False, 0., 0.)


def test_legacy_wrapped_motor_state_never_refreshes_v1_inputs(policy_node, node_module):
    node = policy_node
    module = node_module("policy_controller")
    message = module.Float64MultiArray(data=[1., 2., 3., 4., 5., 6., 7.])
    message.layout.dim = [SimpleNamespace(label="motor_state_v2:legacy")]
    node._motor_callback(message)
    np.testing.assert_array_equal(node._motor_state, np.zeros(4))
    assert node._motor_monotonic == 99.99
