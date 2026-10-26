import math

import numpy as np
import pytest

from trackstand_deploy.contracts import (
    BalanceMode,
    BalanceStateMachine,
    build_observation,
    correct_imu_sample,
    motor_fault_clear_allowed,
    rpy_degrees_to_quaternion_xyzw,
    scale_policy_action,
)


def test_upright_observation_order_and_units():
    observation = build_observation(
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([0.1, 0.2, 0.3]),
        steering_position_rad=0.25,
        rear_wheel_position_rad=4.0,
        steering_velocity_rad_s=0.4,
        rear_wheel_velocity_rad_s=2.0,
    )
    assert observation.shape == (11,)
    assert observation.dtype == np.float32
    assert observation[:5] == pytest.approx([0.0, -1.0, 0.1, 0.2, 0.3])
    assert observation[5] == pytest.approx(0.25)
    assert observation[6] == pytest.approx(4.0)
    assert observation[9] == pytest.approx(math.sin(0.25))
    assert observation[10] == pytest.approx(math.cos(0.25))


def test_imu_mount_correction_matches_identity_case():
    mounting = rpy_degrees_to_quaternion_xyzw([0.0, 0.0, 0.0])
    quaternion, rates = correct_imu_sample(
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([1.0, 2.0, 3.0]),
        mounting,
        np.array([0.1, 0.2, 0.3]),
    )
    assert quaternion == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert rates == pytest.approx([0.9, 1.8, 2.7])


def test_action_scaling_rejects_different_training_position_limit():
    with pytest.raises(ValueError, match="1.57"):
        scale_policy_action(np.array([1.0, -1.0]), math.radians(30.0))


def test_action_scaling_reaches_both_v1_endpoints():
    assert scale_policy_action(np.array([1.0, -1.0])) == pytest.approx((1.57, -6.28))
    assert scale_policy_action(np.array([-1.0, 1.0])) == pytest.approx((-1.57, 6.28))


@pytest.mark.parametrize("raw_action", [[1.0, -1.0], [-1.0, 1.0], [0.5, -0.25]])
def test_action_scaling_matches_bounded_v1_training_scales(raw_action):
    steering, drive = scale_policy_action(np.array(raw_action))
    assert steering == pytest.approx(raw_action[0] * 1.57)
    assert drive == pytest.approx(raw_action[1] * 6.28)


@pytest.mark.parametrize("raw_action", [[math.nan, 0.0], [0.0, math.inf], [-math.inf, 0.0], [0.0]])
def test_action_scaling_rejects_invalid_policy_outputs(raw_action):
    with pytest.raises(ValueError, match="policy action must contain two finite values"):
        scale_policy_action(np.array(raw_action), math.radians(90.0))


@pytest.mark.parametrize("limit", [0.0, -1.0, math.nan, math.inf])
def test_action_scaling_rejects_invalid_target_limits(limit):
    with pytest.raises(ValueError, match="steering target limit"):
        scale_policy_action(np.zeros(2), limit)


def test_state_machine_requires_down_cycle_after_fault():
    machine = BalanceStateMachine()
    assert machine.set_rc(True, ready=True) == BalanceMode.BALANCING
    assert machine.update_ready(False) == BalanceMode.FAULT_LATCHED
    assert machine.set_rc(True, ready=True) == BalanceMode.FAULT_LATCHED
    assert machine.set_rc(False, ready=True) == BalanceMode.DOWN
    assert machine.set_rc(True, ready=True) == BalanceMode.BALANCING


def test_motor_fault_clear_requires_disabled_and_steering_within_five_degrees():
    limit = math.radians(5.0)
    assert motor_fault_clear_allowed(False, math.radians(4.99), limit)
    assert motor_fault_clear_allowed(False, math.radians(-4.99), limit)
    assert not motor_fault_clear_allowed(False, limit, limit)
    assert not motor_fault_clear_allowed(False, math.radians(-6.0), limit)
    assert not motor_fault_clear_allowed(True, 0.0, limit)
