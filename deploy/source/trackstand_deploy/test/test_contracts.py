import math

import numpy as np
import pytest

from trackstand_deploy.contracts import (
    BalanceMode,
    BalanceStateMachine,
    build_observation,
    correct_imu_sample,
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
        wheel_radius_m=0.07,
    )
    assert observation.shape == (13,)
    assert observation.dtype == np.float32
    assert observation[:7] == pytest.approx([0.0, -1.0, 0.1, 0.2, 0.3, 0.14, 0.0])
    assert observation[7] == pytest.approx(0.25)
    assert observation[8] == pytest.approx(4.0 - 2.0 * math.pi)
    assert observation[11] == pytest.approx(math.sin(0.25))
    assert observation[12] == pytest.approx(math.cos(0.25))


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


def test_action_scaling_enforces_hardware_steering_limit():
    steering, drive = scale_policy_action(np.array([2.0, -2.0]), math.radians(30.0))
    assert steering == pytest.approx(math.radians(30.0))
    assert drive == pytest.approx(-6.28)


def test_state_machine_requires_down_cycle_after_fault():
    machine = BalanceStateMachine()
    assert machine.set_rc(True, ready=True) == BalanceMode.BALANCING
    assert machine.update_ready(False) == BalanceMode.FAULT_LATCHED
    assert machine.set_rc(True, ready=True) == BalanceMode.FAULT_LATCHED
    assert machine.set_rc(False, ready=True) == BalanceMode.DOWN
    assert machine.set_rc(True, ready=True) == BalanceMode.BALANCING

