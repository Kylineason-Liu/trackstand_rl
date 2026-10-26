"""Exercise the actual policy constructor with ROS stand-ins and real ONNX."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def startup(node_module, monkeypatch):
    overrides = {}

    class Node:
        def __init__(self, name):
            self.parameters = {}
            self.descriptors = {}
            self.create_publisher = Mock(side_effect=lambda *args: Mock())
            self.create_subscription = Mock()
            self.create_timer = Mock()
            self.get_logger = Mock(return_value=Mock())

        def declare_parameter(self, name, default, descriptor):
            self.parameters[name] = overrides.get(name, default)
            self.descriptors[name] = descriptor

        def get_parameter(self, name):
            return SimpleNamespace(value=self.parameters[name])

    monkeypatch.setattr(sys.modules["rclpy.node"], "Node", Node)
    monkeypatch.setattr(sys.modules["ament_index_python.packages"], "get_package_share_directory",
                        lambda _: str(Path(__file__).parents[1]))
    module = node_module("policy_controller")
    return module, overrides


def test_v1_constructor_loads_embedded_actor_and_starts_disabled(startup):
    module, _ = startup
    node = module.PolicyController()
    assert node._last_observation.shape == (11,)
    assert not node._state_machine.enabled and node._rear_episode_zero_rad is None
    assert node._episode_index == 0
    assert all(descriptor.read_only for descriptor in node.descriptors.values())
    assert [call.args[0] for call in node.create_timer.call_args_list] == [.02, .5]
    assert node._motor_publisher.publish.call_args.args[0].data == [0., 0., 0.]
    assert node._kickstand_publisher.publish.call_args.args[0].data == "down"
    subscriptions = {call.args[1]: call.args[3] for call in node.create_subscription.call_args_list}
    assert subscriptions["/trackstand/imu"] == subscriptions["/trackstand/motor_state"] == 1


@pytest.mark.parametrize("key,value", [
    ("policy_rate_hz", 100.), ("max_steering_target_rad", math.pi / 2),
    ("imu_timeout_s", -1), ("max_roll_rad", math.nan), ("max_steering_start_rad", 2.),
])
def test_incompatible_startup_parameters_are_rejected(startup, key, value):
    module, overrides = startup
    overrides[key] = value
    with pytest.raises(ValueError):
        module.PolicyController()


def test_wrong_model_checksum_prevents_any_enabled_command(startup):
    module, overrides = startup
    overrides["model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        module.PolicyController()


def test_v0_model_is_rejected_even_with_its_matching_checksum(startup):
    module, overrides = startup
    overrides["model_path"] = str(Path(__file__).parents[1] / "models" /
                                  "trackstand_rl_rsl_2026-09-22_19-45-06_model_91850.onnx")
    overrides["model_sha256"] = "9232a0723df8806e8b5f6fb4e39a7a11ccb5af8bf058d9026eddb57dab8fd806"
    with pytest.raises(ValueError, match="11"):
        module.PolicyController()
