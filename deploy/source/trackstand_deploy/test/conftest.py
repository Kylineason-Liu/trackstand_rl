"""Isolated ROS/CAN stand-ins for deterministic node tests without hardware."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


class JointStateStub:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None)
        self.name = []
        self.effort = []
        self.position = []
        self.velocity = []


class ArrayStub:
    def __init__(self, data=None):
        self.data = [] if data is None else data
        self.layout = SimpleNamespace(dim=[], data_offset=0)


@pytest.fixture
def node_module(monkeypatch):
    for name in (
        "can", "rclpy", "rclpy.node", "rclpy.qos", "rcl_interfaces", "rcl_interfaces.msg", "sensor_msgs", "sensor_msgs.msg",
        "std_msgs", "std_msgs.msg", "ament_index_python", "ament_index_python.packages",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["can"].CanError = RuntimeError
    sys.modules["can"].Message = SimpleNamespace
    sys.modules["rclpy.node"].Node = object
    sys.modules["rcl_interfaces.msg"].ParameterDescriptor = SimpleNamespace
    sys.modules["rclpy.qos"].QoSProfile = SimpleNamespace
    sys.modules["rclpy.qos"].ReliabilityPolicy = SimpleNamespace(BEST_EFFORT=0, RELIABLE=1)
    sys.modules["rclpy.qos"].DurabilityPolicy = SimpleNamespace(TRANSIENT_LOCAL=1)
    sys.modules["sensor_msgs.msg"].JointState = JointStateStub
    sys.modules["sensor_msgs.msg"].Imu = SimpleNamespace
    sys.modules["std_msgs.msg"].Bool = SimpleNamespace
    sys.modules["std_msgs.msg"].String = SimpleNamespace
    sys.modules["std_msgs.msg"].MultiArrayDimension = SimpleNamespace
    sys.modules["std_msgs.msg"].Float64MultiArray = ArrayStub
    sys.modules["std_msgs.msg"].Float32MultiArray = ArrayStub
    sys.modules["ament_index_python.packages"].get_package_share_directory = lambda _: "unused"

    def load(filename, canonical=False):
        if filename == "motor_io_position":
            load("motor_io", canonical=True)  # Bind inherited methods to this test's ROS/CAN stubs.
        path = Path(__file__).parents[1] / "trackstand_deploy" / f"{filename}.py"
        name = f"trackstand_deploy.{filename}" if canonical else f"trackstand_deploy._{filename}_telemetry_test"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    return load
