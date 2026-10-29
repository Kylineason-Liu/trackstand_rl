"""Actual ROS codecs with synthetic readings; no Node, UART or CAN is opened."""

import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "deploy/source/trackstand_deploy"))

import numpy as np
from rclpy.serialization import deserialize_message, serialize_message
from rclpy.time import Time
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

# Serial driver imports I2C transitively; no I2C implementation is exercised.
fake_smbus = ModuleType("smbus2")
fake_smbus.SMBus = Mock()
sys.modules["smbus2"] = fake_smbus

from trackstand_deploy.imu_publisher import ImuPublisher
from trackstand_deploy.motor_mit_io import MOTOR_CONTROL_FIELDS
from trackstand_deploy.policy_controller import PolicyController
from trackstand_deploy.telemetry import MOTOR_STATE_FIELDS, MOTOR_STATE_LABEL, telemetry_values


class Capture:
    def publish(self, message):
        self.message = deserialize_message(serialize_message(message), type(message))


source_at = time.monotonic() - .01
clock = SimpleNamespace(now=lambda: Time(nanoseconds=123456000000000))
sensor = SimpleNamespace(
    _imu=SimpleNamespace(get_timed_sample=lambda: (source_at, (1., 0., 0., 0.), (.2, -.3, .4))),
    _last_source_time=0., _source_timeout_s=.25, _sensor_to_bike=np.array([0., 0., 0., 1.]),
    _gyro_bias_rad_s=np.zeros(3), _frame_id="imu_link", _publisher=Capture(), get_clock=lambda: clock,
    get_logger=lambda: Mock(),
)
ImuPublisher._publish(sensor)
message = sensor._publisher.message
assert isinstance(message, Imu)
assert message.orientation.w == 1. and message.angular_velocity.y == -.3
age = (123456000000000 - message.header.stamp.sec * 10**9 - message.header.stamp.nanosec) / 1e9
assert .009 < age < .06
policy = SimpleNamespace(_imu_timeout_s=.06, _motor_timeout_s=.06, get_clock=lambda: clock)
PolicyController._imu_callback(policy, message)
assert .009 < time.monotonic() - policy._imu_monotonic < .06

state = Float64MultiArray(data=[.1, -.2, .3, -.4, time.time(), float("nan"), float("nan")])
state.layout.dim = [MultiArrayDimension(label=MOTOR_STATE_LABEL, size=7, stride=7)]
state = deserialize_message(serialize_message(state), Float64MultiArray)
PolicyController._motor_callback(policy, state)
np.testing.assert_array_equal(policy._motor_state, [.1, -.2, .3, -.4])
assert len(MOTOR_STATE_FIELDS) == 7

control = Float64MultiArray(data=telemetry_values(MOTOR_CONTROL_FIELDS, {
    "sequence": 1, "steering_driver_status_age_ms": 1., "drive_driver_status_age_ms": 2.,
}))
control.layout.dim = [MultiArrayDimension(
    label="motor_control_mit_v1:" + ",".join(MOTOR_CONTROL_FIELDS),
    size=len(MOTOR_CONTROL_FIELDS), stride=len(MOTOR_CONTROL_FIELDS),
)]
restored = deserialize_message(serialize_message(control), Float64MultiArray)
assert list(restored.data[-2:]) == [1., 2.]
assert restored.layout.dim[0].size == len(MOTOR_CONTROL_FIELDS)

result = dict(real_ROS_codecs=True, synthetic_IMU_backdated_stamp_seconds=age,
              policy_preserves_serial_age=True, motor_state_fields=7,
              motor_control_fields=len(MOTOR_CONTROL_FIELDS), no_ROS_node_or_hardware_opened=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
