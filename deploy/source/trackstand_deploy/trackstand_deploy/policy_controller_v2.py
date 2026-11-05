"""V2 Beta policy with the existing sensor, kickstand and fault state machine."""

import json
import math
import time

import rclpy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String

from .contracts import BalanceMode
from .contracts_v2 import (
    COMMAND_LABEL, COMMAND_TOPIC, TASK, encoded_drive_kd, scale_policy_action_v2, validate_policy_manifest,
)
from .policy import OnnxPolicy
from .policy_controller import PolicyController

MODEL_FILENAME = "trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx"
MODEL_SHA256 = "a43583ec06d3c441d74d93057378fc25d8d79255a4ad312aeca6f59ad1c25c33"


class PolicyControllerV2(PolicyController):
    task_name = TASK
    node_name = "trackstand_policy_controller_v2"
    model_filename, model_sha256 = MODEL_FILENAME, MODEL_SHA256
    command_topic = COMMAND_TOPIC
    observation_label = "trackstand_v2:"
    action_scaler = staticmethod(scale_policy_action_v2)

    def __init__(self):
        self._backend_ready = False
        self._backend_monotonic = 0.
        super().__init__()
        self.create_subscription(String, "/trackstand/motor_status", self._backend_callback, 1)

    def _backend_callback(self, message):
        """A v1 motor-state publisher must never make a velocity policy ready."""
        try:
            status = json.loads(message.data)
            required = {"task": TASK, "drive_action": "velocity_rad_s", "motor_command_topic": COMMAND_TOPIC,
                        "mit_config_ready": True, "drive_mit_config_ready": True, "fault_latched": False,
                        "mit_restart_required": False, "recover_pending_settings": False}
            gains_rates = {"drive_mit_kd": encoded_drive_kd(), "drive_integral_gain_nm_per_rad": 3.3,
                           "drive_integral_rate_hz": 200., "policy_command_rate_hz": 50.,
                           "state_query_rate_hz": 100., "mit_command_rate_hz": 200.}
            ready = (all(type(status.get(key)) is type(value) and status[key] == value
                         for key, value in required.items()) and
                     all(math.isclose(float(status.get(key, math.nan)), value, rel_tol=0., abs_tol=1e-9)
                         for key, value in gains_rates.items()))
        except (ValueError, TypeError, AttributeError, OverflowError):
            ready = False
        self._backend_ready, self._backend_monotonic = ready, time.monotonic()
        if not ready and self._state_machine.enabled:
            self._state_machine.mode = BalanceMode.FAULT_LATCHED
            self._policy_fault_reasons = ["v2_motor_backend_not_ready"]
            self._rear_episode_zero_rad = None
            self._publish_motor_command(False, 0., 0.)

    def _readiness_reasons(self, now):
        reasons = super()._readiness_reasons(now)
        if not self._backend_ready:
            reasons.append("v2_motor_backend_not_ready")
        elif not 0 < self._backend_monotonic <= now or now-self._backend_monotonic > self._motor_timeout_s:
            reasons.append("v2_motor_backend_timeout")
        return reasons

    def _load_policy(self, model_path):
        self._policy_manifest = validate_policy_manifest(model_path)
        return OnnxPolicy(model_path)

    def _publish_motor_command(self, enable, steering_target, drive_target):
        message = Float64MultiArray(data=[float(enable), float(steering_target), float(drive_target)])
        message.layout.dim = [MultiArrayDimension(label=COMMAND_LABEL, size=3, stride=3)]
        self._motor_publisher.publish(message)


def main(args=None):
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = PolicyControllerV2()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            rclpy.try_shutdown()
