"""Versioned, unit-explicit diagnostic schemas and transport results (no ROS dependency)."""

from dataclasses import dataclass
import math


MOTOR_CONTROL_FIELDS = (
    "sequence", "ros_time_s", "monotonic_time_s", "cycle_interval_ms", "work_duration_ms",
    "dry_run", "requested_enable", "gate_enabled", "outputs_active", "fault_latched",
    "command_age_ms", "command_ready", "steering_feedback_ready", "drive_feedback_ready",
    "steering_pair_age_ms", "drive_pair_age_ms",
    "steering_position_age_ms", "steering_velocity_age_ms", "drive_position_age_ms", "drive_velocity_age_ms",
    "steering_position_rad", "steering_velocity_rad_s", "drive_position_rad", "drive_velocity_rad_s",
    "requested_steering_target_rad", "requested_drive_target_nm", "bounded_steering_target_rad",
    "bounded_drive_target_nm", "steering_target_clipped", "drive_target_clipped",
    "slewed_steering_target_rad", "steering_slew_limited", "steering_error_rad",
    "steering_p_torque_nm", "steering_d_torque_nm", "steering_unclipped_torque_nm",
    "steering_final_torque_nm", "steering_torque_saturated",
    "drive_damping_torque_nm", "drive_unclipped_torque_nm", "drive_final_torque_nm", "drive_torque_saturated",
    "steering_current_command_a", "steering_encoded_current_a", "steering_encoded_torque_nm",
    "steering_send_attempted", "steering_send_success", "steering_send_duration_ms",
    "drive_current_command_a", "drive_encoded_current_a", "drive_encoded_torque_nm",
    "drive_send_attempted", "drive_send_success", "drive_send_duration_ms",
    "disable_attempted", "steering_zero_send_success", "steering_disable_send_success",
    "drive_zero_send_success", "drive_disable_send_success",
    "steering_position_query_ms", "steering_velocity_query_ms", "drive_position_query_ms", "drive_velocity_query_ms",
    "steering_position_query_failures", "steering_velocity_query_failures",
    "drive_position_query_failures", "drive_velocity_query_failures",
)

POLICY_TIMING_FIELDS = (
    "sequence", "ros_time_s", "monotonic_time_s", "cycle_interval_ms", "work_duration_ms",
    "observation_build_ms", "inference_ms", "action_processing_ms", "inference_success",
    "ready", "policy_enabled", "imu_age_ms", "motor_age_ms", "rc_age_ms", "motor_fault",
)


def age_ms(now: float, sample_time: float) -> float:
    """Monotonic sample age, or NaN when a sample has never arrived."""
    return math.nan if sample_time <= 0.0 else max(0.0, now - sample_time) * 1000.0


def telemetry_values(fields: tuple[str, ...], values: dict) -> list[float]:
    """Unavailable/not-applicable measurements are NaN, never fabricated zeros."""
    unknown = set(values).difference(fields)
    if unknown:
        raise ValueError(f"unknown telemetry fields: {sorted(unknown)}")
    return [float(values.get(field, math.nan)) for field in fields]


def motor_disable_reasons(
    requested_enable: bool,
    command_present: bool,
    command_ready: bool,
    steering_constant_ready: bool,
    drive_constant_ready: bool,
    steering_fresh: bool,
    drive_fresh: bool,
    fault_latched: bool,
) -> list[str]:
    reasons = []
    if not requested_enable:
        reasons.append("enable_not_requested")
    if not command_present:
        reasons.append("command_not_received")
    elif not command_ready:
        reasons.append("command_timeout")
    if not steering_constant_ready:
        reasons.append("steering_torque_constant_unavailable")
    if not drive_constant_ready:
        reasons.append("drive_torque_constant_unavailable")
    if not steering_fresh:
        reasons.append("steering_feedback_missing_or_stale")
    if not drive_fresh:
        reasons.append("drive_feedback_missing_or_stale")
    if fault_latched:
        reasons.append("fault_latched")
    return reasons


@dataclass
class MotorSendResult:
    attempted: bool = False
    success: bool = False
    current_command_a: float = math.nan
    encoded_current_a: float = math.nan
    encoded_torque_nm: float = math.nan
    duration_ms: float = math.nan
    error: str | None = None


@dataclass
class DisableResult:
    attempted: bool = False
    steering_zero_success: bool | None = None
    steering_disable_success: bool | None = None
    drive_zero_success: bool | None = None
    drive_disable_success: bool | None = None
    errors: tuple[str, ...] = ()


def send_result_values(prefix: str, result: MotorSendResult) -> dict:
    return {
        f"{prefix}_current_command_a": result.current_command_a,
        f"{prefix}_encoded_current_a": result.encoded_current_a,
        f"{prefix}_encoded_torque_nm": result.encoded_torque_nm,
        f"{prefix}_send_attempted": result.attempted,
        f"{prefix}_send_success": result.success,
        f"{prefix}_send_duration_ms": result.duration_ms,
    }
