import math

import pytest

from trackstand_deploy.telemetry import (
    MOTOR_CONTROL_FIELDS,
    POLICY_TIMING_FIELDS,
    age_ms,
    motor_disable_reasons,
    telemetry_values,
)


def test_schemas_are_unique_and_unavailable_values_are_nan():
    for fields in (MOTOR_CONTROL_FIELDS, POLICY_TIMING_FIELDS):
        assert len(fields) == len(set(fields))
        values = telemetry_values(fields, {"sequence": 1})
        assert values[fields.index("sequence")] == 1.0
        assert math.isnan(values[fields.index("cycle_interval_ms")])


def test_unknown_fields_cannot_silently_change_schema():
    with pytest.raises(ValueError, match="unknown telemetry fields"):
        telemetry_values(POLICY_TIMING_FIELDS, {"wrong_units": 1})


def test_age_units_missing_sample_and_concurrent_sample():
    assert age_ms(10.0, 9.995) == pytest.approx(5.0)
    assert math.isnan(age_ms(10.0, 0.0))
    assert age_ms(10.0, 10.001) == 0.0


def test_command_timeout_has_explicit_reason_without_a_fault_latch():
    assert motor_disable_reasons(True, True, False, True, True, True, True, False) == ["command_timeout"]
    assert motor_disable_reasons(True, True, True, True, True, True, True, False) == []


def test_startup_and_missing_feedback_reasons():
    reasons = motor_disable_reasons(False, False, False, False, False, False, False, True)
    assert "enable_not_requested" in reasons and "command_not_received" in reasons
    assert "steering_torque_constant_unavailable" in reasons
    assert "drive_feedback_missing_or_stale" in reasons and "fault_latched" in reasons
