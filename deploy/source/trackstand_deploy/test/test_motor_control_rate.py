"""Backend-selected diagnostic sampling must never throttle commands or events."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trackstand_deploy.telemetry import (
    MOTOR_CONTROL_RATE_HZ, DisableResult, MotorControlPublishGate, MotorSendResult,
)


@pytest.mark.parametrize("tick_rate_hz", [50, 200, 250, 1000])
def test_publish_gate_keeps_fifty_hz_phase(tick_rate_hz):
    gate = MotorControlPublishGate()
    published = [index for index in range(2 * tick_rate_hz) if gate.due(100 + index / tick_rate_hz)]
    assert MOTOR_CONTROL_RATE_HZ == 50
    assert len(published) == 100
    assert published == list(range(0, 2 * tick_rate_hz, tick_rate_hz // 50))


def test_publish_gate_jitter_and_extra_mit_ticks_do_not_change_phase():
    gate = MotorControlPublishGate()
    ticks = sorted([100 + index * .005 + (.0002 if index % 4 == 0 else 0) for index in range(400)]
                   + [100 + index * .02 + .001 for index in range(100)])
    published = [stamp for stamp in ticks if gate.due(stamp)]
    assert len(published) == 100
    assert all(abs(stamp - (100.0002 + index * .02)) < .005 for index, stamp in enumerate(published))


def test_publish_gate_skips_missed_slots_without_catchup_burst():
    gate = MotorControlPublishGate()
    assert gate.due(0.0)
    assert not gate.due(.019)
    assert gate.due(.101)
    assert not gate.due(.101) and not gate.due(.119)
    assert gate.due(.120)  # Original phase, not .101 + .020.
    assert not gate.due(.120)


def test_publish_gate_cannot_invent_samples_when_control_is_slower():
    gate = MotorControlPublishGate()
    assert sum(gate.due(index / 20) for index in range(20)) == 20


@pytest.mark.parametrize("rate_hz", [50, 100, 200])
def test_publish_gate_supports_independent_selected_rates(rate_hz):
    gate = MotorControlPublishGate(rate_hz)
    assert sum(gate.due(100 + index / 1000) for index in range(1000)) == rate_hz
    assert gate.due(102.0)
    assert not gate.due(102.0)  # A delayed tick cannot produce a catch-up burst.


@pytest.mark.parametrize("rate_hz", [0, -1, float("nan"), float("inf")])
def test_publish_gate_rejects_invalid_rates(rate_hz):
    with pytest.raises(ValueError):
        MotorControlPublishGate(rate_hz)


@pytest.mark.parametrize("filename,classname", [
    ("motor_io", "MotorIo"), ("motor_io_position", "MotorIoPosition"), ("motor_mit_io", "MotorMitIo"),
])
def test_all_backends_sample_numeric_only_and_keep_immediate_events(node_module, filename, classname, monkeypatch):
    module = node_module(filename)
    node = object.__new__(getattr(module, classname))
    node._initialize_command_state()
    node._dry_run = False
    node._outputs_active = True
    node._control_publisher, node._event_publisher = Mock(), Mock()
    node._control_dimension = SimpleNamespace()
    node._mit_encoded_command = None
    node._disable_wait, node._disable_confirmed = {}, False
    # Legacy backends retain 50 Hz; MIT now explicitly requests 200 Hz.
    rate_hz = 200 if filename == "motor_mit_io" else 50
    emitter_module = sys.modules[node._publish_control_telemetry.__module__]
    encode = Mock(wraps=emitter_module.telemetry_values)
    monkeypatch.setattr(emitter_module, "telemetry_values", encode)
    for index in range(200):
        node._control_sequence += 1
        stamp = 100 + index * .005
        enabled = index not in (1, 2)
        node._fault_latched = index == 2
        reasons = [] if enabled else ["injected_fault" if index == 2 else "enable_not_requested"]
        disable = DisableResult(errors=("injected send error",)) if index == 3 else DisableResult()
        node._publish_control_telemetry(stamp, {
            "sequence": node._control_sequence, "ros_time_s": stamp,
            "monotonic_time_s": stamp, "gate_enabled": enabled,
        }, reasons, MotorSendResult(), MotorSendResult(), disable)
        if index in (1, 2, 3):
            assert node._control_publisher.publish.call_count == (index + 1 if rate_hz == 200 else 1)
            event = json.loads(node._event_publisher.publish.call_args.args[0].data)
            assert event["sequence"] == index + 1  # Event delivered on the unsampled tick.
            if index == 3:
                assert "disable_send_failed" in event["disable_reasons"]
                assert event["disable_send_errors"] == ["injected send error"]
    assert node._control_publisher.publish.call_count == encode.call_count == rate_hz
    assert node._event_publisher.publish.call_count == 5
    records = [dict(zip(node._control_fields, call.args[0].data))
               for call in node._control_publisher.publish.call_args_list]
    assert [record["sequence"] for record in records] == list(range(1, 201, 200 // rate_hz))
    assert isinstance(node._control_publish_gate, MotorControlPublishGate)
