"""Raw C2 waveform/transport checks; never open a real CAN socket."""

import math
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "experiments" / "steering_sine_position.py"


@pytest.fixture
def code():
    source = SCRIPT.read_text()
    namespace = {"__name__": "position_script_test"}
    exec(compile(source, str(SCRIPT), "exec"), namespace)
    return namespace


def test_execute_needs_no_confirmation_and_defaults_to_one_rad(code, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--execute"])
    args = code["parse_args"]()
    assert args.execute and not args.preview
    assert code["RATE_HZ"] == 200.0
    assert (args.amplitude, args.period, args.duration, args.motor_id, args.sign) == (1, 10, 0, 2, -1)


def test_c2_sine_extrema_units_and_sign(code):
    args = SimpleNamespace(amplitude=1, period=10, gear_ratio=1, sign=-1)
    payload = code["c2_payload"]
    for elapsed, count in ((0, 0), (2.5, -2608), (5, 0), (7.5, 2608), (10, 0)):
        assert payload(args, elapsed) == b"\xc2" + struct.pack("<i", count)
    args.amplitude = math.pi / 2
    assert payload(args, 2.5) == b"\xc2" + struct.pack("<i", -4096)
    args.sign = 1  # Native positive position: left from the rider's view.
    assert payload(args, 2.5) == b"\xc2" + struct.pack("<i", 4096)


@pytest.mark.parametrize("zero_count", [123456789, -2097153])
def test_c2_uses_native_startup_count_without_sign_inverting_it(code, zero_count):
    args = SimpleNamespace(amplitude=1, period=10, gear_ratio=1, sign=-1)
    for elapsed, offset in ((0, 0), (2.5, -2608), (7.5, 2608)):
        assert code["c2_payload"](args, elapsed, zero_count) == b"\xc2" + struct.pack("<i", zero_count + offset)


def test_preview_without_can():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--preview", "--duration", "0.01"],
        check=True, capture_output=True, text=True,
    )
    assert result.stdout.splitlines() == [
        "PREVIEW ONLY: startup count assumed 0; execution reads A3 once.",
        "t=0.000s 102#C200000000", "t=0.005s 102#C2F8FFFFFF",
    ]


@pytest.fixture
def fake_can(code):
    state = SimpleNamespace(now=0.0, frames=[], closed=False, zero_count=123456789, fault=None, timeouts=[])

    class Bus:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            state.closed = True

        def bind(self, address):
            assert address == ("can0",)

        def setsockopt(self, level, option, data):
            assert struct.unpack("=II", data) == (2, 0xC00007FF)

        def settimeout(self, timeout):
            state.timeouts.append(timeout)

        def send(self, frame):
            state.frames.append((state.now, frame))

        def recv(self, size):
            assert size == 16
            if state.fault == "no_reply":
                raise TimeoutError("no A3 reply")
            host = 1 if state.fault == "wrong_id" else 2
            length = 5 if state.fault == "short_reply" else 7
            command = 0xC2 if state.fault == "wrong_command" else 0xA3
            # Nonzero single-turn field must NOT be used as the startup origin.
            data = bytes([command]) + struct.pack("<Hi", 15300, state.zero_count)
            state.now += 2.0 if state.fault else 0.5  # Sine phase must start AFTER the startup read.
            return struct.pack("=IB3x8s", host, length, data)

    code["socket"] = SimpleNamespace(socket=lambda *args: Bus(), AF_CAN=29, SOCK_RAW=3, CAN_RAW=1,
                                      SOL_CAN_RAW=101, CAN_RAW_FILTER=1)
    code["time"] = SimpleNamespace(monotonic=lambda: state.now,
                                   sleep=lambda dt: setattr(state, "now", state.now + dt))
    return state


@pytest.mark.parametrize("zero_count", [123456789, -2097153])
def test_transport_reads_a3_once_then_sends_c2_at_200_hz(code, fake_can, zero_count):
    state = fake_can
    state.zero_count = zero_count
    args = SimpleNamespace(interface="can0", motor_id=2, amplitude=1, period=10, duration=0.08,
                           gear_ratio=1, sign=-1)
    code["execute"](args)
    host, length, data = struct.unpack("=IB3x8s", state.frames[0][1])
    assert host == 0x102 and length == 1 and data[0] == 0xA3
    motion = state.frames[1:]
    assert [timestamp for timestamp, _ in motion] == pytest.approx([0.5 + i * 0.005 for i in range(16)])
    assert struct.unpack("<i", struct.unpack("=IB3x8s", motion[0][1])[2][1:5])[0] == zero_count
    for timestamp, frame in motion:
        host, length, payload = struct.unpack("=IB3x8s", frame)
        assert host == 0x102 and length == 5
        assert payload[:length] == code["c2_payload"](args, timestamp - 0.5, zero_count)
    assert state.timeouts[-1] is None
    assert state.closed


@pytest.mark.parametrize("fault", ["no_reply", "wrong_id", "short_reply", "wrong_command"])
def test_missing_or_invalid_startup_feedback_never_sends_c2(code, fake_can, fault):
    state = fake_can
    state.fault = fault
    with pytest.raises(TimeoutError):
        code["execute"](SimpleNamespace(interface="can0", motor_id=2))
    assert len(state.frames) == 1
    assert struct.unpack("=IB3x8s", state.frames[0][1])[2][0] == 0xA3
    assert state.timeouts[-1] is None
    assert state.closed
