"""Standalone feedback collector checks with fake CAN only; no hardware I/O."""

import csv
import importlib.util
import math
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE.parents[1] / "experiments" / "motor_feedback.py"


def load_file(path, name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module(monkeypatch):
    return load_file(SCRIPT, "standalone_motor_feedback_test", monkeypatch)


@pytest.fixture
def fake_can(module, monkeypatch):
    state = SimpleNamespace(now=0.0, sent=[], pending=[], timeout=None, closed=False, suppress=set(), prefix=[],
                            filters=None, counts={1: 20000, 2: -20000}, kt=1.25, speed_raw=-12345, current_raw=-1234,
                            target_filters=None, target_closed=False, target_pending=queue.Queue(),
                            targets_done=threading.Event(), wait_for_targets=False)

    class Bus:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            state.closed = True

        def bind(self, address):
            assert address == ("can0",)

        def setsockopt(self, level, option, payload):
            state.filters = payload

        def setblocking(self, blocking):
            state.timeout = None if blocking else 0

        def settimeout(self, timeout):
            state.timeout = timeout

        def send(self, frame):
            if state.wait_for_targets:
                assert state.targets_done.wait(1.0)
            host, length, data = struct.unpack("=IB3x8s", frame)
            assert length == 1 and data[0] in (0xA1, 0xA2, 0xA3, 0xB0)  # No motor writes, even at shutdown.
            device_id, command = host & 0xFF, data[0]
            assert host == 0x100 | device_id
            state.sent.append((state.now, device_id, command))
            if command in state.suppress:
                return
            if command == 0xB0:
                payload = b"\xb0\x00" + struct.pack("<f", state.kt) + b"\x00"
            elif command == 0xA3:
                payload = b"\xa3\x00\x01" + struct.pack("<i", state.counts[device_id])
            else:
                value = state.speed_raw if command == 0xA2 else state.current_raw
                payload = bytes([command]) + struct.pack("<i", value)
            state.pending.extend(state.prefix)
            state.pending.append(struct.pack("=IB3x8s", device_id, len(payload), payload))

        def recv(self, length):
            assert length == 16
            if state.pending:
                if state.timeout != 0:
                    state.now += 0.0001
                return state.pending.pop(0)
            if state.timeout == 0:
                raise BlockingIOError()
            state.now += state.timeout
            raise TimeoutError()

    bus = Bus()

    class TargetBus:
        """A receive-only stand-in; deliberately has no send method."""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state.target_closed = True

        def bind(self, address):
            assert address == ("can0",)

        def setsockopt(self, level, option, payload):
            state.target_filters = payload

        def settimeout(self, timeout):
            self.timeout = timeout

        def recv(self, size):
            assert size == 16
            if state.target_pending.empty():
                state.targets_done.set()  # Prior command rows have already reached the logging queue.
            try:
                return state.target_pending.get(timeout=self.timeout)
            except queue.Empty:
                raise TimeoutError()

    sockets = iter((bus, TargetBus()))
    monkeypatch.setattr(module, "socket", SimpleNamespace(
        socket=lambda *args: next(sockets), AF_CAN=29, SOCK_RAW=3, CAN_RAW=1, SOL_CAN_RAW=101, CAN_RAW_FILTER=1,
        timeout=TimeoutError,
    ))
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: state.now, time=lambda: 1700000000 + state.now,
        sleep=lambda dt: setattr(state, "now", state.now + dt),
    ))
    return bus, state


@pytest.mark.parametrize("name,device_id,encoder_sign", [("steering", 2, -1), ("drive", 1, 1)])
def test_conversions_match_motor_io_protocol(module, fake_can, monkeypatch, name, device_id, encoder_sign):
    reference = load_file(PACKAGE / "trackstand_deploy" / "can_protocol.py", "feedback_protocol_reference", monkeypatch)
    bus, state = fake_can
    motor = module.Motor(name, device_id, 1.7, encoder_sign, -1)
    first = module.collect_motor(bus, motor, 0, 0.002)
    assert first["position_rad"] == first["position_unwrapped_rad"] == 0
    assert first["zero_count"] == state.counts[device_id]
    state.counts[device_id] += 40000
    row = module.collect_motor(bus, motor, 0, 0.002)
    a3 = b"\xa3\x00\x01" + struct.pack("<i", state.counts[device_id])
    a2 = b"\xa2" + struct.pack("<i", state.speed_raw)
    a1 = b"\xa1" + struct.pack("<i", state.current_raw)
    # The historical experiment retains both columns; v1 motor-state is continuous.
    assert row["position_unwrapped_rad"] == pytest.approx(
        reference.parse_a3_position(a3, motor.zero_count, 1.7, encoder_sign))
    assert row["position_unwrapped_rad"] == pytest.approx(encoder_sign * 40000 * module.RADIANS_PER_COUNT / 1.7)
    assert row["velocity_rad_s"] == pytest.approx(reference.parse_a2_velocity(a2, 1.7, encoder_sign))
    current = reference.parse_current_feedback(a1)
    assert row["current_a"] == pytest.approx(current)
    assert row["torque_nm"] == pytest.approx(reference.current_feedback_to_torque_nm(current, 1.25, -1))
    assert row["position_elapsed_s"] < row["velocity_elapsed_s"] < row["current_elapsed_s"]
    assert [command for _, _, command in state.sent].count(0xB0) == 1


def test_missing_samples_are_not_cached_or_fabricated(module, fake_can):
    bus, state = fake_can
    motor = module.Motor("steering", 2, 1, -1, -1)
    module.collect_motor(bus, motor, 0, 0.002)
    state.suppress.update((0xA1, 0xA2, 0xA3))
    row = module.collect_motor(bus, motor, 0, 0.002)
    assert row["encoder_count"] is None and row["zero_count"] == -20000
    for field in ("position_rad", "position_unwrapped_rad", "velocity_rad_s", "current_a", "torque_nm",
                  "position_elapsed_s", "velocity_elapsed_s", "current_elapsed_s"):
        assert math.isnan(row[field])


@pytest.mark.parametrize("kt", [0, math.nan, math.inf, 100])
def test_b0_invalid_no_guessed_torque_and_retry(module, fake_can, kt):
    bus, state = fake_can
    state.kt = kt
    motor = module.Motor("steering", 2, 1, -1, -1)
    row = module.collect_motor(bus, motor, 0, 0.002)
    assert row["current_a"] == pytest.approx(-1.234) and math.isnan(row["torque_nm"])
    module.collect_motor(bus, motor, 0, 0.002)
    assert [command for _, _, command in state.sent].count(0xB0) == 1
    state.now = 1.1
    state.kt = 1.25
    row = module.collect_motor(bus, motor, 0, 0.002)
    assert row["torque_nm"] == pytest.approx(1.5425)
    assert [command for _, _, command in state.sent].count(0xB0) == 2


def test_query_ignores_c2_wrong_id_malformed_and_old_replies(module, fake_can):
    bus, state = fake_can
    old = struct.pack("=IB3x8s", 2, 5, b"\xa1" + struct.pack("<i", 9999))
    state.pending.append(old)
    state.prefix = [
        struct.pack("=IB3x8s", 2, 7, b"\xc2" + bytes(6)),
        struct.pack("=IB3x8s", 1, 5, b"\xa1" + bytes(4)),
        struct.pack("=IB3x8s", 2, 3, b"\xa1" + bytes(2)),
        struct.pack("=IB3x8s", 0x80000002, 5, b"\xa1" + bytes(4)),
    ]
    data = module.query(bus, 2, 0xA1, 0.002)
    assert data == b"\xa1" + struct.pack("<i", -1234)
    with pytest.raises(KeyError):
        module.query(bus, 2, 0xC2, 0.002)
    assert len(state.sent) == 1


def test_query_timeout_is_bounded(module, fake_can):
    bus, state = fake_can
    state.suppress.add(0xA1)
    assert module.query(bus, 2, 0xA1, 0.002) is None
    assert state.now == pytest.approx(0.002)


def test_record_csv_both_motors_at_target_200_hz(module, fake_can, tmp_path):
    _, state = fake_can
    output = tmp_path / "feedback.csv"
    module.run(module.parse_args(["--duration", "0.015", "--output", str(output)]))
    with output.open() as log:
        rows = list(csv.DictReader(log))
    assert len(rows) == 6
    assert all(row["record_type"] == "feedback" and row["c2_target_count"] == "" for row in rows)
    assert [row["motor"] for row in rows] == ["steering", "drive"] * 3
    assert all(math.isfinite(float(row["torque_nm"])) for row in rows)
    steering_times = [stamp for stamp, device_id, command in state.sent if device_id == 2 and command == 0xA3]
    assert [b - a for a, b in zip(steering_times, steering_times[1:])] == pytest.approx([0.0049, 0.005])
    # First cycle reads B0 before A3; subsequent cycles use the latched constant.
    assert state.filters == struct.pack("=II", 2, 0xC00007FF) + struct.pack("=II", 1, 0xC00007FF)
    assert state.target_filters == struct.pack("=II", 0x102, 0xC00007FF) + struct.pack("=II", 0x101, 0xC00007FF)
    assert state.closed and state.target_closed


def test_existing_recording_not_overwritten(module, fake_can, tmp_path):
    output = tmp_path / "keep.csv"
    output.write_text("existing data\n")
    with pytest.raises(FileExistsError):
        module.run(module.parse_args(["--output", str(output)]))
    assert output.read_text() == "existing data\n"
    assert fake_can[1].sent == []


def test_capture_decodes_only_outgoing_c2_and_receipt_times(module, fake_can):
    _, state = fake_can
    state.now = 10.0
    frames = [
        struct.pack("=IB3x8s", 0x102, 5, b"\xc2" + struct.pack("<i", -500000)),
        struct.pack("=IB3x8s", 2, 7, b"\xc2" + bytes(6)),  # Motor's reply, not a command.
        struct.pack("=IB3x8s", 0x103, 5, b"\xc2" + bytes(4)),  # Unselected motor.
        struct.pack("=IB3x8s", 0x102, 3, b"\xc2" + bytes(2)),  # Wrong DLC.
        struct.pack("=IB3x8s", 0x102, 5, b"\xc0" + bytes(4)),  # Not C2.
        struct.pack("=IB3x8s", 0x80000102, 5, b"\xc2" + bytes(4)),  # Extended.
        struct.pack("=IB3x8s", 0x40000102, 5, b"\xc2" + bytes(4)),  # RTR.
        struct.pack("=IB3x8s", 0x20000102, 5, b"\xc2" + bytes(4)),  # Error.
        struct.pack("=IB3x8s", 0x101, 5, b"\xc2" + struct.pack("<i", 987654321)),
    ]
    pending, stopped = queue.Queue(), threading.Event()

    class CaptureBus:
        def recv(self, size):
            assert size == 16
            state.now += 0.005
            frame = frames.pop(0)
            if not frames:
                stopped.set()
            return frame

    motors = [module.Motor("steering", 2, 1, -1, -1), module.Motor("drive", 1, 1, 1, -1)]
    module.receive_c2_targets(CaptureBus(), motors, 10.0, stopped, pending)
    rows = [pending.get_nowait(), pending.get_nowait()]
    assert pending.empty()
    assert [row["record_type"] for row in rows] == ["c2_target", "c2_target"]
    assert [row["motor"] for row in rows] == ["steering", "drive"]
    assert [row["c2_target_count"] for row in rows] == [-500000, 987654321]
    assert [row["elapsed_s"] for row in rows] == pytest.approx([0.005, 0.045])
    assert [row["time_unix_s"] for row in rows] == pytest.approx([1700000010.005, 1700000010.045])
    assert all(math.isnan(row["position_rad"]) and row["encoder_count"] is None for row in rows)
    assert state.sent == []  # Passive capture has not issued any CAN request.


def test_csv_keeps_every_target_event_separate_from_feedback(module, fake_can, tmp_path):
    _, state = fake_can
    counts = [-20000, -22608, -20000, -17392]
    for count in counts:
        state.target_pending.put(struct.pack("=IB3x8s", 0x102, 5, b"\xc2" + struct.pack("<i", count)))
    state.wait_for_targets = True
    output = tmp_path / "targets_and_feedback.csv"
    module.run(module.parse_args(["--duration", "0.015", "--output", str(output)]))
    with output.open() as log:
        rows = list(csv.DictReader(log))
    targets = [row for row in rows if row["record_type"] == "c2_target"]
    feedback = [row for row in rows if row["record_type"] == "feedback"]
    assert len(feedback) == 6 and len(targets) == 4
    assert [int(row["c2_target_count"]) for row in targets] == counts
    assert all(row["motor"] == "steering" and row["motor_id"] == "2" for row in targets)
    assert all(float(row["elapsed_s"]) == 0 for row in targets)
    assert all(float(row["time_unix_s"]) == 1700000000 for row in targets)
    assert all(row["c2_target_count"] == "" for row in feedback)
    assert all(command in (0xA1, 0xA2, 0xA3, 0xB0) for _, _, command in state.sent)
    assert state.closed and state.target_closed


def test_pending_target_events_are_saved_on_interrupt(module, fake_can, tmp_path, monkeypatch):
    _, state = fake_can
    state.target_pending.put(struct.pack("=IB3x8s", 0x102, 5, b"\xc2" + struct.pack("<i", -12345)))

    def interrupt(*args):
        assert state.targets_done.wait(1.0)
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "collect_motor", interrupt)
    output = tmp_path / "interrupted.csv"
    with pytest.raises(KeyboardInterrupt):
        module.run(module.parse_args(["--output", str(output)]))
    with output.open() as log:
        rows = list(csv.DictReader(log))
    assert len(rows) == 1 and rows[0]["record_type"] == "c2_target"
    assert rows[0]["c2_target_count"] == "-12345"
    assert state.closed and state.target_closed and state.sent == []


def test_capture_socket_failure_is_reported(module, fake_can):
    _, state = fake_can
    pending = queue.Queue()

    class BrokenCapture:
        def recv(self, size):
            raise OSError("capture failed")

    module.receive_c2_targets(BrokenCapture(), [], 0, threading.Event(), pending)
    with pytest.raises(OSError, match="capture failed"):
        module.write_c2_targets(None, pending)
    assert state.sent == []


def test_defaults_match_deployment_and_steering_only(module):
    args = module.parse_args([])
    assert args.motors == ["steering", "drive"] and args.rate == 200
    assert (args.steering_id, args.drive_id) == (2, 1)
    assert (args.steering_encoder_sign, args.drive_encoder_sign) == (-1, -1)
    assert args.steering_torque_sign == args.drive_torque_sign == -1
    assert module.parse_args(["--motors", "steering"]).motors == ["steering"]


@pytest.mark.parametrize("options", [
    ["--rate", "0"], ["--rate", "nan"], ["--timeout", "inf"], ["--duration", "-1"],
    ["--steering-gear-ratio", "0"], ["--steering-id", "0"], ["--drive-id", "2"],
    ["--motors", "steering", "steering"],
])
def test_invalid_args(module, options):
    with pytest.raises(SystemExit) as error:
        module.parse_args(options)
    assert error.value.code == 2


def test_help_needs_no_ros_or_can_interface():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], check=True, capture_output=True, text=True)
    assert "without ROS" in result.stdout and "--motors" in result.stdout
