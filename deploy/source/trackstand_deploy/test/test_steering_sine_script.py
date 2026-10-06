"""Pure waveform/CLI checks; never initialize ROS or command hardware."""

import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "steering_sine.py"


@pytest.fixture(scope="module")
def sine_code():
    python = SCRIPT.read_text()
    namespace = {"__name__": "steering_sine_test"}
    exec(compile(python, str(SCRIPT), "exec"), namespace)
    return namespace


def test_defaults_match_requested_command(sine_code):
    args = sine_code["parse_args"](["--execute"])
    assert args.amplitude == 1.0
    assert args.period == 10.0
    assert args.duration == 0.0
    assert sine_code["RATE_HZ"] == 50.0
    assert sine_code["TOPIC"] == "/trackstand/motor_command"
    assert sine_code["MESSAGE_TYPE"] == "std_msgs/msg/Float64MultiArray"


def test_waveform_extrema_and_period(sine_code):
    target = sine_code["steering_target"]
    assert target(0.0, 1.0, 10.0) == 0.0
    assert target(2.5, 1.0, 10.0) == pytest.approx(1.0)
    assert target(5.0, 1.0, 10.0) == pytest.approx(0.0, abs=1e-15)
    assert target(7.5, 1.0, 10.0) == pytest.approx(-1.0)
    assert target(10.0, 1.0, 10.0) == 0.0
    values = [target(index / 50.0, 1.0, 10.0) for index in range(500)]
    assert min(values) == -1.0
    assert max(values) == 1.0
    assert all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values)


def test_preview_has_50_hz_samples_and_no_ros(sine_code, capsys):
    args = sine_code["parse_args"](["--preview", "--duration", "0.08"])
    sine_code["preview"](args)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("PREVIEW ONLY")
    assert len(lines) == 5
    assert lines[1] == "t=0.000s {data: [1.0, 0.000000000, 0.0]}"
    assert lines[2].startswith("t=0.020s")
    assert lines[3].startswith("t=0.040s")
    assert lines[4].startswith("t=0.060s")


@pytest.mark.parametrize("options", [
    [], ["--execute", "--preview"], ["--execute", "--period", "0"],
    ["--execute", "--period", "0.04"], ["--execute", "--amplitude", "1.1"],
    ["--execute", "--amplitude", "0"], ["--execute", "--duration", "-1"],
    ["--execute", "--period", "nan"], ["--execute", "--duration", "inf"],
])
def test_rejects_invalid_or_unconfirmed_execution(sine_code, options):
    with pytest.raises(SystemExit) as error:
        sine_code["parse_args"](options)
    assert error.value.code == 2


def test_python_help():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], check=True, capture_output=True, text=True)
    assert "BOTH motors" in result.stdout
    assert "--execute" in result.stdout


@pytest.fixture
def fake_ros(sine_code, monkeypatch):
    """Simulate graph discovery, timer ticks, transport failures, and signals."""
    code = sine_code.copy()
    # Functions need the isolated globals when time/signal are replaced.
    python = SCRIPT.read_text()
    exec(compile(python, str(SCRIPT), "exec"), code)
    state = SimpleNamespace(
        now=0.0, timer=None, records=[], handlers={}, initialized=False, destroyed=False,
        competing=False, fail_send=False, stop_after=None, motor_present=True,
    )

    class Array:
        def __init__(self, data=None):
            self.data = [] if data is None else data

    class Publisher:
        def publish(self, message):
            if state.fail_send and message.data[0] == 1.0 and len(state.records) == 2:
                raise RuntimeError("simulated send failure")
            state.records.append((state.now, list(message.data)))

        def get_subscription_count(self):
            return 1 if state.motor_present else 0

    class Node:
        def create_publisher(self, message_type, topic, qos):
            assert message_type is Array
            assert topic == "/trackstand/motor_command"
            assert qos.depth == 1 and qos.durability == "volatile" and qos.reliability == "reliable"
            return Publisher()

        def get_name(self):
            return "sine_test_node"

        def get_publishers_info_by_topic(self, topic):
            names = [self.get_name()]
            if state.competing:
                names.append("trackstand_policy_controller")
            return [SimpleNamespace(node_name=name) for name in names]

        def get_subscriptions_info_by_topic(self, topic):
            if not state.motor_present:
                return []
            return [SimpleNamespace(
                node_name="trackstand_motor_io", node_namespace="/", topic_type=code["MESSAGE_TYPE"],
            )]

        def create_timer(self, interval_s, callback, clock):
            assert interval_s == 0.02
            assert clock.clock_type == "steady"
            state.timer = SimpleNamespace(callback=callback, canceled=False)
            state.timer.cancel = lambda: setattr(state.timer, "canceled", True)
            return state.timer

        def destroy_node(self):
            state.destroyed = True

    def spin_once(node, timeout_sec):
        state.now += timeout_sec
        if state.stop_after is not None and len(state.records) >= state.stop_after:
            state.handlers[code["signal"].SIGTERM](15, None)
        if state.timer is not None and not state.timer.canceled:
            state.timer.callback()

    def signal_handler(number, handler):
        previous = state.handlers.get(number)
        state.handlers[number] = handler
        return previous

    code["time"] = SimpleNamespace(
        monotonic=lambda: state.now, sleep=lambda duration: setattr(state, "now", state.now + duration),
    )
    code["signal"] = SimpleNamespace(SIGINT=2, SIGTERM=15, signal=signal_handler)
    for name in ("rclpy", "rclpy.clock", "rclpy.qos", "rclpy.signals", "std_msgs", "std_msgs.msg"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    ros = sys.modules["rclpy"]
    ros.init = lambda **kwargs: setattr(state, "initialized", True)
    ros.ok = lambda: state.initialized
    ros.try_shutdown = lambda: setattr(state, "initialized", False)
    ros.create_node = lambda *args, **kwargs: Node()
    ros.spin_once = spin_once
    sys.modules["rclpy.clock"].Clock = SimpleNamespace
    sys.modules["rclpy.clock"].ClockType = SimpleNamespace(STEADY_TIME="steady")
    sys.modules["rclpy.qos"].QoSProfile = SimpleNamespace
    sys.modules["rclpy.qos"].ReliabilityPolicy = SimpleNamespace(RELIABLE="reliable")
    sys.modules["rclpy.qos"].DurabilityPolicy = SimpleNamespace(VOLATILE="volatile")
    sys.modules["rclpy.signals"].SignalHandlerOptions = SimpleNamespace(NO="no_ros_signal_handlers")
    sys.modules["std_msgs.msg"].Float64MultiArray = Array
    return code, state


def test_execution_waveform_timer_and_disable(fake_ros):
    code, state = fake_ros
    code["execute"](code["parse_args"](["--execute", "--duration", "10"]))
    enabled = [(stamp, data) for stamp, data in state.records if data[0] == 1.0]
    disabled = [data for _, data in state.records if data[0] == 0.0]
    assert enabled[0][0] >= 2.0  # No command during discovery.
    assert enabled[0][1] == [1.0, 0.0, 0.0]
    assert 499 <= len(enabled) <= 501
    for (previous, _), (current, data) in zip(enabled, enabled[1:]):
        assert current - previous == pytest.approx(0.02)
        elapsed_s = current - enabled[0][0]
        assert data == pytest.approx([1.0, math.sin(2.0 * math.pi * elapsed_s / 10.0), 0.0])
    assert disabled == [[0.0, 0.0, 0.0]] * 10
    assert state.timer.canceled and state.destroyed and not state.initialized


@pytest.mark.parametrize("problem", ["competing", "motor_missing"])
def test_bad_graph_never_publishes(fake_ros, problem):
    code, state = fake_ros
    state.competing = problem == "competing"
    state.motor_present = problem != "motor_missing"
    with pytest.raises(RuntimeError):
        code["execute"](code["parse_args"](["--execute"]))
    assert state.records == []
    assert state.destroyed and not state.initialized


def test_disable_after_publication_error(fake_ros):
    code, state = fake_ros
    state.fail_send = True
    with pytest.raises(RuntimeError, match="simulated send failure"):
        code["execute"](code["parse_args"](["--execute"]))
    assert [data for _, data in state.records if data[0] == 0.0] == [[0.0, 0.0, 0.0]] * 10
    assert state.timer.canceled and state.destroyed and not state.initialized


def test_sigterm_stops_and_disables(fake_ros):
    code, state = fake_ros
    state.stop_after = 5
    code["execute"](code["parse_args"](["--execute"]))
    assert len([data for _, data in state.records if data[0] == 1.0]) == 5
    assert [data for _, data in state.records if data[0] == 0.0] == [[0.0, 0.0, 0.0]] * 10
    assert state.timer.canceled and state.destroyed and not state.initialized
