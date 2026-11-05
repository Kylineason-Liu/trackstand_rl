"""Real pipe/terminal input and existing policy routing, without ROS or hardware."""

import os
from pathlib import Path
import signal
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def terminal(node_module, monkeypatch):
    class Node:
        def __init__(self, name):
            self.name = name
            self.context = SimpleNamespace(ok=lambda: True)
            self.create_publisher = Mock(return_value=Mock())
            self.create_timer = Mock()
            self.get_logger = Mock(return_value=Mock())

        def destroy_node(self):
            pass

    monkeypatch.setattr(sys.modules['rclpy.node'], 'Node', Node)
    module = node_module('manual_kickstand')
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, 'r')
    writer = SimpleNamespace(fd=write_fd)
    try:
        yield module.ManualKickstand(reader), module, writer
    finally:
        reader.close()
        if writer.fd is not None:
            os.close(writer.fd)


def states(node):
    return [call.args[0].data for call in node._publisher.publish.call_args_list]


def write_tick(terminal, data):
    node, _, writer = terminal
    os.write(writer.fd, data)
    node._tick()


def test_default_down_exact_rc_topic_type_and_50hz_heartbeat(terminal):
    node, module, _ = terminal
    from std_msgs.msg import Bool
    node.create_publisher.assert_called_once_with(Bool, '/trackstand/kickstands_up', 10)
    assert node.name == 'trackstand_manual_kickstand' and not node.finished
    assert callable(module.main)
    assert node.create_timer.call_args.args[0] == .02
    for _ in range(10):
        node._tick()
    assert states(node) == [False] * 11


def test_up_down_case_whitespace_and_hold_while_operator_is_idle(terminal):
    node, _, _ = terminal
    write_tick(terminal, b'  UP \r\n')
    assert states(node)[-1] is True
    for _ in range(10):
        node._tick()
    assert states(node)[-10:] == [True] * 10
    write_tick(terminal, b'down\n')
    assert states(node)[-1] is False and not node.finished


def test_partial_input_never_blocks_heartbeat_or_runs_before_enter(terminal):
    node, _, _ = terminal
    write_tick(terminal, b'u')
    assert states(node)[-1] is False
    node._tick()  # This would hang if select+readline were used.
    write_tick(terminal, b'p\n')
    assert states(node)[-1] is True


def test_batched_down_up_retains_both_transitions(terminal):
    node, _, _ = terminal
    write_tick(terminal, b'up\n')
    node._publisher.publish.reset_mock()
    write_tick(terminal, b'down\nup\n')
    assert states(node) == [False, True, True]


@pytest.mark.parametrize('command', [b'\n', b'left\n', b'true\n', b'up down\n', b'\xffup\n'])
def test_invalid_or_empty_command_does_not_change_switch_state(terminal, command):
    node, _, _ = terminal
    write_tick(terminal, b'up\n')
    write_tick(terminal, command)
    assert states(node)[-1] is True and not node.finished


@pytest.mark.parametrize('command', [b'quit\nup\n', b'exit\n'])
def test_quit_requests_down_and_cannot_process_later_up(terminal, command):
    node, _, _ = terminal
    write_tick(terminal, b'up\n')
    write_tick(terminal, command)
    assert states(node)[-1] is False and node.finished
    before = len(states(node))
    node._tick()
    assert len(states(node)) == before


def test_eof_after_up_requests_down_and_exits(terminal):
    node, _, writer = terminal
    write_tick(terminal, b'up\n')
    os.close(writer.fd)
    writer.fd = None
    node._tick()
    assert node.finished and states(node)[-1] is False


def test_unterminated_up_at_eof_never_arms(terminal):
    node, _, writer = terminal
    write_tick(terminal, b'up')
    os.close(writer.fd)
    writer.fd = None
    node._tick()
    assert node.finished and not any(states(node))


def test_terminal_error_requests_down(terminal, monkeypatch):
    node, module, _ = terminal
    write_tick(terminal, b'up\n')
    monkeypatch.setattr(module.select, 'select', Mock(side_effect=OSError('terminal disconnected')))
    node._tick()
    assert node.finished and states(node)[-1] is False


def test_temporary_input_interrupt_keeps_heartbeat(terminal, monkeypatch):
    node, module, _ = terminal
    write_tick(terminal, b'up\n')
    monkeypatch.setattr(module.select, 'select', Mock(side_effect=InterruptedError()))
    node._tick()
    assert not node.finished and states(node)[-1] is True


def test_long_incomplete_line_is_bounded_and_stops_down(terminal):
    node, _, _ = terminal
    write_tick(terminal, b'x' * 4096)
    write_tick(terminal, b'x')
    assert node.finished and states(node)[-1] is False


def test_output_drives_existing_policy_callback_and_servo_command(terminal, node_module):
    node, _, _ = terminal
    module = node_module('policy_controller_v2')
    policy = object.__new__(module.PolicyControllerV2)
    from trackstand_deploy.contracts import BalanceStateMachine
    policy._state_machine = BalanceStateMachine()
    policy._rear_episode_zero_rad = None
    policy._ready = Mock(return_value=True)
    policy._begin_episode = Mock()
    policy._command_kickstands = Mock()
    policy._publish_motor_command = Mock()
    write_tick(terminal, b'up\n')
    policy._rc_callback(node._publisher.publish.call_args.args[0])
    assert policy._state_machine.enabled
    policy._command_kickstands.assert_called_with(True)
    write_tick(terminal, b'down\n')
    policy._rc_callback(node._publisher.publish.call_args.args[0])
    assert not policy._state_machine.enabled
    policy._publish_motor_command.assert_called_with(False, 0, 0)
    policy._command_kickstands.assert_called_with(False)


def test_existing_policy_still_gates_manual_up_when_unready(terminal, node_module):
    node, _, _ = terminal
    module = node_module('policy_controller_v2')
    policy = object.__new__(module.PolicyControllerV2)
    from trackstand_deploy.contracts import BalanceStateMachine
    policy._state_machine = BalanceStateMachine()
    policy._ready = Mock(return_value=False)
    policy._command_kickstands = Mock()
    write_tick(terminal, b'up\n')
    policy._rc_callback(node._publisher.publish.call_args.args[0])
    assert not policy._state_machine.enabled
    policy._command_kickstands.assert_called_with(False)


@pytest.mark.parametrize('ending', ['signal', 'constructor', 'external_shutdown', 'cleanup'])
def test_entrypoint_sends_down_before_ros_shutdown_and_restores_signals(terminal, monkeypatch, ending):
    _, module, _ = terminal
    executors, signals = ModuleType('rclpy.executors'), ModuleType('rclpy.signals')
    executors.ExternalShutdownException = type('ExternalShutdownException', (Exception,), {})
    signals.SignalHandlerOptions = SimpleNamespace(NO='no_ros_signal_handlers')
    monkeypatch.setitem(sys.modules, executors.__name__, executors)
    monkeypatch.setitem(sys.modules, signals.__name__, signals)
    handlers = {number: object() for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    originals = dict(handlers)

    def register(number, handler):
        previous, handlers[number] = handlers[number], handler
        return previous

    monkeypatch.setattr(module.signal, 'signal', register)
    clock = SimpleNamespace(value=10.)
    module.time = SimpleNamespace(monotonic=lambda: clock.value)
    node = Mock(finished=False)
    node.context.ok.return_value = True
    node.stop.side_effect = lambda: setattr(node, 'finished', True)
    module.ManualKickstand = Mock(return_value=node)

    def spin(node, timeout_sec):
        clock.value += timeout_sec
        if not node.finished:
            if ending == 'external_shutdown':
                raise executors.ExternalShutdownException()
            handlers[signal.SIGINT](signal.SIGINT, None)

    module.rclpy = SimpleNamespace(init=Mock(), try_shutdown=Mock(), ok=lambda: True, spin_once=Mock(side_effect=spin))
    if ending == 'constructor':
        module.ManualKickstand.side_effect = RuntimeError('constructor failure')
    elif ending == 'cleanup':
        node.stop.side_effect = RuntimeError('cleanup failure')
    if ending in ('constructor', 'cleanup'):
        with pytest.raises(RuntimeError):
            module.main()
    else:
        module.main()
    module.rclpy.init.assert_called_once_with(args=None, signal_handler_options='no_ros_signal_handlers')
    module.rclpy.try_shutdown.assert_called_once()
    assert handlers == originals
    if ending != 'constructor':
        node.stop.assert_called_once()
        node.destroy_node.assert_called_once()


def test_v2_launch_can_exclude_rc_without_launching_an_interactive_reader(monkeypatch):
    import runpy
    captured = []
    for name in ('ament_index_python', 'ament_index_python.packages', 'launch', 'launch.actions',
                 'launch.conditions', 'launch.substitutions', 'launch_ros', 'launch_ros.actions'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    package = Path(__file__).parents[1]
    sys.modules['ament_index_python.packages'].get_package_share_directory = lambda _: str(package)
    sys.modules['launch'].LaunchDescription = lambda actions: actions
    sys.modules['launch.actions'].DeclareLaunchArgument = lambda *args, **kwargs: ('argument', args, kwargs)
    sys.modules['launch.conditions'].IfCondition = lambda value: ('if', value)
    sys.modules['launch.substitutions'].LaunchConfiguration = lambda name: ('configuration', name)
    sys.modules['launch_ros.actions'].Node = lambda **kwargs: captured.append(kwargs) or kwargs
    actions = runpy.run_path(str(package/'launch/trackstand_v2.launch.py'))['generate_launch_description']()
    assert actions[0][1] == ('use_rc',) and actions[0][2]['default_value'] == 'true'
    rc = next(node for node in captured if node['executable'] == 'rc_kickstand')
    assert rc['condition'] == ('if', ('configuration', 'use_rc'))
    assert len(captured) == 5 and not any(node['executable'] == 'manual_kickstand' for node in captured)
