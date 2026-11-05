"""Actual v2 nodes with deterministic ROS/CAN firmware stand-ins, no hardware."""

from collections import deque
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trackstand_deploy.contracts_v2 import COMMAND_LABEL, COMMAND_TOPIC
from trackstand_deploy.mit_protocol import parse_f0, parse_mit_motion


def f1(ranges, status=0):
    p, v, t = 32768, 2048, 2048
    return bytes((0xf1, p >> 8, p & 255, v >> 4, ((v & 15) << 4) | (t >> 8), t & 255, status))


@pytest.fixture
def nodes(node_module, monkeypatch, tmp_path):
    overrides = {'settings_journal_path': str(tmp_path / 'motor-settings.json')}
    clock = SimpleNamespace(value=100.)

    class Node:
        def __init__(self, name):
            self.node_name = name
            self.parameters = {}
            self.context = SimpleNamespace(ok=lambda: True)
            self.create_publisher = Mock(side_effect=lambda *args: Mock())
            self.create_subscription = Mock()
            self.create_timer = Mock()
            self.get_logger = Mock(return_value=Mock())
            self.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(clock.value * 1e9)))

        def declare_parameter(self, name, default, descriptor):
            self.parameters[name] = overrides.get(name, default)

        def get_parameter(self, name):
            return SimpleNamespace(value=self.parameters[name])

        def destroy_node(self):
            pass

    monkeypatch.setattr(sys.modules['rclpy.node'], 'Node', Node)
    monkeypatch.setattr(sys.modules['ament_index_python.packages'], 'get_package_share_directory',
                        lambda _: str(Path(__file__).parents[1]))
    module = node_module('motor_mit_io_v2')
    base = sys.modules['trackstand_deploy.motor_mit_io']
    monkeypatch.setattr(base.threading, 'Thread', Mock(return_value=Mock()))
    for mod in (module, base):
        mod.time = SimpleNamespace(monotonic=lambda: clock.value, time=lambda: clock.value)
    node = module.MotorMitIoV2()
    return node, module, base, overrides, clock, node_module


class Firmware:
    def __init__(self, node, base, clock):
        self.node, self.base, self.clock = node, base, clock
        self.sent, self.queue = [], deque()
        self.settings = {i: {0xf0: b'\xf0' + struct.pack('<HHH', 955, 4500, 1800),
                             0xcd: b'\xcd\0\0\0\0'} for i in (1, 2)}
        self.originals = {i: dict(values) for i, values in self.settings.items()}
        self.count = {1: -900000, 2: 0}
        self.active = {1: False, 2: False}
        self.closed = False

    def send(self, can_id, payload):
        self.sent.append((can_id, payload))
        device = can_id & 255
        if can_id & 0x400:
            assert len(payload) == 8
            self.active[device] = True
            response = f1(parse_f0(self.settings[device][0xf0]), 1)
        else:
            op = payload[0]
            if op in (0xf0, 0xcd):
                if len(payload) > 1:
                    self.settings[device][op] = payload
                response = self.settings[device][op]
            elif op in (0xcf, 0xae):
                if op == 0xcf:
                    self.active[device] = False
                response = bytes([op]) + bytes(5) + bytes([5, 0])  # Mode retained after CF.
            elif op == 0xb1:
                self.count[device] = 0
                response = b'\xb1\0\0'
            elif op == 0xa3:
                response = b'\xa3' + struct.pack('<Hi', self.count[device] % 16384, self.count[device])
            elif op == 0xa2:
                response = b'\xa2' + struct.pack('<i', 0)
            elif op == 0xa1:
                response = b'\xa1' + struct.pack('<i', -100)
            elif op == 0xb0:
                response = b'\xb0\0' + struct.pack('<f', .3 if device == 1 else .78) + b'\0'
            elif op == 0xf1:
                response = f1(parse_f0(self.settings[device][0xf0]), int(self.active[device]))
            else:
                raise AssertionError(f'unexpected v2 CAN command: {can_id:x} {payload.hex()}')
        self.clock.value += .00001
        self.queue.append(self.base.CanReply(device, response, False, 0, self.clock.value))

    def receive(self, timeout):
        if self.queue:
            return self.queue.popleft()
        self.clock.value += max(timeout, .000001)
        return None

    def close(self):
        self.closed = True


def startup(nodes):
    node, module, base, overrides, clock, loader = nodes
    bus = Firmware(node, base, clock)
    node._tx_bus = bus
    node._startup()
    # Actual state polling creates fresh position/speed pairs before enabling.
    for motor in (node._steering, node._drive):
        node._query(motor, 0xa3, 7, .1)
        node._query(motor, 0xa2, 5, .1)
    return node, module, base, bus, clock


def command(module, enable=1, steering=.1, drive=1):
    from std_msgs.msg import Float64MultiArray, MultiArrayDimension
    message = Float64MultiArray(data=[enable, steering, drive])
    message.layout.dim = [MultiArrayDimension(label=COMMAND_LABEL, size=3, stride=3)]
    return message


def test_actual_constructor_versioned_interfaces_and_startup(nodes):
    node, module, base, bus, clock = startup(nodes)
    assert node.node_name == 'trackstand_motor_io_v2'
    assert [call.args[1] for call in node.create_subscription.call_args_list] == [COMMAND_TOPIC]
    assert node.create_timer.call_args.args[0] == .01
    assert node._rear_config_ready and node._mit_config_ready
    assert node._drive.zero_count == -900000 and node._drive.position_rad == 0
    assert node._settings.path.exists()
    assert (0x102, b'\xb1') in bus.sent and (0x101, b'\xb1') not in bus.sent
    assert parse_f0(bus.settings[1][0xf0]).velocity_rad_s >= 61.6
    assert all(payload[0] != 0xc0 for cid, payload in bus.sent if cid < 0x400)
    assert bus.settings[1][0xcd][1:4] == b'\1\xf4\1'


def test_dry_run_does_not_change_firmware_or_send_motion(nodes):
    node, module, base, overrides, clock, _ = nodes
    node._dry_run = True
    node, module, base, bus, clock = startup(nodes)
    node._command_callback(command(module))
    node._control_tick()
    assert bus.settings == bus.originals and not node._settings.path.exists()
    assert all(cid < 0x400 and len(payload) == 1 and payload[0] != 0xb1 for cid, payload in bus.sent)
    assert node._rear_command_ranges.velocity_rad_s > node._rear_ranges.velocity_rad_s


def test_velocity_packets_pi_history_and_no_hidden_half_rate(nodes):
    node, module, base, bus, clock = startup(nodes)
    node._command_callback(command(module))
    node._control_tick()
    node._drain_received()
    clock.value += .005
    node._control_tick()
    node._drain_received()
    assert node._rear_integral.torque_nm > 0
    first = node._rear_integral.torque_nm
    node._command_callback(command(module, drive=.5))
    clock.value += .005
    node._control_tick()
    motion = [(cid, payload) for cid, payload in bus.sent if cid >= 0x400]
    assert [cid for cid, _ in motion] == [0x502, 0x501] * 3
    assert node._rear_integral.torque_nm > first
    decoded = parse_mit_motion(motion[-1][1], node._rear_command_ranges)
    assert decoded.velocity_rad_s < 0 and decoded.kp_nm_per_rad == 0
    assert decoded.feedforward_torque_nm < 0
    node._publish_status()
    status = json.loads(node._status_publisher.publish.call_args.args[0].data)
    assert status['drive_action'] == 'velocity_rad_s' and not status['mit_total_torque_clamp_verified']


def test_disable_resets_integral_cf_only_and_waits_for_both_f1(nodes):
    node, module, base, bus, clock = startup(nodes)
    node._command_callback(command(module))
    node._control_tick()
    node._drain_received()
    clock.value += .005
    node._control_tick()
    node._drain_received()
    assert node._rear_integral.torque_nm > 0
    node._command_callback(command(module, enable=0))
    node._control_tick()
    assert node._rear_integral.torque_nm == 0
    assert node._disable_wait
    node._drain_received()
    assert node._disable_confirmed and not node._disable_wait
    assert all(payload[0] != 0xc0 for cid, payload in bus.sent if cid < 0x400)
    node._command_callback(command(module))
    node._control_tick()
    assert node._rear_integral.torque_nm == 0  # Prime anew.


def test_down_up_within_one_cycle_cannot_skip_reset(nodes):
    node, module, base, bus, clock = startup(nodes)
    node._command_callback(command(module))
    node._control_tick()
    node._drain_received()
    node._command_callback(command(module, enable=0))
    node._command_callback(command(module, enable=1))
    node._control_tick()
    assert node._fault_latched and node._rear_integral.previous_time is None


def test_enable_withdrawn_between_sends_never_sends_rear_motion(nodes):
    node, module, base, bus, clock = startup(nodes)
    node._command_callback(command(module))
    real_send = bus.send

    def withdraw(cid, payload):
        real_send(cid, payload)
        if cid == 0x502:
            node._command_callback(command(module, enable=0))

    bus.send = withdraw
    node._control_tick()
    assert not any(cid == 0x501 for cid, _ in bus.sent)
    assert node._rear_integral.torque_nm == 0 and node._disable_wait


def test_stale_rear_a2_blocks_commands_and_clears_pi(nodes):
    node, module, base, bus, clock = startup(nodes)
    node._command_callback(command(module))
    node._rear_a2_at -= .1
    node._control_tick()
    assert not any(cid >= 0x400 for cid, _ in bus.sent)
    assert node._feedback_gate_fault and node._rear_integral.torque_nm == 0


def test_large_p_demand_matches_velocity_contract_and_exposes_missing_total_clamp(nodes):
    node, module, base, bus, clock = startup(nodes)
    node._command_callback(command(module, drive=61.6))
    node._control_tick()
    assert not node._fault_latched and node._rear_packet.estimated_total_torque_nm > 6.28
    diag = dict(zip(node._control_fields, node._control_publisher.publish.call_args.args[0].data))
    assert diag['drive_sim_reference_clipped_torque_nm'] == 6.28
    assert not diag['mit_total_torque_clamp_verified']


def test_shutdown_restores_originals_with_motion_loop_stopped(nodes):
    node, module, base, bus, clock = startup(nodes)
    node._running = False
    node._restore_backend_configuration()
    assert bus.settings == bus.originals and not node._settings.path.exists()
    assert not node._running and not node._restoring_configuration


def test_complete_worker_run_disables_both_restores_and_closes(nodes, monkeypatch):
    node, module, base, _, clock, _ = nodes
    bus = Firmware(node, base, clock)
    monkeypatch.setattr(base, 'SocketCan', lambda *args: bus)
    original_startup, original_tick = node._startup, node._control_tick

    def start():
        original_startup()
        for motor in (node._steering, node._drive):
            node._query(motor, 0xa3, 7, .1)
            node._query(motor, 0xa2, 5, .1)
        node._command_callback(command(module))

    def one_tick():
        original_tick()
        node._running = False

    node._startup, node._control_tick = start, one_tick
    node._run_io()
    assert not node._fault_latched, node._fault_reason
    assert bus.closed and bus.settings == bus.originals
    assert not node._settings.path.exists() and node._disable_confirmed
    assert set(cid for cid, _ in bus.sent if cid >= 0x400) == {0x501, 0x502}


def test_uncertain_disable_never_restores_persistent_settings(nodes):
    node, module, base, bus, clock = startup(nodes)
    node._disable_confirmed = False
    with pytest.raises(RuntimeError, match='before both motors'):
        node._restore_backend_configuration()
    assert node._settings.path.exists()


def test_stale_journal_blocks_normal_start_and_explicit_recovery_is_motion_free(nodes):
    node, module, base, bus, clock = startup(nodes)
    journal = node._settings.path
    node._settings.document = None  # Simulate a process that died after writes.
    node._mit_config_ready = node._rear_config_ready = False
    with pytest.raises(RuntimeError, match='pending motor settings'):
        node._startup_disable()
    bus.sent.clear()
    node._recover_pending = True
    node._startup()
    assert bus.settings == bus.originals and not journal.exists() and not node._running
    assert all(cid < 0x400 and payload[0] != 0xb1 for cid, payload in bus.sent)


def test_rear_a3_rollover_continuity_and_unsolicited_rejection(nodes):
    node, module, base, bus, clock = startup(nodes)
    for count in (2**31-2, -2**31+2):
        bus.count[1] = count
        node._query(node._drive, 0xa3, 7, .1)
        node._query(node._drive, 0xa2, 5, .1)
        if count > 0:
            before = node._drive.position_rad
    from trackstand_deploy.can_protocol import RADIANS_PER_COUNT
    assert node._drive.position_rad - before == pytest.approx(-4 * RADIANS_PER_COUNT)
    last = node._rear_last_count
    frame = base.CanReply(1, b'\xa3' + struct.pack('<Hi', 0, 0), False, 0, clock.value)
    node._consume(frame, clock.value)
    assert node._rear_last_count == last


def test_anchored_scheduler_skips_missed_ticks_without_bursts(nodes):
    node, _, _, _, clock, _ = nodes
    clock.value = 100.017
    assert node._next_control_deadline(100.) == pytest.approx(100.020)


def test_v2_policy_actual_constructor_versioned_output_and_model(nodes):
    _, _, _, _, _, loader = nodes
    module = loader('policy_controller_v2')
    node = module.PolicyControllerV2()
    assert node.node_name == 'trackstand_policy_controller_v2'
    assert node._model_path.endswith('model_50100.onnx')
    output = node._motor_publisher.publish.call_args.args[0]
    assert output.data == [0, 0, 0] and output.layout.dim[0].label == COMMAND_LABEL
    assert node.action_scaler([1, 1]) == (1.57, 61.6)


def test_actual_policy_tick_scales_velocity_and_slews_steering_at_policy_rate(nodes, monkeypatch):
    import numpy as np
    motor, _, _, _, clock = startup(nodes)
    module = nodes[-1]('policy_controller_v2')
    times = SimpleNamespace(monotonic=lambda: clock.value)
    module.time = times
    monkeypatch.setattr(sys.modules['trackstand_deploy.policy_controller'], 'time', times)
    policy = module.PolicyControllerV2()
    motor._publish_status()
    policy._backend_callback(motor._status_publisher.publish.call_args.args[0])
    policy._world_from_bike = np.array([0., 0., 0., 1.])
    policy._body_rate_rad_s = np.zeros(3)
    policy._motor_state = np.array([0., 100., 0., 0.])
    policy._imu_monotonic = policy._motor_monotonic = policy._rc_monotonic = clock.value
    policy._policy = Mock(act=Mock(return_value=np.array([1., .5], dtype=np.float32)))
    policy._state_machine.set_rc(True, True)
    for step in range(2):
        policy._policy_tick()
        output = policy._motor_publisher.publish.call_args.args[0]
        assert output.data == pytest.approx([1., .478 * (step+1), 30.8])
        assert output.layout.dim[0].label == COMMAND_LABEL
        observation = policy._policy.act.call_args.args[0]
        assert observation.shape == (11,) and observation[6] == 0
        clock.value += .02


@pytest.mark.parametrize('device', [1, 2])
def test_mit_fault_from_either_motor_disables_both(nodes, device):
    node, module, base, bus, clock = startup(nodes)
    node._command_callback(command(module))
    node._control_tick()
    node._drain_received()
    data = f1(parse_f0(bus.settings[device][0xf0]), 3)
    node._consume(base.CanReply(device, data, False, 0, clock.value), clock.value)
    node._control_tick()
    assert node._hard_fault and node._rear_integral.torque_nm == 0
    assert set(node._disable_wait) == {'steering', 'drive'}


def test_policy_requires_fresh_matching_backend_status_and_latches_its_loss(nodes):
    motor, _, _, _, clock = startup(nodes)
    loader = nodes[-1]
    module = loader('policy_controller_v2')
    module.time = SimpleNamespace(monotonic=lambda: clock.value)
    policy = module.PolicyControllerV2()
    assert 'v2_motor_backend_not_ready' in policy._readiness_reasons(clock.value)
    motor._publish_status()
    policy._backend_callback(motor._status_publisher.publish.call_args.args[0])
    assert policy._backend_ready
    clock.value += .061
    assert 'v2_motor_backend_timeout' in policy._readiness_reasons(clock.value)
    policy._state_machine.mode = module.BalanceMode.BALANCING
    policy._backend_callback(SimpleNamespace(data='{"task":"Trackstand-Balance-v1"}'))
    assert policy._state_machine.mode == module.BalanceMode.FAULT_LATCHED
    assert policy._motor_publisher.publish.call_args.args[0].data == [0, 0, 0]


def test_unversioned_torque_message_is_rejected(nodes):
    node, module, *_ = nodes
    message = command(module)
    message.layout.dim = []
    node._command_callback(message)
    assert not node._requested_enable and node._fault_latched
