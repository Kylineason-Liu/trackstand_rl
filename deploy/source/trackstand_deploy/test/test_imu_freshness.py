"""Exercise real serial parsing with a fake port; no UART/I2C device is opened."""

from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import math
import struct
import sys

import pytest
import serial


@pytest.fixture
def driver(node_module, monkeypatch):
    # The vendor package eagerly imports its unused I2C backend. Keep that
    # dependency isolated while executing the actual serial decoder below.
    smbus = ModuleType("smbus2")
    smbus.SMBus = Mock()
    monkeypatch.setitem(sys.modules, "smbus2", smbus)
    monkeypatch.setattr(serial, "Serial", Mock(return_value=Mock()))
    module = node_module("imu_serial")
    clock = SimpleNamespace(value=100.)
    module.time = SimpleNamespace(monotonic=lambda: clock.value)
    imu = module.YbImuSerial("fake-port")
    return imu, clock


def raw(imu, gyro=(100, -200, 300)):
    imu._parse_data(imu.FUNC_REPORT_IMU_RAW, struct.pack("<9h", 0, 0, 0, *gyro, 0, 0, 0))


def quat(imu, value=(1., 0., 0., 0.)):
    imu._parse_data(imu.FUNC_REPORT_IMU_QUAT, struct.pack("<4f", *value))


@pytest.mark.parametrize("first", ["gyro", "quaternion"])
def test_no_sample_until_both_required_channels_are_received(driver, first):
    imu, clock = driver
    (raw if first == "gyro" else quat)(imu)
    assert imu.get_timed_sample() is None
    clock.value += .01
    (quat if first == "gyro" else raw)(imu)
    time, quaternion, gyro = imu.get_timed_sample()
    assert time == 100. and quaternion == (1., 0., 0., 0.)
    assert gyro == pytest.approx(tuple(v * 2000 / 32767 * math.pi / 180 for v in (100, -200, 300)))


@pytest.mark.parametrize("frozen", ["gyro", "quaternion"])
def test_one_channel_and_unrelated_frames_cannot_refresh_frozen_channel(driver, frozen):
    imu, clock = driver
    raw(imu)
    quat(imu)
    before = imu.get_timed_sample()
    clock.value += 1
    (quat if frozen == "gyro" else raw)(imu)
    imu._parse_data(imu.FUNC_REPORT_IMU_EULER, struct.pack("<3f", 0., 0., 0.))
    imu._parse_data(imu.FUNC_VERSION, b"\x01\x02\x03")
    imu._parse_data(0xFF, b"anything")
    assert imu.get_timed_sample()[0] == before[0]


@pytest.mark.parametrize("value", [(0., 0., 0., 0.), (math.nan, 0., 0., 0.), (math.inf, 0., 0., 0.)])
def test_invalid_quaternion_cannot_replace_last_valid_snapshot(driver, value):
    imu, clock = driver
    raw(imu)
    quat(imu)
    before = imu.get_timed_sample()
    clock.value += 1
    quat(imu, value)
    assert imu.get_timed_sample() == before


@pytest.mark.parametrize("opcode,length", [(0x04, 0), (0x04, 12), (0x16, 4), (0x26, 3), (0x81, 0)])
def test_checksum_valid_but_short_frame_does_not_kill_decoder_or_make_sample(driver, opcode, length):
    imu, _ = driver
    imu._parse_data(opcode, bytes(length))
    assert imu.get_timed_sample() is None


def test_snapshot_tuples_are_immutable_and_do_not_alias_later_vendor_updates(driver):
    imu, clock = driver
    raw(imu)
    quat(imu)
    before = imu.get_timed_sample()
    clock.value += .01
    raw(imu, (400, 500, 600))
    quat(imu, (0., 1., 0., 0.))
    assert before[0] == 100. and before[1] == (1., 0., 0., 0.)
    assert imu.get_timed_sample()[0] == pytest.approx(100.01)
    assert imu.get_timed_sample()[1] == (0., 1., 0., 0.)
