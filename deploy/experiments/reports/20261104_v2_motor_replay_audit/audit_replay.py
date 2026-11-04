"""Independent offline wire, reference and two-device lifecycle audit.

This program never opens SocketCAN. Simulated replies implement the documented
register/frame layouts independently of the replay decoder.
"""

from collections import deque
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import struct
import sys
from unittest.mock import patch


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SCRIPT = ROOT / "deploy/experiments/v2_motor_replay.py"
spec = importlib.util.spec_from_file_location("v2_replay_audit_subject", SCRIPT)
v2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v2
spec.loader.exec_module(v2)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_wire(payload, ranges):
    p = (payload[0] << 8) | payload[1]
    v = (payload[2] << 4) | (payload[3] >> 4)
    kp = ((payload[3] & 15) << 8) | payload[4]
    kd = (payload[5] << 4) | (payload[6] >> 4)
    t = ((payload[6] & 15) << 8) | payload[7]
    return ((p * 2 / 65535 - 1) * ranges[0], (v * 2 / 4095 - 1) * ranges[1],
            kp * 500 / 4095, kd * 5 / 4095, (t * 2 / 4095 - 1) * ranges[2])


def feedback(device):
    pmax, vmax, tmax = (x * scale for x, scale in zip(device["f0"], (.1, .01, .01)))
    # The fake plant is intentionally held fixed; this is a protocol/controller
    # exercise, not a hardware-response or dynamics simulation.
    position = device["count"] * 2 * math.pi / 16384
    p = round((max(-pmax, min(pmax, position)) + pmax) / (2 * pmax) * 65535)
    velocity = round(4095 / 2)
    torque = round(4095 / 2)
    return bytes((0xF1, p >> 8, p & 255, velocity >> 4,
                  ((velocity & 15) << 4) | (torque >> 8), torque & 255, int(device["active"])))


class Clock:
    def __init__(self):
        self.now = float(os.environ.get("AUDIT_CLOCK_ORIGIN", "100"))

    def monotonic(self):
        return self.now

    def time(self):
        return 1800000000. + self.now

    def sleep(self, seconds):
        self.now += max(0., seconds)


class Log:
    def __init__(self):
        self.rows = []

    def emit(self, row):
        self.rows.append(row)

    def check(self):
        return None


class Bus:
    def __init__(self, clock):
        self.clock, self.queue, self.tx = clock, deque(), []
        self.devices = {
            1: {"count": 2147483645, "f0": (955, 4500, 1800), "cd": (0, 5000, 0), "active": False},
            2: {"count": -7454, "f0": (955, 4500, 1800), "cd": (1, 20, 0), "active": False},
        }

    def send(self, frame):
        cid, length, padded = struct.unpack("=IB3x8s", frame)
        data = padded[:length]
        self.tx.append((self.clock.monotonic(), cid, data))
        self.clock.sleep(.00015)  # Nonzero submission overhead must not accumulate scheduler drift.
        identity = cid & 255
        device = self.devices[identity]
        if cid & 0x400:
            assert len(data) == 8
            device["active"] = True
            reply = feedback(device)
        else:
            op = data[0]
            if op == 0xCF:
                device["active"] = False
                reply = bytes((op, 0, 8, 0, 0, 30, 5, 0))  # Retained mode5 must use inactive F1.
            elif op == 0xF0:
                if length > 1:
                    assert not device["active"]
                    device["f0"] = struct.unpack("<HHH", data[1:])
                reply = bytes([op]) + struct.pack("<HHH", *device["f0"])
            elif op == 0xCD:
                if length > 1:
                    assert not device["active"]
                    device["cd"] = (data[1], int.from_bytes(data[2:4], "little"), data[4])
                enabled, timeout, action = device["cd"]
                reply = bytes([op, enabled]) + struct.pack("<H", timeout) + bytes([action])
            elif op == 0xB0:
                reply = bytes([op, 0]) + struct.pack("<f", .3 if identity == 1 else .78) + bytes([0])
            elif op == 0xB3:
                reply = bytes([op]) + struct.pack("<I", 1000)
            elif op == 0xB1:
                assert identity == 2 and not device["active"], "B1 must never reset rear or active motor"
                device["count"] = 0
                reply = bytes([op, 0, 0])
            elif op == 0xA3:
                reply = bytes([op]) + struct.pack("<Hi", device["count"] % 16384, device["count"])
            elif op in (0xA1, 0xA2):
                reply = bytes([op]) + struct.pack("<i", 0)
            elif op == 0xAE:
                reply = bytes((op, 0, 8, 0, 0, 30, 5 if device["active"] else 0, 0))
            elif op == 0xF1:
                reply = feedback(device)
            else:
                raise AssertionError(f"Unexpected request {op:02X}")
        self.queue.append(struct.pack("=IB3x8s", identity, len(reply), reply))
        return len(frame)

    def recvmsg(self, *_args):
        if not self.queue:
            raise BlockingIOError
        return self.queue.popleft(), [], 0, None


def selected(_read, _write, _error, timeout):
    clock.sleep(max(0., timeout))
    return [], [], []


start_hash = digest(SCRIPT)
with patch.object(v2.socket, "socket", side_effect=AssertionError("Preview attempted SocketCAN")):
    with redirect_stdout(io.StringIO()) as captured:
        assert v2.main([]) == 0
preview = captured.getvalue()
assert "PREVIEW" in preview

rows, manifest = v2.load_reference(v2.HERE / "replay_targets/v2_26600_no_dr/policy_targets_50hz.csv", 10.)
assert len(rows) == 500 and rows[0]["time_s"] == 0. and rows[-1]["time_s"] == 9.98
reference_path = v2.HERE / "replay_targets/v2_26600_no_dr/policy_targets_50hz.csv"
source_path = ROOT / "isaaclab_train/reports/20261103_v2_26600_replay_capture/policy_targets_50hz.csv"
assert reference_path.read_bytes() == source_path.read_bytes()
assert manifest["task"] == "Trackstand-Balance-v2"

rng = random.Random(42)
wire_checks = 0
for ranges in ((95.5, 45., 18.), (12.5, 3., 6.28), (1.6, 25., 5.6)):
    for _ in range(1000):
        values = (rng.uniform(-ranges[0], ranges[0]), rng.uniform(-ranges[1], ranges[1]),
                  rng.uniform(0., 500.), rng.uniform(0., 5.), rng.uniform(-ranges[2], ranges[2]))
        payload, recorded = v2.packet(*values, ranges)
        independent = decode_wire(payload, ranges)
        assert all(abs(a-b) < 1e-12 for a, b in zip(recorded, independent))
        for actual, request, lsb in zip(independent, values,
                                        (2*ranges[0]/65535, 2*ranges[1]/4095, 500/4095, 5/4095,
                                         2*ranges[2]/4095)):
            assert abs(actual-request) <= lsb * .5000001
        wire_checks += 1

clock, log = Clock(), Log()
bus = Bus(clock)
motors = [v2.Motor("steering", 2, 23.9, 5.6), v2.Motor("drive", 1, 61.6, 6.28)]
args = v2.parse_args([])
io_owner = v2.ReplayIO(args, bus, log, motors, clock=clock)
journals = []


def persist():
    journals.append({m.name: {"restore": [x.hex() for x in m.restore], "settings": dict(m.settings)} for m in motors})


with patch.object(v2.select, "select", selected):
    v2.setup(io_owner, rows, persist)
    originals = {m.name: dict(m.settings) for m in motors}
    assert bus.devices[1]["cd"][:2] == (1, 500) and bus.devices[2]["cd"][:2] == (1, 500)
    motion_start = clock.monotonic()
    v2.motion(io_owner, rows)
    elapsed = clock.monotonic() - motion_start
    assert 9.999 <= elapsed <= 10.025, elapsed
    assert all(not m.disabled_confirmed for m in motors)
    v2.cleanup(io_owner)
assert all(m.disabled_confirmed and not m.restore for m in motors)
assert all(not d["active"] for d in bus.devices.values())
assert bus.devices[1]["f0"] == (955, 4500, 1800)
assert bus.devices[1]["cd"] == (0, 5000, 0) and bus.devices[2]["cd"] == (1, 20, 0)
b1_commands = [(cid, payload) for _, cid, payload in bus.tx if payload[0] == 0xB1 and not cid & 0x400]
assert b1_commands == [(0x102, bytes([0xB1]))]
assert motors[0].origin["verified"] and motors[1].origin["verified"]
assert motors[1].zero_count == 2147483645
bus.devices[1]["count"] = -2147483646
with patch.object(v2.select, "select", selected):
    io_owner.request(motors[1], 0xA3)
assert motors[1].travel_count == 5
assert abs(motors[1].position + 5 * 2 * math.pi / 16384) < 1e-15
packet_rows = [r for r in log.rows if r["record_type"] == "command"]
for r in packet_rows:
    index = min(499, int(r["replay_elapsed_s"] / .02 + 1e-8))
    assert r["target_index"] == index
    if r["motor"] == "steering":
        assert abs(r["encoded_position_rad"]-rows[index]["held_steering_position_target_rad"]) <= 95.5/65535 + 1e-12
        assert r["encoded_kp"] > 9.9 and r["encoded_kd"] > .19
    else:
        assert r["encoded_kp"] == 0. and abs(r["encoded_kd"] - round(.3*4095/5)*5/4095) < 1e-12
        assert abs(r["effective_velocity_target_rad_s"]-rows[index]["drive_velocity_target_rad_s"]) <= 3./4095 + 1e-12
integral_rows = [r for r in log.rows if r["record_type"] == "integral_update"]
assert abs(len(integral_rows) - 2000) <= 1
assert all(0. <= r["integral_dt_s"] <= .006 for r in integral_rows)
assert digest(SCRIPT) == start_hash, "Source changed during audit; rerun on final source"
result = dict(passed=True, source_sha256=start_hash, reference_sha256=digest(reference_path),
              preview_opens_no_socket=True, reference_prefix_rows=500,
              wire_property_cases=wire_checks, mocked_devices=2,
              source_prefix_seconds=10., measured_mock_motion_seconds=elapsed,
              monotonic_origin_s=float(os.environ.get("AUDIT_CLOCK_ORIGIN", "100")),
              steering_packets=sum(r["motor"] == "steering" for r in packet_rows),
              drive_packets=sum(r["motor"] == "drive" for r in packet_rows),
              integral_updates=len(integral_rows), b1_motor_ids=[2],
              both_cf_disabled_and_settings_restored=True,
              rear_int32_rollover_unwrapped=True,
              original_short_watchdog_restored=True, real_CAN_opened=False,
              physical_response_validated=False)
suffix = "" if result["monotonic_origin_s"] == 100. else "_" + str(int(result["monotonic_origin_s"]))
(OUT / ("independent_audit" + suffix + ".json")).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
