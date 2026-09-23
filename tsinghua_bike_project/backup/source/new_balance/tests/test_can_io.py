"""Tests for new_balance.can_io — the CAN request/response helpers.

These cover the failure mode the module exists to prevent: two nodes sharing
one SocketCAN interface, where every socket sees every frame, so pairing a
request to a response by arrival order silently slips out of phase.

No hardware and no CAN interface are needed.  python-can is used if present;
otherwise a minimal stand-in supplies the two names can_io touches.

    python -m unittest discover -s tests        # from source/new_balance/
"""
import sys
import types
import unittest
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]     # source/new_balance/
sys.path.insert(0, str(_PKG_ROOT))

try:                                                # real python-can if installed
    import can
except ImportError:                                 # otherwise a stand-in
    can = types.ModuleType("can")

    class _Message:
        def __init__(self, arbitration_id=0, data=b"", is_extended_id=False):
            self.arbitration_id = arbitration_id
            self.data = bytearray(data)
            self.is_extended_id = is_extended_id

    can.Message = _Message
    can.CanError = type("CanError", (Exception,), {})
    can.BusABC = object
    sys.modules["can"] = can

from new_balance.can_io import await_response, drain, query  # noqa: E402

STEER_FB, STEER_HOST = 0x001, 0x101
REAR_FB, REAR_HOST = 0x002, 0x102


def msg(can_id, cmd, payload=b"\x00" * 6):
    """Build a frame.  Keyword args only — real can.Message takes timestamp first."""
    return can.Message(arbitration_id=can_id,
                       data=bytes([cmd]) + payload,
                       is_extended_id=False)


class FakeBus:
    """recv() drains a scripted queue; send() enqueues that request's replies."""

    def __init__(self, on_send=None):
        self.queue = []
        self.sent = []
        self._on_send = on_send or (lambda m: [])

    def send(self, message):
        self.sent.append(bytes(message.data))
        self.queue.extend(self._on_send(message))

    def recv(self, timeout=0.0):
        return self.queue.pop(0) if self.queue else None


class TestQuery(unittest.TestCase):

    def test_foreign_frames_never_shift_a_request_response_pair(self):
        """The original bug: rear's traffic interleaved into steer's queue.

        Pairing by arrival order let one foreign frame put A3/A4 permanently
        out of phase, so _parse_a3 was handed the A4 reply forever after.
        """
        def on_send(message):
            cmd = message.data[0]
            if cmd == 0xA3:
                return [msg(REAR_HOST, 0xA2),                  # looped-back query
                        msg(STEER_FB, 0xA3, b"\x11" * 6),      # the reply we want
                        msg(REAR_FB, 0xA2)]                    # rear motor's reply
            if cmd == 0xA4:
                return [msg(STEER_FB, 0xA4, b"\x22" * 6)]
            return []

        bus = FakeBus(on_send)
        req_a3 = can.Message(arbitration_id=STEER_HOST, data=bytes([0xA3]),
                             is_extended_id=False)
        req_a4 = can.Message(arbitration_id=STEER_HOST, data=bytes([0xA4]),
                             is_extended_id=False)

        for cycle in range(5):
            a3 = query(bus, req_a3, STEER_FB, 0.002)
            a4 = query(bus, req_a4, STEER_FB, 0.002)
            self.assertEqual(a3[0], 0xA3, f"cycle {cycle}: A3 slot got 0x{a3[0]:02X}")
            self.assertEqual(a4[0], 0xA4, f"cycle {cycle}: A4 slot got 0x{a4[0]:02X}")
            self.assertEqual((a3[1], a4[1]), (0x11, 0x22))

    def test_command_echo_on_our_own_feedback_id_is_ignored(self):
        """main_balance's C1/C0 echoes share the feedback ID — no filter drops them."""
        bus = FakeBus(lambda m: [msg(STEER_FB, 0xC1),
                                 msg(STEER_FB, 0xA3, b"\x77" * 6)])
        req = can.Message(arbitration_id=STEER_HOST, data=bytes([0xA3]),
                          is_extended_id=False)
        resp = query(bus, req, STEER_FB, 0.002)
        self.assertEqual((resp[0], resp[1]), (0xA3, 0x77))

    def test_drain_discards_a_late_reply_from_the_previous_cycle(self):
        """A reply that missed its window carries the same command byte, so
        only the pre-send drain can tell it from a fresh one."""
        stale = msg(REAR_FB, 0xA2, b"\xAA" * 6)
        fresh = msg(REAR_FB, 0xA2, b"\xBB" * 6)
        req = can.Message(arbitration_id=REAR_HOST, data=bytes([0xA2]),
                          is_extended_id=False)

        bus = FakeBus(lambda m: [fresh])
        bus.queue.append(stale)
        self.assertEqual(query(bus, req, REAR_FB, 0.002)[1], 0xBB,
                         "served a stale reply from the previous cycle")

        # Without the drain the stale frame wins — this is what drain buys.
        bus2 = FakeBus(lambda m: [fresh])
        bus2.queue.append(stale)
        self.assertEqual(
            query(bus2, req, REAR_FB, 0.002, drain_first=False)[1], 0xAA)

    def test_timeout_returns_none(self):
        req = can.Message(arbitration_id=STEER_HOST, data=bytes([0xA3]),
                          is_extended_id=False)
        self.assertIsNone(query(FakeBus(), req, STEER_FB, 0.002))

    def test_send_failure_returns_none(self):
        class DeadBus(FakeBus):
            def send(self, message):
                raise can.CanError("no link")

        req = can.Message(arbitration_id=STEER_HOST, data=bytes([0xA3]),
                          is_extended_id=False)
        self.assertIsNone(query(DeadBus(), req, STEER_FB, 0.002))

    def test_empty_request_is_rejected(self):
        bus = FakeBus()
        empty = can.Message(arbitration_id=STEER_HOST, data=b"",
                            is_extended_id=False)
        with self.assertRaises(ValueError):
            query(bus, empty, STEER_FB, 0.002)


class TestDrain(unittest.TestCase):

    def test_is_bounded(self):
        bus = FakeBus()
        bus.queue = [msg(STEER_FB, 0xA3) for _ in range(500)]
        self.assertEqual(drain(bus, max_frames=64), 64)

    def test_reports_zero_on_an_empty_queue(self):
        self.assertEqual(drain(FakeBus()), 0)


class TestAwaitResponse(unittest.TestCase):

    def test_rejects_other_feedback_ids(self):
        bus = FakeBus()
        bus.queue = [msg(REAR_FB, 0xA3), msg(STEER_FB, 0xA3, b"\x99" * 6)]
        self.assertEqual(await_response(bus, STEER_FB, 0xA3, 0.002)[1], 0x99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
