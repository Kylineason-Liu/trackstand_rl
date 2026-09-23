#!/usr/bin/env python3
"""Shared SocketCAN request/response helpers for the motor publisher nodes.

Why this module exists
----------------------
SocketCAN is a broadcast stack: every socket opened on ``can0`` receives
**every** frame on the interface — the other motor's replies, and, via
the kernel's local loopback, the query frames *other processes* send.
(``receive_own_messages=False`` only suppresses a socket's echo of its
own transmissions; it says nothing about other sockets.)

Two consequences shape the helpers below:

1. A kernel filter (`apply_feedback_filter`) removes the bulk of that
   traffic before it ever reaches Python.

2. The filter is **not sufficient**.  Frames carrying our own feedback ID
   arrive that are not replies to our query — ``main_balance`` commands
   the same motors with C1/C0, and the motor echoes those on the same
   feedback ID.  No ID filter can separate them.  So a node must never
   assume the next frame on its socket is the reply it just asked for:
   `query` matches on the echoed command byte and discards the rest.

Getting (2) wrong is subtle.  Code that pairs requests to responses by
arrival order works perfectly until a second node joins the bus, then
slips one frame out of phase and stays there.
"""

from __future__ import annotations

import time

import can

_STD_ID_MASK = 0x7FF


def apply_feedback_filter(bus: can.BusABC, feedback_id: int) -> None:
    """Restrict ``bus`` to frames from ``feedback_id`` (kernel-side).

    Everything else on the interface — the other motor, other nodes'
    looped-back queries — is dropped before reaching userspace.
    """
    bus.set_filters(
        [{"can_id": feedback_id, "can_mask": _STD_ID_MASK, "extended": False}])


def drain(bus: can.BusABC, max_frames: int = 64) -> int:
    """Discard whatever is already queued on ``bus``; return the count.

    Called just before sending a query, so anything still queued arrived
    *before* the request and therefore cannot be its reply — keeping it
    would mean serving a reply from an earlier cycle.  Bounded so a
    babbling bus cannot trap the caller here.
    """
    dropped = 0
    while dropped < max_frames:
        try:
            if bus.recv(timeout=0.0) is None:
                break
        except can.CanError:
            break
        dropped += 1
    return dropped


def await_response(
    bus: can.BusABC, feedback_id: int, cmd: int, timeout: float
) -> bytes | None:
    """Read frames until the one from ``feedback_id`` echoing ``cmd``.

    Returns its payload, or None if ``timeout`` elapses first.  Frames
    that are neither are discarded, so a request/response pair cannot
    slip out of phase.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return None
        try:
            msg = bus.recv(timeout=remaining)
        except can.CanError:
            return None
        if msg is None:
            return None
        if msg.arbitration_id != feedback_id:
            continue
        if len(msg.data) < 1 or msg.data[0] != cmd:
            continue
        return bytes(msg.data)


def query(
    bus: can.BusABC,
    request: can.Message,
    feedback_id: int,
    timeout: float,
    drain_first: bool = True,
) -> bytes | None:
    """Send ``request`` and return the reply echoing its command byte.

    ``request`` is taken prebuilt so callers can hoist it out of their
    poll loop.  Returns None if the send fails or no matching reply
    arrives within ``timeout``.
    """
    if len(request.data) < 1:
        raise ValueError("request must carry at least a command byte")
    cmd = request.data[0]

    if drain_first:
        drain(bus)

    try:
        bus.send(request)
    except can.CanError:
        return None

    return await_response(bus, feedback_id, cmd, timeout)
