# Strict MIT feedback collector audit — 2026-10-04

The original findings/results below describe the 2026-10-04 versions. See the
2026-10-05 compatibility supplement at the end for the current sender's F0
write/read-back sequence and updated recorder behavior.

Audited `motor_feedback_mit.py` against the local GIM CAN V3.09b0 PDF,
the supplied Python driver, `steering_sine_mit.py`, Linux CAN headers and
kernel documentation/source. Findings below are source/protocol and offline
transport results, not measurements of real motor firmware or CAN delivery.

## Confirmed defects and corrections

| Finding | Correction and regression evidence |
| --- | --- |
| Critical receive filter: `0xE00007FF` includes `CAN_ERR_FLAG`, registering error filters instead of normal motor-data filters. | Use `0xC00007FF` on both sockets. Simulated transport now models Linux's separate error-filter path, and checks every installed ID receives standard data while rejecting EFF/RTR/error frames. |
| MIT broadcast/public frames were excluded. | Observe `0x400`/`0x4FF` and normal `0x00`/`0xFF`; group frames are routed to each selected motor using its own configuration. |
| Shutdown exited immediately, abandoning already queued frames. | Nonblocking bounded drain followed by final CSV flush; regression includes final CF request. A sustained flood produces an explicit drain error. |
| Inherited read helper unpacked short frames without a length check. | Standalone guarded decoder and exact device/opcode/DLC matching; malformed, wrong-ID, wrong-opcode and local-host packets cannot become measurements. |
| Stale draining and CSV writing could continue under sustained traffic, starving sampling. | Bound both operations; no new query while stale backlog remains. A writer backlog over 20,000 rows is an explicit incomplete-recording error. |
| Local native-ID F0 writes looked like confirmed motor replies. | Inspect `recvmsg` direction flags; writes invalidate ranges until a motor reply. |
| CF acknowledgements, unsolicited AE status, and other selected-motor commands were silently discarded. | Decode AE/CF status and retain unknown/interfering commands as raw diagnostic rows. Test covers all 256 opcode bytes at every payload length 1–8, plus empty frames. |
| No visibility into kernel receive-queue overflow. | Enable `SO_RXQ_OVFL` on both sockets; reported drops or truncated frame/ancillary metadata terminate with error. Preserve the observer's overflow counter on the current row. |
| B1 made existing startup-relative zeros invalid; an interim implementation stopped recording on every B1. The current sender sends B1 on every execution. | Capture request and acknowledgement, pause that motor's measurements, increment its reference epoch, reacquire ranges/zeros/constant after acknowledgement, and discard any cycle crossing the reset. End-to-end simulated CSV test covers CF → B1 → acknowledgement → F0/F1 → MIT targets/replies → CF/status. |
| B0 replies of undocumented lengths 6/8 could be accepted. | Require the V3.09b0 length of 7; invalid constants leave torque NaN. Missing measurement replies are NaN, not zero or reused samples. |

Before repair, offline reproductions returned `None` for a MIT broadcast,
read zero already-queued tail frames after stop, and raised `unpack requires
a buffer of 16 bytes` for a short query frame. The filter defect was confirmed
separately against Linux's receive-list implementation, not just a generic
bitwise-filter model.

## Command/reply and unit audit

The recorder transmits **only** one-byte reads on `0x100 | device_id`.
It does not send MIT motion, CF, B1, reboot, or configuration writes, including
on shutdown. Read replies must arrive on the selected native device ID.

| Message | Required length and decoded payload (zero-based byte indexes) |
| --- | --- |
| A1 | Reply 5: signed little-endian int32 `[1:5]` × 0.001 A; measured torque = native Q-axis current × B0 constant × torque sign. |
| A2 | Reply 5: signed little-endian int32 `[1:5]` × 0.01 rpm; convert × 2π/60, encoder sign, ÷ gear ratio. |
| A3 | Reply 7: signed little-endian multi-turn int32 `[3:7]`, 16,384 counts/revolution; subtract first valid count in this epoch, apply encoder sign and gear ratio. Single-turn `[1:3]` is not used for this position. |
| B0 | Reply 7: little-endian float32 `[2:6]`; require a finite 0.01–10 Nm/A constant, otherwise do not fabricate torque. |
| F0 | Read reply/configuration write 7: three little-endian uint16 fields `[1:3]`, `[3:5]`, `[5:7]`, scales 0.1 rad, 0.01 rad/s, 0.01 Nm. Writes are not confirmation. |
| MIT command | 8, **no opcode**; IDs `0x400|id`/`0x500|id`, plus documented group IDs. High-bit-first fields p16/v12/Kp12/Kd12/torque12; p/v/torque symmetric ranges from F0; Kp 0–500, Kd 0–5, unsigned. |
| F1 / automatic MIT reply | 7: opcode F1, p16 `[1:3]`, v12 `[3]` and high nibble `[4]`, torque12 low nibble `[4]` and `[5]`; status `[6]` bit0 MIT active, bit1 system fault. |
| CF request / AE or CF status | Request CF 1; status 8: uint16 `[1:3]` × 0.01 V, uint16 `[3:5]` × 0.01 A, temperature `[5]`, mode `[6]`, fault `[7]`. Request and acknowledgement remain separate records. |
| B1 | Request 1; acknowledgement 3. Both are raw-recorded; acknowledgement completes the origin transition for that motor. |
| Reboot | Exact 8-byte `00FF00FF00FF00FF` request. Preserve it and terminate; there is no documented acknowledgement to establish readiness. |
| Other selected-motor traffic | Raw ID, direction, opcode when applicable, length and payload are retained. C0/C1/C2/C3 and arbitrary firmware parameters are **not** interpreted as MIT targets. This is not a universal semantic decoder for every GIM operation. |

Position/velocity/torque nibble packing, gain endpoints, signs, gear conversion,
native absolute origin, independent A3/F1 zeros and nearest-encoded-zero
residuals passed checks. No confirmed original MIT scaling/packing defect was
found. Native absolute positions are never sign-flipped before subtracting
their native zero. Torque fields remain motor torque, not a claimed gearbox
output torque or independently measured shaft torque.

The PDF's p.21 F0/F1 reply tables list several IDs, but p.2 explicitly specifies
native device IDs for all replies; the supplied `CANInterface.receive_response`
agrees. This collector follows that explicit convention. Prefixed F0 is treated
as a host write, not silently reclassified as a motor reply. The p.26 example
also conflicts with the main status table's current scale; the main table and
vendor `motor.py` AE/CF decoders both use 0.01 A. Raw current is retained.

## Concurrency, timing and incomplete recordings

An observer-only thread processes CAN frames in receive order and maintains
locked per-motor ranges/reference state. The main thread alone writes CSV and
issues reads. An origin epoch guards A3/B0 commits; a reset-crossing compound
sample is discarded rather than mixing coordinates. While B1 is pending,
measurements and units requiring references remain unavailable, but raw
capture and other motors continue. Missing B1 acknowledgement is visible in
`origin_pending` and does not silently resume measurement.

Each A3/A2/A1 query has its own host receipt timestamp, and query flags indicate
received/missing/not-attempted replies. Observer and compound measurement rows
can be out of timestamp order; sort before analysis, and separate epochs.
Nominal 200 Hz is a requested cycle rate per motor, not a measured guarantee:
six sequential read round trips for two motors plus startup queries may overrun
5 ms. The loop does not issue catch-up bursts. F1 is captured passively after
initialization; the collector does not add continuous F1 polling.

Existing outputs are refused before opening CAN. Capture socket errors reach
the main writer. Shutdown drains queued traffic for at most 0.25 seconds and
joins the receiver with a bound, then flushes captured rows. Reported overflow,
writer backlog, drain failure and truncation return an error instead of a
success message implying a complete recording.

## What code and offline tests cannot establish

- No sequence number exists: delayed earlier same-opcode replies cannot be
  uniquely correlated to a later query. F1 query and automatic replies also
  have identical wire formats.
- An external controller's unprefixed native-ID F0 write is indistinguishable
  from an F0 reply. Local-origin flags distinguish local writes; using the
  normal host-prefixed IDs avoids this ambiguity in the supplied setup.
- Host receipt time is not hardware or motor acquisition time. Direction
  flags identify local-host origin, not a particular sending process.
- `SO_RXQ_OVFL` reports socket queue drops, not every interface/hardware loss.
  A final drop without another delivered packet may lack a notification.
  A sender that disables loopback prevents local target observation.
- Unobserved power cycling, firmware variants, actual query throughput,
  physical torque calibration and complete real CAN delivery need hardware
  evidence. No hardware commands were sent during this audit.

## Verification and sources

Regression checks use simulated CAN transports, kernel-aware filter semantics,
threaded reception and temporary CSVs. Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s deploy/experiments/tests -p 'test_*mit.py' -v
python3 -m pyflakes deploy/experiments/motor_feedback_mit.py \
  deploy/experiments/tests/test_motor_feedback_mit.py
```

Final result: **59 tests passed** (38 collector tests and 21 sender tests),
including full CSV startup/shutdown capture, reference changes during a query,
overflow/truncation, signed int32 boundaries and signal-handler cleanup.
Pyflakes reported no issues; the edited files contain no trailing whitespace.
The original C2 collector's SHA-256 remains
`8567f24684c4a5925e9b6884de6410ba52e801fa535c2b9fea0f33c356f5b20e`.
The sender was preserved at the version inspected during final verification,
SHA-256 `6378d84ae0d47e387f4ff4c9414234c8da86c9802c0c2fddc3f890329785b3e3`.

Local primary sources:

- GIM `自定义CAN通信协议_3.09b0.pdf` in
  `tsinghua_bike_project/assign_2_Hardwares/assign_2_Hardwares/3.RearSteer Servo【GIM3510-8】/CAN Protocol Doc/`:
  pp.2, 4–8, 16, 20–22; p.26 inconsistency cross-checked against the driver.
- Adjacent `GIM_Motor_Driver/motor_driver_python/can_interface.py` and `motor.py`.
- `/usr/include/linux/can.h`, `/usr/include/linux/can/raw.h`,
  `/usr/include/asm-generic/socket.h`, and the local `socket(7)` man page.
- [Linux SocketCAN filters and message flags](https://docs.kernel.org/networking/can.html),
  and [Linux CAN receive-list implementation](https://github.com/torvalds/linux/blob/master/net/can/af_can.c)
  (`can_rcv_list_find`: ERR-masked filters use a separate receive list).

Only the MIT collector, its regression tests and the related recording/audit
documentation were edited here. The original C2 collector and MIT sender
implementation were not edited by this audit.

## Compatibility supplement — 2026-10-05

The newly inspected, user-confirmed functional sender performs:
CF/disabled acknowledgement → B1/acknowledgement → F0 read → F0 write with
Pos_Max=3 rad and preserved velocity/torque bytes → exact write reply →
independent F0 read-back → A3 zero check → stationary/inactive/fault-free F1
check → MIT motion → exit CF. Its SHA-256 is
`71d9ddf0d50099d46771986bc8d11fcdd38c8302ae5b5ded4efeef3eaeaafc08`;
the sender implementation was not changed for this update.

The recorder previously resumed queries at B1 acknowledgement. That could
interleave its F0/F1 reads with the sender's write/read-back verification.
It also retained an F1 zero decoded under an earlier position range after
an F0 write, and could use its first post-motion A3 poll instead of the sender's
pre-motion sample as zero. These are compatibility defects, not claims that
the proven sender or the motor has failed.

Corrections:

- Pause the affected motor's read queries from observed startup CF/B1 through
  the first MIT target with confirmed ranges. B1/F0 replies alone do not resume
  polling. Raw reception continues; other selected motors still poll.
- Track requested F0 bytes separately; accept a confirming reply only when all
  fields match. Mismatches retain raw data and leave physical decoding unknown.
  Do not hardcode 3 rad as a fallback or overwrite velocity/torque ranges.
- Clear the saved F1 zero on a write or confirmed-range change, and latch the
  new-range preflight F1 position. Capture the sender's A3 startup counts after
  F0 confirmation; a later poll does not redefine this zero.
- Guard compound samples with a measurement-state epoch as well as the origin
  epoch. Discard cycles crossing preflight/configuration changes.
- Show startup/origin pause state in console and CSV. Record requested ranges,
  F0 confirmation state and A3 single-turn counts alongside existing telemetry.
  Exit CF after observed motion leaves read-only polling enabled.

The compatibility tests run the **actual sender functions** against simulated
CAN, with initial F0 tuples `(955,4500,1800)`, `(120,800,300)` and `(30,1234,456)`.
Each produces the expected 3-rad position range without changing its other
ranges. No recorder reads are issued during observed preflight. Tests cover
rejected/missing startup replies, old F0 replies, distinct old/new quantized F1
zeros, pre-motion A3 versus later moved samples, independent motor pauses and
configuration changes during an in-flight query. A threaded CSV test captures
the new write/acknowledgement/read-back/A3/F1 sequence and shutdown traffic.
Validation: **72 tests passed** (43 collector and 29 current-sender tests);
Pyflakes reported no issues. No hardware commands were sent for this update.

The recorder does not validate the sender's physical alignment or reproduce
its preflight safety decisions; it records them. Starting it before the sender
is required for complete startup capture. An already in-flight read cannot be
cancelled when CF is first observed, and the protocol still lacks request IDs.
If startup fails before motion, the recorder stays passive until a later
successful startup; this is explicitly visible, not a receive failure.
