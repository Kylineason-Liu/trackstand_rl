# MIT motor feedback collection

`motor_feedback_mit.py` is a self-contained, read-only MIT experiment recorder
using SocketCAN and Python's standard library. No ROS nodes, package build,
third-party CAN library, or local measurement helper is required. The CAN
interface must already be up.

Start the recorder before `steering_sine_mit.py` to capture startup, motion,
and shutdown. The sender resets its origin with B1 at each execution; the
recorder tracks this transition without mixing pre/post-reset measurements.
The current sender also writes/verifies a +/-3 rad F0 position range, keeping
its actual velocity/torque ranges. The recorder observes those settings rather
than writing or assuming them, and pauses its queries throughout preflight.
From `deploy/experiments/`:

```bash
# Terminal 1: steering only, 200 Hz target query rate.
python3 motor_feedback_mit.py --motors steering \
  --output logs/motor_feedback_mit_test.csv

# Terminal 2: example +/-0.1 rad, 2 Hz sine for 30 seconds.
python3 steering_sine_mit.py --execute --amplitude 0.1 --period 0.5 --duration 30
```

Stop the recorder after the sender has stopped, to capture its CF request and
subsequent measurements. Default collection selects both steering and drive.
`--duration 40` stops after 40 seconds. `--rate`, `--timeout`, `--interface`,
and the per-motor ID, gear-ratio, encoder-sign and torque-sign flags match the
original collector; `--help` lists them. Existing output files are never overwritten.

## CSV records

| `record_type` | Information |
| --- | --- |
| `feedback` | Original A3 encoder counts and startup-relative position; A2 velocity; A1 Q-axis current; B0 constant and current-derived torque. Each measurement has its own receipt timestamp. |
| `mit_target` | Each observed eight-byte MIT command: absolute position, velocity, Kp, Kd, feed-forward torque, unsigned codes, CAN ID and payload bytes. |
| `mit_feedback` | Each observed F1 reply: position, velocity, torque, raw codes, MIT-active and system-fault bits, with one host receipt timestamp. |
| `mit_config` | Observed F0 replies and configuration writes, including payload and active decoding-range state. |
| `mit_disable` | Observed host CF request. This records the request, not proof that output was disabled. |
| `motor_status` | Eight-byte CF acknowledgement or AE status, including voltage, bus current, temperature, mode and fault code. |
| `origin_reset` | B1 request/acknowledgement or reboot, with reference epoch and transition note. |
| `read_query` | One-byte A1/A2/A3/B0/F0/F1 request observed on a host ID. |
| `motor_reply` | Raw matching A1/A2/A3/B0 reply. A3 rows also contain single-/multi-turn counts and relative position, including the sender's pre-motion zero sample. |
| `can_frame` | Other selected-motor traffic, including C0/C1/C2/C3, unexpected opcodes or wrong lengths, preserved without invented decoding. |

Both MIT unicast IDs (`0x402` and `0x502` for steering address 2), broadcast
`0x400`, and public address `0x4FF` are recognized. Normal command group IDs
`0x00`/`0xFF` are also captured. Each group frame produces one row per selected
motor, decoded with that motor's own ranges. Unrelated motors and extended,
RTR, error, and CAN FD frames are outside this collector's data-frame scope.
The complete field list is `FIELDS` in the script.
CAN/device IDs are decimal integers in the CSV; `mit_payload_hex` is hexadecimal.
For compatibility, `mit_can_id`/`mit_payload_hex` also store non-MIT frames.
`can_direction` distinguishes local-host transmission from bus reception; it
does not identify the sending process. Query receipt flags are `True` for a
matching reply, `False` for a missing reply, and blank when not attempted.
`startup_pending` marks intentional passive-only preflight;
`mit_config_write_pending` marks an unconfirmed/mismatched F0 write.
`mit_requested_*_max_*` records requested ranges separately from confirmed
`mit_*_max_*` ranges. `measurement_epoch` identifies changes in query/startup
state, independently of the physical-origin `reference_epoch`.

## Units and position references

The original measurement units remain unchanged: `position_unwrapped_rad` uses
A3's first valid observed signed multi-turn count in the current reference epoch;
`position_rad` wraps it into [-pi, pi). Velocity is
joint rad/s. `current_a` is native measured Q-axis current; `torque_nm` is
`torque_sign * current_a * B0_constant`, in motor N m.

MIT fields ending in `_motor_rad`, `_motor_rad_s`, or `_motor_nm` retain native
absolute motor position, motor velocity and torque. `mit_kp_nm_per_motor_rad`
and `mit_kd_nm_s_per_motor_rad` are decoded gains in motor coordinates.
`mit_feedforward_torque_motor_nm` is only the additional torque field; it is
not the total torque produced by MIT's position/velocity feedback controller.

`mit_target_position_rad` and `mit_feedback_position_rad` both use the first
valid decoded F1 position in the current reference epoch as their zero:

```text
joint_relative_position_rad = encoder_sign * (native_motor_rad - first_F1_motor_rad) / gear_ratio
```

Their velocities also use encoder sign and gear ratio; signed MIT torque fields
use torque sign, as in the original collector. No angle wrapping is applied to
the MIT relative positions. This does not extend the finite MIT encoding range.
A3 and F1 have independent startup zeros; their relative positions must not be
treated as identical without checking the actual recordings. The current
sender reads A3/F1 during preflight. Capturing these replies uses the pre-motion
A3 count as zero rather than a later poll after movement. During preflight,
A3 cannot establish that zero until the observed F0 write is confirmed. Native
MIT targets and F1 feedback preserve the shared coordinate system even if
startup samples are missed or differ. Start the recorder first: it cannot
recover earlier frames.

An observed B1 request increments `reference_epoch`, sets `origin_pending=True`,
clears ranges/zeros/torque constant, and pauses measurement queries for that
motor. Its three-byte acknowledgement clears `origin_pending`, but does **not**
resume queries: the observer continues through F0 read/write/confirmation,
read-back and A3/F1 checks. The first observed MIT target with confirmed ranges
ends `startup_pending` and permits B0/A3/A2/A1 polling. An acknowledgement
without a captured request also starts a new epoch. A measurement cycle crossing
an origin, configuration or preflight-state change is discarded as NaN with a
`can_note`. A missing acknowledgement leaves that motor paused, while raw frames
and other selected motors continue recording. Group B1 acknowledgements complete
independently for each selected motor. Do not
combine relative positions across epochs. An observed reboot ends recording
with an explicit error because that command has no acknowledgement.

## Capture and timing details

The recorder sends only one-byte A1/A2/A3/B0/F0/F1 **read** queries. It never
enables/disables motors, sets origins, or writes gains, limits or MIT targets.
Shutdown sends no motor commands. A3/A2/A1 cycles target 200 Hz
per selected motor, with B0 initialization/retries. F0/F1 initialization reads
retry once a second until ranges/zero are known; F1 motion replies are then
captured passively without adding continuous F1 polling.

For the current sender, observing startup CF pauses reads on that motor before
B1/F0 verification; B1 alone also begins a pause. No additional recorder queries
are deliberately issued during this observed preflight. A query already in
flight before the observer sees CF cannot be cancelled. The sender's A3/F1
queries continue and are recorded passively. The first MIT target resumes
polling only with confirmed ranges and no pending origin/F0 write. CF after
observed motion is treated as shutdown, so read-only measurements continue
after the sine stops. If the sender aborts before motion, the recorder stays
passive, with that state visible in CSV and console; a later successful sender
startup can resume it. These distinctions follow this sender's sequence,
not an assumption that every arbitrary CF means startup or shutdown.

Actual F0 ranges are decoded as little-endian uint16 with units 0.1 rad,
0.01 rad/s and 0.01 N m. MIT target/F1 fields use high-bit-first packing.
Unknown ranges leave physical fields as NaN while retaining raw codes and
bytes; gains can still be decoded. No assumed defaults or later ranges are
silently applied to earlier rows. An observed F0 write invalidates decoding
until a valid reply exactly matches the latest observed write, including
preserved velocity/torque bytes. A mismatched/stale F0 reply leaves decoding
unavailable and adds a diagnostic note. The saved F1 zero is cleared on writes
or changes to confirmed ranges, so a zero decoded with +/-95.5 rad cannot be
retained after switching to +/-3 rad. Each row records the most
recent observed configuration timestamp in `mit_config_elapsed_s`.

`elapsed_s`, individual measurement timestamps and `mit_feedback_elapsed_s`
use the same recorder monotonic clock. Command and F1 times are **host receipt
times**, not sender timestamps, hardware timestamps or motor acquisition times.
F1 query replies and automatic MIT motion replies have the same wire format and
are not distinguished. Rows from the query loop and observer may be out of
timestamp order; sort by elapsed time before analysis. There is no resampling,
phase adjustment, interpolation or repeated cached feedback.

The query socket accepts only standard, externally received native-ID replies
with the requested opcode and exact documented length. Stale-frame draining
is bounded; a new request is not sent if the old queue cannot be cleared.
The protocol has no sequence number: a delayed earlier reply with the same
opcode cannot be uniquely associated with a new request. Local-native-ID F0
writes are distinguished using `recvmsg` direction flags; an external host's
unprefixed F0 write can be indistinguishable from a reply. Use host-prefixed
normal commands as the supplied sender does.

Both sockets enable Linux's receive-queue overflow notification. Reported drops,
truncated receive metadata, capture-thread failures or a CSV backlog exceeding
20,000 rows terminate recording with an error. `socket_dropped_frames` preserves
the observer's reported counter. CSV writes are bounded during sampling, and
shutdown drains queued observer frames for at most 0.25 s before flushing them;
an unfinished drain is an error. Zero reported drops cannot prove lossless
delivery: interface losses, sender-disabled loopback, or a final drop without
a subsequent frame are not necessarily observable here.

Measured torque is current/firmware-derived, not an independent shaft-torque
measurement; F1's conversion relies on the motor's configured torque constant.
Restart collection after motor power/reboot events. This collector is intended
for `steering_sine_mit.py`; the existing C2-specific plot/analysis script does
not accept its MIT record types yet.

## Offline verification

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s deploy/experiments/tests -p test_motor_feedback_mit.py -v
```

Tests use simulated transport and temporary CSVs. Protocol source: local
GIM CAN V3.09b0 pp.2, 4–8, 16 and 20–22. They verify parsing, Linux filter
semantics, origin transitions and recording behavior without hardware commands.
Compatibility tests execute the actual current sender functions using simulated
transport, covering custom velocity/torque ranges and failed preflight sequences.
See [the strict audit report](MOTOR_FEEDBACK_MIT_AUDIT.md) for findings and limits.
