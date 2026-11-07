# Rear v2 controller characterization

From the repository root:

```bash
python3 deploy/experiments/drive_v2_characterization.py
python3 deploy/experiments/drive_v2_characterization.py --execute
```

Requires Python 3 with NumPy and the repository's adjacent experiment helpers and
`deploy/source/trackstand_deploy` package. No ROS nodes are required. Stop other
CAN writers. Support the frame and leave the rear wheel airborne/free. This script
never queries, disables or commands steering; arrange its condition independently.

The default plan contains 300 positive velocity targets, 0.1 through 30.0 rad/s in
0.1 increments. Each is held for five seconds, then CF disables the rear motor,
disable is confirmed, and a nominal two-second coasting window follows. There is
no rest-speed requirement or braking command. Thus the next trial can start with
nonzero velocity. Then five ten-second sine trials use amplitude 4 rad/s at
0.5, 1, 2, 5 and 10 Hz, with the same disabled rests. Nominal total: 2160 seconds
(36 minutes), plus setup, disable acknowledgements and query/printing overhead.
`--amplitude`, `--frequencies` and `--sine-duration` customize only the sine suite.

MIT packets and the host integrator update at nominal 200 Hz; A1/A2/A3/AE queries
run at nominal 100 Hz. Late slots are skipped/logged, without catch-up bursts or
lateness aborts. This is not hard real-time scheduling. The disabled window is
scheduled for exactly 2 seconds, not extended until the wheel stops.

The controller directly imports the current v2 helpers: Kp=0, requested Kd=0.3
(encoded 0.3003663), Ki=3.3, actual-elapsed-time trapezoidal integration,
anti-windup, native sign -1 and feedforward quantization compensation. Integral
resets on disable/start of a new trial; it persists throughout each sine. F0
preserves position range and uses the full v2 velocity range, rather than a
narrower experimental codec. CD is temporarily 500 ms. B3/B5/B8/B9 are read only.
Original F0/CD are durably saved before writes, verified, then restored after CF
on normal completion, interruption or errors. Failed restoration remains in JSON
as `pending_restore`; power loss cannot run cleanup, so retain this JSON.

A 30 rad/s step from rest initially demands about 9 Nm from the proportional
term. The integral bound is **not** a verified firmware bound on total MIT torque.
The CSV logs both estimated P+I torque and its clipped simulation reference;
the latter is diagnostic and is not sent. Existing firmware current limits are
read/logged, not changed. There are no experiment speed/current/torque/angle,
tracking, rest-speed or tight-deadline aborts. Motor AE/F1 faults, malformed
frames, competing rear writers, transport/logging failures and unavailable A2
feedback still stop the test. A2 must be finite and younger than the configurable
`--feedback-timeout` (default one second), because integration needs valid velocity.

Three outputs are saved under `deploy/experiments/logs/` (or `--output FILE.csv`):

- CSV: every raw TX/RX frame plus decoded feedback, requested/wire/effective
  velocity, current-derived torque, PI state, timestamps, trial/phase and timing.
- JSON: plan, arguments, source/contract hashes, original/applied settings,
  phase durations and rates/counts, completion and restoration status.
- `_plan.csv`: targets, sine frequencies, durations and rests.

A3 uses signed multi-turn counts with rollover handling. The first startup A3 is
only a software log zero; no B1 is sent. Each trial also logs displacement from
its own freshly sampled start position. This does not physically reposition the
wheel or zero its velocity. Airborne measurements do not establish loaded-bike
performance.

If shutdown could not restore settings, use the saved JSON explicitly:

```bash
python3 deploy/experiments/drive_v2_characterization.py --recover-settings path/to/run.json --execute
```

Recovery sends rear CF, verifies disable, restores and verifies the recorded
pending F0/CD values, and creates a separate CSV/JSON/plan record. It sends no
motion packets and does not edit the original log. Check the new JSON's
`disabled_and_restored` before starting another run. After a hard interruption,
recover the interrupted run before accepting new firmware settings as originals.
