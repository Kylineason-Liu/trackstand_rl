# Recorded policy steering replay: implementation and audit

Implemented on 2026-10-26. This is an offline software verification; no CAN
interface or real motor was used by the implementation/testing work.

See [the operating guide](../../STEERING_POLICY_REPLAY.md) and
[the script](../../steering_policy_replay.py). The
[exported CSVs and manifest](../../replay_targets/manifest.json) contain both
command levels from the actual model_2500 first episodes, env 0, in Isaac400,
MuJoCo400 and MuJoCo800: 1,500 samples each. Both steering levels are identical
throughout these three episodes. Earlier commentary suggesting startup
differences for these particular episodes was corrected after inspecting exports.

The default 12.50–17.50 s core plus preparation/holds reproduces all 400 targets
of each completed simulator replay exactly at float32 precision. The scaled and
conditioned selection remains explicit, with optional once-per-20-ms v1 software
conditioning available only for scaled inputs. Input samples never advance at
the packet rate; four repeated packets carry the same target at 200 Hz.

Verified with **21 new tests**, plus **29 existing shared-helper tests**:

- All exported raw/scaled/conditioned/submitted values versus original NPZs.
- All three default plans versus the existing simulator replay input arrays.
- Actual saved CSV fields, signs/units, decoded MIT gains and packet intervals.
- One selected-motor CAN socket; no rear, B1 or configuration-write packets.
- Byte-identical four-packet groups, both input selectors, and 50 Hz slew updates.
- Fresh query correlation, preserved kernel queue age and missing timestamp refusal.
- Missed deadlines without target skipping/bursts; a second check before send.
- F1 faults/inactive state, stale F1/A1, transient current/angle excursions,
  range/interval errors and existing-output refusal.
- Disable attempts and fresh CF/AE confirmation after failures, including a
  simulated background CSV-writer failure. Logging failure cannot bypass cleanup.

The audit fixed preparation-ramp float32 rounding to match the saved simulator
inputs, removed inherited query matching that could accept an old kernel arrival,
and made current/angle aborts act on every received sample. Syntax, Pyflakes,
120-column and whitespace checks passed. `python3 -S` preview confirms the Pi
runtime has no site-package/ROS/NumPy dependency; the adjacent helper is required.

**228 protected files** are unchanged during the final audit, including production
code, existing shared helper, original captures/replays, both progress ledgers
and Git index. Only new replay/export/data/docs/tests/audit files and the
experiment README were added/changed. Existing unrelated submodule state was
preserved. See [verification.json](verification.json),
[new test log](replay_tests.log), and [shared-helper test log](shared_transport_tests.log).

These tests use virtual CAN/clock responses, not a motor-response model or real
hardware evidence. Actual Pi scheduling, loaded response, effective MIT torque
protection, firmware feedback frequency and real tracking accuracy remain to be
measured. F0 encodes ranges; the script does not certify a physical torque cap.
Default current/angle aborts stop the experiment rather than regulating torque.
The script does not enable a watchdog or change existing origin/range/limit settings.
