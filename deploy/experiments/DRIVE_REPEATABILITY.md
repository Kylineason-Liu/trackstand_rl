# Controlled rear-wheel C0 repeatability experiment

`drive_repeatability.py` tests the low-torque regime implicated by the original
50/200 Hz replay discrepancy. It commands rear motor **ID 1 only**, using the
verified signs: positive policy torque/angle/speed means forward wheel motion;
native C0 current and A2/A3 signs are reversed. No ROS or package build is needed.
Python's standard library is sufficient on the Pi.

This battery can distinguish starting-angle, command-rate, direction, breakaway
and history effects. It cannot establish the motor's complete operating envelope,
ground-contact behavior, independent shaft torque or peak-torque duration.

## Setup and first run

Keep the rear wheel fully airborne/free, with the frame independently supported.
Independently disable steering and stop all deployment nodes, feedback collectors,
`cansend` loops and other CAN writers. Keep the fixture, power supply and firmware
settings unchanged. Only this script owns rear CAN during a session.

Transfer/update these adjacent files together:

- `drive_repeatability.py`
- `drive_policy_replay.py`
- `steering_policy_replay.py`
- `steering_sine_mit_io.py`
- `replay_targets/` including its manifest and selected CSV

From the repository root:

```bash
# Show the complete design without opening CAN or creating files.
python3 deploy/experiments/drive_repeatability.py

# Recommended first comparison: 24 trials, 2 angles x 2 rates x 3 repeats x 2 waveform polarities.
python3 deploy/experiments/drive_repeatability.py --execute \
  --families replay --angles 0 180 --rates 50 200 --repeats 3 \
  --session deploy/experiments/logs/rear_matched_replay
```

Put a visible mark **A** on the wheel and a fixed pointer on the frame. After
confirming disabled mode, the script asks you to align A to the pointer, verifies
continuous rest and fresh status, and records the native multi-turn count as the
session reference. For every subsequent trial:

1. The script confirms the motor is disabled before asking you to touch the wheel.
2. Physically rotate the wheel to the requested angle **forward from A**. Quarter
   turns are 90/180/270 degrees. Do not reset the firmware encoder with B1.
3. Remove hands and press Enter. The script verifies the circular encoder phase
   (default tolerance 2 degrees) and continuous rest for 1 second before any C0.
   A phase mismatch asks you to reposition; failure to settle stops the session.
   After zero-current preparation, phase/speed are checked again before the first
   nonzero target. Unexpected movement during preparation stops that trial.
4. The trial runs automatically, followed by CF disable and verified CD restoration.
   Wait for the next **DISABLED** prompt before repositioning. A wheel may coast
   after torque is removed. Enter `q` at a prompt or Ctrl-C to stop.

The first mark is an arbitrary repeatable physical orientation, not an encoder
origin reset. Every trial separately records a fresh software displacement zero.
During manual waits only stdin uses a worker thread; CAN remains owned by the main
thread and disabled-status queries maintain communication. The script never moves
the wheel into position automatically.

## Test cases

The full default design contains **504 trials**. Each condition has three repeats
at 0/90/180/270 degrees and 50/100/200 Hz C0 packet rates. Target waveforms update
on a 50 Hz clock. Rates rotate across repeats within each matched condition;
condition blocks are shuffled reproducibly with seed 42.

| Family | Full trials | Input and purpose |
| --- | ---: | --- |
| `replay` | 72 | Original model_2500 source window 12.5–17.5 s and its sign-reversed copy; compare historical behavior and direction |
| `breakaway` | 72 | Both signs, 0.02 Nm steps every 0.20 s up to 0.36 Nm; identify near-zero motion onset versus phase/rate |
| `pulse` | 288 | Both signs at 0.06/0.12/0.24/0.36 Nm for 0.20 s; compare acceleration and subsequent zero-current deceleration |
| `reversal` | 72 | +0.24 then −0.24 Nm, and the opposite sequence, each leg 0.20 s; expose direction/history effects |

Every trial starts with 2 s zero current and ends with 3 s zero current. The latter
is **enabled C0 zero-current observation**, not disabled coasting or C1 zero-speed
control. Pulse amplitudes/durations are experimental settings, not motor ratings.
The whole planned waveform is rejected above 0.44 Nm or the measured-B0-derived
1.5 A command bound; no waveform is silently clipped. This intentionally covers
the previously problematic low-torque band rather than maximum motor torque.

For breakaway, two fresh A3 readings after the first nonzero command must exceed
0.01 rad from the **ramp-start position**, sampled after zero-current preparation.
Preparation displacement is retained in feedback, but cannot trigger breakaway.
The script then immediately requests zero current and observes for the configured
coast period, without continuing the ramp. The JSON records the last delivered
torque and detection time. This finite-speed, delayed criterion is **not an exact
static-friction coefficient or motor latency**. If no onset is detected, the
trial is recorded as censored; no threshold is fabricated. Read A1 around onset
to distinguish delivered current from commanded torque.

## Full suite, subsets and resume

```bash
# Full 504-trial design; pause after 12 trials to check the fixture and inspect data.
python3 deploy/experiments/drive_repeatability.py --execute \
  --session deploy/experiments/logs/rear_full --max-trials 12

# Continue the exact same design/reference/configuration.
python3 deploy/experiments/drive_repeatability.py --execute \
  --resume deploy/experiments/logs/rear_full --max-trials 12

# Preview saved progress.
python3 deploy/experiments/drive_repeatability.py \
  --resume deploy/experiments/logs/rear_full

# A focused breakaway sweep at finer starting-angle resolution.
python3 deploy/experiments/drive_repeatability.py --execute \
  --families breakaway --angles 0 45 90 135 180 225 270 315 \
  --rates 50 200 --repeats 3 --session deploy/experiments/logs/rear_breakaway
```

`--preset quick` gives 16 replay/breakaway trials with only one repeat per
condition; it is a functional check. `--preset screen` gives 96 trials with three
repeats, two angles, two rates and replay/breakaway/two pulse magnitudes.
Individual `--families`, `--angles`, `--rates`, `--repeats` and `--pulse-torques`
override presets. `--help` lists durations, ramp settings and thresholds.

Resume skips completed trials and retries failed/interrupted trials using new
attempt filenames, retaining every old attempt. Restore mark A before resuming;
the script checks it against the saved phase. Saved source hashes reject changed
scripts/helpers/target data. Resume accepts only mode and `--max-trials` overrides;
start a new session to change the design. Record power cycles, supply/fixture
changes and firmware changes as separate sessions with `--note`.

Executing processes exclusively lock the session directory. Resume revalidates
saved configuration/order and hashes of completed CSV/plan files and metadata;
changed or missing completed records stop resume. Configuration state is saved
atomically and synced before hardware work. The first initialized trial records
Kt/B3/B4/original-CD settings; changes in later trials stop before C0. Firmware
gains not queried by the existing replay transport still require unchanged settings
and an honest `--note`; the script does not verify every firmware register.

## Timing, feedback and protection

- Allowed send lateness defaults to **20 ms** (`--max-lateness-ms 20`).
- The 50 Hz target timeline is fixed to elapsed wall-clock time. A delayed packet
  does not stretch the waveform. Missed packet slots/target indices are recorded;
  the scheduler does not send a backlog burst, keeping at least 80% of one packet
  interval between sends. This differs from the older replay's stretched clock.
- `timing_valid=false` flags skipped slots or unintentionally missed target indices.
  Keep such trials for timing diagnosis; do not treat them as clean rate comparisons.
  Breakaway's intentional early ramp termination is recorded separately.
  It cannot conceal missing targets that occurred before detection. An excessive
  host stall across the experiment's end is also a failure, not normal completion.
- A3/A2/A1 queries target **100 Hz each** by default at every command rate; AE 5 Hz,
  B0/B4/CD 1 Hz. These are requested host rates, not hard real-time guarantees.
- Host current/speed/temperature trips default to **2 A /10 rad/s /60 C**. Feedback
  freshness is 100 ms and query lateness 20 ms. Exceeding a bound stops the session
  and retains partial data. These are experimental aborts, not silent motor limits.
- The script reads B3/B4 but does not change them, gains or encoder origins. B3 is
  not treated as a C0 current ceiling. Temporary CD settings are verified and
  restored after disable. Existing shorter watchdogs are not lengthened.
- CF confirms disabled mode; it does not guarantee the wheel has stopped. Software
  cannot guarantee cleanup after power loss/SIGKILL; independently verify the bike
  before touching it after abnormal termination.

## Recorded data and interpretation

The session directory contains `session.json` (ordered design, reference, source
hashes, arguments and attempt results), `summary.csv`, reference-capture CSVs and,
for every attempt, a feedback CSV, a JSON outcome and the complete planned CSV.
Feedback includes exact TX/RX bytes, send/kernel-RX times, native A3 counts,
unwrapped relative displacement, A2 speed, independent A1 and C0 current reports,
B0, voltage, temperature, mode/fault, query outcomes and timing diagnostics.

The summary compares displacement with velocity integration, speed extrema,
current-derived torque error, actual packet rate, timing validity and onset.
`A1_torque_error_rms_nm` includes zero-current phases;
`core_A1_torque_error_rms_nm` excludes preparation/coasting so a short pulse's
tracking error is not diluted by long zero-current intervals.
C0 reply populations differ by packet rate; compare independent A1
samples or first replies after target changes. A1/B0 torque is not an independent
shaft-torque measurement. RX/send timestamps are not motor acquisition/wire-TX
timestamps. Source columns remain the original recording; `polarity` and
`requested_torque_nm` describe the waveform actually tested.

At fixed rate, consistent angle dependence supports a resistance/phase effect.
At matched angle, a reproducible rate effect supports refresh/timing/firmware
investigation. Large scatter within one condition supports investigation of hidden
history or unresolved current dynamics. Compare voltage/temperature and actual
timing before making either inference. Use the results to identify a model and
measured DR/compensation ranges; no such cause is assumed by this script.

## Audit and verification (2026-10-28)

The audit corrected preparation displacement triggering false breakaway, incomplete
stationarity/status checks when capturing mark A, unreported end-of-run stalls,
missing-target accounting during early ramp termination, primary failures being
masked by cleanup/analysis failures, and unvalidated saved state on resume.
It also added session exclusion, completed-record integrity checks, measured-setting
consistency, malformed-A3 rejection, and separate core torque RMS. Source replay
rows are validated once and retained in memory, preventing later file edits from
changing an ongoing waveform. Original errors, cleanup/recording/analysis errors
and partial files are retained separately. The 20 ms command tolerance remains.

There are **28 suite regression tests**, including failure injection, startup-phase
shifts, over-deadline stalls, setting changes, damaged resume records, concurrent
session attempts and full pause/resume. The existing **56 replay regression tests**
also pass. All use virtual CAN; hardware/firmware behavior remains to be measured.

Virtual-CAN regression checks, without hardware:

```bash
python3 -B -m unittest discover -s deploy/experiments/tests -p 'test_drive_repeatability.py' -v
```
