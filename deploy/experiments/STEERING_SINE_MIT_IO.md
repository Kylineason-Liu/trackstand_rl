# Integrated MIT steering sine experiment

`steering_sine_mit_io.py` controls **steering only**, at a fixed nominal
**50 Hz**, and automatically records feedback to CSV. Python stdlib and Linux
SocketCAN are sufficient; no ROS node, package build or separate recorder is
needed. The original MIT sender/recorder remain unchanged.

## Run on the bike

Support the bicycle securely, start with steering physically centred, and leave
it free to move. B1 treats the current angle as zero; it cannot verify actual
straight-ahead alignment. Stop all other
selected-motor CAN writers, including `motor_io`, `motor_io_position`, sine
senders and feedback collectors. Keep an independent physical means of stopping
the motor: software cannot guarantee disable after a CAN or power failure.

From `deploy/experiments/`:

```bash
# Preview five seconds of targets; opens neither CAN nor CSV.
python3 steering_sine_mit_io.py --preview --amplitude 0.1 --period 1 --duration 5

# Initial supported test: +/-0.1 joint rad, 1 Hz sine, 10 seconds.
python3 steering_sine_mit_io.py --amplitude 0.1 --period 1 --duration 10 \
  --output logs/mit_50hz_1hz01rad.csv

# F1-only motion telemetry; omit continuous A3/A2/A1 reads.
python3 steering_sine_mit_io.py --feedback mit --amplitude 0.1 --period 1 --duration 10

# Additionally record drive feedback, without controlling/resetting/disabling it.
python3 steering_sine_mit_io.py --include-drive --amplitude 0.1 --period 1 --duration 10

python3 steering_sine_mit_io.py --help
```

The output's parent directory must exist. An existing output is refused **before
CAN opens**. Without `--output`, a timestamped CSV is created in the current
directory. Execution is the default; `--execute` is optional. Duration measures
the motion phase, excluding startup/shutdown. `--duration 0` runs until Ctrl-C.

## Settings and rates

| Setting | Default / meaning |
| --- | --- |
| MIT command rate | Fixed 50 Hz; not configurable through a rate flag |
| `--interface`, `--motor-id` | `can0`, steering address `2` |
| `--amplitude`, `--period` | Peak **1 joint rad**, period **10 s**; frequency = 1/period |
| `--sign`, `--gear-ratio` | `-1`, `1`; positive joint sine maps to negative native motor position |
| `--torque-sign` | `-1`, retained from the existing recorder; independent of encoder sign |
| Kp / Kd | Requested `10 Nm/motor-rad`, `0.2 Nm s/motor-rad` |
| Target velocity / feedforward torque | Requested zero; encoded residuals are recorded |
| `--feedback` | `extended`: automatic F1 plus A3/A2/A1 reads; `mit`: automatic F1 only during motion |
| `--feedback-rate` | Nominal **200 A3/A2/A1 cycles/s per selected motor**, extended mode only |
| `--query-timeout` | `0.002 s`, asynchronous motion-time read deadline |
| `--feedback-timeout` | `0.25 s` without steering F1 causes motion to stop |
| `--include-drive` | Off; requires extended mode. Drive defaults: ID 1, gear 1, encoder sign +1, torque sign -1 |

The sine frequency must be below 25 Hz at this 50 Hz sampling rate. Native
amplitude plus the verified zero must fit the +/-3-rad encoding range. This is
**not** a verified mechanical steering limit or current/torque safety limit.

MIT's PD/current control happens in firmware as shown in the protocol diagram;
its internal update frequency is not documented. Automatic F1 replies are
expected from each MIT command, nominally 50/s, not independent 200-Hz F1
sampling. Extended queries obtain additional measurements between setpoints.
Real achieved command/read rates depend on the Pi, CAN and firmware. Auxiliary
timeouts do not block command scheduling; missed deadlines cause no catch-up
command burst. Inspect actual CSV intervals rather than assuming exact rates.

## Startup, recording and shutdown

One CAN-owning thread runs verified CF/disabled acknowledgement -> B1/origin
acknowledgement -> F0 read -> F0 write -> matching reply -> independent read-back
-> A3 stationary-origin check -> F1 inactive/fault-free/stationary-zero check.
F0 Pos_Max becomes **3 native motor rad**; actual velocity/torque bytes are
preserved. B1/F0 modify persistent settings. B2/B3 limits and firmware PI gains
are **not** changed or asserted safe by this script.

The verified F1 reply supplies the native absolute sine centre; its sign and
gear ratio are applied to the sine offset, not to that native centre. The
verified A3 reply supplies the independent encoder zero. B0 is read in extended
mode; invalid/missing B0 leaves current-derived torque NaN and is retried.

A bounded CSV-writer thread receives scalar dictionaries, never CAN sockets or
motor state. Recording failure/backlog, reported RX overflow/truncation,
observed other selected-motor writers, steering fault/unexpected MIT exit, origin/range
changes or missing steering F1 abort the experiment. Once CAN setup completes,
normal completion, Ctrl-C/SIGTERM and failures attempt steering CF. Shutdown
records/drains replies for up to 0.25 s; missing disabled-mode acknowledgement
is an explicit error. An attempted CF is not proof of physical disable. A
logger failure cannot prevent the physical CF send attempt, but can prevent
its receipt from being saved. No command is sent to enable or disable drive.

## CSV interpretation

| `record_type` | Contents |
| --- | --- |
| `mit_target` | Requested relative sine position; actual encoded position/velocity/gains/feedforward torque; payload; host send timestamp, interval, lateness and skipped ticks |
| `mit_feedback` | Native/relative F1 position, velocity and torque, raw codes, MIT-active/fault bits; one host receipt timestamp |
| `feedback` | **One** A3, A2 or A1 reply per row: encoder position, speed, or Q-axis current/current-derived torque; not a simultaneous compound sample |
| `can_tx`, `can_rx` | Other transmitted/received raw selected-motor frames, including B1/F0/B0 and malformed/unclassified replies |
| `motor_status` | AE/CF voltage, bus current, temperature, mode and fault code |
| `query_result` | Actual read request and `received`, `late_reply`, `timed_out` or `cancelled` status; host round-trip time where available |
| `event` | Verified reference, stop reason and shutdown acknowledgement result |

Units match the old recorder: A3 int32 multi-turn counts at bytes `[3:7]`,
16,384/revolution; A2 int32 `[1:5]` x0.01 rpm -> rad/s; A1 int32 `[1:5]`
x0.001 A; B0 float32 `[2:6]` Nm/A. Position/velocity use encoder sign and
divide by gear ratio. Torque uses the **separate torque sign**, remains motor
torque (not gearbox-output torque), and is firmware/current-derived rather
than independently measured shaft torque. A3 `position_unwrapped_rad` retains
multi-turn motion; `position_rad` wraps to [-pi, pi). MIT positions do not wrap.

Rows are sparse: blank fields do not apply; NaN represents unavailable numeric
conversion. Missing replies never create invented or cached feedback samples.
A3 and F1 references are independent, even though both are checked near B1 zero.
Times are host send/receipt times, not CAN wire or motor acquisition times.
F1 has no sequence number: never assume a row uniquely acknowledges a particular
MIT command. Delayed same-opcode read replies cannot be uniquely correlated
either. This new sparse schema is not identical to the old compound collector;
existing C2 plotting tools do not automatically support it.

## Offline validation

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s deploy/experiments/tests -p 'test_*mit*.py'
python3 -m pyflakes deploy/experiments/steering_sine_mit_io.py \
  deploy/experiments/tests/test_steering_sine_mit_io.py
```

Run these from the repository root. Tests use simulated CAN/virtual clocks and
temporary CSVs, including 20-ms commands with missing auxiliary responses or
30-ms F1 latency, exact protocol packing, startup rejection, reference handling,
read-only drive, writer failure, overflow, watchdog and CF cleanup. These do not
prove actual Pi scheduling, firmware PD frequency, torque calibration or motor
performance. No hardware commands were used during implementation validation.
