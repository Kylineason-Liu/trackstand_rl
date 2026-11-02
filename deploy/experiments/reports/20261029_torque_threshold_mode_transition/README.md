# Rear torque-threshold abort: CF mode transition

Analyzed `20261029_172218_464098_drive_torque_threshold` (the uploaded 17:22:18 run, not the earlier 17:18:40 terminal
example). Its four runtime script hashes match the pre-fix local sources saved
under `/tmp/threshold_mode_transition_20261029_gtzh78z6/`.

The first +0.12 Nm trial completed a 5.000084 s hold. The
script then aborted during its first rest, before trial 2. The immediate CF
reply still reported current-control mode 2; the decoder required disabled mode
0 immediately. This was an overly strict transition check in the script.

| CSV line | Event | Time from first rest CF send (ms) | Mode | Fault |
| --- | --- | ---: | ---: | ---: |
| 5086 | First rest CF send | 0.000 | — | — |
| 5087 | CF reply; old code aborts | 0.698051 | 2 | 0 |
| 5091 | Disabled CF status during cleanup | 1.774987 | 0 | 0 |
| 5092 | Disabled AE status during cleanup | 1.986226 | 0 | 0 |

[Full event timeline](event_timeline.csv) and [raw-data hashes/diagnosis](diagnosis.json).
Additional CF commands were sent during cleanup, including an unlogged direct
send in the shared shutdown helper. Therefore the table does not isolate the
first command's physical disable delay, firmware execution time, or which CF
produced the first mode-0 status. It does prove that the immediate mode-2 reply
triggered the abort and subsequent cleanup obtained disabled status.

All recorded fault codes are zero. Fresh A1 peak absolute current was 0.689 A,
A2 peak speed 29.543537 rad/s and temperature 30–32°C, below
this run's limits. All 1,522 logged query outcomes are received. Final metadata
confirms disable and watchdog restoration. This error is not an angle threshold,
current/speed/temperature trip or query-timeout report.

One trial's last measured displacement is 115.550171 rad and
mean A1-current-derived torque 0.117586 Nm. Starting speed
was -0.015708 rad/s. These measurements do not determine
a repeatable breakaway coefficient or provide the missing other 49 trials.

## Fix

`drive_torque_threshold.py` now tolerates the previous mode 2 only while awaiting
fresh disabled confirmation, within the existing 500 ms bound. CF retries occur
at nominal 20 ms intervals until a fresh current-phase CF/AE reports mode 0.
Other modes still fail; active mode after confirmation still fails. No next
trial starts if disable remains unconfirmed. Two-second rest is still measured
from the first CF send; confirmation time is not added. The console now says
“CF sent, awaiting disabled confirmation” instead of prematurely “disabled”.
Torque levels, holds, rates, signs, references and other abort limits are unchanged.

Eighteen software/virtual-CAN tests pass, including the exact logged CF payload,
old-mode acknowledgement followed by AE confirmation, retries without extending
rest, persistent failure at the deadline, and stale-status rejection. Syntax,
Pyflakes, 120-column and whitespace checks pass. Seven protected original log/
helper files are unchanged. No hardware or simulator was operated for diagnosis
or verification; the corrected script still needs a physical rerun.
