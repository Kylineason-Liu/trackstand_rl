# CAN/ROS latency A/B report: motors disabled, 2026-11-06

**Result: with motors disabled and deployment-level CAN load, steady-state latency is low and `rosbag` recording has only a small effect.** The CAN round trip is 0.42 ms at p50 and about 1.3 ms at p99, unchanged by recording. There were no socket drops, CAN error frames or reply timeouts.

Three things remain open: a one-off stall of about 340 ms at startup, irregular bursts of late queries, and the Pi running at up to 83.7 °C. This test does not cover actuation or policy inference.

## Test setup

| Item | Value |
|---|---|
| Command | `bash deploy/experiments/run_latency_ab.sh 120` (ROS Jazzy sourced) |
| Commit | `b0141b7`, plus the `run_latency_ab.sh` fix described below |
| Trials | 120 s each: minimal, full, full, minimal (ABBA order) |
| Only difference between conditions | "Full" runs `ros2 bag record`; "minimal" does not |
| CAN | `can0`, 1 Mbit/s, MCP251x on SPI, `txqueuelen` 10 |
| CAN load | Eight `A3/A2/A1/AE` queries at 100 batches/s, `F1` reads at 270.9 Hz per motor, `F0/CD` at 1 Hz |
| ROS load | Synthetic 200 Hz and 100 Hz publishers, plus a 50 Hz consumer with no inference |
| Not running | Motor and policy nodes, kickstand scripts, IMU publisher |

Both motors were disabled throughout. Only `CF` disable and one-byte read requests were sent.

Raw logs and analyzer output are in `deploy/experiments/logs/20261106_151814_latency_ab/` on the Pi. That directory is 309 MB and gitignored, so it is not in this repository.

## Results

All values are in milliseconds unless stated. Each cell shows the two trials of that condition: trial 1 / trial 4 for minimal, trial 2 / trial 3 for full. Metrics are defined in the next section.

| Metric | Stat | Minimal (1 / 4) | Full (2 / 3) |
|---|---|---|---|
| Query to reply | p50 | 0.417 / 0.416 | 0.420 / 0.424 |
| | p99 | 1.36 / 1.33 | 1.30 / 1.36 |
| | max | 11.0 / 9.5 | 7.0 / 4.4 |
| Kernel receive backlog | p50 | 0.030 / 0.030 | 0.030 / 0.031 |
| | p99 | 1.26 / 1.22 | 1.45 / 1.53 |
| | max | 23.7 / 24.1 | **338** / 24.9 |
| Dequeue to processed | p99 | 0.062 / 0.062 | 0.065 / 0.066 |
| Send call duration | p99 | 0.78 / 0.76 | 0.87 / 1.02 |
| | max | 9.5 / 9.8 | **333** / 9.8 |
| Query dispatch lateness | p50 | 1.06 / 1.07 | 1.13 / 1.16 |
| | p99 | 9.6 / 8.2 | 10.1 / 10.2 |
| | max | 148 / 86 | **380** / 82 |
| ROS publish to callback | p50 | 0.47 / 0.47 | 0.55 / 0.53 |
| | p99 | 1.66 / 1.67 | 2.04 / 2.14 |
| | max | 8.5 / 9.1 | 11.8 / 8.0 |
| Data age at 50 Hz consumer* | p50 | 13.2 / 16.1 | 16.5 / 17.2 |
| | p99 | 26.1 / 24.6 | 27.5 / 25.9 |
| | max | 109 / 82 | 75 / 58 |

\* Recomputed from `ros.csv` using only ticks from 2 s after the probe started until it stopped. See "Reading the analyzer output".

| Bus and health check | Trial 1 | Trial 2 | Trial 3 | Trial 4 |
|---|---|---|---|---|
| Achieved vs planned airtime | 99.3% | 99.0% | 99.3% | 99.5% |
| Estimated bit/s | 205,780 | 205,204 | 205,748 | 206,177 |
| Frames/s (TX + RX) | 2,665 | 2,658 | 2,665 | 2,670 |
| Socket drops | 0 | 0 | 0 | 0 |
| CAN error frames | 0 | 0 | 0 | 0 |
| Reply timeouts | 0 | 0 | 0 | 0 |
| Queries over 2 ms budget | 151 | 99 | 131 | 140 |
| Invalid clock samples | 0 | 0 | 0 | 0 |
| Pi temperature, start to peak (°C) | 72.2 to 79.3 | 77.7 to 82.1 | 80.4 to 83.7 | 80.4 to 83.7 |

## What each metric means

**Reading the statistics.** p50 is the typical value: half of samples are at or below it. p95 and p99 are the values that 95% and 99% of samples stay under. max is the single worst sample. At about 160,000 frames per trial, p99 still leaves about 1,600 samples above it, and it hides rare stalls, so check max as well.

### CAN timing

| Metric | Measured from and to | What it tells you | What it does not tell you |
|---|---|---|---|
| Query to reply (`query_to_kernel_reply_ms_trusted`) | Start of the send call to the kernel timestamp on the matching reply | The full round trip through host TX, adapter, wire, motor and RX | Firmware time alone. It is an aggregate of all those stages. |
| Kernel receive backlog (`receive_queue_ms`) | Kernel receive timestamp to Python reading the frame | How long frames wait for the process, which shows host scheduling delay | Delay before the kernel timestamp, such as time in the adapter or SPI transfer |
| Dequeue to processed (`dequeue_to_processed_ms`) | Reading the frame to finishing its parsing | The probe's own per-frame processing cost | The production node's cost, which runs different code |
| Send call duration (`send_syscall_ms`) | Start to end of the `send()` call | How long the host socket call blocks | Wire transmit time. The call returns once the frame is handed to the kernel. |
| Query dispatch lateness (`query_dispatch_lateness_ms`) | Scheduled time to the start of the send call | How far sending falls behind schedule | Pure host delay. All items in a batch share one scheduled time and go out one at a time, so later items are late by design. `A3` p50 is about 1 ms and `AE` p50 is about 5.5 ms for this reason. |

Round-trip caveat: CAN replies carry no transaction ID. Once a query overlaps or times out, the analyzer stops trusting that motor and opcode for the rest of the trial. `F1` on both motors became untrusted in every trial, because it runs at 271 Hz, so round-trip samples are mainly from the other opcodes. Trusted samples numbered 96,258 to 120,629 per trial.

### ROS timing

| Metric | Measured from and to | What it tells you |
|---|---|---|
| ROS publish to callback (`publish_to_callback_ms`) | Just before publish to the subscriber callback starting | The cost of serialization, DDS and the executor on one Pi |
| Callback source age (`callback_source_age_ms`) | Oldest of the four latest state reply timestamps to the callback | How old the CAN data is when it reaches a subscriber. This is in the analyzer output but not in the table above: p50 was 6.6–9.2 ms. |
| Callback to consumer (`callback_to_50hz_consumer_ms`) | Latest callback to the next 50 Hz tick | Wait for the consumer's timer. Up to about 20 ms is normal at 50 Hz. |
| Data age at 50 Hz consumer (`consumer_source_age_ms`) | Oldest of the four latest state reply timestamps to the 50 Hz tick | The end-to-end age of data a policy would see, excluding inference. Its p50 depends on the timer phase between publisher and consumer, so small p50 differences between trials are not meaningful. |

### Bus and health checks

| Metric | Meaning | Good value |
|---|---|---|
| Achieved vs planned airtime | Estimated bus bit rate divided by the planned 207,200 bit/s. The estimate is 47 + 8×DLC bits per frame, excluding bit stuffing and retries. | 90–110%. Below that, the run does not prove anything about deployment load. |
| Frames/s | Sent plus received frames per second. The plan is about 2,684, against about 2,400 in real deployment. | Near plan |
| Socket drops (`rx_dropped`) | Frames lost because the socket's receive buffer overflowed | 0 |
| CAN error frames | Bus or controller errors delivered to the socket | 0 |
| Reply timeouts | Queries with no reply within 50 ms | 0 |
| Queries over 2 ms budget | A query's reply did not arrive within the 2 ms slot before the next query. This was 0.10–0.16% of the roughly 95,000 batch queries per trial. | Low. It is logged, not fatal. |
| Clock samples | Timestamps whose wall-to-monotonic conversion was uncertain by more than 1 ms, or invalid | 0 |

## Findings

### `rosbag` has a small effect
The round trip is unchanged. Recording adds about 0.2–0.3 ms to p99 receive backlog, 0.4 ms to p99 ROS callback and about 1 ms to p99 dispatch lateness. Consumer data age at p99 rises by about 1.3 ms. The direction held in both pairs for backlog and ROS callback, but two pairs is a small sample.

Heat does not explain the difference: trial 4 was among the hottest and matched trial 1, which was the coolest.

### Startup stall of about 340 ms
A single stall of 330–380 ms hit about 0.5–0.6 s after the probe started, in the first full trial. It appeared in trial 2 of this run and also in trial 2 of the earlier hung run, but not in trial 3 or either minimal trial. In this run, one `send()` call blocked for 333 ms.

After the first 2 s, worst backlog in all trials was 13–26 ms. The timing suggests the recorder connecting or starting up, but this is not proven. If the production stack starts recording at the same point, it could be exposed to the same stall.

### Irregular bursts of late queries
After startup, each trial had 4–5 bursts in which a whole batch went out 60–148 ms late. Times are irregular, and the bursts appear in both conditions. The `F1` stream, on its own schedule, stayed within 18–28 ms during the same trials.

The cause is unknown. Because `F1` stayed closer to schedule, part of this may come from the probe's sequential query queue rather than from a host stall. This needs tracing before treating it as a production risk.

### Temperature
The Pi climbed from 72 °C to 83.7 °C over about 9 minutes and was still at 83.7 °C after the run, without policy inference. That is close to its throttling range. Throttle flags could not be read because `vcgencmd` cannot open `/dev/vcio`, so it is unknown whether throttling occurred.

## Reading the analyzer output
- **Teardown inflates ROS maximums:** the observer keeps ticking briefly after the probe stops, which adds samples of up to 570 ms. `analysis.json` includes them. With teardown excluded, callback-to-consumer maximum is 15–20 ms and consumer data age p99 is as shown in the table above. The raw `analysis.json` p99 for consumer age reads 34–36 ms in the full trials, which is misleading.
- **Dispatch lateness is not a host-delay figure:** see the note in the CAN timing table.

## Wrapper fix
`run_latency_ab.sh` hung at the end of the first full trial on its first live run. Bash starts `&` jobs with SIGINT ignored and `ros2 bag record` kept that setting, so stopping it with `kill -INT` did nothing and `wait` never returned. Cleanup now uses `kill -TERM`, and all four trials then completed with both bags closed properly (41 MB each).

Two other log directories from that session should not be used for conclusions:
- `20261106_145646_latency_ab`: the hung run. Trials 1 and 2 have valid CAN data, but trial 2's ROS consumer data includes the hang.
- `20261106_150729_latency_ab`: a 25.8 s partial run while the hung run's recorder was still active.

## Limits
- **Motors were disabled.** This says nothing about firmware behaviour while actuating. `F1` reads replace MIT commands, which matches estimated airtime but has different payloads and firmware paths.
- **No policy inference ran**, and the ROS load is synthetic, so production CPU load will be higher.
- **Bit stuffing and retries are not measured.** Actual bus load is higher than the estimate, and external CAN instrumentation would be needed to measure it.
- **The kernel timestamp is taken after the MCP251x SPI transfer**, so that delay falls inside the round-trip figure and outside the backlog figure.

## Next steps
1. Run the passive capture during a supported replay (`--mode passive`) to measure timing under real actuation.
2. Add active cooling, then repeat the A/B so temperature stays stable across trials.
3. Trace the batch lateness bursts in `can_latency_probe.py` to separate probe queueing from host stalls.
4. Check whether the startup stall is tied to recorder startup, for example by delaying probe start until the recorder has subscribed.
5. Exclude post-run ticks from the ROS metrics in `analyze_latency.py`.
