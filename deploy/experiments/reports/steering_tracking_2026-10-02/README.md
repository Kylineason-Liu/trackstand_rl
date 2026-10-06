# Steering tracking: are four torque updates enough?

## Answer

**The current controller does not consistently attain a new steering target within one 20 ms policy period. However, the recordings do not establish that 200 Hz is intrinsically insufficient, or that increasing the rate would fix it.**

Four is the nominal number of *new torque calculations per policy period*, not a four-iteration deadline for completing a position move. The motor continues moving and the controller continues correcting error across policy periods. Receiving another copy of the same target does not reset motion or discard the previous correction. Stable tracking requires an appropriate actuator response and delay, not simply a particular ratio of clocks.

## New measurements from the original bags

All times below are relative to the first enabled motor-control tick, using the node's monotonic clock. Position is the signed A3 feedback available to the controller; its CAN receipt age is recorded, but the motor's internal acquisition timestamp is unknown. First fractional crossing is **not** settling. Settling requires all subsequent recorded positions within the specified band and at least 250 ms of confirmation. These bands are analytical definitions, not certified hardware or balancing requirements.

| Stage C interval | Target / initial angle | What happens | Result |
| --- | --- | --- | --- |
| 2, positive step | +1.146° / +0.088° | First four ticks (0–16.93 ms) all report +0.088°; first changed position at 25.26 ms; first 90% displacement crossing at 47.72 ms (10th tick) | 410 successful steering sends over 2.055 s. Final 0.5 s reports +0.901°, leaving 0.245° target error. Never settles within ±0.1° of target; stays within a loose ±0.3° band from 47.72 ms. Settles around its *biased final position* to ±one encoder count from 110.68 ms. |
| 3, first negative step | −1.146° / +0.901° | First changed position at 25.82 ms; first 90% crossing at 46.91 ms, then overshoots and oscillates | The step is approximately twice the positive step's displacement, so this is not a matched direction comparison. Later watchdog disable separates this interval from the next. |
| 4, negative-target re-enable | −1.146° / −11.382° | Same target for 364 successful steering sends over 1.824 s; first changed position at 159.60 ms | Position spans −18.918° to +19.556° and does not settle within either ±0.1° or ±0.3°. The initial displacement and firmware enable history differ from interval 2. |

See [initial response/current plots](step_and_reenable_response.png), [long constant-target plot](constant_target_long_hold.png), and [full numeric results](tracking_results.json).

### A newly isolated current-delivery observation

During interval 4's first 150 ms, the loop sends **29 steering commands** requesting approximately **1.185–1.794 N·m**. The **29 current-feedback samples** received in that same window imply **0–0.00546 N·m**, median zero. The first received magnitude of at least 0.01 N·m occurs at **153.85 ms**, and the first changed encoder position available to the loop occurs at **159.60 ms**.

In contrast, interval 2's first feedback magnitude of at least 0.01 N·m arrives at **21.11 ms**. This is a measured startup/re-enable discrepancy, **not** an identified firmware enable time, pure CAN transit delay, current-loop time constant, or proof of mechanical stiction. Host send success is not proof that current is being applied. The current-derived estimate is not a calibrated shaft-torque measurement. Firmware mode/status and raw CAN are needed to explain the discrepancy. Unlike ordered command/reply correlations in the original report, these initial-window statistics use receipt timestamps and require no per-command reply pairing assumption.

### Four is not four fresh measurements

For the positive step, observed complete received-message groups span **3 ticks (15 groups), 4 (70), or 5 (13)**, excluding the two edge groups. The nominal four is therefore approximate, not a synchronized guarantee.

**31.0%** of that interval's control ticks reuse the previous A3 receipt timestamp. Distinct A3 receipt intervals have median **6.21 ms**, maximum **24.25 ms**; position receipt age reaches **26.23 ms**. The negative long hold similarly reuses A3 timestamps on **31.0%** of ticks. These are host receipt measurements, not firmware sampling frequencies. Faster commands alone do not ensure fresher feedback.

### What A and B can and cannot add

- A's unpowered encoder consistency check supports correctly related A2/A3 speed and angle units/signs outside its damaged interval. It cannot measure powered steering response.
- B uses model_41050 but is a dry run. Its contiguous enabled target increments have median absolute magnitude **0.01084°**, 95th percentile **0.04126°**, maximum **0.09327°**. Thus its target is usually changing only slightly every 20 ms, not repeatedly demanding a fresh one-degree step. This supported-state experiment does **not** establish live-balancing target bandwidth or tracking performance.
- C provides powered response at only **one configured low-level rate, 200 Hz**. There is no matched 100/200/400 Hz experiment, and no powered policy run in these bags.

## Interpretation

1. **Not demonstrated:** settling to each commanded angle in 20 ms. Even the favorable positive step takes roughly 48 ms merely to cross 90% of its initial-to-target displacement and retains steady error.
2. **Established:** a fixed-target steering/current-control problem persists beyond one policy period. Interval 4 gives the same target about 91 nominal policy periods, yet does not settle. Rapidly changing policy targets cannot be its sole cause.
3. **Unresolved:** whether 200 Hz is an adequate rate for a properly tuned version of this actuator-control system. Changing rate affects delay in samples, feedback reuse, scheduling, and closed-loop behavior. More ticks are not automatically more stable.
4. **Additional concern:** current delivery after a disable/re-enable can lag the host's requested torque by many policy periods. A powered test must distinguish enable transition from already-enabled tracking.
5. A position servo need not finish every move before its next target arrives. What matters for policy transfer is stable tracking with compatible gain, lag, limits and noise. Training uses an implicit simulator servo, whereas deployment's external sampled PD uses CAN feedback. A known 50 Hz policy interface does not establish compatible actuator dynamics.

## Evidence / verification

- Source: [target reception](../../source/trackstand_deploy/trackstand_deploy/motor_io.py#L331) stores the latest target without resetting actuator state; [continuous torque loop](../../source/trackstand_deploy/trackstand_deploy/motor_io.py#L445) recomputes from stored target and feedback. The slew state resets to measured position only on disabled ticks.
- Previous [A/B/C report](../rosbag_ABC_2026-10-02/README.md) documents 50 Hz policy / 200 Hz motor timing, the torque equations, feedback delays, counts and recording limitations.
- All three original MCAP SHA-256 values were rechecked against the decoder's saved hashes. No original recording was changed. No runtime, training, simulation, model or configuration file was edited for this analysis.
- Five numerical-method tests pass; eleven existing motor-control tests also pass. These validate analysis definitions/software arithmetic, not physical stability.

Reproduce from the repository root, using caches produced by the original report's decoder:

```bash
python3 deploy/reports/steering_tracking_2026-10-02/analyze_tracking.py --cache-dir /path/to/ABC_cache
python3 -m pytest -q deploy/reports/steering_tracking_2026-10-02/test_tracking_analysis.py
PYTHONPATH=deploy/source/trackstand_deploy python3 -m pytest -q deploy/source/trackstand_deploy/test/test_motor_control.py
```

## Next real-bike test

See [Stage D protocol](STAGE_D_PROTOCOL.md). Its disabled scheduling test is usable with existing nodes. Its powered rate/response comparisons require a guarded steering-only bench mode that does **not currently exist**; do not substitute the unmodified, known-oscillating production motor loop. This report does not silently implement a new motor-control mode or disable safety checks.
