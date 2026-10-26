# Steering replay initialization correction

Implemented 2026-10-26. This corrects the clipped initial F1 reference found in
the [four hardware recordings](../20261026_policy_replay_hardware/README.md).
No real CAN interface or motor was operated during implementation or validation.

## Corrected startup

`steering_policy_replay.py --execute` now:

1. Confirms CF disabled/fault-free; reads existing F0/B0/B3/CD. Rejects an
   impossible target range before changing the origin.
2. Records the old A3 count and checks A2/A1/AE for stationary, disabled setup.
3. Sends **B1 exactly once** and requires its three-byte acknowledgement.
   The acknowledgement carries a mechanical offset; it is not assumed zero.
4. Requires fresh A3 near the new native zero, then fresh inactive/fault-free F1
   with an unsaturated position consistent with A3. The tolerance derives from
   two encoder counts plus one F1 position encoding step. F1 speed must indicate
   stationary steering within the existing helper's quantization-aware tolerance.
5. Brackets F1 with another A3 read, rechecks A2/A1/AE and unchanged F0, then
   validates every absolute MIT target before starting the replay clock.

The verified post-reset A3/F1 samples define the fixed per-run references. The
command remains `F1_zero + encoder_sign * target_rad`, with target velocity and
feedforward torque zero. Initial F1 endpoints are rejected rather than accepted
as valid origins. Missing B1 replies are not retried; cleanup still attempts CF
and confirms disabled. Setup flags, old count, acknowledgement and final origins
are saved under JSON `steering_origin_setup`, including failed setup.

**B1 stores a persistent steering encoder origin.** Start physically straight
and stationary for every executed run. The origin is not restored on exit;
failed setup may also leave the new origin stored. This matches the existing
working MIT sine/deployment startup method. F0 ranges, gains/settings, rear
commands, targets, packet rates, timing abort and current abort are unchanged.
Preview opens no CAN and sends no B1. No measured-angle travel abort was restored.

## Evidence and checks

Protocol Rev.3.09b0, PDF pages labelled **8/26** and **20–21/26**, documents B1's
persistent origin, three-byte reply and possible ~35 ms second-encoder delay;
F0's persistent encoding ranges; F1's 16-bit position and state fields. The
already working `steering_sine_mit_io.py` and normal `motor_mit_io.py` use the
same B1/A3/F1 origin validation pattern. Neither was modified by this fix.

**86 tests passed:** 24 steering, 20 rear, 29 shared-helper and 13 historical
analysis tests. New cases replay the actual problematic 8282/8380-count starts
with F1 code 65535, at 50/200 Hz, including the documented 35 ms B1 delay. They
verify one reset, unchanged F0, near-zero references and mock A3 tracking within
encoder/encoding precision. Negative cases cover missing/non-applied reset,
bad A3, both F1 endpoints, inconsistent F1, moving setup, active/faulted F1 and
movement while capturing references. No MIT packet is sent on those failures.
Unexpected B1 acknowledgements during motion also stop replay.

Fresh-copy `python3 -S -B --preview` checks pass for both replay entry points.
AST, Pyflakes, 120-column and whitespace checks pass. Only `startup`, `execute`
and the execution message changed among steering functions; the receive-length
table additionally allows B1 replies. Rear runtime and shared transport source
remain byte-identical. **176 protected existing files remain unchanged**, including
all 12 raw replay files, original plot/statistic artifacts, production code,
models, both progress ledgers and Git index.

The hardware report now checks the [archived runtime sources](../20261026_policy_replay_hardware/runtime_snapshot/)
that produced its historical measurements. Its motor statistics and figures
were not regenerated or changed. This preserves provenance without comparing
old recording hashes against updated runtime code.

See [verification.json](verification.json) and the four accompanying test logs.
These are software/virtual-CAN checks. Repeat the physical test to verify the
corrected initial reference and tracking; the separate 3 ms deadline issue is
not changed by this correction.

## Raspberry Pi

Copy the updated `deploy/experiments/steering_policy_replay.py` to the Pi; its
existing adjacent helper and target files are still required. No ROS build is
needed. With other writers stopped, the bike supported, rear disabled, front
wheel airborne/free and steering physically straight:

```bash
cd ~/trackstand_rl/deploy/experiments
python3 steering_policy_replay.py --preview
python3 steering_policy_replay.py --execute --level scaled --packet-rate 50
```

Check `Steering origin verified after B1: A3=... counts, F1=... rad` and the saved
`steering_origin_setup.verified` flag. Both references should be near zero;
F1 need not be exactly zero because its symmetric encoding is quantized.
Physically straighten again with the motor disabled before a separate 200 Hz run:

```bash
python3 steering_policy_replay.py --execute --level scaled --packet-rate 200
```
