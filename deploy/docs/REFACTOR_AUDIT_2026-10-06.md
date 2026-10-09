# Deployment Refactoring Audit

Date: 2026-10-06. Scope: all 53 pre-existing Python files under `deploy/`, including runtime nodes, experiments, tests, launch/setup code, hardware libraries and historical analysis programs; one regression-test file was added. Generated build/install directories are excluded.

Subsequent same-day change: at the user's request, `motor_mit_io.py` was separated
from `MotorIo` and now inherits directly from ROS `Node`. The counts and shared-code
descriptions below record the earlier refactor, not the later independent MIT file.
See [MIT node documentation](MOTOR_MIT_IO.md) for the separation and its verification.

## Result

Removed **493 physical lines from runtime Python**, without changing configured hardware, policy, control equations, CAN payloads, public ROS topics or intended operating sequence. The complete Python tree decreased from **14,184 to 13,832 lines**, including added regression coverage. These counts include comments and blank lines; they are not a runtime-performance measurement.

| Runtime file | Before | After | Main change |
| --- | ---: | ---: | --- |
| `motor_io.py` | 665 | 675 | Shared defaults/initialization, ROS interfaces, snapshot layout and telemetry hooks |
| `motor_io_position.py` | 888 | 465 | Inherits common motor I/O instead of copying it |
| `motor_mit_io.py` | 940 | 879 | Reuses common defaults, command-state initialization, ROS interfaces and telemetry emission |
| `crsf.py` | 174 | 157 | Direct 11-bit channel extraction; one shared debounce-reset branch |
| `imu_publisher.py` | 110 | 107 | Shared component assignment; unchanged quaternion order and angular-rate units |
| `kickstand_driver.py` | 144 | 145 | One pose-command helper; both servos still enabled before either moves |

The C2 class still owns firmware-limit configuration, absolute position encoding, current queries, unwrapped-position send guards and its additional telemetry. Its channel dataclass retains the original field/constructor order. Shared snapshots retain all original tuple indices and acquire the same channel lock.

MIT still owns its single CAN receiver, kernel receive timestamps, startup/preflight, 50 Hz steering schedule, 200 Hz rear/safety schedule, F1 handling, acknowledged-disable state machine and hard-fault rules. These methods were not replaced with the older C0/C2 implementation. Each variant retains its versioned telemetry field order/label and variant-specific event/status data.

## Preserved boundaries

- Configurations and bundled models are byte-identical to the pre-edit snapshot. Motor IDs/signs, IMU correction, RC mappings, servo targets, watchdogs, gains and limits are unchanged.
- Policy inference, observation construction, action scaling and kickstand state-machine logic were reviewed but not changed.
- Vendor IMU/servo libraries and historical report programs/artifacts are byte-identical. Unused vendor imports are not removed merely to shorten a third-party copy.
- Independent experimental controllers/collectors keep their standalone interfaces and differing synchronization/safety contracts. Similar-looking CAN code is not automatically interchangeable: native MIT units, C2 startup zero, passive observers, query matching and feedback clocks differ.
- Default launch still starts `motor_io`. Use exactly one of `motor_io`, `motor_io_position` and `motor_mit_io` as the motor-command writer.

Three test imports still referenced the former `deploy/scripts/` locations; they now load the actual files in `deploy/experiments/`. The integrated MIT sine script's help documentation and tests were corrected to its existing **1-second default period**. Its actual CLI defaults, 50 Hz command rate and behavior were not changed.

## Verification

| Check | Result |
| --- | --- |
| Deployment pytest suite | **352 passed, 1 skipped** |
| Experiment unittest suite | **117 passed** |
| Historical tracking-analysis tests | **5 passed** |
| Independent old/new comparison | **3,000 matching control ticks**: 1,000 per motor backend, seeds 0–9 |
| Syntax compilation | All **54** current Python files compile |
| Package build | Isolated ROS 2 Humble `colcon build --symlink-install` succeeded |
| Whitespace check | `git diff --check` passed |
| Static checks | No Pyflakes findings outside unchanged vendor IMU files |

The independent comparison loads the actual pre-edit classes and current classes with identical synthetic state and clocks. It compares CAN frames, control/event/fault/state/status publications and internal latch/disable transitions, including stale inputs, clipped targets, overspeed, dry-run, injected CAN failures and MIT CF confirmations. All 18 common parameter defaults and 20 MIT defaults also match their original values and declaration order. These checks supplement—not replace—the existing focused protocol/startup/timeout tests. New regression checks cover shared ROS schemas/QoS, CRSF random channel payloads, servo ordering, IMU fields/freshness and both legacy encoder pollers' startup-zero handling.

Re-run the permanent tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=deploy/source/trackstand_deploy \
  python3 -m pytest -q -p no:cacheprovider deploy/source/trackstand_deploy/test
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s deploy/experiments/tests
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  deploy/experiments/reports/steering_tracking_2026-10-02/test_tracking_analysis.py
```

Local comparison/build artifacts and the unedited baseline are under `/tmp/trackstand_deploy_refactor.ILIycU/`; these temporary files are not a permanent repository dependency. The sole skipped test needs local socket/kernel timestamp operations denied by this sandbox. Matplotlib reports an unrelated unavailable 3-D projection; the tested analysis is 2-D. The vendor Pyflakes findings are an unused `threading` import and two intentional package re-exports.

## Existing limitations intentionally not changed

Fixing these would change behavior, exceeding an exact-functionality refactor. They remain separate follow-up work; passing these tests does not certify unsupported balancing safety.

| Priority / area | Concrete evidence and consequence |
| --- | --- |
| High: MIT torque cap | `motor_mit_io.py` sends firmware PD gains/position, not a host-clamped steering torque. `mit_protocol.py` F0 ranges encode values; they do not establish a total ±5.6 N·m PD torque cap. Firmware current limiting still needs verification. |
| High: legacy disable/fault feedback | `motor_io.py::_disable_outputs` retries failed socket writes but does not await CF mode confirmation. Its receiver accepts A1/C0 torque feedback, while A3/A2/B0 pollers do not implement MIT's AE/CF hardware-fault monitoring. C2 inherits these existing limitations. A successful send is not confirmed disable. |
| Medium: legacy latch clearing | C0/C2 control ticks test `motor_fault_clear_allowed` after detecting overspeed. With enable off and steering within ±5°, an overspeed latch can clear that tick without requiring fresh feedback or speed recovery. The disabled gate remains off; MIT's stricter clearing logic is intentionally separate. |
| Medium: IMU freshness/coherence | `imu_serial.py::_parse_data` updates one freshness timestamp after any vendor-parsed frame, rather than independent quaternion/gyro timestamps. A fresh unrelated report can coexist with older observation components. Vendor parsing/receive-thread error recovery is also unchanged. |
| Medium: servo confirmation | `kickstand_driver.py` does not require torque-enable acknowledgements before moving. `sms_sts.py::_ack` validates packet checksum/ID/length but does not reject a nonzero status error byte. Commands do not prove physical support. |
| Low: legacy ROS shutdown | Several original node `main()` functions still call `rclpy.shutdown()` after signal-driven shutdown; C2/MIT have their existing different shutdown handling. This can produce the previously observed double-shutdown exception. |
| Offline tools: relocated report paths | Historical `plot_command_torque.py` resolves its default repository to `deploy/`, as do `analyze_bags.py` and `analyze_tracking.py`; the real rosbags/models are under the repository root. Use explicit `--bag`/`--repository` where available. The ABC analyzer's embedded replay path needs a separate correction before rerunning it. Historical sources/outputs were preserved in this refactor. |

## Raspberry Pi handoff

Rebuild in `deploy/` with `colcon build --symlink-install`, then source `install/setup.bash`. CLI entry points and parameter flags are unchanged. Start with the bike secured and kickstands down, run only one motor I/O variant, and verify dry-run has no actuation. Repeat the supported command/feedback and RC-down disable checks before an unsupported test. Record actual motor/policy intervals, telemetry sequence gaps and MIT disable confirmations on the Pi; desktop stub tests and a package build cannot establish real hardware timing or response.
