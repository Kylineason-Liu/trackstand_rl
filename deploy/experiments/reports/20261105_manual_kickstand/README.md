# Manual kickstand replacement — 2026-11-05

Implemented `manual_kickstand` as a terminal replacement for the RC reader.
Same `/trackstand/kickstands_up` Bool authority, down at startup, held state
published at nominal 50 Hz. Complete `up` / `down` lines publish immediately;
partial input never blocks the heartbeat. Existing policy readiness/fault gates
and servo-command routing remain intact. No application subscriptions added.

Quit/EOF/input failure and handled signals request down before shutdown while
ROS is usable. Delivery is best effort, not a servo acknowledgement. Unexpected
publisher loss still invokes the existing motor timeout; it does not itself
command kickstands down. Pi-local heartbeat is not an SSH liveness guarantee.

V2 launch now accepts `use_rc:=false`; default remains true. Start the manual
node in a separate foreground terminal. See the guide below for usage.

[Usage](../../../docs/MANUAL_KICKSTAND.md)

Validation command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=deploy/source/trackstand_deploy python3 -B -m pytest -q -p no:cacheprovider deploy/source/trackstand_deploy/test
```

Result: **718 passed, 1 skipped in 4.45 s**. Includes 23 new manual tests:
real pipe input, held state, down/up ordering, EOF/error/partial lines, existing
v2 policy callback integration, cleanup/signals and launch selection. Packaging
checks include the new entry point. The existing kernel timestamp socket test
is skipped under sandbox restrictions. ROS interactions use test stand-ins;
no ROS 2 runtime, Pi build, live CAN or servo movement was tested here.

Baseline comparison confirms policy, motor, RC, servo, hardware configuration
and v1 launch files unchanged. Existing-file changes are listed in
`preservation.json`; new files are the manual node, tests, guide and this report.
