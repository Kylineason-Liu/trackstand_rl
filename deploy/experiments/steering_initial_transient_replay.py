#!/usr/bin/env python3
"""Replay the recorded steering startup from source time zero, with feedback logging.

Default: model_2500 Isaac400 targets, source 0-5 s, then 1 s final-position hold.
No preparation ramp or pre-hold: the first recorded position is applied directly
from the verified physically straight B1 zero. References nominally update at 50 Hz;
--packet-rate 200 repeats each reference four times. MIT Kp=10/Kd=0.2 by default.
Packet rates/duration are nominal; delays stretch replay instead of catch-up bursts.
Only steering ID 2 is addressed. Disable the rear independently, support the
bike, lift the front wheel clear, and physically straighten before every run.
Preview is the default; no ROS, NumPy or package build is required on the Pi.
"""

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import signal
import sys

import steering_policy_replay as replay


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run steering and save CSV/JSON/plan")
    mode.add_argument("--preview", action="store_true", help="Inspect targets without CAN or files (default)")
    parser.add_argument("--duration", type=replay.finite, default=5., help="Recorded seconds from spawn (default 5)")
    parser.add_argument("--post-hold", type=replay.finite, default=1., help="Final hold seconds (default 1)")
    parser.add_argument("--packet-rate", type=int, choices=(50, 200), default=50)
    parser.add_argument("--targets", type=Path, help="Alternative source CSV, beginning at source time zero")
    parser.add_argument("--level", choices=("scaled", "conditioned"), default="scaled")
    parser.add_argument("--interface", help="SocketCAN interface (existing default can0)")
    parser.add_argument("--kp", type=replay.finite, help="MIT Kp (existing default 10 Nm/rad)")
    parser.add_argument("--kd", type=replay.finite, help="MIT Kd (existing default 0.2 Nm s/rad)")
    parser.add_argument("--current-abort-a", type=replay.finite, help="Existing measured-current trip (default 3 A)")
    parser.add_argument("--max-lateness-ms", type=replay.finite, help="Packet-deadline trip (default 20 ms)")
    parser.add_argument("--note", help="Record actual fixture, supply and firmware conditions")
    parser.add_argument("--output", type=Path, help="Exclusive output CSV path; otherwise timestamped under logs/")
    options = parser.parse_args(argv)
    arguments = ["--execute" if options.execute else "--preview", "--start", "0", "--ramp", "0", "--pre-hold", "0"]
    for key, value in vars(options).items():
        if key not in ("execute", "preview") and value is not None:
            arguments.extend(("--" + key.replace("_", "-"), str(value)))
    args = replay.parse_args(arguments)
    if options.output is None:
        name = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.level}_{args.packet_rate}hz_steering_initial_transient.csv"
        args.output = replay.HERE / "logs" / name
    # The backend records its own hash; preserve this entry point's provenance too.
    args.entry_point = Path(__file__).name
    args.entry_point_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return args


def main(argv=None):
    args = parse_args(argv)

    def stop(*_):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        plan, digest = replay.load_plan(args)
        core = [row for row in plan if row["phase"] == "core"]
        first = core[0]["target_rad"]
        print(f"{'EXECUTE' if args.execute else 'PREVIEW'} steering initial transient: {args.targets.name}")
        print(f"Source [0, {args.duration:g}) s; {len(core)} recorded targets at 50 Hz; "
              f"nominal MIT packets {args.packet_rate} Hz; nominal total {len(plan) * replay.DT:g} s")
        print(f"Lateness trip {args.max_lateness_ms:g} ms; "
              "delayed sends stretch replay, without skipped targets/bursts.")
        print(f"No preparation ramp/pre-hold. First target from straight zero: {first:.6f} rad; "
              f"target range [{min(row['target_rad'] for row in plan):.6f}, "
              f"{max(row['target_rad'] for row in plan):.6f}] rad")
        print(f"Initial nominal stationary Kp*error: {args.kp * first:.6f} Nm (not measured motor torque). "
              f"Measured-current trip remains {args.current_abort_a:g} A.")
        if args.execute:
            print("Front wheel airborne/free, steering straight, independent support, rear disabled, "
                  "all other writers stopped. B1 persistently resets steering zero once.", flush=True)
            replay.execute(args, plan, digest)
        else:
            print("No CAN opened; no files created. Add --execute for steering motion.")
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, OverflowError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
