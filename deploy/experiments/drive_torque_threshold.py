#!/usr/bin/env python3
"""Rear C0 constant-torque trials, with integrated raw CAN/feedback recording.

Default: +/-0.12, 0.13, 0.14, 0.15, 0.16 Nm, 5 seconds each, 5 repetitions
(50 trials). Between trials send CF and coast disabled for 2 seconds. No active
braking or automatic waiting for zero speed; a coasting start is not breakaway
from rest. Rear wheel airborne/free, frame supported, steering independently
disabled, all other CAN writers stopped. Preview by default; --execute runs CAN.
Uses the three adjacent replay helpers, Python stdlib only; no ROS or policy.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import sys
import time

import drive_policy_replay as rear


HERE = Path(__file__).resolve().parent
TORQUES_NM = (.12, .13, .14, .15, .16)
COUNTS_PER_TURN = 16384
EXTRA_FIELDS = ("trial_id", "repeat", "torque_level_nm", "direction", "trial_start_count",
                "trial_position_rad", "phase_time_s")
PLAN_FIELDS = ("trial_id", "repeat", "torque_level_nm", "direction", "requested_torque_nm",
               "hold_s", "rest_after_s", "nominal_start_s")
SUMMARY_FIELDS = ("trial_id", "repeat", "torque_level_nm", "direction", "requested_torque_nm",
                  "encoded_torque_nm", "hold_actual_s", "start_count", "end_count", "start_phase_deg",
                  "start_velocity_rad_s", "start_velocity_age_s", "start_position_age_s",
                  "end_velocity_rad_s", "end_velocity_age_s", "end_position_age_s",
                  "displacement_rad", "speed_min_rad_s", "speed_max_rad_s", "torque_mean_nm",
                  "torque_min_nm", "torque_max_nm", "A1_samples", "A2_samples", "A3_samples",
                  "C0_packets", "max_command_gap_s")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run rear motor; automatically save CSV/JSON/plan/summary")
    mode.add_argument("--preview", action="store_true", help="Default: show plan without CAN or files")
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--packet-rate", type=int, choices=(50, 200), default=50,
                        help="Nominal C0 refresh rate; default 50 matches the seven previous replays")
    parser.add_argument("--hold", type=rear.shared.finite, default=5., help="Seconds at each constant torque")
    parser.add_argument("--rest", type=rear.shared.finite, default=2., help="Disabled coast seconds BETWEEN trials")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-lateness-ms", type=rear.shared.finite, default=20.)
    parser.add_argument("--current-abort-a", type=rear.shared.finite, default=2.)
    parser.add_argument("--speed-abort-rad-s", type=rear.shared.finite, default=61.6,
                        help="Abort, do not clamp, above the configured rear motor speed limit")
    parser.add_argument("--temperature-abort-c", type=rear.shared.finite, default=60.)
    parser.add_argument("--note", default="", help="Physical fixture, supply, firmware settings or other conditions")
    parser.add_argument("--output", type=Path, help="New CSV path; default logs/<timestamp>_drive_torque_threshold.csv")
    args = parser.parse_args(argv)
    if (min(args.hold, args.rest, args.max_lateness_ms, args.current_abort_a,
            args.speed_abort_rad_s, args.temperature_abort_c) <= 0 or args.repeats < 1):
        parser.error("durations, repeats and abort thresholds must be positive")
    if args.output is None:
        args.output = HERE / "logs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_drive_torque_threshold.csv"
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must end in .csv")
    # Fixed, previously verified hardware mapping. These values do not load a policy/YAML.
    args.motor_id, args.encoder_sign, args.torque_sign, args.level = 1, -1, -1, "constant"
    args.query_rate, args.query_timeout, args.feedback_timeout = 100., .02, .1
    args.command_current_limit_a, args.watchdog_ms = 1.5, 500
    if args.current_abort_a <= args.command_current_limit_a:
        parser.error("--current-abort-a must exceed the 1.5 A encoded-command acceptance limit")
    return args


def build_plan(args):
    plan, nominal_start = [], 0.
    for repeat in range(1, args.repeats + 1):
        for level in TORQUES_NM:
            for sign, direction in ((1, "forward"), (-1, "backward")):
                plan.append(dict(trial_id=len(plan) + 1, repeat=repeat, torque_level_nm=level,
                                 direction=direction, requested_torque_nm=sign * level,
                                 hold_s=args.hold, rest_after_s=args.rest, nominal_start_s=nominal_start))
                nominal_start += args.hold + args.rest
    plan[-1]["rest_after_s"] = 0.  # No extra rest after the final trial.
    return plan


class CsvLog(rear.CsvLog):
    fields = rear.CsvLog.fields + EXTRA_FIELDS


class ThresholdIO(rear.DriveIO):
    def __init__(self, *args):
        super().__init__(*args)
        self.trial, self.trial_zero, self.phase_start = {}, None, None
        self.last_count = self.total_count = self.startup_total = None
        self.count_stamp = self.velocity_stamp = -math.inf
        self.samples = {op: [] for op in (0xA1, 0xA2, 0xA3)}
        self.last_disabled_stamp, self.last_status = -math.inf, None
        self.phase_end = math.inf

    def row(self, motor, kind, stamp=None, **values):
        row = super().row(motor, kind, stamp, **values)
        row.update({key: self.trial.get(key) for key in EXTRA_FIELDS[:4]})
        row.update(trial_start_count=self.trial.get("start_count"),
                   phase_time_s=(row["elapsed_s"] + self.start - self.phase_start
                                 if self.phase_start is not None else None))
        return row

    def decode_reply(self, motor, data, stamp, row):
        op = data[0]
        if op == 0xA3 and struct.unpack("<H", data[1:3])[0] >= COUNTS_PER_TURN:
            return "invalid A3 single-turn count"
        previous_kt, previous_slew = motor.torque_constant, self.current_slew_raw
        fatal = super().decode_reply(motor, data, stamp, row)
        if fatal:
            return fatal
        if self.phase in ("torque", "rest"):
            if op in (0xA1, 0xC0) and abs(row["current_a"]) > self.args.current_abort_a:
                return "rear current exceeds experiment abort threshold"
            if op == 0xA2 and abs(row["velocity_rad_s"]) > self.args.speed_abort_rad_s:
                return "rear speed exceeds experiment abort threshold"
        if op == 0xA3:
            count = row["encoder_count"]
            delta = 0 if self.last_count is None else (count - self.last_count + 2**31) % 2**32 - 2**31
            self.total_count = count if self.total_count is None else self.total_count + delta
            self.last_count, self.count_stamp = count, stamp
            if self.startup_total is not None:
                self.position = -(self.total_count - self.startup_total) * 2 * math.pi / COUNTS_PER_TURN
                row.update(position_unwrapped_rad=self.position,
                           position_rad=(self.position + math.pi) % (2 * math.pi) - math.pi)
            if self.trial_zero is not None:
                row["trial_position_rad"] = -(self.total_count - self.trial_zero) * 2 * math.pi / COUNTS_PER_TURN
        elif op == 0xA2:
            self.velocity_stamp = stamp
        if self.phase in ("torque", "rest"):
            if op == 0xB0 and motor.torque_constant != previous_kt:
                return "B0 torque constant changed during experiment"
            if op == 0xB4 and self.current_slew_raw != previous_slew:
                return "B4 current slew changed during experiment"
            if op == 0xCD and data != self.expected_cd:
                return "CD watchdog changed during experiment"
        if op in (0xAE, 0xCF):
            self.last_status = data
            if self.phase in ("torque", "rest") and (data[7] or data[5] >= self.args.temperature_abort_c):
                return "rear motor fault or excessive temperature"
            if op == 0xCF and data[6] == 0:
                self.last_disabled_stamp = stamp
            # A reply to a query from the PREVIOUS phase can arrive after the transition.
            fresh_mode_query = (self.pending is not None and self.pending.opcode == op
                                and self.phase_start is not None
                                and self.phase_start <= self.pending.sent_at <= stamp)
            if fresh_mode_query and self.phase in ("torque", "rest"):
                if data[6] != (2 if self.phase == "torque" else 0):
                    return "rear motor mode differs from current experiment phase"
        if (self.phase == "torque" and stamp >= self.phase_start and op in self.samples):
            column = {0xA1: "torque_nm", 0xA2: "velocity_rad_s", 0xA3: "trial_position_rad"}[op]
            self.samples[op].append(row[column])
        return ""

    def drain(self):
        for _ in range(rear.mit.MAX_RX_BATCH):
            try:
                self.receive()
            except BlockingIOError:
                return True
            if self.phase in ("torque", "rest") and time.monotonic() >= min(self.next_command, self.phase_end):
                return False
        return False

    def guard(self):
        self.log.check()
        now = time.monotonic()
        for op in (0xA1, 0xA2, 0xA3):
            if now - self.sample_times.get(op, -math.inf) > self.args.feedback_timeout:
                raise RuntimeError(f"stale rear {op:02X} feedback")
        if now - self.sample_times.get(0xAE, -math.inf) > .6:
            raise RuntimeError("stale rear motor status")
        if self.phase == "torque":
            if now - max(self.sample_times.get(0xC0, -math.inf), self.phase_start) > self.args.feedback_timeout:
                raise RuntimeError("stale C0 acknowledgement")
        if not math.isfinite(self.current) or abs(self.current) > self.args.current_abort_a:
            raise RuntimeError("rear current exceeds experiment abort threshold")
        if not math.isfinite(self.velocity) or abs(self.velocity) > self.args.speed_abort_rad_s:
            raise RuntimeError("rear speed exceeds experiment abort threshold")


def begin_rest(io):
    """Disable before printing, summary writes or waiting; return actual CF send time."""
    io.phase, io.phase_start = "rest", time.monotonic()
    if io.pending is not None:
        io.finish_query("cancelled_phase_boundary")
    sent = io.send(io.motors[0], b"\xcf")
    io.phase_start, io.next_command = sent, math.inf
    io.pending = rear.mit.PendingRead(io.motors[0], 0xCF, sent, sent + .5)
    return sent


def run_interval(io, end, packet=None):
    """Fixed elapsed interval; refresh constant torque without catch-up bursts."""
    args, motor = io.args, io.motors[0]
    io.phase_end = end
    periods = {0xA3: .01, 0xA2: .01, 0xA1: .01, 0xAE: .2, 0xB0: 1., 0xB4: 1., 0xCD: 1.}
    due, jobs = dict.fromkeys(periods, time.monotonic()), deque()
    previous, packets, max_gap = io.phase_start if packet else None, 1 if packet else 0, 0.
    while time.monotonic() < end:
        empty = io.drain()
        io.guard()
        now = time.monotonic()
        if now >= end:
            break
        if packet is not None and now >= io.next_command:
            data, fields = packet
            sent = io.send(motor, data, command_previous_s=previous, command_deadline_s=io.next_command,
                           record_type="drive_target", requested_torque_nm=io.trial["requested_torque_nm"], **fields)
            max_gap, previous, packets = max(max_gap, sent - previous), sent, packets + 1
            io.next_command = min(end, time.monotonic() + 1 / args.packet_rate)
        if io.pending is not None and now >= io.pending.deadline:
            opcode = io.pending.opcode
            io.finish_query("timed_out")
            if opcode == 0xCF:
                raise RuntimeError("inter-trial CF disable not acknowledged")
        if not jobs and io.pending is None:
            for opcode, period in periods.items():
                if now >= due[opcode]:
                    jobs.append(opcode)
                    due[opcode] = now + period
        if empty and io.pending is None and jobs:
            opcode = jobs.popleft()
            sent = io.send(motor, bytes([opcode]))
            io.pending = rear.mit.PendingRead(motor, opcode, sent, sent + args.query_timeout)
        wake = min(end, io.next_command, io.pending.deadline if io.pending else min(due.values()))
        io.wait(0 if not empty or jobs and io.pending is None else max(0., wake - time.monotonic()))
    # A fixed-duration test must not silently become a much longer hold after a host stall.
    if (time.monotonic() - end) * 1000 > args.max_lateness_ms:
        raise RuntimeError("missed interval-end deadline")
    if packet is None and io.last_disabled_stamp < io.phase_start:
        raise RuntimeError("disabled rest not confirmed; next trial cancelled")
    return packets, max_gap


def run_trial(io, trial, packet):
    io.guard()
    if io.pending is not None:
        io.finish_query("cancelled_phase_boundary")
    io.trial = dict(trial, start_count=io.last_count)
    io.trial_zero = io.total_count
    io.samples = {op: [] for op in (0xA1, 0xA2, 0xA3)}
    initial_speed, initial_speed_stamp, initial_pos_stamp = io.velocity, io.velocity_stamp, io.count_stamp
    io.phase, io.phase_start = "torque", time.monotonic()
    data, fields = packet
    sent = io.send(io.motors[0], data, record_type="drive_target",
                   requested_torque_nm=trial["requested_torque_nm"], **fields)
    io.phase_start = sent
    io.next_command = time.monotonic() + 1 / io.args.packet_rate
    packets, max_gap = run_interval(io, sent + trial["hold_s"], packet)
    # Capture the last actual samples, then turn output OFF before any summary I/O.
    last_count, last_speed = io.last_count, io.velocity
    last_pos_stamp, last_speed_stamp = io.count_stamp, io.velocity_stamp
    displacement = -(io.total_count - io.trial_zero) * 2 * math.pi / COUNTS_PER_TURN
    disabled_at = begin_rest(io)
    a1, a2, a3 = (io.samples[op] for op in (0xA1, 0xA2, 0xA3))
    if not a1 or not a2 or not a3:
        raise RuntimeError("trial lacks fresh A1/A2/A3 feedback")
    summary = {key: trial[key] for key in PLAN_FIELDS[:5]}
    summary.update(encoded_torque_nm=fields["encoded_command_torque_nm"], hold_actual_s=disabled_at - sent,
                   start_count=io.trial["start_count"], end_count=last_count,
                   start_phase_deg=io.trial["start_count"] % COUNTS_PER_TURN * 360 / COUNTS_PER_TURN,
                   start_velocity_rad_s=initial_speed, start_velocity_age_s=sent - initial_speed_stamp,
                   start_position_age_s=sent - initial_pos_stamp, end_velocity_rad_s=last_speed,
                   end_velocity_age_s=disabled_at - last_speed_stamp,
                   end_position_age_s=disabled_at - last_pos_stamp, displacement_rad=displacement,
                   speed_min_rad_s=min(a2), speed_max_rad_s=max(a2), torque_mean_nm=sum(a1) / len(a1),
                   torque_min_nm=min(a1), torque_max_nm=max(a1), A1_samples=len(a1), A2_samples=len(a2),
                   A3_samples=len(a3), C0_packets=packets, max_command_gap_s=max_gap)
    return summary, disabled_at


def execute(args, plan):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(".json")
    plan_path = args.output.with_name(args.output.stem + "_plan.csv")
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    if any(path.exists() for path in (args.output, meta_path, plan_path, summary_path)):
        raise FileExistsError("CSV/JSON/plan/summary exists; refuse overwrite")
    metadata = dict(completed=False, completed_trials=0, disable_confirmed=False, watchdog_restored=False,
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                   for p in (__file__, rear.__file__, rear.shared.__file__, rear.mit.__file__)},
                    rest="CF-disabled free coast, fixed duration; no guarantee of stationary next start",
                    command_schedule="fixed-duration holds, send-relative refresh without catch-up bursts",
                    position_reference="startup and per-trial software A3 multi-turn references; no B1 reset",
                    torque_definition="signed measured A1 current times measured B0 Kt; not shaft/net torque",
                    timestamps="host send-call and kernel RX; not motor acquisition or on-wire TX",
                    steering_motor_commanded=False, trial_results=[])
    with plan_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(plan)
    with meta_path.open("x+") as meta, summary_path.open("x", newline="") as summary_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        summary_writer.writeheader()
        json.dump(metadata, meta, indent=2)
        meta.flush()
        try:
            with CsvLog(args.output) as log, rear.shared.open_bus(args) as bus:
                motor = rear.mit.Motor("drive", 1, 1., -1, -1)
                io = ThresholdIO(bus, log, motor, args)
                try:
                    packets = rear.startup(io, plan)
                    # Preserve the exact first A3 zero saved by startup if a later sample differs.
                    startup_delta = (io.last_count - motor.zero_count + 2**31) % 2**32 - 2**31
                    io.startup_total = io.total_count - startup_delta
                    metadata.update(encoder_zero_count=motor.zero_count,
                                    torque_constant_nm_per_a=motor.torque_constant)
                    for trial, packet in zip(plan, packets):
                        summary, disabled_at = run_trial(io, trial, packet)
                        summary_writer.writerow(summary)
                        summary_file.flush()
                        metadata["trial_results"].append(summary)
                        metadata["completed_trials"] += 1
                        print(f"Trial {trial['trial_id']}/{len(plan)}: {trial['requested_torque_nm']:+.2f} Nm; "
                              f"start speed {summary['start_velocity_rad_s']:+.3f} rad/s; "
                              f"displacement {summary['displacement_rad']:+.4f} rad; disabled", flush=True)
                        if trial["rest_after_s"]:
                            run_interval(io, disabled_at + trial["rest_after_s"])
                    metadata["completed"] = True
                finally:
                    metadata["disable_confirmed"] = rear.shutdown(io)
                    metadata.update(watchdog_restored=io.watchdog_restored, CD_restore_error=io.restore_error,
                                    original_CD_hex=io.original_cd.hex() if io.original_cd else None,
                                    B3_position_speed_limit_a=io.current_limit, B4_current_slew_raw=io.current_slew_raw,
                                    final_fault_code=getattr(io, "final_fault_code", None))
                    if not metadata["disable_confirmed"] or not io.watchdog_restored:
                        raise RuntimeError("rear disable/CD restore not confirmed; inspect motor/settings")
                    if metadata["final_fault_code"]:
                        raise RuntimeError("rear fault present at shutdown")
        except BaseException as error:
            metadata.update(completed=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            meta.seek(0)
            json.dump(metadata, meta, indent=2, allow_nan=False)
            meta.truncate()
            meta.flush()


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    duration = sum(row["hold_s"] + row["rest_after_s"] for row in plan)
    print(f"{'EXECUTE' if args.execute else 'PREVIEW'} rear ID 1, C0 {args.packet_rate} Hz: "
          f"{len(plan)} tests, {args.hold:g} s each, {args.rest:g} s CF-disabled rest BETWEEN tests; "
          f"nominal {duration:g} s, excluding startup/shutdown")
    levels = ", ".join(f"+/-{value:.2f}" for value in TORQUES_NM)
    print(f"Levels: {levels} Nm per repetition; "
          "+ = forward, - = backward. Actual breakaway threshold is not assumed.")
    if not args.execute:
        for row in plan:
            print(f"{row['trial_id']:02d}: repeat {row['repeat']} {row['requested_torque_nm']:+.2f} Nm "
                  f"for {row['hold_s']:g} s -> rest {row['rest_after_s']:g} s")
        print("No CAN/output files opened. Add --execute to run and save feedback.")
        return 0
    print("Rear wheel airborne/free, frame supported, steering disabled, other writers stopped.\n"
          f"CSV: {args.output}", flush=True)

    def stop(*_):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        execute(args, plan)
        print("Finished; motor disabled and original CD watchdog restored.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; disable/restore attempted, partial logs retained.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, OverflowError) as error:
        print(f"ERROR: {error}; partial logs retained if created.", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
