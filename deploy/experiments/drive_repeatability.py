#!/usr/bin/env python3
"""Controlled rear C0 experiment; preview by default, stdlib + adjacent replay helpers.

No ROS or motor-mode changes beyond C0/CF and verified temporary CD watchdog.
The rear wheel must be airborne/free, frame independently supported, steering
independently disabled and all other CAN writers stopped. Each trial requires a
manual physical starting-angle reset WHILE DISABLED; software zeros do not move
the wheel. Tests cannot establish complete motor behavior or shaft torque.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, deque
from contextlib import contextmanager
import csv
from datetime import datetime
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import struct
import sys
import threading
import time

import drive_policy_replay as rear


HERE, DT = Path(__file__).resolve().parent, rear.DT
COUNTS = 16384
EXTRA_FIELDS = ("trial_id", "family", "repeat", "angle_deg", "polarity", "amplitude_nm",
                "reference_phase_deg", "angle_error_deg")
SUMMARY_FIELDS = ("trial_id", "family", "repeat", "angle_deg", "polarity", "amplitude_nm", "packet_rate",
                  "attempt", "status", "timing_valid", "actual_packet_hz", "max_lateness_ms",
                  "skipped_packet_slots", "missing_target_indices", "start_count", "end_count",
                  "displacement_rad", "velocity_integral_rad", "speed_min_rad_s", "speed_max_rad_s",
                  "A1_torque_error_rms_nm", "core_A1_torque_error_rms_nm", "breakaway_detected_s",
                  "breakaway_command_nm", "breakaway_censored",
                  "voltage_min_v", "voltage_max_v", "temperature_max_c", "csv", "error")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Open CAN and run with manual prompts")
    mode.add_argument("--preview", action="store_true", help="Default: no CAN/files")
    parser.add_argument("--preset", choices=("quick", "screen", "full"), default="full")
    parser.add_argument("--families", nargs="+", choices=("replay", "breakaway", "pulse", "reversal"))
    parser.add_argument("--angles", type=rear.shared.finite, nargs="+", help="Forward-positive degrees from mark A")
    parser.add_argument("--rates", type=int, nargs="+", choices=(50, 100, 200))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--pulse-torques", type=rear.shared.finite, nargs="+")
    parser.add_argument("--reversal-torque", type=rear.shared.finite, default=.24)
    parser.add_argument("--pulse-seconds", type=rear.shared.finite, default=.20)
    parser.add_argument("--ramp-step-nm", type=rear.shared.finite, default=.02)
    parser.add_argument("--ramp-max-nm", type=rear.shared.finite, default=.36)
    parser.add_argument("--ramp-dwell", type=rear.shared.finite, default=.20)
    parser.add_argument("--movement-rad", type=rear.shared.finite, default=.01,
                        help="Ramp termination after two fresh A3 samples beyond this displacement")
    parser.add_argument("--pre-zero", type=rear.shared.finite, default=2.)
    parser.add_argument("--coast", type=rear.shared.finite, default=3., help="Zero-current observation, not disabled")
    parser.add_argument("--rest-seconds", type=rear.shared.finite, default=1., help="Continuous disabled rest")
    parser.add_argument("--rest-timeout", type=rear.shared.finite, default=20.)
    parser.add_argument("--rest-speed", type=rear.shared.finite, default=.10)
    parser.add_argument("--angle-tolerance-deg", type=rear.shared.finite, default=2.)
    parser.add_argument("--targets", type=Path, default=HERE / "replay_targets/2500_isaac400_env0_episode1.csv")
    parser.add_argument("--start", type=rear.shared.finite, default=12.5)
    parser.add_argument("--duration", type=rear.shared.finite, default=5.)
    parser.add_argument("--level", choices=("scaled", "submitted"), default="scaled")
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--query-rate", type=rear.shared.finite, default=100.)
    parser.add_argument("--max-lateness-ms", type=rear.shared.finite, default=20.)
    parser.add_argument("--speed-abort-rad-s", type=rear.shared.finite, default=10.)
    parser.add_argument("--current-abort-a", type=rear.shared.finite, default=2.)
    parser.add_argument("--command-current-limit-a", type=rear.shared.finite, default=1.5)
    parser.add_argument("--temperature-abort-c", type=rear.shared.finite, default=60.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--note", default="", help="Supply, fixture, firmware changes and ambient conditions")
    parser.add_argument("--session", type=Path, help="New exclusive output directory")
    parser.add_argument("--resume", type=Path,
                        help="Reuse saved config/order/reference; only mode/max-trials may change")
    parser.add_argument("--max-trials", type=int, help="Pause after this many trials; later use --resume")
    args = parser.parse_args(argv)
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be positive")
    if args.resume:
        # Resuming with silently changed conditions would spoil a controlled comparison.
        raw = list(sys.argv[1:] if argv is None else argv)
        permitted = {"--resume", "--execute", "--preview", "--max-trials"}
        if any(token.split("=")[0] not in permitted for token in raw if token.startswith("--")):
            parser.error("--resume accepts only --execute/--preview and --max-trials")
        return args
    defaults = {
        "quick": ([0., 180.], [50, 200], 1, ["replay", "breakaway"], [.12, .24]),
        "screen": ([0., 180.], [50, 200], 3, ["replay", "breakaway", "pulse"], [.12, .24]),
        "full": ([0., 90., 180., 270.], [50, 100, 200], 3,
                 ["replay", "breakaway", "pulse", "reversal"], [.06, .12, .24, .36]),
    }[args.preset]
    for key, value in zip(("angles", "rates", "repeats", "families", "pulse_torques"), defaults):
        if getattr(args, key) is None:
            setattr(args, key, value)
    try:
        for value in (args.start, args.duration, args.pre_zero, args.coast, args.pulse_seconds, args.ramp_dwell):
            rear.shared.ticks(value)
        if (args.repeats < 1 or args.duration <= 0 or args.pre_zero < .2 or args.coast < .2
                or min(args.pulse_seconds, args.ramp_dwell, args.ramp_step_nm, args.ramp_max_nm) <= 0
                or args.ramp_step_nm > args.ramp_max_nm or args.ramp_max_nm / args.ramp_step_nm > 100
                or min(args.pulse_torques + [args.reversal_torque]) <= 0
                or max(args.pulse_torques + [args.reversal_torque, args.ramp_max_nm]) > .44
                or not .002 <= args.movement_rad <= .1
                or not 0 < args.angle_tolerance_deg <= 10 or not 0 < args.rest_speed <= .2
                or not .2 <= args.rest_seconds < args.rest_timeout or not 0 < args.query_rate <= 200
                or 1 / args.query_rate >= .1 or args.max_lateness_ms <= 0
                or not 0 < args.command_current_limit_a < args.current_abort_a
                or min(args.speed_abort_rad_s, args.temperature_abort_c) <= 0
                or any(not 0 <= a < 360 for a in args.angles)
                or len(set(args.angles)) != len(args.angles) or len(set(args.rates)) != len(args.rates)
                or len(set(args.families)) != len(args.families)
                or len(set(args.pulse_torques)) != len(args.pulse_torques)
                or args.max_trials is not None and args.max_trials < 1):
            raise ValueError("invalid durations, duplicate conditions, torque range, rates or thresholds")
    except ValueError as error:
        parser.error(str(error))
    return args


def serializable(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")}


def build_trials(args):
    """Randomized blocks, with rate order rotated across repeats for every condition."""
    rng, trials = random.Random(args.seed), []
    blocks = [(angle, family, polarity, amplitude) for angle in args.angles for family in args.families
              for polarity in (-1, 1)
              for amplitude in (args.pulse_torques if family == "pulse" else [args.reversal_torque]
                                if family == "reversal" else [0.])]
    offsets = {block: rng.randrange(len(args.rates)) for block in blocks}
    for repeat in range(args.repeats):
        order = list(blocks)
        rng.shuffle(order)
        for block in order:
            angle, family, polarity, amplitude = block
            offset = (offsets[block] + repeat) % len(args.rates)
            for rate in args.rates[offset:] + args.rates[:offset]:
                trials.append(dict(trial_id=f"{len(trials) + 1:04d}", repeat=repeat + 1, angle_deg=angle,
                                   family=family, polarity=polarity, amplitude_nm=amplitude,
                                   packet_rate=rate, attempts=[]))
    return trials


def build_plan(args, trial):
    plan = []

    def add(phase, torque, source=None):
        item = dict.fromkeys(rear.PLAN_FIELDS)
        item.update(target_index=len(plan), target_time_s=len(plan) * DT, phase=phase,
                    requested_torque_nm=rear.shared.f32(torque))
        if source:
            item.update({key: source[key] for key in ("source_time_s", "raw_drive", "scaled_drive_nm",
                                                     "submitted_drive_nm")})
        plan.append(item)

    def hold(phase, torque, seconds):
        for _ in range(rear.shared.ticks(seconds)):
            add(phase, torque)

    hold("pre_zero", 0., args.pre_zero)
    sign, family = trial["polarity"], trial["family"]
    if family == "replay":
        rows = getattr(args, "_target_rows", None)
        if rows is None:
            rows, _ = rear.shared.load_target_rows(args.targets)
        key = "scaled_drive_nm" if args.level == "scaled" else "submitted_drive_nm"
        for row in rear.shared.select_window(rows, args.start, args.duration):
            add("core", sign * row[key], row)
    elif family == "breakaway":
        steps = math.ceil(args.ramp_max_nm / args.ramp_step_nm - 1e-9)
        for step in range(1, steps + 1):
            hold("ramp", sign * min(step * args.ramp_step_nm, args.ramp_max_nm), args.ramp_dwell)
    else:
        hold("pulse", sign * trial["amplitude_nm"], args.pulse_seconds)
        if family == "reversal":
            hold("reverse", -sign * trial["amplitude_nm"], args.pulse_seconds)
    hold("coast", 0., args.coast)
    if max(abs(row["requested_torque_nm"]) for row in plan) > rear.shared.f32(.44):
        raise ValueError("experiment rejects torques above 0.44 Nm; select a smaller recorded window")
    return plan


def phase_error(count, reference_count, angle_deg):
    actual = (-1 * (count - reference_count) * 360 / COUNTS) % 360
    return actual, (actual - angle_deg + 180) % 360 - 180


class CsvLog(rear.CsvLog):
    fields = rear.CsvLog.fields + EXTRA_FIELDS


class ExperimentIO(rear.DriveIO):
    def row(self, motor, kind, stamp=None, **values):
        row = super().row(motor, kind, stamp, **values)
        trial = getattr(self, "trial", {})
        row.update({key: trial.get(key) for key in EXTRA_FIELDS[:6]})
        # DriveIO marks old preserved-target replays as zero skipped ticks; here actual skips are explicit.
        if "skipped_command_ticks" in values:
            row["skipped_command_ticks"] = values["skipped_command_ticks"]
        return row

    def decode_reply(self, motor, data, stamp, row):
        if data[0] == 0xA3 and struct.unpack("<H", data[1:3])[0] >= COUNTS:
            return "invalid A3 single-turn count"
        fatal = super().decode_reply(motor, data, stamp, row)
        if data[0] == 0xA3:
            self.count = row["encoder_count"]
            self.last_position_stamp = stamp
        return fatal


def operator_input(io, prompt):
    """Only stdin runs in a worker; keep disabled-mode CAN alive during long manual waits."""
    ready, result = threading.Event(), []

    def read():
        try:
            result.append(input(prompt))
        except BaseException as error:
            result.append(error)
        finally:
            ready.set()

    threading.Thread(target=read, daemon=True).start()
    while not ready.wait(.05):
        status = io.request(io.motors[0], 0xAE)
        if status[6] or status[7] or status[5] >= io.args.temperature_abort_c:
            raise RuntimeError("manual placement: motor left disabled/fault-free/cool state")
    if isinstance(result[0], BaseException):
        raise result[0]
    if result[0].strip().lower() == "q":
        raise KeyboardInterrupt
    return result[0]


def disabled_rest(io):
    """Require continuous rest plus fresh disabled/fault/temperature confirmations."""
    motor, args = io.motors[0], io.args
    stable_since, anchor = None, None
    deadline = time.monotonic() + args.rest_timeout
    while time.monotonic() < deadline:
        a3 = io.request(motor, 0xA3)
        count = struct.unpack("<i", a3[3:])[0]
        io.request(motor, 0xA2)
        status = io.request(motor, 0xAE)
        if status[6] or status[7] or status[5] >= args.temperature_abort_c:
            raise RuntimeError("rest check: motor must remain disabled, cool and fault-free")
        now = time.monotonic()
        if abs(io.velocity) > args.rest_speed:
            stable_since, anchor = None, None
        elif anchor is None or abs(count - anchor) > 2:
            stable_since, anchor = now, count
        elif now - stable_since >= args.rest_seconds:
            return count
        io.wait(.02)
    raise RuntimeError("wheel did not remain at rest; retry when fully stopped")


def rest_and_check(io, reference, trial):
    """Disabled-only manual placement, then fresh stationary/phase verification before any C0."""
    motor, args = io.motors[0], io.args
    while True:
        operator_input(io, f"DISABLED. Place wheel at {trial['angle_deg']:g} deg FORWARD from mark A. "
                           "Remove hands, then Enter (q to stop): ")
        count = disabled_rest(io)
        actual, error = phase_error(count, reference, trial["angle_deg"])
        io.log.emit(io.row(motor, "placement", reference_phase_deg=actual, angle_error_deg=error))
        if abs(error) <= args.angle_tolerance_deg:
            motor.zero_count, motor.reference_epoch = count, 1
            io.request(motor, 0xA3)
            io.request(motor, 0xA1)
            return count, error
        print(f"Angle mismatch: measured {actual:.2f} deg, error {error:+.2f} deg; reposition.")


def motion(io, plan, packets, trial):
    """Wall-clock 50-Hz targets independent of C0 refresh. Missed slots are logged, never burst-replayed."""
    motor, args = io.motors[0], io.args
    io.phase, io.motion_start = "motion", time.monotonic()
    io.next_command = io.motion_start
    period, packet_index, previous = 1 / args.packet_rate, 0, None
    periods = {0xA3: 1 / args.query_rate, 0xA2: 1 / args.query_rate, 0xA1: 1 / args.query_rate,
               0xAE: .2, 0xB0: 1., 0xB4: 1., 0xCD: 1.}
    due, jobs = dict.fromkeys(periods, io.motion_start), deque()
    end = io.motion_start + len(plan) * DT
    detected, confirm, last_stamp, last_torque = None, 0, None, 0.
    seen, skipped, max_late, expected_stop = set(), 0, 0., len(plan)
    start_count, core_start = motor.zero_count, None
    first_nonzero_sent = math.inf
    while time.monotonic() < end:
        empty = io.drain()
        io.guard()
        now = time.monotonic()
        elapsed = now - io.motion_start
        index = min(int(elapsed / DT + 1e-8), len(plan) - 1)
        if core_start is None and plan[index]["phase"] != "pre_zero":
            angle_error = phase_error(io.count, io.reference_count, trial["angle_deg"])[1]
            if abs(angle_error) > args.angle_tolerance_deg or abs(io.velocity) > args.rest_speed:
                raise RuntimeError("wheel moved out of the matched starting condition during zero-current preparation")
            core_start = dict(core_start_count=io.count, core_start_angle_error_deg=angle_error,
                              core_start_velocity_rad_s=io.velocity)
            # Zero-current preparation can shift the wheel within the allowed phase tolerance.
            # Count only subsequent ramp motion, never that preparation displacement.
            start_count, last_stamp = io.count, io.last_position_stamp
        if (trial["family"] == "breakaway" and detected is None and plan[index]["phase"] == "ramp"
                and getattr(io, "last_position_stamp", -math.inf) >= first_nonzero_sent
                and io.last_position_stamp != last_stamp):
            last_stamp = getattr(io, "last_position_stamp", None)
            confirm = confirm + 1 if abs(io.count - start_count) * 2 * math.pi / COUNTS >= args.movement_rad else 0
            if confirm >= 2:
                # Use the last SENT command, not a ramp level merely due but not yet delivered.
                detected = dict(breakaway_detected_s=elapsed, breakaway_command_nm=last_torque)
                expected_stop = index  # Current/future ramp samples are intentionally cancelled.
                end = now + args.coast
                io.next_command = min(io.next_command, now)  # Prompt zero after movement; no further ramp.
        if now >= io.next_command and now < end:
            sample = dict(plan[index])
            if detected:
                sample.update(phase="early_coast", requested_torque_nm=0.)
                data, fields = rear.current_payload(0., motor.torque_constant, motor.torque_sign)
            else:
                data, fields = packets[index]
                seen.add(index)
            late = max(0., now - io.next_command)
            missed = int((late + 1e-10) / period)
            previous = io.send(motor, data, command_previous_s=previous, command_deadline_s=io.next_command,
                               record_type="drive_target", packet_index=packet_index,
                               skipped_command_ticks=missed, **sample, **fields)
            last_torque = fields["encoded_command_torque_nm"]
            if last_torque and first_nonzero_sent == math.inf:
                first_nonzero_sent = previous
            skipped += missed
            max_late = max(max_late, previous - io.next_command)
            packet_index += 1
            # Preserve the clock for ordinary jitter; never follow a delayed packet with a near-zero gap.
            io.next_command = max(io.next_command + (missed + 1) * period, previous + .8 * period)
        if io.pending is not None and now >= io.pending.deadline:
            io.finish_query("timed_out")
        if not jobs and io.pending is None:
            for opcode, query_period in periods.items():
                if now >= due[opcode]:
                    jobs.append(opcode)
                    due[opcode] = now + query_period
        if empty and io.pending is None and jobs:
            opcode = jobs.popleft()
            sent = io.send(motor, bytes([opcode]))
            io.pending = rear.mit.PendingRead(motor, opcode, sent, sent + args.query_timeout)
        wake = min(end, io.next_command, io.pending.deadline if io.pending else min(due.values()))
        io.wait(0 if not empty or jobs and io.pending is None else max(0., wake - time.monotonic()))
    # A host stall crossing the end must not bypass deadline/freshness checks and look successful.
    io.guard()
    if (time.monotonic() - end) * 1000 > args.max_lateness_ms:
        raise RuntimeError("missed experiment end deadline")
    io.log.emit(io.row(motor, "motion_end"))
    missing = len(set(range(expected_stop)) - seen)
    return dict(core_start or {}, breakaway_detected_s=detected["breakaway_detected_s"] if detected else None,
                breakaway_command_nm=detected["breakaway_command_nm"] if detected else None,
                breakaway_censored=trial["family"] == "breakaway" and detected is None,
                packet_count=packet_index, skipped_packet_slots=skipped, missing_target_indices=missing,
                intentional_ramp_cancel=detected is not None,
                timing_valid=skipped == 0 and missing == 0, max_lateness_ms=max_late * 1000,
                actual_motion_duration_s=time.monotonic() - io.motion_start)


def save_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        # Reserve attempt/reference state durably before proceeding with hardware.
        os.fsync(stream.fileno())
    temp.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def summarize(path, outcome):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    commands = [r for r in rows if r["record_type"] == "drive_target"]
    active = [r for r in rows if r["can_direction"] == "bus_rx" and r["phase"] == "motion"]
    a3 = [r for r in active if r["can_opcode"] == "163"]
    speeds = [(float(r["replay_time_s"]), float(r["velocity_rad_s"])) for r in active if r["can_opcode"] == "162"]
    status = [r for r in active if r["can_opcode"] == "174"]
    times = [float(r["replay_time_s"]) for r in commands]
    errors, core_errors = [], []
    for row in active:
        if row["can_opcode"] == "161":
            index = bisect_right(times, float(row["replay_time_s"])) - 1
            if index >= 0:
                error = float(row["torque_nm"]) - float(commands[index]["encoded_command_torque_nm"])
                errors.append(error)
                if commands[index]["phase"] not in ("pre_zero", "coast", "early_coast"):
                    core_errors.append(error)
    result = {key: outcome.get(key) for key in SUMMARY_FIELDS}
    result.update(start_count=outcome.get("encoder_zero_count"), end_count=int(a3[-1]["encoder_count"]) if a3 else None,
                  displacement_rad=float(a3[-1]["position_unwrapped_rad"]) if a3 else None,
                  velocity_integral_rad=(sum((b[0] - a[0]) * (b[1] + a[1]) / 2
                                             for a, b in zip(speeds, speeds[1:])) if len(speeds) > 1 else None),
                  speed_min_rad_s=min((s[1] for s in speeds), default=None),
                  speed_max_rad_s=max((s[1] for s in speeds), default=None),
                  A1_torque_error_rms_nm=math.sqrt(sum(x * x for x in errors) / len(errors)) if errors else None,
                  core_A1_torque_error_rms_nm=(math.sqrt(sum(x * x for x in core_errors) / len(core_errors))
                                               if core_errors else None),
                  actual_packet_hz=(len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else None,
                  voltage_min_v=min((float(r["bus_voltage_v"]) for r in status), default=None),
                  voltage_max_v=max((float(r["bus_voltage_v"]) for r in status), default=None),
                  temperature_max_c=max((float(r["motor_temperature_c"]) for r in status), default=None))
    return result


def trial_args(args, trial, output):
    # Reuse proven rear transport/cleanup, including independent A1/C0 freshness and CD rollback.
    values = vars(rear.parse_args([]))
    values.update(vars(args))
    values.update(packet_rate=trial["packet_rate"], output=output, motor_id=1, encoder_sign=-1, torque_sign=-1)
    return argparse.Namespace(**values)


def run_trial(args, trial, session, reference, attempt, baseline=None):
    path = session / f"{trial['trial_id']}_{trial['family']}_{trial['packet_rate']}hz_attempt{attempt}.csv"
    if any(p.exists() for p in (path, path.with_suffix(".json"), path.with_name(path.stem + "_plan.csv"))):
        raise FileExistsError("trial artifacts already exist; refuse overwrite")
    config = trial_args(args, trial, path)
    plan = build_plan(args, trial)
    with path.with_name(path.stem + "_plan.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rear.PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(plan)
    outcome = dict({k: v for k, v in trial.items() if k != "attempts"},
                   attempt=attempt, csv=path.name, status="failed", disable_confirmed=False,
                   watchdog_restored=False, timestamps="host send calls/kernel RX; not shaft torque or wire TX",
                   command_schedule="wall_clock_50hz_targets_no_packet_catchup", max_lateness_ms=config.max_lateness_ms)
    error = None
    try:
        with CsvLog(path) as log, rear.shared.open_bus(config) as bus:
            motor = rear.mit.Motor("drive", 1, 1., -1, -1)
            io = ExperimentIO(bus, log, motor, config)
            io.trial, io.reference_count = trial, reference
            try:
                status = io.request(motor, 0xCF)
                if status[6] or status[7]:
                    raise RuntimeError("placement requires disabled/fault-free rear motor")
                zero, angle_error = rest_and_check(io, reference, trial)
                packets = rear.startup(io, plan)  # Enable temporary watchdog AFTER the manual placement wait.
                if abs(motor.zero_count - zero) > 2:
                    raise RuntimeError("wheel moved after placement verification")
                outcome.update(encoder_zero_count=motor.zero_count, angle_error_deg=angle_error,
                               torque_constant_nm_per_a=motor.torque_constant)
                hardware = dict(Kt=motor.torque_constant, B3=io.current_limit, B4=io.current_slew_raw,
                                CD=io.original_cd.hex())
                outcome["hardware_settings"] = hardware
                if baseline is not None and hardware != baseline:
                    raise RuntimeError("motor settings changed within session; start a separate controlled session")
                outcome.update(motion(io, plan, packets, trial))
                outcome["status"] = "completed"
            except BaseException as caught:
                error = caught
                outcome["error"] = f"{type(caught).__name__}: {caught}"
            finally:
                try:
                    outcome["disable_confirmed"] = rear.shutdown(io)
                    if not outcome["disable_confirmed"] or not io.watchdog_restored:
                        raise RuntimeError("disable/CD restoration not confirmed; inspect before continuing")
                    if getattr(io, "final_fault_code", None):
                        raise RuntimeError("motor fault present at shutdown; session stopped")
                except BaseException as caught:
                    outcome["cleanup_error"] = f"{type(caught).__name__}: {caught}"
                    error = error or caught
                finally:
                    outcome.update(watchdog_restored=io.watchdog_restored, CD_restore_error=io.restore_error,
                                   original_CD_hex=io.original_cd.hex() if io.original_cd else None,
                                   B4_current_slew_raw=io.current_slew_raw, B3_position_speed_limit_a=io.current_limit,
                                   final_fault_code=getattr(io, "final_fault_code", None))
    except BaseException as caught:
        if error is not None:
            outcome["recording_error"] = f"{type(caught).__name__}: {caught}"
        error = error or caught
    if error:
        outcome.update(status="failed", error=f"{type(error).__name__}: {error}")
    save_json(path.with_suffix(".json"), outcome)
    try:
        if path.exists():
            outcome["summary"] = summarize(path, outcome)
        outcome["artifacts_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (path, path.with_name(path.stem + "_plan.csv")) if p.exists()}
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as caught:
        outcome["analysis_error"] = f"{type(caught).__name__}: {caught}"
        error = error or caught
        outcome.update(status="failed", error=f"{type(error).__name__}: {error}")
    if "summary" in outcome:
        outcome["summary"].update(status=outcome["status"], error=outcome.get("error"))
    save_json(path.with_suffix(".json"), outcome)
    return outcome, error


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), Path(rear.__file__), Path(rear.shared.__file__), Path(rear.mit.__file__))}


def saved_config(values):
    """Run persisted values through the same parser/range checks as a new session."""
    if set(values) != set(vars(parse_args([]))):
        raise ValueError("invalid saved argument schema")
    argv = []
    for key, value in values.items():
        if value is None or key in ("execute", "preview", "resume"):
            continue
        option = "--" + key.replace("_", "-")
        argv.extend([option] + [str(v) for v in value] if isinstance(value, list) else [f"{option}={value}"])
    try:
        return parse_args(argv)
    except SystemExit as error:
        raise ValueError("invalid saved configuration") from error


def validate_trials(args, session, saved):
    expected = build_trials(args)
    trials = saved["trials"]
    if len(trials) != len(expected):
        raise ValueError("saved trial design changed")
    reference = saved["reference_count"]
    if reference is not None and (type(reference) is not int or not -(2**31) <= reference < 2**31):
        raise ValueError("invalid saved encoder reference")
    for trial, design in zip(trials, expected):
        if {k: v for k, v in trial.items() if k != "attempts"} != {k: v for k, v in design.items() if k != "attempts"}:
            raise ValueError("saved trial design changed")
        for number, attempt in enumerate(trial["attempts"], 1):
            if (reference is None or attempt["attempt"] != number
                    or attempt["status"] not in ("interrupted", "failed", "completed")):
                raise ValueError("invalid saved trial attempt")
            if attempt["status"] == "completed":
                stem = f"{trial['trial_id']}_{trial['family']}_{trial['packet_rate']}hz_attempt{number}"
                artifacts = attempt.get("artifacts_sha256", {})
                if (set(artifacts) != {stem + ".csv", stem + "_plan.csv"}
                        or not attempt.get("disable_confirmed") or not attempt.get("watchdog_restored")):
                    raise ValueError("completed trial lacks verified artifacts/cleanup")
                for name, digest in artifacts.items():
                    if hashlib.sha256((session / name).read_bytes()).hexdigest() != digest:
                        raise ValueError("completed trial artifact changed")
                if json.loads((session / (stem + ".json")).read_text()) != attempt:
                    raise ValueError("completed trial metadata changed")


def load_session(args):
    if args.resume:
        session = args.resume.resolve()
        saved = json.loads((session / "session.json").read_text())
        config = saved_config(saved["arguments"])
        config.execute, config.max_trials, config.session = args.execute, args.max_trials, session
        config.resume = session
        if saved["source_sha256"] != source_hashes():
            raise ValueError("script/helpers changed since session began; start a new controlled session")
        args = config
    else:
        session = (args.session or HERE / "logs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_drive_repeatability").resolve()
        args.targets = args.targets.resolve()
        saved = dict(arguments=serializable(args), source_sha256=source_hashes(), reference_count=None,
                     trials=build_trials(args), meaning="current-derived torque; no independent shaft torque sensor")
    # Validate all targets before touching CAN, including a resumed source.
    digest = None
    if "replay" in args.families:
        args._target_rows, digest = rear.shared.load_target_rows(args.targets)
    if args.resume and digest != saved["targets_sha256"]:
        raise ValueError("source targets changed since session began")
    saved["targets_sha256"] = digest
    validate_trials(args, session, saved)
    for trial in saved["trials"]:
        build_plan(args, trial)
    return args, session, saved


def capture_reference(args, session, expected=None):
    config = trial_args(args, {"packet_rate": 50}, session / f"reference_{datetime.now():%H%M%S_%f}.csv")
    with CsvLog(config.output) as log, rear.shared.open_bus(config) as bus:
        motor = rear.mit.Motor("drive", 1, 1., -1, -1)
        io = ExperimentIO(bus, log, motor, config)
        try:
            status = io.request(motor, 0xCF)
            if status[6] or status[7]:
                raise RuntimeError("reference capture requires disabled/fault-free motor")
            operator_input(io, "Rear airborne/free, frame supported, other writers OFF. DISABLED: "
                               "place mark A at 0 deg, remove hands, then Enter (q to stop): ")
            count = disabled_rest(io)
            if expected is not None and abs(phase_error(count, expected, 0.)[1]) > args.angle_tolerance_deg:
                raise ValueError("mark A no longer matches saved encoder phase; restore it or start a new session")
            return count
        finally:
            if not rear.shared.shutdown(io):
                raise RuntimeError("reference capture disable not confirmed")


def write_summary(session, saved):
    with (session / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for trial in saved["trials"]:
            for attempt in trial["attempts"]:
                if "summary" in attempt:
                    writer.writerow(attempt["summary"])


@contextmanager
def session_lock(session):
    """Never let two executing/resuming processes write one session concurrently."""
    with (session / ".run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("session is already running in another process") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def execute_session(args, session, saved):
    if not args.resume:
        session.mkdir(parents=True, exist_ok=False)
    with session_lock(session):
        # A different runner could have updated this session between preview/load and lock acquisition.
        if args.resume:
            args, session, saved = load_session(args)
        todo = [t for t in saved["trials"] if not t["attempts"] or t["attempts"][-1]["status"] != "completed"]
        save_json(session / "session.json", saved)
        write_summary(session, saved)  # Rebuild derived summary after a previous interrupted write.
        if not todo:
            print("All trials already completed.")
            return
        ref = capture_reference(args, session, saved["reference_count"])
        if saved["reference_count"] is None:
            saved["reference_count"] = ref
        save_json(session / "session.json", saved)
        for trial in todo[:args.max_trials]:
            if source_hashes() != saved["source_sha256"]:
                raise RuntimeError("script/helpers changed during session; no next trial started")
            print(f"Trial {trial['trial_id']}: {trial['family']}, {trial['angle_deg']:g} deg, "
                  f"{trial['packet_rate']} Hz, polarity {trial['polarity']:+}, repeat {trial['repeat']}", flush=True)
            attempt = len(trial["attempts"]) + 1
            marker = dict(status="interrupted", attempt=attempt)
            trial["attempts"].append(marker)
            save_json(session / "session.json", saved)
            outcome, error = run_trial(args, trial, session, saved["reference_count"], attempt,
                                       saved.get("hardware_settings"))
            trial["attempts"][-1] = outcome
            if "hardware_settings" in outcome and "hardware_settings" not in saved:
                saved["hardware_settings"] = outcome["hardware_settings"]
            save_json(session / "session.json", saved)
            write_summary(session, saved)
            if error:
                raise error
            print(f"Disabled/CD restored. Timing valid: {outcome['timing_valid']}; "
                  f"displacement {outcome['summary']['displacement_rad']} rad", flush=True)
    print(f"Session saved: {session}. Resume to continue pending trials.")


def main(argv=None):
    args = parse_args(argv)
    def stop(*_):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        args, session, saved = load_session(args)
        todo = [t for t in saved["trials"] if not t["attempts"] or t["attempts"][-1]["status"] != "completed"]
        print(f"{'EXECUTE' if args.execute else 'PREVIEW'}: {len(saved['trials'])} trials, {len(todo)} pending; "
              f"rates {args.rates}, angles {args.angles}, repeats {args.repeats}; "
              f"tolerance {args.max_lateness_ms:g} ms")
        print(f"Families: {dict(Counter(t['family'] for t in saved['trials']))}; directory: {session}")
        print("Each trial needs manual placement verification. Full suite can be split with --max-trials/--resume.")
        if not args.execute:
            print("No CAN/files opened. See DRIVE_REPEATABILITY.md for protocol and interpretation.")
            return 0
        execute_session(args, session, saved)
        return 0
    except KeyboardInterrupt:
        print("Stopped; partial results retained. A spinning wheel may still coast.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, OverflowError, KeyError, EOFError, TypeError) as error:
        print(f"ERROR: {error}; results retained if created.", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
