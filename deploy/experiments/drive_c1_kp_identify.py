#!/usr/bin/env python3
"""Identify rear ID 1 C1 effective velocity Kp; preview unless --execute.

NEVER writes B8 Kp or B9 Ki: jointly identifies effective P/I with gains as found.
Original CD is restored after confirmed CF disable. Rear wheel airborne/free, frame supported, steering
disabled and other CAN writers stopped. Logs every packet and feedback to CSV.
No angle/tracking/lateness/rest-speed trips; late slots are skipped and recorded.
Physical speed/torque, driver/bus/log faults and 1/2-s feedback/status stops remain.
Needs adjacent drive_velocity_sine.py only; Python stdlib on the Pi.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys
import threading

import drive_velocity_sine as shared

HERE = Path(__file__).resolve().parent


@dataclass
class Segment:
    name: str
    mode: str
    kind: str
    repeat: int
    base_rad_s: float
    amplitude_rad_s: float
    frequency_hz: float
    duration_s: float
    settle_s: float
    pulse_start_s: float = 0.
    pulse_end_s: float = 0.

    def target(self, elapsed):
        if self.kind == "sine":
            return self.base_rad_s + self.amplitude_rad_s * math.sin(2*math.pi*self.frequency_hz*elapsed)
        if self.kind == "step":
            return self.base_rad_s + (self.amplitude_rad_s if self.pulse_start_s <= elapsed < self.pulse_end_s else 0.)
        return self.base_rad_s


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--suite", choices=("all", "steady", "steps", "sine"), default="all")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--levels", nargs="+", type=shared.finite, default=[.05, .1, .2, .4, .8, 1.5, 3., 6., 10.])
    p.add_argument("--step-bases", nargs="+", type=shared.finite, default=[-6., -2., 0., 2., 6.])
    p.add_argument("--step-sizes", nargs="+", type=shared.finite, default=[.1, .25, .5, 1.])
    p.add_argument("--sine-bases", nargs="+", type=shared.finite, default=[-3., 0., 3.])
    p.add_argument("--sine-amplitudes", nargs="+", type=shared.finite, default=[.25, 1.])
    p.add_argument("--frequencies", nargs="+", type=shared.finite, default=[.5, 1., 2., 5., 10., 20.])
    p.add_argument("--steady-settle", type=shared.finite, default=2.)
    p.add_argument("--steady-measure", type=shared.finite, default=3.)
    p.add_argument("--step-base-hold", type=shared.finite, default=.6)
    p.add_argument("--step-pulse-hold", type=shared.finite, default=.4)
    p.add_argument("--sine-settle", type=shared.finite, default=1.)
    p.add_argument("--sine-cycles", type=int, default=4)
    p.add_argument("--sine-min-measure", type=shared.finite, default=4.)
    p.add_argument("--packet-rate", type=int, choices=(50, 200, 400), default=200)
    p.add_argument("--steady-feedback-rate", type=int, choices=(100, 200, 400), default=100)
    p.add_argument("--dynamic-feedback-rate", type=int, choices=(100, 200, 400), default=400)
    p.add_argument("--feedback-timeout", type=shared.finite, default=1.)
    p.add_argument("--status-timeout", type=shared.finite, default=2.)
    p.add_argument("--interface", default="can0")
    p.add_argument("--note", default="", help="Fixture, load, supply and any firmware differences")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.repeats < 1 or a.sine_cycles < 1:
        p.error("repeats and sine cycles must be positive")
    if min(a.steady_settle, a.steady_measure, a.step_base_hold, a.step_pulse_hold,
           a.sine_settle, a.sine_min_measure, a.feedback_timeout, a.status_timeout) <= 0:
        p.error("durations/timeouts must be positive")
    if min(a.levels+a.step_sizes+a.sine_amplitudes+a.frequencies) <= 0:
        p.error("levels, step sizes, sine amplitudes and frequencies must be positive")
    if max(a.frequencies) >= a.packet_rate/2:
        p.error("sine frequencies must be below half the packet rate")
    if a.output is None:
        a.output = HERE / "logs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_drive_c1_kp_identify.csv"
    if a.output.suffix != ".csv":
        p.error("output must end in .csv")
    a.method = "c1"
    return a


def make_plan(args):
    rng, plan = random.Random(args.seed), []

    def add(mode, kind, repeat, base=0., amplitude=0., frequency=0.):
        if kind == "sine":
            settle = args.sine_settle
            duration = settle + max(args.sine_cycles, math.ceil(args.sine_min_measure*frequency))/frequency
        elif kind == "step":
            settle, duration = args.step_base_hold, 2*args.step_base_hold+args.step_pulse_hold
        elif kind == "zero":
            settle, duration = 0., 2.
        else:
            settle, duration = args.steady_settle, args.steady_settle+args.steady_measure
        segment = Segment(f"{mode}_{kind}_r{repeat}_{len(plan):04d}", mode, kind, repeat, base, amplitude,
                          frequency, duration, settle, args.step_base_hold,
                          args.step_base_hold+args.step_pulse_hold)
        if abs(base)+abs(amplitude) > shared.SPEED_LIMIT_RAD_S:
            raise ValueError("planned target exceeds 61.6 rad/s; targets are never clipped")
        plan.append(segment)

    for repeat in range(1, args.repeats+1):
        add("as_found_pi", "zero", repeat)
        if args.suite in ("all", "steady"):
            cases = [sign*level for level in args.levels for sign in (-1, 1)]
            rng.shuffle(cases)
            for speed in cases:
                add("as_found_pi", "zero", repeat)
                add("as_found_pi", "steady", repeat, base=speed)
        if args.suite in ("all", "steps"):
            cases = [(base, sign*step) for base in args.step_bases for step in args.step_sizes for sign in (-1, 1)]
            rng.shuffle(cases)
            for base, step in cases:
                add("as_found_pi", "step", repeat, base=base, amplitude=step)
        if args.suite in ("all", "sine"):
            cases = [(b, a, f) for b in args.sine_bases for a in args.sine_amplitudes for f in args.frequencies]
            rng.shuffle(cases)
            for base, amplitude, frequency in cases:
                add("as_found_pi", "sine", repeat, base, amplitude, frequency)
        add("as_found_pi", "zero", repeat)
    return plan


def save_report(path, report):
    """Durable original settings/plan record; update only this run's JSON."""
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(shared.json_ready(report), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


class IdentificationIO(shared.VelocityTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_confirmed = False

    def receive(self):
        data = super().receive()
        if data[0] == 0xA3:
            self.position_at = self.clock.monotonic()
        return data

    def send(self, payload, *args, **kwargs):
        # Enforce read-only gains/settings even if a future helper changes.
        if payload[0] not in (0xA1, 0xA2, 0xA3, 0xAE, 0xB0, 0xB3, 0xB5, 0xB8, 0xB9, 0xC1, 0xCD, 0xCF):
            raise ValueError("opcode outside this experiment's read/C1/CD/CF contract")
        if len(payload) > 1 and payload[0] not in (0xC1, 0xCD):
            raise ValueError("B8/B9 and other motor settings are READ ONLY")
        if payload[0] == 0xC1:
            self.disable_confirmed = False
        return super().send(payload, *args, **kwargs)

    def disable(self):
        super().disable()
        self.disable_confirmed = True

    def check_feedback(self, now):
        if now-min(self.velocity_at, self.current_at) > self.args.feedback_timeout:
            raise RuntimeError("velocity/current feedback missing beyond configured timeout")
        if now-self.status_at > self.args.status_timeout:
            raise RuntimeError("driver status missing beyond configured timeout")

    def run_segment(self, segment):
        self.drain()
        self.method = segment.name
        self.phase_start = self.next_command = self.clock.monotonic()
        end, slot, count, skipped = self.phase_start+segment.duration_s, 0, 0, 0
        rate = (self.args.steady_feedback_rate if segment.kind in ("steady", "zero")
                else self.args.dynamic_feedback_rate)
        periods = {0xA1: 1/rate, 0xA2: 1/rate, 0xA3: .02, 0xAE: .1, 0xB8: 1., 0xB9: 1.}
        due = dict.fromkeys(periods, self.phase_start)
        state = dict(segment=asdict(segment), start_monotonic_s=self.phase_start,
                     initial_position_rad=self.position_rad, initial_velocity_rad_s=self.velocity,
                     initial_position_age_s=self.phase_start-getattr(self, "position_at", self.phase_start),
                     query_rates_hz={f"{op:02X}": 1/period for op, period in periods.items()})
        intervals, previous, lateness = [], None, 0.
        self.active = True
        try:
            while self.clock.monotonic() < end:
                if self.stop_requested():
                    raise KeyboardInterrupt
                self.log.check()
                now = self.clock.monotonic()
                if now >= end:
                    break
                self.check_feedback(now)
                if now >= self.next_command:
                    elapsed = now-self.phase_start
                    desired = segment.target(elapsed)
                    payload, encoded = shared.c1_payload(desired)
                    current_slot = max(slot, int(elapsed*self.args.packet_rate+1e-9))
                    omitted = current_slot-slot
                    sent = self.send(payload, scheduled_at=self.next_command, command_index=count,
                                     target_velocity_rad_s=desired, encoded_target_velocity_rad_s=encoded,
                                     target_sample_time_s=elapsed, skipped_slots=omitted)
                    lateness = max(lateness, sent-self.next_command)
                    if previous is not None:
                        intervals.append(sent-previous)
                    previous, count, skipped = sent, count+1, skipped+omitted
                    slot = max(current_slot+1,
                               int((self.clock.monotonic()-self.phase_start)*self.args.packet_rate+1e-9)+1)
                    extra = slot-current_slot-1
                    if extra:
                        skipped += extra
                        self.emit("event", skipped_slots=extra, note="Skipped slots during send; no catch-up burst")
                    self.next_command = min(end, self.phase_start+slot/self.args.packet_rate)
                for op, period in periods.items():
                    now = self.clock.monotonic()
                    if now >= due[op] and now < min(end, self.next_command):
                        self.send(bytes([op]))
                        due[op] = now+period
                self.drain()
                deadline = min(end, self.next_command, *due.values())
                shared.select.select([self.bus], [], [], max(0., min(.01, deadline-self.clock.monotonic())))
        finally:
            self.active = False
            state.update(actual_duration_s=self.clock.monotonic()-self.phase_start, commands=count,
                         skipped_slots=skipped, maximum_logged_lateness_s=lateness,
                         maximum_packet_interval_s=max(intervals) if intervals else None,
                         achieved_rate_hz=len(intervals)/sum(intervals) if intervals else None)
            self.phases.append(state)

def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    counts = {kind: sum(s.kind == kind for s in plan) for kind in ("steady", "step", "sine")}
    print(f"Rear ID 1: as-found PI cases {counts}; {args.repeats} repeats; "
          f"nominal {sum(s.duration_s for s in plan):.1f} s")
    print(f"C1 {args.packet_rate} Hz; A1/A2 steady {args.steady_feedback_rate}, dynamic "
          f"{args.dynamic_feedback_rate} Hz; B8/B9 strictly READ ONLY")
    if not args.execute:
        print("PREVIEW: no CAN/files. Add --execute with rear wheel airborne/free and other writers stopped.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path, plan_path = args.output.with_suffix(".json"), args.output.with_name(args.output.stem+"_plan.csv")
    if any(p.exists() for p in (args.output, report_path, plan_path, report_path.with_suffix(".json.tmp"))):
        raise FileExistsError("refusing to overwrite previous experiment files")
    with plan_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(plan[0])))
        writer.writeheader()
        writer.writerows(asdict(segment) for segment in plan)
    report = dict(schema="drive_c1_kp_identify_v1", completed=False, disabled_and_restored=False,
                  gains_read_only=True, error=None, motor_id=1, encoder_sign=-1, torque_sign=-1,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest(),
                  plan=[asdict(s) for s in plan])
    save_report(report_path, report)
    stopped = threading.Event()
    old_handlers = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    motor = bus = None

    def persist():
        if motor is not None:
            report.update(original_settings=motor.settings, applied_settings=motor.applied_settings,
                          raw_gains=motor.raw_gains.copy(), torque_constant_nm_per_a=motor.kt,
                          initial_multi_turn_count=motor.zero_count, phases=motor.phases)
        save_report(report_path, report)

    try:
        with shared.CsvLog(args.output) as log:
            print(f"CSV: {args.output}\nRear airborne/free; frame supported; steering disabled; other writers stopped.")
            try:
                bus = shared.open_bus(args.interface)
                motor = IdentificationIO(args, bus, log, stopped.is_set)
                motor.startup()
                for index, segment in enumerate(plan):
                    print(f"{index+1}/{len(plan)} {segment.name}: base {segment.base_rad_s:g}, "
                          f"amplitude {segment.amplitude_rad_s:g} rad/s, {segment.frequency_hz:g} Hz")
                    motor.run_segment(segment)
                report["completed"] = True
            except BaseException as error:
                report["execution_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                if motor is not None:
                    try:
                        motor.cleanup()
                        report.update(disabled_and_restored=True)
                    except BaseException as error:
                        report["cleanup_error"] = f"{type(error).__name__}: {error}"
                        raise
    except BaseException as error:
        report.update(completed=False, error=f"{type(error).__name__}: {error}")
        print(f"STOPPED: {report['error']}; partial logs retained", file=sys.stderr)
    finally:
        if bus is not None:
            try:
                bus.close()
            except Exception as error:
                report.update(completed=False, close_error=f"{type(error).__name__}: {error}")
                if report["error"] is None:
                    report["error"] = report["close_error"]
                print(f"CAN close error: {error}; recording outcome retained", file=sys.stderr)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        report["disable_confirmed"] = bool(motor and motor.disable_confirmed)
        persist()
    if report["completed"]:
        print("Completed; CF disable and original CD restoration verified. B8/B9/B3/B5/origins NEVER written.")
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
