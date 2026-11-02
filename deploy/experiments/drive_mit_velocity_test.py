#!/usr/bin/env python3
"""Rear MIT velocity-control tests: holds, steps and sines with adjustable P/I.

Preview by default; --execute runs rear ID 1 only. Needs adjacent
drive_c1_kp_identify.py and drive_velocity_sine.py, Python stdlib, no ROS.
--velocity-kp (alias --kd) is MIT Kd in Nm/(rad/s); MIT position Kp is exactly 0.
--velocity-ki is host integral gain in Nm/rad; default 0 preserves P-only tests.
Host integral torque is sent as MIT feedforward, with physical-limit anti-windup.
Feedforward quantization residual is compensated in velocity encoding only.
F0 Vel_Max is narrowed/read back, then original F0/CD restored after CF disable.
F0 persists in firmware: original settings are saved durably BEFORE any write.
Rear airborne/free, frame supported, steering disabled, all other writers stopped.
No angle/tracking/rest-speed/tight-lateness abort or hidden target ramp/clipping.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import csv
import hashlib
import math
from pathlib import Path
import signal
import struct
import sys
import threading

import drive_c1_kp_identify as c1
import drive_velocity_sine as shared

HERE = Path(__file__).resolve().parent
# Recorded v1 model_2500 startup/settling velocity peaks at 2.483943 rad/s.
# Full C1 identification also requests +/-10 and measured 12.687846 rad/s.
POLICY_RANGE_MIN_RAD_S = 3.0
RANGE_HEADROOM = 1.30


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--velocity-kp", "--kd", dest="velocity_kp", type=shared.finite, default=.30,
                   help="Velocity P = MIT Kd, Nm/(rad/s), default 0.30; position Kp stays zero")
    p.add_argument("--velocity-ki", type=shared.finite, default=0.,
                   help="Host velocity I, Nm/rad; default 0 is P-only; not the raw B9 gain")
    p.add_argument("--velocity-max", type=shared.finite,
                   help="F0 symmetric velocity range, rad/s; default max(3, 1.3*plan peak, encoding headroom)")
    p.add_argument("--suite", choices=("all", "steady", "steps", "sine"), default="all")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    for name, values in (
        ("levels", [.05, .1, .2, .4, .8, 1.5, 3., 6., 10.]),
        ("step-bases", [-6., -2., 0., 2., 6.]), ("step-sizes", [.1, .25, .5, 1.]),
        ("sine-bases", [-3., 0., 3.]), ("sine-amplitudes", [.25, 1.]),
        ("frequencies", [.5, 1., 2., 5., 10., 20.]),
    ):
        p.add_argument("--"+name, nargs="+", type=shared.finite, default=values)
    for name, value in (("steady-settle", 2.), ("steady-measure", 3.), ("step-base-hold", .6),
                        ("step-pulse-hold", .4), ("sine-settle", 1.), ("sine-min-measure", 4.),
                        ("feedback-timeout", 1.), ("status-timeout", 2.)):
        p.add_argument("--"+name, type=shared.finite, default=value)
    p.add_argument("--sine-cycles", type=int, default=4)
    p.add_argument("--packet-rate", type=int, choices=(50, 200, 400), default=200)
    p.add_argument("--steady-feedback-rate", type=int, choices=(100, 200, 400), default=100)
    p.add_argument("--dynamic-feedback-rate", type=int, choices=(100, 200, 400), default=400)
    p.add_argument("--interface", default="can0")
    p.add_argument("--note", default="", help="Fixture, load, supply or firmware differences")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    try:
        kd_encoding(a.velocity_kp)
    except ValueError as error:
        p.error(str(error))
    if a.velocity_ki < 0:
        p.error("velocity I must be nonnegative, in Nm/rad")
    if a.velocity_max is not None and not .01 <= a.velocity_max <= 655.35:
        p.error("F0 --velocity-max must be in [0.01,655.35] rad/s")
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
        a.output = HERE / "logs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_drive_mit_velocity_test.csv"
    if a.output.suffix != ".csv":
        p.error("output must end in .csv")
    return a


def make_plan(args):
    # Reuse exact randomization, waveforms, counts and durations; change labels only.
    mode = "mit_velocity_pi" if args.velocity_ki else "mit_velocity_p"
    return [replace(s, name=s.name.replace("as_found_pi", mode), mode=mode)
            for s in c1.make_plan(args)]


def kd_encoding(kd):
    """Validate the requested SI gain before quantization, without clipping."""
    if not math.isfinite(kd) or not 0 < kd <= 5:
        raise ValueError("velocity P/MIT Kd must be finite and in (0,5] Nm/(rad/s)")
    code = round(kd*4095/5)
    if code == 0:
        raise ValueError("velocity P is too small to encode a nonzero MIT Kd")
    return code, code*5/4095


def velocity_range(plan, requested=None, kd=.3, torque_max=0.):
    peak = max(abs(s.base_rad_s)+abs(s.amplitude_rad_s) for s in plan)
    _, actual_kd = kd_encoding(kd)
    # Signed 12-bit torque has no exact-zero code; leave space for its cancellation.
    compensation = torque_max/4095/actual_kd
    wanted = (max(POLICY_RANGE_MIN_RAD_S, RANGE_HEADROOM*peak, peak+compensation+.01)
              if requested is None else requested)
    code = math.ceil(wanted*100-1e-9)
    if not 1 <= code <= 65535 or code*.01 < peak+compensation:
        raise ValueError("F0 velocity range cannot encode all targets plus zero-torque quantization correction")
    return code*.01


def velocity_payload(target, measured, kp, ranges, integral_nm=0., integral_error_rad=0.):
    """Internal P + host I. Cancel only the FF quantization error in velocity.

    effective_P_target = wire_target + SIGN*(wire_ff - SIGN*integral_nm)/actual_Kd.
    Total user torque = actual_Kd*(effective_P_target - measured) + integral_nm.
    With I=0 this produces the exact previous P-only packet and correction.
    """
    pmax, vmax, tmax = ranges
    d, actual_kd = kd_encoding(kp)
    t = shared.encode(shared.SIGN*integral_nm, tmax, 12)
    ff = shared.decode(t, tmax, 12)
    residual = ff-shared.SIGN*integral_nm
    p = shared.encode(0., pmax, 16)  # Position irrelevant: Kp bits are exactly zero.
    v = shared.encode(shared.SIGN*target-residual/actual_kd, vmax, 12)
    wire_velocity = shared.decode(v, vmax, 12)
    encoded = shared.SIGN*wire_velocity
    effective = encoded+shared.SIGN*residual/actual_kd
    torque = actual_kd*(effective-measured)+integral_nm
    if not math.isfinite(torque) or abs(torque) > shared.TORQUE_LIMIT_NM:
        raise RuntimeError("estimated total MIT torque exceeds 6.28 Nm or feedback is invalid")
    packet = bytes((p >> 8, p & 255, v >> 4, (v & 15) << 4, 0,
                    d >> 4, ((d & 15) << 4) | (t >> 8), t & 255))
    fields = dict(encoded_target_velocity_rad_s=encoded, mit_kp=0., mit_kd=actual_kd,
                  mit_feedforward_native_nm=ff, integral_error_rad=integral_error_rad,
                  integral_torque_nm=integral_nm,
                  estimated_mit_total_torque_nm=torque)
    return packet, fields, effective


def open_bus(interface):
    bus = shared.open_bus(interface)
    try:
        # The protocol accepts both 0x401 and 0x501 for rear MIT commands.
        # Monitor both so a second legal MIT writer cannot evade detection.
        ids = (1, 0x101, 0x401, 0x501, 0, 0xFF, 0x400, 0x4FF)
        bus.setsockopt(shared.socket.SOL_CAN_RAW, shared.socket.CAN_RAW_FILTER,
                       b"".join(struct.pack("=II", cid, 0xC00007FF) for cid in ids))
        return bus
    except BaseException:
        bus.close()
        raise


class MitVelocityTestIO(c1.IdentificationIO):
    """Use the existing phase scheduler; replace C1 packets before any bus send."""

    def __init__(self, args, *values, persist=lambda: None, **kwargs):
        super().__init__(args, *values, **kwargs)
        self.plan, self.persist = make_plan(args), persist
        self.kd, self.ki = args.velocity_kp, args.velocity_ki
        self.command_ranges = None
        self.first_mit_at = None
        self.encoder_zero_at = None
        self.disable_confirmation_method = None
        self.maximum_effective_target_error_rad_s = 0.
        self.reset_integral()

    def reset_integral(self):
        self.integral = shared.Integral()
        self.integral_nm, self.integral_limited_updates = 0., 0

    def update_integral(self, target, stamp):
        if not self.ki:
            return
        error, previous = target-self.velocity, self.integral.error_rad
        proposed = self.ki*self.integral.update(error, stamp)
        _, kd = kd_encoding(self.kd)
        # Reserve only worst-case velocity quantization error, not an arbitrary torque margin.
        limit = shared.TORQUE_LIMIT_NM-kd*self.command_ranges[1]/4095
        i_limit = min(limit, self.command_ranges[2])
        candidate = max(-i_limit, min(i_limit, proposed))
        delta = candidate-self.ki*previous
        total = kd*error+candidate
        if (total > limit and delta > 0) or (total < -limit and delta < 0):
            candidate = self.ki*previous  # Freeze only accumulation further into saturation; allow unwinding.
        self.integral_limited_updates += int(candidate != proposed)
        self.integral.error_rad = candidate/self.ki
        self.integral_nm = candidate

    def run_segment(self, segment):
        self.reset_integral()
        before = len(self.phases)
        try:
            super().run_segment(segment)
        finally:
            if len(self.phases) > before:
                self.phases[-1].update(integral_reset_at_start=True,
                                      final_integral_error_rad=self.integral.error_rad,
                                      final_integral_torque_nm=self.integral_nm,
                                      integral_limited_updates=self.integral_limited_updates)

    def send(self, payload, can_id=0x101, **fields):
        if can_id != 0x101:
            raise ValueError("only this adapter may create rear MIT packets")
        if payload[0] == 0xC1 and len(payload) == 5:
            target = fields["target_velocity_rad_s"]
            if self.command_ranges is None:
                raise RuntimeError("MIT encoding ranges not configured")
            self.update_integral(target, self.clock.monotonic())
            packet, encoded_fields, effective = velocity_payload(
                target, self.velocity, self.kd, self.command_ranges, self.integral_nm, self.integral.error_rad)
            fields.update(encoded_fields)
            self.maximum_effective_target_error_rad_s = max(
                self.maximum_effective_target_error_rad_s, abs(effective-target))
            self.disable_confirmed = False
            self.disable_confirmation_method = None
            sent = shared.VelocityTest.send(self, packet, 0x501, **fields)
            if self.first_mit_at is None:
                self.first_mit_at = sent
            return sent
        if payload[0] not in (0xA1, 0xA2, 0xA3, 0xAE, 0xB0, 0xB3, 0xB5, 0xB8, 0xB9, 0xCD, 0xCF, 0xF0, 0xF1):
            raise ValueError("opcode outside MIT experiment contract")
        if len(payload) > 1 and payload[0] not in (0xCD, 0xF0):
            raise ValueError("B8/B9/B3/B5 and origins are READ ONLY")
        if len(payload) > 1 and not self.disable_confirmed:
            raise RuntimeError("F0/CD writes require confirmed disabled motor")
        return shared.VelocityTest.send(self, payload, can_id, **fields)

    def disable(self):
        self.active, self.disable_confirmed = False, False
        self.reset_integral()
        deadline = self.clock.monotonic()+1.
        while self.clock.monotonic() < deadline:
            try:
                status = self.request(0xCF, timeout=min(.1, deadline-self.clock.monotonic()))
                if status[6] == 0:
                    self.disable_confirmed, self.disable_confirmation_method = True, "CF mode 0"
                    return
                # Prior MIT hardware retains mode 5 after CF. Do not equate it with
                # disable without a subsequent inactive/fault-free F1 query.
                if status[6] == 5 and not status[7] and self.clock.monotonic() < deadline:
                    inactive = self.request(0xF1, timeout=min(.1, deadline-self.clock.monotonic()))
                    if inactive[6] == 0:
                        self.disable_confirmed, self.disable_confirmation_method = True, "CF mode 5 + F1 inactive"
                        return
            except TimeoutError:
                pass
            retry_at = min(deadline, self.clock.monotonic()+.05)
            while self.clock.monotonic() < retry_at:
                self.drain()
                shared.select.select([self.bus], [], [], max(0., retry_at-self.clock.monotonic()))
        raise RuntimeError("CF did not confirm disable (mode 0 or retained MIT mode with inactive F1) within 1 s")

    def configure_setting(self, wanted):
        reply = self.request(wanted[0], wanted)
        reread = self.request(wanted[0])
        if wanted != reply or wanted != reread:
            raise RuntimeError(f"{wanted[0]:02X} readback mismatch; correct ranges/watchdog not established")
        return reread

    def startup(self):
        self.cleaning = False
        self.disable()
        self.first_mit_at = None
        for op in (0xB0, 0xB8, 0xB9, 0xB3, 0xB5, 0xCD, 0xF0):
            self.settings[f"{op:02X}"] = self.request(op).hex().upper()
        original_f0 = bytes.fromhex(self.settings["F0"])
        vmax = velocity_range(self.plan, self.args.velocity_max, self.kd, self.ranges[2])
        wanted_f0 = original_f0[:3]+struct.pack("<H", round(vmax*100))+original_f0[5:]
        cd = bytes.fromhex(self.settings["CD"])
        if cd[1] not in (0, 1):
            raise ValueError("invalid original CD watchdog setting")
        timeout_ms = min(struct.unpack("<H", cd[2:4])[0], 500) if cd[1] else 500
        if timeout_ms <= 0:
            raise ValueError("invalid enabled CD watchdog timeout")
        wanted_cd = b"\xcd\x01"+struct.pack("<H", timeout_ms)+cd[4:]
        # Save originals even if the first persistent write succeeds but its reply is lost.
        self.persist()
        print(f"Original F0={self.settings['F0']} (saved to JSON before configuration)", flush=True)
        for original, wanted in ((original_f0, wanted_f0), (cd, wanted_cd)):
            if original != wanted:
                self.saved.append(original)
                self.persist()
                self.configure_setting(wanted)
            self.applied_settings[f"{wanted[0]:02X}"] = wanted.hex().upper()
            if wanted[0] == 0xF0:
                self.command_ranges = self.ranges
        # Start capture's software origin with the next A3 received, rather
        # than an earlier initialization reply. Never send a physical B1 reset.
        self.last_count = self.zero_count = None
        self.travel_counts, self.position_rad = 0, math.nan
        self.encoder_zero_at = None
        self.request(0xA3)
        self.encoder_zero_at = self.position_at
        self.emit("event", multi_turn_count=self.zero_count,
                  note="Capture software zero: A3 after configuration; no B1; measured velocity not re-zeroed")
        for op in (0xA2, 0xA1, 0xAE):
            self.request(op)
        # No synchronous JSON/fsync after enabling CD: slow Pi storage must not
        # consume the watchdog budget before the first motion packet.
        _, actual_kd = kd_encoding(self.kd)
        print(f"MIT Kp=0, velocity P/Kd={self.kd:g} (encoded {actual_kd:.9g}) Nm/(rad/s), "
              f"host Ki={self.ki:g} Nm/rad, requested FF=host integral torque")
        print(f"Verified F0 velocity +/-{self.ranges[1]:g} rad/s; resolution {2*self.ranges[1]/4095:.9g} rad/s")
        print("FF quantization residual is cancelled in velocity encoding; wire values and integral are recorded.")

    def configuration(self):
        if self.command_ranges is None:
            return None
        pmax, vmax, tmax = self.command_ranges
        code, actual_kd = kd_encoding(self.kd)
        ff = shared.decode(shared.encode(0., tmax, 12), tmax, 12)
        return dict(position_kp=0., velocity_kp_requested_nm_per_rad_s=self.kd,
                    velocity_kd_encoded_nm_per_rad_s=actual_kd, velocity_kd_code=code,
                    integral_gain_nm_per_rad=self.ki,
                    integral_update="Host packet rate, monotonic elapsed dt, trapezoidal A2 velocity error",
                    integral_reset="Each segment start and motor disable; retained within steps/sines",
                    anti_windup="I bounded by min(F0 torque range, 6.28 minus codec margin); "
                                "freeze increments that push P+I beyond physical torque headroom",
                    feedforward_requested="SIGN * host integral torque",
                    zero_integral_feedforward_encoded_native_nm=ff,
                    zero_integral_native_velocity_codec_offset_rad_s=-ff/actual_kd,
                    position_range_rad=pmax, velocity_range_rad_s=vmax, torque_range_nm=tmax,
                    velocity_resolution_rad_s=2*vmax/4095,
                    position_reference="Kp=0; absolute target irrelevant; first capture A3 is log origin only")

    def check_feedback(self, now):
        super().check_feedback(now)
        if self.first_mit_at is not None and now-self.first_mit_at > self.args.feedback_timeout:
            if now-self.f1_at > self.args.feedback_timeout or not self.f1_status & 1:
                raise RuntimeError("MIT feedback missing/stale or driver not in active MIT mode")

    def cleanup(self):
        self.cleaning, self.active = True, False
        try:
            self.disable()
        except Exception:
            self.disable()
        failures, pending = [], []
        for original in reversed(self.saved):
            try:
                self.configure_setting(original)
            except Exception as error:
                failures.append(f"{original[0]:02X}: {error}")
                pending.append(original)
        self.saved[:] = reversed(pending)
        if failures:
            raise RuntimeError("disabled, but settings restoration not confirmed: "+"; ".join(failures))
        self.emit("event", note="CF disabled confirmed; original F0/CD restored; B3/B5/B8/B9/origins unchanged")


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    planned_range = velocity_range(plan, args.velocity_max, args.velocity_kp)
    counts = {kind: sum(s.kind == kind for s in plan) for kind in ("steady", "step", "sine", "zero")}
    print(f"Rear ID 1 MIT P={args.velocity_kp:g} Nm/(rad/s), host I={args.velocity_ki:g} Nm/rad: "
          f"{counts}; {sum(s.duration_s for s in plan):g} s")
    print(f"MIT {args.packet_rate} Hz; A1/A2 steady {args.steady_feedback_rate}, dynamic "
          f"{args.dynamic_feedback_rate} Hz; planned F0 velocity +/-{planned_range:g} rad/s")
    if not args.execute:
        print("PREVIEW: no CAN/files. Actual F0 torque range determines final FF quantization headroom.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path, plan_path = args.output.with_suffix(".json"), args.output.with_name(args.output.stem+"_plan.csv")
    if any(p.exists() for p in (args.output, report_path, plan_path, report_path.with_suffix(".json.tmp"))):
        raise FileExistsError("refusing to overwrite previous experiment files")
    with plan_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(plan[0])))
        writer.writeheader()
        writer.writerows(asdict(s) for s in plan)
    report = dict(schema="drive_mit_velocity_test_v2", completed=False, disabled_and_restored=False,
                  gains_read_only=True, error=None, motor_id=1, encoder_sign=-1, torque_sign=-1,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scheduler_sha256=hashlib.sha256(Path(c1.__file__).read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest(),
                  control_law="MIT Kp=0; Kd=velocity_kp; host FF=velocity_ki*integral(target-A2) dt; "
                              "physical-limit anti-windup; FF quantization residual cancelled",
                  target_semantics="encoded_target_velocity is wire velocity; "
                                   "effective P target = wire + SIGN*(wire_ff-SIGN*integral_torque_nm)/Kd; "
                                   "total torque = Kd*(effective P target-A2)+integral_torque_nm",
                  plan=[asdict(s) for s in plan])
    c1.save_report(report_path, report)
    stopped = threading.Event()
    handlers = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    motor = bus = None

    def persist():
        if motor is not None:
            report.update(original_settings=motor.settings, applied_settings=motor.applied_settings,
                          raw_gains=motor.raw_gains.copy(), torque_constant_nm_per_a=motor.kt,
                          mit_encoding_ranges=motor.command_ranges, final_readback_ranges=motor.ranges,
                          mit_configuration=motor.configuration(),
                          disable_confirmation_method=motor.disable_confirmation_method,
                          initial_multi_turn_count=motor.zero_count,
                          encoder_zero_monotonic_s=motor.encoder_zero_at,
                          maximum_effective_target_error_rad_s=motor.maximum_effective_target_error_rad_s,
                          pending_restore_payloads=[p.hex().upper() for p in motor.saved], phases=motor.phases)
        c1.save_report(report_path, report)

    try:
        with shared.CsvLog(args.output) as log:
            print(f"CSV: {args.output}\nRear airborne/free; frame supported; steering disabled; other writers stopped.")
            try:
                bus = open_bus(args.interface)
                motor = MitVelocityTestIO(args, bus, log, stopped.is_set, persist=persist)
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
                        report["disabled_and_restored"] = True
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
                report["error"] = report["error"] or report["close_error"]
                print(f"CAN close error: {error}", file=sys.stderr)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        report["disable_confirmed"] = bool(motor and motor.disable_confirmed)
        try:
            persist()
        except Exception as error:
            report["completed"] = False
            print(f"Cannot finalize JSON: {error}; pre-write settings snapshot retained if saved", file=sys.stderr)
    if report["completed"]:
        print("Completed; CF disable/original F0/CD restoration verified. Wheel may coast.")
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
