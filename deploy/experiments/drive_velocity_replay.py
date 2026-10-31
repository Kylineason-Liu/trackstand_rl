#!/usr/bin/env python3
"""Replay simulated actual rear velocity through C1; automatically record feedback.

Pi: stdlib only, plus adjacent drive_velocity_sine.py for the tested CAN/logger.
Preview by default. PC plotting needs NumPy/Matplotlib. Targets are held at
50 Hz and resent at 200 Hz by default, with no interpolation or hidden scaling.
No angle, tracking-error, small-current or packet-lateness abort. Late slots are
logged/skipped. Existing physical speed/torque, driver/bus and log-failure trips
remain; feedback/status tolerances are widened to 1/2 seconds by default.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import threading

import drive_velocity_sine as shared


HERE = Path(__file__).resolve().parent
DEFAULT_TARGETS = HERE / "replay_targets/2500_isaac_velocity_startup.csv"


class Reference:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.times = [float(row["time_s"]) for row in rows]
        self.velocity = [float(row["velocity_rad_s"]) for row in rows]
        if len(rows) < 2 or abs(self.times[0]) > 1e-9:
            raise ValueError("reference must start at t=0 and have an endpoint sample")
        if not all(math.isfinite(x) for x in self.times + self.velocity):
            raise ValueError("reference contains a non-finite value")
        if any(abs(t - i * .02) > 1e-8 for i, t in enumerate(self.times)):
            raise ValueError("expected consecutive 50-Hz reference samples, including the endpoint")
        if max(map(abs, self.velocity)) > shared.SPEED_LIMIT_RAD_S:
            raise ValueError("reference exceeds 61.6 rad/s; no clipping/scaling is applied")
        self.duration = self.times[-1]
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def at(self, elapsed):
        index = max(0, min(len(self.times) - 1, bisect_right(self.times, elapsed + 1e-10) - 1))
        return self.velocity[index], self.times[index]


class ReplayIO(shared.VelocityTest):
    def receive(self):
        data = super().receive()
        if data[0] == 0xA3:
            self.position_at = self.clock.monotonic()
        return data

    def check_feedback(self, now):
        if now - min(self.velocity_at, self.current_at) > self.args.feedback_timeout:
            raise RuntimeError(f"no fresh velocity/current for {self.args.feedback_timeout:g} s")
        if now - self.status_at > self.args.status_timeout:
            raise RuntimeError(f"no fresh driver status for {self.args.status_timeout:g} s")

    def run_segment(self, name, duration, reference=None):
        if duration <= 0:
            return
        self.drain()
        # Individual missing queries during motion are tolerated; startup reads
        # still establish Kt, a multi-turn origin and status before any command.
        self.method, self.phase_start = name, self.clock.monotonic()
        self.next_command = self.phase_start
        end, slot, count, skipped_total = self.phase_start + duration, 0, 0, 0
        periods = {0xA2: .005, 0xA1: .01, 0xA3: .02, 0xAE: .1}
        due = {op: self.phase_start for op in periods}
        intervals, previous, peak_lateness = [], None, 0.
        initial_position, initial_velocity = self.position_rad, self.velocity
        initial_position_at = getattr(self, "position_at", None)
        self.active = True
        print(f"{name}: {duration:g} s, C1 {self.args.packet_rate:g} Hz")
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
                    elapsed = now - self.phase_start
                    target, sample_time = reference.at(elapsed) if reference is not None else (0., 0.)
                    payload, encoded = shared.c1_payload(target)
                    current_slot = max(slot, int(elapsed * self.args.packet_rate + 1e-9))
                    skipped = current_slot - slot
                    peak_lateness = max(peak_lateness, now - self.next_command)
                    stamp = self.send(payload, scheduled_at=self.next_command, command_index=count,
                                      target_velocity_rad_s=target, target_sample_time_s=sample_time,
                                      encoded_target_velocity_rad_s=encoded, skipped_slots=skipped)
                    if previous is not None:
                        intervals.append(stamp - previous)
                    previous, count = stamp, count + 1
                    skipped_total += skipped
                    slot = max(current_slot + 1,
                               int((self.clock.monotonic() - self.phase_start) * self.args.packet_rate + 1e-9) + 1)
                    skipped_during_send = slot - current_slot - 1
                    if skipped_during_send:
                        skipped_total += skipped_during_send
                        self.emit("event", skipped_slots=skipped_during_send,
                                  note="packet slots skipped during CAN send; no catch-up burst")
                    self.next_command = min(end, self.phase_start + slot / self.args.packet_rate)
                for op, period in periods.items():
                    now = self.clock.monotonic()
                    if now >= due[op] and now < min(end, self.next_command):
                        self.send(bytes([op]))
                        due[op] = now + period
                self.drain()
                deadline = min(end, self.next_command, *due.values())
                shared.select.select([self.bus], [], [], max(0., min(.01, deadline - self.clock.monotonic())))
        finally:
            self.active = False
            self.phases.append(dict(name=name, start_monotonic_s=self.phase_start,
                                    planned_duration_s=duration,
                                    actual_duration_s=self.clock.monotonic() - self.phase_start,
                                    commands=count, skipped_slots=skipped_total, max_lateness_s=peak_lateness,
                                    max_packet_interval_s=max(intervals) if intervals else None,
                                    achieved_rate_hz=1 / (sum(intervals) / len(intervals)) if intervals else None,
                                    initial_position_rad=initial_position, initial_velocity_rad_s=initial_velocity,
                                    initial_position_sample_time_s=initial_position_at,
                                    initial_position_age_s=(self.phase_start - initial_position_at
                                                            if initial_position_at is not None else None)))


def plot_run(log_path):
    """PC only: retain raw timestamps; compare against actual sent held targets."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_path = Path(log_path)
    report = json.loads(log_path.with_suffix(".json").read_text())
    reference = Reference(log_path.with_name(log_path.stem + "_reference.csv"))
    if reference.sha256 != report["reference_sha256"]:
        raise ValueError("saved reference hash differs from replay metadata")
    phase = next((p for p in report.get("phases", []) if p["name"] == "core"), None)
    if phase is None:
        raise ValueError("log contains no core replay; inspect its error in JSON")
    origin, duration = phase["start_monotonic_s"], phase["planned_duration_s"]
    with log_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if r["method"] == "core" and r["record_type"] == "command"]
    if not selected:
        raise ValueError("no replay commands were transmitted")
    command_t = np.array([float(r["host_monotonic_s"]) - origin for r in selected])
    command_v = np.array([float(r["encoded_target_velocity_rad_s"]) for r in selected])

    def fresh(opcode, key):
        chosen = [r for r in rows if r["method"] == "core" and r["record_type"] == "rx"
                  and r["opcode"] == opcode and r[key] and math.isfinite(float(r[key]))]
        return (np.array([float(r["host_monotonic_s"]) - origin for r in chosen]),
                np.array([float(r[key]) for r in chosen]))

    speed_t, speed = fresh("A2", "velocity_rad_s")
    torque_t, torque = fresh("A1", "current_torque_nm")
    position_t, position = fresh("A3", "position_rad")
    end = min(duration, phase["actual_duration_s"])
    valid = (speed_t >= command_t[0]) & (speed_t < end)
    indices = np.searchsorted(command_t, speed_t[valid], side="right") - 1
    errors = speed[valid] - command_v[indices]
    # Prefer the full physics-rate simulation curve when available on the PC.
    # The reference copied to the Pi remains the exact 50-Hz command sequence.
    provenance = report.get("source_provenance", {})
    native = HERE / "replay_targets" / Path(provenance.get("native_file", "missing")).name
    simulation_path = reference.path
    if native.is_file() and hashlib.sha256(native.read_bytes()).hexdigest() == provenance.get("native_sha256"):
        simulation_path = native
    with simulation_path.open(newline="") as stream:
        sim_rows = list(csv.DictReader(stream))
    sim_t = np.array([float(r["time_s"]) for r in sim_rows])
    sim_v = np.array([float(r["velocity_rad_s"]) for r in sim_rows])
    # Interpolation is for numerical comparison of asynchronous samples ONLY;
    # CAN targets are never interpolated, retimed or amplitude-adjusted.
    curve_error = speed[valid] - np.interp(speed_t[valid], sim_t, sim_v)
    metrics = dict(completed=report["completed"], comparison="fresh A2 versus last actually transmitted C1 target",
                   time_alignment="same core-start host clock; no fitted time shift or smoothing",
                   compared_samples=int(len(errors)), nominal_duration_s=duration,
                   rms_error_rad_s=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                   p95_absolute_error_rad_s=float(np.percentile(abs(errors), 95)) if len(errors) else None,
                   peak_absolute_error_rad_s=float(max(abs(errors))) if len(errors) else None,
                   simulation_curve=str(simulation_path), simulation_samples=len(sim_rows),
                   rms_error_vs_simulation_curve_rad_s=float(np.sqrt(np.mean(curve_error**2))) if len(errors) else None,
                   simulation_comparison="linear interpolation of saved states at A2 receipt times; no time shift",
                   peak_current_derived_torque_nm=float(max(abs(torque))) if len(torque) else None,
                   phases=report["phases"], execution_error=report.get("error"))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(sim_t, sim_v, color="black", lw=1, label="Simulation actual velocity")
    # Append the endpoint only to display the final held command; not a new TX.
    axes[0].step(np.r_[command_t, end], np.r_[command_v, command_v[-1]], where="post",
                 color="#e58b19", lw=.9, label="Actual transmitted C1 target (held)")
    axes[0].plot(speed_t, speed, color="#2478ba", lw=.7, label="Real motor A2 velocity")
    axes[0].set_ylabel("Velocity (rad/s)")
    axes[1].plot(torque_t, torque, color="#238b45", lw=.8, label="Real A1 current × B0 Kt")
    axes[1].set_ylabel("Torque estimate (Nm)")
    if "position_rad" in sim_rows[0]:
        axes[2].plot(sim_t, [float(r["position_rad"]) for r in sim_rows], color="black",
                     lw=.9, label="Simulation displacement")
    axes[2].plot(position_t, position - phase["initial_position_rad"], color="#2478ba", lw=.8,
                 label="Real A3 displacement from core-start snapshot")
    axes[2].set_ylabel("Displacement (rad)")
    axes[2].set_xlabel("Seconds from core replay start")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(loc="lower right")
        ax.axvline(0, color="red", lw=1)
        ax.margins(y=.12)
    axes[-1].set_xlim(0, duration)
    fig.suptitle(f"C1 rear velocity replay — {report.get('model', 'simulation reference')} — "
                 f"{report['packet_rate_hz']:g} Hz packets — {'complete' if report['completed'] else 'PARTIAL'}")
    fig.tight_layout()
    for suffix in (".png", ".svg"):
        fig.savefig(log_path.with_name(log_path.stem + "_comparison").with_suffix(suffix), dpi=170)
    plt.close(fig)
    metrics_path = log_path.with_name(log_path.stem + "_comparison.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Open CAN and run the motor; otherwise preview")
    mode.add_argument("--plot", type=Path, help="PC only: plot a previously recorded replay CSV")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--packet-rate", type=int, choices=(50, 200), default=200)
    parser.add_argument("--pre-zero", type=shared.finite, default=1., help="C1 zero-speed seconds before replay")
    parser.add_argument("--post-zero", type=shared.finite, default=1., help="C1 zero-speed seconds before CF disable")
    parser.add_argument("--feedback-timeout", type=shared.finite, default=1.)
    parser.add_argument("--status-timeout", type=shared.finite, default=2.)
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--note", default="", help="Record actual support/loading/supply conditions")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.pre_zero, args.post_zero) < 0 or min(args.feedback_timeout, args.status_timeout) <= 0:
        parser.error("pre/post times must be nonnegative; feedback timeouts must be positive")
    if args.output is None:
        args.output = HERE / "logs" / (
            f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.packet_rate}hz_drive_velocity_replay.csv")
    if args.output.suffix != ".csv":
        parser.error("--output must end in .csv")
    args.method = "c1"  # The shared startup never enters MIT or converts unknown B8/B9 units.
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.plot:
        return plot_run(args.plot)
    reference = Reference(args.targets)
    print(f"Rear ID 1 C1: simulation t=0..{reference.duration:g} s, held 50 Hz, packets {args.packet_rate} Hz")
    print(f"Reference speed [{min(reference.velocity):.6g}, {max(reference.velocity):.6g}] rad/s; + = forward")
    if not args.execute:
        print("PREVIEW: no CAN/files. Add --execute to replay and automatically record CSV/JSON.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(".json")
    reference_path = args.output.with_name(args.output.stem + "_reference.csv")
    if any(p.exists() for p in (args.output, meta_path, reference_path)):
        raise FileExistsError("refusing to overwrite previous replay files")
    source_meta = args.targets.with_suffix(".json")
    provenance = json.loads(source_meta.read_text()) if source_meta.exists() else {}
    if provenance.get("reference_sha256", reference.sha256) != reference.sha256:
        raise ValueError("reference differs from its recording manifest")
    reference_path.write_bytes(args.targets.read_bytes())
    report = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  reference_sha256=reference.sha256, source_provenance=provenance,
                  model=provenance.get("model", args.targets.name), source_sha256=hashlib.sha256(
                      Path(__file__).read_bytes()).hexdigest(), helper_sha256=hashlib.sha256(
                      Path(shared.__file__).read_bytes()).hexdigest(),
                  packet_rate_hz=args.packet_rate, target_rate_hz=50, motor_id=1, encoder_sign=-1,
                  completed=False, disabled_and_restored=False, error=None)
    stopped = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    bus = io = None
    try:
        with shared.CsvLog(args.output) as log:
            print(f"CSV: {args.output}\nRear wheel airborne/free; frame supported; "
                  "steering disabled; other writers stopped.")
            try:
                bus = shared.open_bus(args.interface)
                io = ReplayIO(args, bus, log, stopped.is_set)
                io.startup()
                io.run_segment("pre_zero", args.pre_zero)
                io.run_segment("core", reference.duration, reference)
                io.run_segment("post_zero", args.post_zero)
                report["completed"] = True
            except BaseException as error:
                report["execution_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                if io is not None:
                    io.cleanup()
                    report["disabled_and_restored"] = True
    except BaseException as error:
        report.update(completed=False, error=f"{type(error).__name__}: {error}")
        print(f"STOPPED: {report['error']}; partial logs retained", file=sys.stderr)
    finally:
        if bus is not None:
            bus.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if io is not None:
            report.update(phases=io.phases, torque_constant_nm_per_a=io.kt, raw_gains=io.raw_gains,
                          original_settings=io.settings, applied_settings=io.applied_settings,
                          initial_multi_turn_count=io.zero_count)
        meta_path.write_text(json.dumps(shared.json_ready(report), indent=2, allow_nan=False) + "\n")
    if report["completed"]:
        print("Completed; CF disable confirmed, CD restored; wheel may coast. B3/B5/gains/encoder origin unchanged.")
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
