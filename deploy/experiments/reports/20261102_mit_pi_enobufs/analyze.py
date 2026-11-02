"""Offline diagnosis of the 2026-11-02 MIT PI SocketCAN ENOBUFS stop."""

from collections import Counter, defaultdict, deque
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
STEM = "20261102_103508_240825_drive_mit_velocity_test"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    path = EXP/"logs"/(STEM+".csv")
    meta = json.loads(path.with_suffix(".json").read_text())
    last_phase = meta["phases"][-1]
    end = last_phase["start_monotonic_s"]+last_phase["actual_duration_s"]
    counts, pending, pairs = Counter(), defaultdict(deque), defaultdict(list)
    sends, receives, commands, statuses, tail = [], [], [], [], deque()
    peaks, last_rx, row_count = defaultdict(float), {}, 0
    kt = meta["torque_constant_nm_per_a"]
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            row_count += 1
            kind, op, t = row["record_type"], row["opcode"], float(row["host_monotonic_s"])
            counts[f"{kind}_{op}"] += 1
            if t >= end-1:
                tail.append(row)
            data = bytes.fromhex(row["payload_hex"])
            if kind in ("tx", "command"):
                sends.append((t, len(data)))
                pending["F1" if op == "MIT" else op].append(t)
            if kind == "command":
                commands.append(row)
                for k in ("estimated_mit_total_torque_nm", "integral_torque_nm", "velocity_rad_s"):
                    peaks[k] = max(peaks[k], abs(float(row[k])))
                kp = ((data[3] & 15) << 8) | data[4]
                kd = ((data[5] << 4) | (data[6] >> 4))*5/4095
                assert kp == 0 and kd == 246*5/4095
                native_v = (((data[2] << 4) | (data[3] >> 4))*2/4095-1)*13
                native_ff = ((((data[6] & 15) << 8) | data[7])*2/4095-1)*18
                torque = -(kd*(native_v+float(row["velocity_rad_s"]))+native_ff)
                assert abs(torque-float(row["estimated_mit_total_torque_nm"])) < 1e-10
            if kind != "rx":
                continue
            receives.append((t, len(data)))
            assert row["can_id"] == "001", row
            last_rx[op] = dict(time_s=t, payload=row["payload_hex"],
                               age_at_failure_ms=(end-t)*1000)
            if pending[op]:
                pairs[op].append((t, (t-pending[op].popleft())*1000))
            if op == "A1":
                torque = -struct.unpack("<i", data[1:])[0]*.001*kt
                assert abs(torque-float(row["current_torque_nm"])) < 1e-10
                peaks["fresh_a1_torque_nm"] = max(peaks["fresh_a1_torque_nm"], abs(torque))
            elif op == "A2":
                speed = -struct.unpack("<i", data[1:])[0]*.01*2*math.pi/60
                assert abs(speed-float(row["velocity_rad_s"])) < 1e-10
                peaks["fresh_a2_velocity_rad_s"] = max(peaks["fresh_a2_velocity_rad_s"], abs(speed))
            elif op in ("AE", "CF"):
                assert data[7] == 0
                statuses.append(dict(time_s=t, opcode=op, mode=data[6], fault=data[7],
                                     volts=int.from_bytes(data[1:3], "little")*.01, temperature_c=data[5]))
            elif op == "F1":
                assert data[6] == 1

    first = meta["phases"][0]["start_monotonic_s"]
    windows = []
    for seconds in (1., 5., 20.):
        tx = [x for x in sends if end-seconds <= x[0] < end]
        rx = [x for x in receives if end-seconds <= x[0] < end]
        # Conservative classical 11-bit CAN count incl. IFS and worst-case stuffing;
        # excludes retransmissions, error frames and any unobserved bus traffic.
        bits = sum(47+8*n+(33+8*n)//4 for _, n in tx+rx)
        rr = {}
        for op in ("A1", "A2", "A3", "AE", "F1"):
            delay = [d for t, d in pairs[op] if end-seconds <= t < end]
            rr[op] = dict(count=len(delay), median_ms=float(np.median(delay)),
                          p99_ms=float(np.quantile(delay, .99)), max_ms=max(delay))
        windows.append(dict(seconds=seconds, host_tx_hz=len(tx)/seconds, host_rx_hz=len(rx)/seconds,
                            successful_frames_worst_stuffed_load_at_1mbps=bits/(seconds*1e6),
                            fifo_pair_delays=rr))
    last = commands[-1]
    result = dict(stem=STEM, input_sha256={path.name:sha(path), path.with_suffix(".json").name:
                                         sha(path.with_suffix(".json"))},
                  source_sha256_matches_current=meta["source_sha256"]==sha(EXP/"drive_mit_velocity_test.py"),
                  rows=row_count, counts=dict(counts), phases=len(meta["phases"]),
                  completed_phases=sum(p["actual_duration_s"] >= p["segment"]["duration_s"]
                                       for p in meta["phases"]),
                  phase_kind_counts=dict(Counter(p["segment"]["kind"] for p in meta["phases"])),
                  motion_elapsed_s=end-first, estimated_failure_monotonic_s=end,
                  estimated_failure_unix_s=float(last["time_unix_s"])+end-float(last["host_monotonic_s"]),
                  last_phase=last_phase, last_command=last, last_rx=last_rx,
                  last_status=statuses[-1], status_count=len(statuses), peaks=dict(peaks),
                  windows=windows, pending_logged_queries={k:len(v) for k,v in pending.items() if v},
                  pending_logged_query_ages_ms={k:[(end-t)*1000 for t in v] for k,v in pending.items() if v},
                  command_count=len(commands), skipped_slots=sum(p["skipped_slots"] for p in meta["phases"]),
                  anti_windup_limited_updates=sum(p["integral_limited_updates"] for p in meta["phases"]),
                  cleanup={k:meta[k] for k in ("execution_error", "cleanup_error", "disable_confirmed",
                                              "disabled_and_restored", "pending_restore_payloads")},
                  assumptions="FIFO response pairing is descriptive host timing, not acquisition/wire latency. "
                              "Failure time estimated from phase finally timestamp. Failed sends are not logged.",
                  analysis_utc=datetime.now(timezone.utc).isoformat(), hardware_or_simulator_run=False)
    (HERE/"statistics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    with (HERE/"last_second.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tail[0]))
        writer.writeheader()
        writer.writerows(tail)
    print(json.dumps({k:v for k,v in result.items() if k not in ("counts", "last_phase", "windows", "last_rx")},
                     indent=2))
    for w in windows:
        print("WINDOW",json.dumps(w))


if __name__ == "__main__":
    main()
