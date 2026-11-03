#!/usr/bin/env python3
"""Plot a locomotive deployment session: matplotlib PNG pages + one interactive plotly HTML.

Input is a session folder written by ``locomotive.launch.py``
(``~/trackstand_logs/locomotive_YYYYmmdd_HHMMSS/`` with ``bag/`` and ``session.json``)
or a bare rosbag directory. Output goes to ``<session>/analysis/`` (``--out`` to change):

    summary.md / summary.json   headline measures (ride success, lean, tracking, saturation,
                                timing, faults), riding-only where stated
    01_overview.png ...         one matplotlib page per topic group (riding shaded green,
                                faults marked red)
    interactive.html            all time series in one tall, shared-time-axis plotly page plus
                                distributions; plotly.js embedded, works offline

Bag reading uses ``rosbag2_py`` (source ROS first) or, without ROS, the pure-Python
``rosbags`` package (``pip install rosbags``). Field names come from each telemetry
message's layout label, so added fields are picked up automatically.

    source /opt/ros/humble/setup.bash
    python3 deploy/experiments/plot_locomotive_session.py ~/trackstand_logs/locomotive_20261102_183000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "source" / "trackstand_deploy" / "models" / "locomotive_policy_manifest.json"
OBSERVATION_NAMES = ("gravity_y", "ang_vel_x", "ang_vel_y", "ang_vel_z", "speed", "steer", "vel_cmd",
                     "yaw_rate_cmd")
DEFAULT_STEER_CLIP_RAD = math.radians(20.0)
DEFAULT_REAR_CLIP_RAD_S = 45.0
DRIVE_TORQUE_LIMIT_NM = 6.28
STEERING_TORQUE_LIMIT_NM = 5.6
ROLL_GATE_DEG = 30.0  # Controller max_roll_rad 0.5236.
NUMERIC_TOPICS = {
    "/locomotive/observation": "observation", "/locomotive/raw_action": "raw_action",
    "/locomotive/motor_command": "motor_command", "/locomotive/policy_timing": "policy_timing",
    "/locomotive/rc_sticks": "sticks", "/trackstand/motor_control": "motor_control",
    "/trackstand/motor_state": "motor_state",
}
JSON_TOPICS = {
    "/locomotive/policy_event": "policy_event", "/locomotive/status": "status",
    "/trackstand/motor_event": "motor_event", "/trackstand/motor_status": "motor_status",
    "/trackstand/kickstands_status": "kickstands_status",
}
FIXED_NAMES = {
    "observation": OBSERVATION_NAMES, "raw_action": ("steering", "rear"),
    "motor_command": ("enable", "steering_target_rad", "rear_velocity_rad_s"), "sticks": ("throttle", "yaw"),
}
RIDING_COLOR, FAULT_COLOR = "#2ca02c", "#d62728"


# ---- reading ------------------------------------------------------------------------------
def _bag_directory(path: Path) -> Path:
    if (path / "metadata.yaml").is_file():
        return path
    if (path / "bag" / "metadata.yaml").is_file():
        return path / "bag"
    raise FileNotFoundError(f"no rosbag (metadata.yaml) in {path} or {path / 'bag'}")


def _storage_id(bag: Path) -> str:
    for line in (bag / "metadata.yaml").read_text().splitlines():
        if line.strip().startswith("storage_identifier:"):
            return line.split(":", 1)[1].strip()
    return ""


def iter_bag(bag: Path, topics: set[str]):
    """Yield (topic, receive time ns, message) for the selected topics."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        rosbag2_py = None
    if rosbag2_py is not None:
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=_storage_id(bag)),
                    rosbag2_py.ConverterOptions("cdr", "cdr"))
        types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
        classes = {name: get_message(kind) for name, kind in types.items() if name in topics}
        while reader.has_next():
            topic, data, stamp = reader.read_next()
            if topic in classes:
                yield topic, stamp, deserialize_message(data, classes[topic])
        return
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise SystemExit("Reading a bag needs ROS (`source /opt/ros/<distro>/setup.bash`) or "
                         "`pip install rosbags`") from error
    with AnyReader([bag]) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for connection, stamp, raw in reader.messages(connections=connections):
            yield connection.topic, stamp, reader.deserialize(raw, connection.msgtype)


@dataclass
class Table:
    """Time-stamped numeric samples with named columns (t in seconds from session start)."""

    t: np.ndarray
    columns: dict[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns.get(name, np.full(len(self.t), np.nan))

    def __contains__(self, name: str) -> bool:
        return name in self.columns

    def __len__(self) -> int:
        return len(self.t)


def _empty() -> Table:
    return Table(np.zeros(0), {})


@dataclass
class Session:
    path: Path
    meta: dict
    t0_ns: int
    tables: dict[str, Table] = field(default_factory=dict)
    events: dict[str, list[tuple[float, dict]]] = field(default_factory=dict)
    logs: list[tuple[float, int, str, str]] = field(default_factory=list)
    duration_s: float = 0.0

    def table(self, name: str) -> Table:
        return self.tables.get(name, _empty())


def _names_from_label(message, width: int, fallback) -> tuple:
    dims = getattr(getattr(message, "layout", None), "dim", None)
    if dims:
        label = str(dims[0].label)
        if ":" in label:
            names = tuple(label.split(":", 1)[1].split(","))
            if len(names) == width:
                return names
    if fallback and len(fallback) == width:
        return tuple(fallback)
    return tuple(f"c{i}" for i in range(width))


def load_session(path: Path) -> Session:
    path = path.expanduser().resolve()
    bag = _bag_directory(path)
    root = bag.parent if bag.name == "bag" else bag
    meta = {}
    if (root / "session.json").is_file():
        meta = json.loads((root / "session.json").read_text())
    wanted = set(NUMERIC_TOPICS) | set(JSON_TOPICS) | {
        "/trackstand/imu", "/trackstand/kickstands_up", "/trackstand/kickstands_command",
        "/trackstand/motor_fault", "/rosout"}
    numeric: dict[str, list] = {name: [] for name in NUMERIC_TOPICS.values()}
    names: dict[str, tuple] = {}
    imu, bools, json_events, logs = [], {"kickstands_up": [], "motor_fault": []}, {}, []
    kick_cmd = []
    first = None
    last = None
    for topic, stamp, message in iter_bag(bag, wanted):
        first = stamp if first is None else min(first, stamp)
        last = stamp if last is None else max(last, stamp)
        if topic in NUMERIC_TOPICS:
            key = NUMERIC_TOPICS[topic]
            data = np.asarray(message.data, dtype=float)
            if key not in names:
                names[key] = _names_from_label(message, len(data), FIXED_NAMES.get(key))
            if len(data) == len(names[key]):
                numeric[key].append((stamp, data))
        elif topic in JSON_TOPICS:
            try:
                json_events.setdefault(JSON_TOPICS[topic], []).append((stamp, json.loads(message.data)))
            except (TypeError, ValueError):
                json_events.setdefault(JSON_TOPICS[topic], []).append((stamp, {"raw": str(message.data)}))
        elif topic == "/trackstand/imu":
            header = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            q, w = message.orientation, message.angular_velocity
            imu.append((stamp, header, q.x, q.y, q.z, q.w, w.x, w.y, w.z))
        elif topic == "/trackstand/kickstands_command":
            kick_cmd.append((stamp, 1.0 if str(message.data).strip().lower() == "up" else 0.0))
        elif topic == "/rosout":
            if int(message.level) >= 30:  # WARN and above.
                logs.append((stamp, int(message.level), str(message.name), str(message.msg)))
        else:
            bools[topic.rsplit("/", 1)[-1]].append((stamp, float(bool(message.data))))
    if first is None:
        raise SystemExit(f"{bag}: no recognised locomotive topics in the bag")
    session = Session(root, meta, first, duration_s=(last - first) * 1e-9)
    to_s = lambda stamps: (np.asarray(stamps, dtype=np.int64) - first) * 1e-9
    for key, rows in numeric.items():
        if rows:
            values = np.vstack([row[1] for row in rows])
            session.tables[key] = Table(to_s([row[0] for row in rows]),
                                        {name: values[:, i] for i, name in enumerate(names[key])})
    if imu:
        a = np.asarray(imu, dtype=float)
        stamps = np.asarray([row[0] for row in imu], dtype=np.int64)
        headers = np.asarray([row[1] for row in imu], dtype=np.int64)
        roll, pitch, yaw = quaternion_to_rpy(a[:, 2:6])
        session.tables["imu"] = Table(to_s(stamps), {
            "roll_deg": np.degrees(roll), "pitch_deg": np.degrees(pitch), "yaw_deg": np.degrees(np.unwrap(yaw)),
            "gyro_x": a[:, 6], "gyro_y": a[:, 7], "gyro_z": a[:, 8],
            "header_age_ms": (stamps - headers) * 1e-6})
    for key, rows in list(bools.items()) + [("kickstands_command_up", kick_cmd)]:
        if rows:
            session.tables[key] = Table(to_s([r[0] for r in rows]), {"value": np.asarray([r[1] for r in rows])})
    for key, rows in json_events.items():
        session.events[key] = [((stamp - first) * 1e-9, data) for stamp, data in rows]
    session.logs = [((stamp - first) * 1e-9, level, name, text) for stamp, level, name, text in logs]
    return session


def quaternion_to_rpy(q_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ZYX roll/pitch/yaw from the bike-frame gravity vector the policy sees (gravity_y = -sin roll cos pitch)."""
    x, y, z, w = (q_xyzw[:, i] for i in range(4))
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    # Bike-frame gravity = R^T (0, 0, -1).
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    roll = np.arctan2(-gy, -gz)
    pitch = np.arcsin(np.clip(gx, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


# ---- derived state ------------------------------------------------------------------------
def mode_intervals(session: Session) -> list[tuple[float, float, str]]:
    """(start, end, mode) from policy events, falling back to the 10 Hz status mode."""
    samples = [(t, e["mode"]) for t, e in session.events.get("policy_event", []) if "mode" in e]
    samples += [(t, e["mode"]) for t, e in session.events.get("status", []) if "mode" in e]
    samples.sort()
    intervals = []
    for t, mode in samples:
        if intervals and intervals[-1][2] == mode:
            continue
        if intervals:
            intervals[-1] = (intervals[-1][0], t, intervals[-1][2])
        intervals.append((t, session.duration_s, mode))
    return intervals


def mask_for(t: np.ndarray, intervals, mode: str = "riding") -> np.ndarray:
    mask = np.zeros(len(t), dtype=bool)
    for start, end, name in intervals:
        if name == mode:
            mask |= (t >= start) & (t < end)
    return mask


def fault_markers(session: Session) -> list[tuple[float, str]]:
    marks = []
    for t, event in session.events.get("motor_event", []):
        if event.get("fault_latched") and event.get("fault_reason"):
            marks.append((t, f"motor: {event['fault_reason']}"))
    for t, event in session.events.get("policy_event", []):
        if event.get("mode") == "fault_latched":
            marks.append((t, "policy: " + ", ".join(event.get("latched_fault_reasons") or ["fault"])))
    deduplicated, seen = [], set()
    for t, text in sorted(marks):
        if text not in seen:
            deduplicated.append((t, text))
            seen.add(text)
    return deduplicated


def model_limits(session: Session) -> tuple[float, float, str]:
    model = session.meta.get("model_filename")
    if not model:
        for _, status in session.events.get("status", []):
            model = status.get("model") or model
    try:
        entries = json.loads(MANIFEST.read_text())["models"]
        contract = next(e["contract"] for e in entries if e["onnx"] == model)
        return (float(contract["steering_action_clip_rad"][1]), float(contract["drive_action_clip_rad_s"][1]),
                str(model))
    except (OSError, StopIteration, KeyError, ValueError):
        return DEFAULT_STEER_CLIP_RAD, DEFAULT_REAR_CLIP_RAD_S, str(model or "unknown")


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "mean": float(np.mean(values)), "rms": float(np.sqrt(np.mean(values ** 2))),
            "p50": float(np.percentile(values, 50)), "p99": float(np.percentile(values, 99)),
            "max_abs": float(np.max(np.abs(values))), "min": float(np.min(values)), "max": float(np.max(values))}


def _fraction(flags: np.ndarray) -> float | None:
    flags = np.asarray(flags, dtype=float)
    flags = flags[np.isfinite(flags)]
    return float(np.mean(flags > 0.5)) if len(flags) else None


def summarize(session: Session) -> dict:
    intervals = mode_intervals(session)
    steer_clip, rear_clip, model = model_limits(session)
    rides = [(s, e) for s, e, m in intervals if m == "riding"]
    ride_ends = []
    for start, end in rides:
        following = next(((s, m) for s, e, m in intervals if s >= end and m != "riding"), (end, "end_of_bag"))
        ride_ends.append({"start_s": start, "duration_s": end - start, "ended_by": following[1]})
    imu, obs, cmd = session.table("imu"), session.table("observation"), session.table("motor_command")
    control, timing = session.table("motor_control"), session.table("policy_timing")
    riding_imu = mask_for(imu.t, intervals)
    riding_cmd = mask_for(cmd.t, intervals) & (cmd["enable"] > 0.5)
    gate = control["gate_enabled"] > 0.5
    roll = imu["roll_deg"][riding_imu]
    summary = {
        "session": str(session.path), "model": model, "duration_s": session.duration_s,
        "meta": {k: session.meta.get(k) for k in ("started", "rear_control_mode", "drive_mit_kd_nm_s_per_rad",
                                                  "dry_run", "git")},
        "rides": {"count": len(rides), "total_riding_s": float(sum(e - s for s, e in rides)),
                  "longest_s": float(max((e - s for s, e in rides), default=0.0)), "segments": ride_ends},
        "faults": [{"t_s": t, "reason": text} for t, text in fault_markers(session)],
        "lean_riding": {"roll_deg": _stats(roll),
                        "time_above_5deg": _fraction(np.abs(roll) > 5.0) if len(roll) else None,
                        "time_above_10deg": _fraction(np.abs(roll) > 10.0) if len(roll) else None,
                        "peak_margin_to_gate_deg": ROLL_GATE_DEG - float(np.max(np.abs(roll))) if len(roll) else None},
        "tracking_riding": {
            "speed_error_m_s": _stats(obs["speed"] - obs["vel_cmd"]),
            "yaw_rate_error_rad_s": _stats(obs["ang_vel_z"] - obs["yaw_rate_cmd"]),
            "steering_error_rad": _stats(control["steering_error_rad"][gate]),
            "rear_speed_error_rad_s": _stats((control["drive_velocity_target_rad_s"]
                                              - control["drive_velocity_rad_s"])[gate]),
        },
        "saturation_riding": {
            "steering_clip_rad": steer_clip, "rear_clip_rad_s": rear_clip,
            "steering_at_clip": _fraction(np.abs(cmd["steering_target_rad"][riding_cmd]) >= steer_clip - 1e-6),
            "rear_at_clip": _fraction(np.abs(cmd["rear_velocity_rad_s"][riding_cmd]) >= rear_clip - 1e-6),
            "rear_torque_limited": _fraction(control["drive_mit_torque_limited"][gate]),
            "raw_action_steering_abs_gt_1": _fraction(np.abs(session.table("raw_action")["steering"]) > 1.0),
        },
        "timing": {
            "policy_cycle_ms": _stats(timing["cycle_interval_ms"]),
            "inference_ms": _stats(timing["inference_ms"]),
            "motor_cycle_ms": _stats(control["cycle_interval_ms"]),
            "motor_work_ms": _stats(control["work_duration_ms"]),
            "imu_age_ms": _stats(timing["imu_age_ms"]), "motor_age_ms": _stats(timing["motor_age_ms"]),
            "rc_age_ms": _stats(timing["rc_age_ms"]), "imu_header_to_bag_ms": _stats(imu["header_age_ms"]),
            "drive_mit_f1_age_ms": _stats(control["drive_mit_f1_age_ms"]),
        },
        "torque": {
            "steering_measured_nm": _stats(session.table("motor_state")["steering_torque_nm"]),
            "drive_measured_nm": _stats(session.table("motor_state")["drive_torque_nm"]),
            "drive_mit_estimated_nm": _stats(control["drive_mit_estimated_torque_nm"][gate]),
        },
        "warnings_and_errors": [{"t_s": t, "level": level, "node": name, "msg": text[:300]}
                                for t, level, name, text in session.logs[:200]],
    }
    return summary


def _fmt(stats: dict, key: str = "rms", scale: float = 1.0, digits: int = 3) -> str:
    return "n/a" if not stats or stats.get("n", 0) == 0 else f"{stats[key] * scale:.{digits}g}"


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def summary_markdown(s: dict) -> str:
    lean, track, sat, timing = s["lean_riding"], s["tracking_riding"], s["saturation_riding"], s["timing"]
    rows = [
        ("Rides / total riding / longest", f"{s['rides']['count']} / {s['rides']['total_riding_s']:.1f} s / "
                                           f"{s['rides']['longest_s']:.1f} s"),
        ("Faults latched", str(len(s["faults"]))),
        ("Roll RMS / peak (riding)", f"{_fmt(lean['roll_deg'])}° / {_fmt(lean['roll_deg'], 'max_abs')}°"),
        ("Time with roll beyond ±5° / ±10°", f"{_pct(lean['time_above_5deg'])} / {_pct(lean['time_above_10deg'])}"),
        ("Speed error RMS (v − v*)", f"{_fmt(track['speed_error_m_s'])} m/s"),
        ("Yaw-rate error RMS", f"{_fmt(track['yaw_rate_error_rad_s'])} rad/s"),
        ("Steering tracking RMS", f"{_fmt(track['steering_error_rad'], scale=180 / math.pi)}°"),
        ("Rear speed tracking RMS", f"{_fmt(track['rear_speed_error_rad_s'])} rad/s"),
        ("Steering at clip / rear at clip", f"{_pct(sat['steering_at_clip'])} / {_pct(sat['rear_at_clip'])}"),
        ("Rear torque-limited (MIT)", _pct(sat["rear_torque_limited"])),
        ("Policy cycle p50 / p99 / max", f"{_fmt(timing['policy_cycle_ms'], 'p50')} / "
                                         f"{_fmt(timing['policy_cycle_ms'], 'p99')} / "
                                         f"{_fmt(timing['policy_cycle_ms'], 'max')} ms"),
        ("Inference p99 / max", f"{_fmt(timing['inference_ms'], 'p99')} / {_fmt(timing['inference_ms'], 'max')} ms"),
        ("Motor cycle p99 / max", f"{_fmt(timing['motor_cycle_ms'], 'p99')} / "
                                  f"{_fmt(timing['motor_cycle_ms'], 'max')} ms"),
        ("IMU / motor / RC age max", f"{_fmt(timing['imu_age_ms'], 'max')} / {_fmt(timing['motor_age_ms'], 'max')}"
                                     f" / {_fmt(timing['rc_age_ms'], 'max')} ms"),
    ]
    lines = [f"# Locomotive session {Path(s['session']).name}", "",
             f"Model `{s['model']}`, duration {s['duration_s']:.1f} s, rear mode "
             f"`{s['meta'].get('rear_control_mode')}`, dry_run `{s['meta'].get('dry_run')}`.", "",
             "| Measure | Value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
    if s["faults"]:
        lines += ["", "## Faults", ""] + [f"- {f['t_s']:.2f} s: {f['reason']}" for f in s["faults"]]
    if s["rides"]["segments"]:
        lines += ["", "## Ride segments", ""] + [
            f"- {r['start_s']:.2f} s for {r['duration_s']:.2f} s, ended by `{r['ended_by']}`"
            for r in s["rides"]["segments"]]
    if s["warnings_and_errors"]:
        lines += ["", "## Node warnings / errors (first 200)", ""] + [
            f"- {w['t_s']:.2f} s [{w['node']}] {w['msg']}" for w in s["warnings_and_errors"]]
    return "\n".join(lines) + "\n"


# ---- plot specification (shared by matplotlib and plotly) ---------------------------------
@dataclass
class Trace:
    table: str
    column: str
    label: str
    scale: float = 1.0
    step: bool = False
    style: str = "-"


@dataclass
class Panel:
    title: str
    unit: str
    traces: list[Trace]
    hlines: tuple = ()


def page_specs(session: Session) -> list[tuple[str, list[Panel]]]:
    deg = 180.0 / math.pi
    steer_clip, rear_clip, _ = model_limits(session)
    return [
        ("Overview", [
            Panel("Mode / enable", "state", [Trace("mode", "code", "mode (0 down, 1 launch, 2 riding, 3 fault)",
                                                   step=True),
                                             Trace("kickstands_up", "value", "RC enable (CH7 up)", step=True),
                                             Trace("kickstands_command_up", "value", "stands commanded up", step=True),
                                             Trace("motor_control", "gate_enabled", "motor gate", step=True)]),
            Panel("Speed", "m/s", [Trace("speed", "value", "measured speed (0.07 × rear)"),
                                   Trace("observation", "vel_cmd", "v* (policy input)", style="--"),
                                   Trace("status", "launch_speed_m_s", "launch ramp", style=":")]),
            Panel("Roll (IMU)", "deg", [Trace("imu", "roll_deg", "roll")], hlines=(-ROLL_GATE_DEG, ROLL_GATE_DEG)),
            Panel("RC sticks", "[-1, 1]", [Trace("sticks", "throttle", "throttle"), Trace("sticks", "yaw", "yaw")]),
        ]),
        ("Balance", [
            Panel("Roll / pitch", "deg", [Trace("imu", "roll_deg", "roll"), Trace("imu", "pitch_deg", "pitch")],
                  hlines=(-5, 5)),
            Panel("Body rates", "rad/s", [Trace("imu", "gyro_x", "roll rate ω_x"),
                                          Trace("imu", "gyro_y", "pitch rate ω_y"),
                                          Trace("imu", "gyro_z", "yaw rate ω_z")]),
            Panel("Heading (IMU yaw, unwrapped)", "deg", [Trace("imu", "yaw_deg", "yaw")]),
            Panel("Policy input gravity_y", "-", [Trace("observation", "gravity_y", "gravity_y (= −sin roll)")]),
        ]),
        ("Tracking", [
            Panel("Speed tracking (riding)", "m/s", [Trace("observation", "speed", "speed"),
                                                     Trace("observation", "vel_cmd", "v*", style="--")]),
            Panel("Speed error v − v*", "m/s", [Trace("derived", "speed_error", "v − v*")], hlines=(0,)),
            Panel("Yaw-rate tracking (riding)", "rad/s", [Trace("observation", "ang_vel_z", "ω_z"),
                                                          Trace("observation", "yaw_rate_cmd", "ω_z*", style="--")]),
            Panel("Yaw-rate error", "rad/s", [Trace("derived", "yaw_error", "ω_z − ω_z*")], hlines=(0,)),
        ]),
        ("Policy actions", [
            Panel("Raw actions", "-", [Trace("raw_action", "steering", "a0 steering"),
                                       Trace("raw_action", "rear", "a1 rear")], hlines=(-1, 1)),
            Panel("Steering target", "deg", [Trace("motor_command", "steering_target_rad", "commanded", scale=deg)],
                  hlines=(-math.degrees(steer_clip), math.degrees(steer_clip))),
            Panel("Rear speed target", "rad/s", [Trace("motor_command", "rear_velocity_rad_s", "commanded")],
                  hlines=(-rear_clip, rear_clip)),
            Panel("Command enable", "0/1", [Trace("motor_command", "enable", "enable", step=True)]),
        ]),
        ("Steering actuator", [
            Panel("Steering position", "deg", [
                Trace("motor_control", "slewed_steering_target_rad", "target", scale=deg, style="--"),
                Trace("motor_control", "steering_position_rad", "measured", scale=deg)]),
            Panel("Steering error", "deg", [Trace("motor_control", "steering_error_rad", "target − measured",
                                                  scale=deg)], hlines=(0,)),
            Panel("Steering velocity", "rad/s", [Trace("motor_control", "steering_velocity_rad_s", "measured")]),
            Panel("Steering torque (measured A1 × Kt)", "N·m", [
                Trace("motor_state", "steering_torque_nm", "measured"),
                Trace("motor_control", "mit_feedback_torque_motor_nm", "MIT F1 feedback", style=":")],
                  hlines=(-STEERING_TORQUE_LIMIT_NM, STEERING_TORQUE_LIMIT_NM)),
        ]),
        ("Rear actuator", [
            Panel("Rear wheel speed", "rad/s", [
                Trace("motor_control", "drive_velocity_target_rad_s", "requested target", style="--"),
                Trace("motor_control", "drive_mit_effective_target_rad_s", "MIT effective (torque-bounded)",
                      style=":"),
                Trace("motor_control", "drive_velocity_rad_s", "measured (A2)")]),
            Panel("Rear speed error", "rad/s", [Trace("derived", "rear_error", "target − measured")], hlines=(0,)),
            Panel("Rear torque", "N·m", [Trace("motor_control", "drive_mit_estimated_torque_nm", "MIT predicted"),
                                         Trace("motor_state", "drive_torque_nm", "measured A1 × Kt")],
                  hlines=(-DRIVE_TORQUE_LIMIT_NM, DRIVE_TORQUE_LIMIT_NM)),
            Panel("Rear limits / C1", "flag / rad/s", [
                Trace("motor_control", "drive_mit_torque_limited", "MIT torque-limited", step=True),
                Trace("motor_control", "drive_c1_encoded_velocity_rad_s", "C1 encoded target")]),
        ]),
        ("Timing", [
            Panel("Policy loop", "ms", [Trace("policy_timing", "cycle_interval_ms", "cycle interval"),
                                        Trace("policy_timing", "inference_ms", "inference"),
                                        Trace("policy_timing", "work_duration_ms", "work")], hlines=(20,)),
            Panel("Motor loop", "ms", [Trace("motor_control", "cycle_interval_ms", "cycle interval"),
                                       Trace("motor_control", "work_duration_ms", "work"),
                                       Trace("motor_control", "mit_actual_send_interval_ms", "MIT send interval")],
                  hlines=(5,)),
            Panel("Input ages at policy", "ms", [Trace("policy_timing", "imu_age_ms", "IMU (limit 60)"),
                                                 Trace("policy_timing", "motor_age_ms", "motor (limit 60)"),
                                                 Trace("policy_timing", "rc_age_ms", "RC (limit 150)")],
                  hlines=(60, 150)),
            Panel("Feedback ages at motor node", "ms", [
                Trace("motor_control", "steering_pair_age_ms", "steering A3/A2 pair"),
                Trace("motor_control", "drive_pair_age_ms", "drive A3/A2 pair"),
                Trace("motor_control", "drive_mit_f1_age_ms", "drive MIT F1"),
                Trace("imu", "header_age_ms", "IMU header → bag")], hlines=(50,)),
        ]),
        ("Health", [
            Panel("Gate / outputs / fault", "0/1", [Trace("motor_control", "gate_enabled", "gate", step=True),
                                                    Trace("motor_control", "outputs_active", "outputs active",
                                                          step=True),
                                                    Trace("motor_control", "fault_latched", "motor fault",
                                                          step=True),
                                                    Trace("motor_fault", "value", "fault topic", step=True)]),
            Panel("Query failures (cumulative)", "count", [
                Trace("motor_control", "steering_position_query_failures", "steering A3"),
                Trace("motor_control", "steering_velocity_query_failures", "steering A2"),
                Trace("motor_control", "drive_position_query_failures", "drive A3"),
                Trace("motor_control", "drive_velocity_query_failures", "drive A2")]),
            Panel("Policy readiness", "0/1", [Trace("policy_timing", "ready", "ready", step=True),
                                              Trace("policy_timing", "policy_enabled", "riding", step=True),
                                              Trace("policy_timing", "inference_success", "inference ok",
                                                    step=True)]),
            Panel("Send durations", "ms", [Trace("motor_control", "steering_send_duration_ms", "steering"),
                                           Trace("motor_control", "drive_send_duration_ms", "drive")]),
        ]),
    ]


HISTOGRAMS = (
    ("imu", "roll_deg", "Roll while riding (deg)", True),
    ("observation", "steer", "Steering angle while riding (rad)", False),
    ("raw_action", "steering", "Raw action a0 (steering)", False),
    ("raw_action", "rear", "Raw action a1 (rear)", False),
    ("derived", "speed_error", "Speed error v − v* (m/s)", False),
    ("derived", "rear_error", "Rear speed error (rad/s, gate on)", False),
    ("policy_timing", "cycle_interval_ms", "Policy cycle interval (ms)", False),
    ("motor_control", "cycle_interval_ms", "Motor cycle interval (ms)", False),
)
MODE_CODES = {"down": 0.0, "launch": 1.0, "riding": 2.0, "fault_latched": 3.0}


def add_derived(session: Session) -> None:
    intervals = mode_intervals(session)
    if intervals:
        t = np.asarray([s for s, _, _ in intervals] + [intervals[-1][1]])
        code = np.asarray([MODE_CODES.get(m, np.nan) for _, _, m in intervals] + [MODE_CODES.get(intervals[-1][2])])
        session.tables["mode"] = Table(t, {"code": code})
    status = session.events.get("status", [])
    if status:
        session.tables["status"] = Table(np.asarray([t for t, _ in status]), {
            key: np.asarray([float(e.get(key)) if isinstance(e.get(key), (int, float)) else np.nan
                             for _, e in status]) for key in ("launch_speed_m_s", "speed_m_s", "vel_cmd_m_s")})
        session.tables["speed"] = Table(session.tables["status"].t, {"value": session.tables["status"]["speed_m_s"]})
    obs, control = session.table("observation"), session.table("motor_control")
    derived_t = obs.t if len(obs) else np.zeros(0)
    columns = {"speed_error": obs["speed"] - obs["vel_cmd"], "yaw_error": obs["ang_vel_z"] - obs["yaw_rate_cmd"]}
    session.tables["derived"] = Table(derived_t, columns)
    if len(control):
        gate = control["gate_enabled"] > 0.5
        rear = np.where(gate, control["drive_velocity_target_rad_s"] - control["drive_velocity_rad_s"], np.nan)
        # Rear error lives on the motor-control clock; keep it in its own table.
        session.tables["derived_rear"] = Table(control.t, {"rear_error": rear})


def _resolve(session: Session, trace: Trace) -> tuple[np.ndarray, np.ndarray] | None:
    table = session.tables.get("derived_rear") if trace.column == "rear_error" else session.tables.get(trace.table)
    if table is None or not len(table) or trace.column not in table:
        return None
    t, y = table.t, table[trace.column] * trace.scale
    if not np.any(np.isfinite(y)):
        return None
    if trace.step and t[-1] < session.duration_s:
        # Transition-only topics (one message per change): hold the last state to the session end.
        t, y = np.append(t, session.duration_s), np.append(y, y[-1])
    return t, y


def _decimate(x: np.ndarray, y: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope per time bucket: keeps spikes (timing outliers, torque peaks) that striding drops."""
    if limit <= 0 or len(x) <= limit:
        return x, y
    buckets = max(1, limit // 2)
    edges = np.linspace(0, len(x), buckets + 1).astype(int)
    keep = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        chunk = y[start:end]
        if not np.any(np.isfinite(chunk)):
            keep.append(start)
            continue
        low, high = start + int(np.nanargmin(chunk)), start + int(np.nanargmax(chunk))
        keep.extend(sorted({low, high}))
    keep = np.asarray(keep)
    return x[keep], y[keep]


# ---- matplotlib ---------------------------------------------------------------------------
def write_matplotlib(session: Session, pages, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intervals = mode_intervals(session)
    faults = fault_markers(session)
    written = []
    for index, (name, panels) in enumerate(pages, start=1):
        fig, axes = plt.subplots(len(panels), 1, figsize=(15, 3.2 * len(panels)), sharex=True, squeeze=False)
        for axis, panel in zip(axes[:, 0], panels):
            for start, end, mode in intervals:
                if mode == "riding":
                    axis.axvspan(start, end, color=RIDING_COLOR, alpha=0.08, lw=0)
            for t, _ in faults:
                axis.axvline(t, color=FAULT_COLOR, lw=1.0, ls="--", alpha=0.8)
            for level in panel.hlines:
                axis.axhline(level, color="0.5", lw=0.8, ls=":")
            plotted = 0
            for trace in panel.traces:
                resolved = _resolve(session, trace)
                if resolved is None:
                    continue
                x, y = resolved
                if trace.step:
                    axis.step(x, y, where="post", label=trace.label, lw=1.2)
                else:
                    axis.plot(x, y, trace.style, label=trace.label, lw=1.0)
                plotted += 1
            axis.set_title(panel.title, loc="left", fontsize=10)
            axis.set_ylabel(panel.unit)
            axis.grid(True, alpha=0.3)
            if plotted:
                axis.legend(loc="upper right", fontsize=8, ncol=min(plotted, 4))
            else:
                axis.text(0.5, 0.5, "no data in this session", transform=axis.transAxes, ha="center",
                          va="center", color="0.5")
        axes[-1, 0].set_xlabel("time since session start (s)")
        fig.suptitle(f"{session.path.name} — {name}  (green = riding, red dashed = fault)", fontsize=12)
        fig.tight_layout()
        path = out / f"{index:02d}_{name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    riding = mode_intervals(session)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for axis, (table, column, title, riding_only) in zip(axes.flat, HISTOGRAMS):
        resolved = _resolve(session, Trace(table, column, title))
        if resolved is None:
            axis.text(0.5, 0.5, "no data", transform=axis.transAxes, ha="center", va="center", color="0.5")
        else:
            x, y = resolved
            if riding_only:
                y = y[mask_for(x, riding)]
            y = y[np.isfinite(y)]
            if len(y):
                axis.hist(y, bins=60, color="#1f77b4", alpha=0.85)
        axis.set_title(title, fontsize=10)
        axis.grid(True, alpha=0.3)
    fig.suptitle(f"{session.path.name} — distributions", fontsize=12)
    fig.tight_layout()
    path = out / f"{len(pages) + 1:02d}_distributions.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(path)
    return written


# ---- plotly -------------------------------------------------------------------------------
PLOTLY_PLOT_HEIGHT_PX = 300  # Drawing area per subplot: tall enough to read each trace.
PLOTLY_GAP_PX = 80  # Title + tick labels between subplots.


def write_plotly(session: Session, pages, summary: dict, out: Path, max_points: int) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels = [(page, panel) for page, page_panels in pages for panel in page_panels]
    titles = [f"{page} — {panel.title} [{panel.unit}]" for page, panel in panels]
    height = len(panels) * (PLOTLY_PLOT_HEIGHT_PX + PLOTLY_GAP_PX) + 200
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=PLOTLY_GAP_PX / height,
                        subplot_titles=titles)
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2")
    for row, (_, panel) in enumerate(panels, start=1):
        for index, trace in enumerate(panel.traces):
            resolved = _resolve(session, trace)
            if resolved is None:
                continue
            x, y = _decimate(*resolved, max_points)
            dash = {"--": "dash", ":": "dot"}.get(trace.style, "solid")
            # SVG traces: WebGL (Scattergl) misplaces traces across many shared-axis subplots.
            fig.add_trace(go.Scatter(
                x=x, y=y, name=trace.label, legend="legend" if row == 1 else f"legend{row}", mode="lines",
                line={"shape": "hv" if trace.step else "linear", "dash": dash, "width": 1.3,
                      "color": palette[index % len(palette)]},
                hovertemplate=f"{trace.label}: %{{y:.4g}} {panel.unit}<br>t=%{{x:.3f}} s<extra></extra>"),
                row=row, col=1)
        for level in panel.hlines:
            fig.add_hline(y=level, line={"dash": "dot", "color": "gray", "width": 1}, row=row, col=1)
        fig.update_yaxes(title_text=panel.unit, row=row, col=1)
    for start, end, mode in mode_intervals(session):
        if mode == "riding":
            fig.add_vrect(x0=start, x1=end, fillcolor=RIDING_COLOR, opacity=0.08, line_width=0, row="all", col=1)
    for t, text in fault_markers(session):
        fig.add_vline(x=t, line={"color": FAULT_COLOR, "dash": "dash", "width": 1}, row="all", col=1)
        fig.add_annotation(x=t, y=1.0, yref="paper", text=text, showarrow=False, textangle=-90,
                           font={"color": FAULT_COLOR, "size": 10}, xanchor="left", yanchor="top")
    fig.update_xaxes(showticklabels=True, showspikes=True, spikemode="across", spikethickness=1)
    fig.update_xaxes(title_text="time since session start (s)", row=len(panels), col=1)
    fig.update_layout(height=height, hovermode="x unified",
                      title=f"{session.path.name} — green = riding, red dashed = fault",
                      margin={"l": 70, "r": 260, "t": 90, "b": 60}, template="plotly_white")
    # One legend per subplot, pinned beside it (a single shared legend drifts out of line on tall pages).
    for row in range(1, len(panels) + 1):
        axis = fig.layout["yaxis" if row == 1 else f"yaxis{row}"]
        fig.update_layout({"legend" if row == 1 else f"legend{row}": {
            "y": axis.domain[1], "yanchor": "top", "x": 1.01, "xanchor": "left", "font": {"size": 10},
            "bgcolor": "rgba(255,255,255,0.7)"}})

    histogram = make_subplots(rows=2, cols=4, subplot_titles=[h[2] for h in HISTOGRAMS],
                              vertical_spacing=0.18, horizontal_spacing=0.06)
    riding = mode_intervals(session)
    for index, (table, column, title, riding_only) in enumerate(HISTOGRAMS):
        resolved = _resolve(session, Trace(table, column, title))
        if resolved is None:
            continue
        x, y = resolved
        if riding_only:
            y = y[mask_for(x, riding)]
        histogram.add_trace(go.Histogram(x=y[np.isfinite(y)], nbinsx=60, name=title, showlegend=False),
                            row=index // 4 + 1, col=index % 4 + 1)
    histogram.update_layout(height=900, title="Distributions", template="plotly_white")

    table_rows = summary_markdown(summary).split("| Measure | Value |")[1].split("\n\n")[0].strip().splitlines()[1:]
    cells = [row.strip("|").split("|") for row in table_rows]
    summary_table = go.Figure(go.Table(
        header={"values": ["Measure", "Value"], "align": "left", "fill_color": "#dfe6ee"},
        cells={"values": [[c[0].strip() for c in cells], [c[1].strip() for c in cells]], "align": "left",
               "height": 26}))
    summary_table.update_layout(height=60 + 30 * (len(cells) + 1), margin={"l": 20, "r": 20, "t": 40, "b": 10},
                                title="Headline measures")
    path = out / "interactive.html"
    with open(path, "w") as stream:
        stream.write("<html><head><meta charset='utf-8'><title>"
                     f"{session.path.name}</title></head><body style='font-family:sans-serif'>\n")
        stream.write(summary_table.to_html(full_html=False, include_plotlyjs=True))
        stream.write(fig.to_html(full_html=False, include_plotlyjs=False))
        stream.write(histogram.to_html(full_html=False, include_plotlyjs=False))
        stream.write("</body></html>\n")
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path, help="Session folder (with bag/) or a rosbag directory")
    parser.add_argument("--out", type=Path, help="Output folder (default <session>/analysis)")
    parser.add_argument("--no-png", action="store_true", help="Skip matplotlib pages")
    parser.add_argument("--no-html", action="store_true", help="Skip the plotly page")
    parser.add_argument("--max-points", type=int, default=8000,
                        help="Per-trace point cap in the HTML (min/max envelope per time bucket; 0 = all)")
    args = parser.parse_args(argv)

    session = load_session(args.session)
    add_derived(session)
    out = (args.out or session.path / "analysis").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(session)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (out / "summary.md").write_text(summary_markdown(summary))
    pages = page_specs(session)
    written = [out / "summary.md", out / "summary.json"]
    if not args.no_png:
        written += write_matplotlib(session, pages, out)
    if not args.no_html:
        written.append(write_plotly(session, pages, summary, out, args.max_points))
    print(summary_markdown(summary).split("\n## ")[0])
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
