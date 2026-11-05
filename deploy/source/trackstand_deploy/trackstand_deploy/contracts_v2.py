"""Trackstand v2 velocity-action contract and pure MIT PI helpers (no ROS)."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from .contracts import OBSERVATION_NAMES, STEERING_ACTION_SCALE_RAD
from .mit_protocol import MitCommand, MitRanges, decode_symmetric, encode, mit_command_payload, parse_mit_motion

TASK = "Trackstand-Balance-v2"
COMMAND_TOPIC = "/trackstand/motor_command_v2"
COMMAND_LABEL = "motor_command_velocity_v2:enable,steering_target_rad,drive_target_rad_s"
DRIVE_VELOCITY_SCALE_RAD_S = 61.6
DRIVE_KD_NM_S_PER_RAD = 0.30
DRIVE_KI_NM_PER_RAD = 3.3
DRIVE_EFFORT_LIMIT_NM = 6.28
INTEGRAL_RATE_HZ = 200.0


def scale_policy_action_v2(raw_action, max_steering_target_rad=STEERING_ACTION_SCALE_RAD):
    action = np.asarray(raw_action, dtype=np.float32)
    if (action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.)
            or max_steering_target_rad != STEERING_ACTION_SCALE_RAD):
        raise ValueError("v2 requires two bounded Beta means and steering limit 1.57 rad")
    return (float(np.clip(float(action[0]) * 1.57, -1.57, 1.57)),
            float(np.clip(float(action[1]) * DRIVE_VELOCITY_SCALE_RAD_S,
                          -DRIVE_VELOCITY_SCALE_RAD_S, DRIVE_VELOCITY_SCALE_RAD_S)))


def validate_policy_manifest(model_path):
    path = Path(model_path)
    manifest = json.loads(path.with_suffix(".contract.json").read_text())
    if (manifest.get("task") != TASK or manifest.get("drive_action") != "velocity"
            or tuple(manifest.get("observations", ())) != OBSERVATION_NAMES
            or manifest.get("scales_baked_into_onnx") is not False):
        raise ValueError("Expected unscaled v2 velocity policy with the ordered eleven observations")
    expected = {
        "drive_velocity_scale_rad_s": DRIVE_VELOCITY_SCALE_RAD_S,
        "drive_velocity_p_nm_s_per_rad": DRIVE_KD_NM_S_PER_RAD,
        "drive_integral_gain_nm_per_rad": DRIVE_KI_NM_PER_RAD,
        "drive_integral_rate_hz": INTEGRAL_RATE_HZ, "drive_effort_limit_nm": DRIVE_EFFORT_LIMIT_NM,
    }
    for name, value in expected.items():
        supplied = manifest.get(name)
        if not isinstance(supplied, (int, float)) or not math.isclose(supplied, value, rel_tol=0., abs_tol=1e-9):
            raise ValueError(f"v2 policy contract mismatch: {name}")
    if manifest.get("policy_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("v2 policy contract checksum mismatch")
    return manifest


class VelocityIntegral:
    """Trapezoidal host I; retained across targets, primed after each disable."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.error_rad = self.torque_nm = self.previous_error = self.last_dt_s = 0.
        self.previous_time = None
        self.limited_updates = 0

    def update(self, error_rad_s, now, kd, limit_nm):
        if not all(map(math.isfinite, (error_rad_s, now, kd, limit_nm))) or kd <= 0 or limit_nm <= 0:
            raise ValueError("invalid PI sample, gain or effort bound")
        dt = now - self.previous_time if self.previous_time is not None else 0.
        if dt < 0:
            raise ValueError("PI monotonic clock moved backwards")
        increment = .5 * (error_rad_s + self.previous_error) * dt if self.previous_time is not None else 0.
        proposed = DRIVE_KI_NM_PER_RAD * (self.error_rad + increment)
        candidate = max(-limit_nm, min(limit_nm, proposed))
        delta = candidate - self.torque_nm
        total = kd * error_rad_s + candidate
        if (total > limit_nm and delta > 0) or (total < -limit_nm and delta < 0):
            candidate = self.torque_nm
        self.limited_updates += int(candidate != proposed)
        self.torque_nm, self.error_rad = candidate, candidate / DRIVE_KI_NM_PER_RAD
        self.previous_error, self.previous_time, self.last_dt_s = error_rad_s, now, dt


def encoded_drive_kd():
    return round(DRIVE_KD_NM_S_PER_RAD * 4095 / 5) * 5 / 4095


def rear_f0_payload(original):
    """Full +/-61.6-rad/s contract plus only codec headroom; preserve Pos_Max."""
    if len(original) != 7 or original[0] != 0xF0:
        raise ValueError("invalid rear F0")
    p, v, t = struct.unpack("<HHH", original[1:])
    if min(p, v, t) <= 0:
        raise ValueError("invalid F0 ranges")
    t = max(t, math.ceil(DRIVE_EFFORT_LIMIT_NM * 100))
    headroom = (t * .01) / 4095 / encoded_drive_kd() + .01
    v = math.ceil((DRIVE_VELOCITY_SCALE_RAD_S + headroom) * 100)
    if not 1 <= v <= 65535:
        raise ValueError("rear velocity range cannot encode the v2 contract")
    return bytes([0xF0]) + struct.pack("<HHH", p, v, t)


def integral_limit(ranges):
    # Reserve the worst-case velocity quantization contribution, as in the
    # tested MIT experiment. This is not a verified total motor torque clamp.
    return min(DRIVE_EFFORT_LIMIT_NM - encoded_drive_kd() * ranges.velocity_rad_s / 4095, ranges.torque_nm)


@dataclass(frozen=True)
class VelocityPacket:
    payload: bytes
    command: MitCommand
    effective_target_rad_s: float
    estimated_total_torque_nm: float


def velocity_packet(target_rad_s, measured_rad_s, integral_nm, ranges: MitRanges, sign=-1.):
    if (not all(map(math.isfinite, (target_rad_s, measured_rad_s, integral_nm))) or sign not in (-1., 1.)
            or abs(target_rad_s) > DRIVE_VELOCITY_SCALE_RAD_S or abs(integral_nm) > integral_limit(ranges) + 1e-10):
        raise ValueError("invalid v2 velocity/PI request")
    kd = encoded_drive_kd()
    ff = decode_symmetric(encode(sign * integral_nm, -ranges.torque_nm, ranges.torque_nm, 12), ranges.torque_nm, 12)
    residual = ff - sign * integral_nm
    payload = mit_command_payload(MitCommand(0., sign * target_rad_s - residual / kd,
                                             0., DRIVE_KD_NM_S_PER_RAD, sign * integral_nm), ranges)
    command = parse_mit_motion(payload, ranges)
    effective = sign * command.velocity_rad_s + sign * residual / kd
    estimate = kd * (effective - measured_rad_s) + integral_nm
    return VelocityPacket(payload, command, effective, estimate)
