"""Check action meaning: v1 and v2 Beta checkpoints have identical tensor shapes."""

import hashlib
import json
import math
from pathlib import Path

from . import contract as c


def validate_policy_contract(policy_path, config_path=None):
    """Read saved Isaac YAML or an explicit JSON sidecar without importing Isaac.

    Metadata establishes the declared training contract, not an independently
    proved relation between an arbitrary export and its training checkpoint.
    A JSON sidecar may bind itself to an export with policy_sha256.
    """
    policy_path = Path(policy_path).resolve()
    if config_path is None:
        candidates = (policy_path.with_suffix(".contract.json"), policy_path.parent / "params/env.yaml")
        config_path = next((path for path in candidates if path.is_file()), None)
    if config_path is None:
        raise ValueError(
            "V2 policy metadata is required: keep adjacent params/env.yaml, provide --policy_config, "
            "or add <policy-stem>.contract.json; 11-input Beta shapes cannot distinguish v1 torque from v2 velocity"
        )
    config_path = Path(config_path).resolve()
    text = config_path.read_text()
    if config_path.suffix.lower() == ".json":
        metadata = json.loads(text)
        if metadata.get("task") != c.TASK or metadata.get("drive_action") != "velocity":
            raise ValueError("Policy sidecar must declare Trackstand-Balance-v2 and drive_action=velocity")
        if tuple(metadata.get("observations", ())) != c.OBSERVATIONS:
            raise ValueError("Policy sidecar observation ordering differs from v2")
        values = {name: metadata.get(name) for name in (
            "drive_velocity_scale_rad_s", "drive_velocity_p_nm_s_per_rad", "drive_integral_gain_nm_per_rad",
            "drive_integral_rate_hz", "drive_effort_limit_nm",
        )}
        expected = (c.DRIVE_SCALE, c.DRIVE_KD, c.DRIVE_INTEGRAL_GAIN,
                    c.DRIVE_INTEGRAL_RATE_HZ, c.DRIVE_EFFORT_LIMIT)
        supplied_hash = metadata.get("policy_sha256")
        if supplied_hash and hashlib.sha256(policy_path.read_bytes()).hexdigest() != supplied_hash:
            raise ValueError("Policy sidecar SHA256 does not match the loaded policy")
    else:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("Install requirements_v2.txt (PyYAML is needed for saved env.yaml)") from error
        # BaseLoader does not construct Python objects from Isaac's YAML tags.
        metadata = yaml.load(text, Loader=yaml.BaseLoader)
        try:
            action = metadata["actions"]["drive"]
            actuator = metadata["scene"]["robot"]["actuators"]["drive"]
            if not action["class_type"].endswith(":IntegralJointVelocityAction"):
                raise ValueError("Checkpoint is not a v2 velocity-integral policy (v1 torque is incompatible)")
            if not actuator["class_type"].endswith(":IdealPDActuator") or float(actuator["stiffness"]) != 0.:
                raise ValueError("V2 expects IdealPD with zero rear position stiffness")
            if str(action.get("use_default_offset")).lower() != "false" or float(action["offset"]) != 0.:
                raise ValueError("V2 expects rear velocity with zero offset")
            clip = [float(x) for x in action["clip"]["rear_wheel_joint"]]
            if clip != [-c.DRIVE_SPEED_LIMIT, c.DRIVE_SPEED_LIMIT]:
                raise ValueError("Saved rear velocity clip differs from v2")
            values = dict(scale=action["scale"], proportional_gain=actuator["damping"],
                          integral_gain=action["integral_gain_nm_per_rad"],
                          integral_rate=action["integral_rate_hz"], effort_limit=actuator["effort_limit"])
            expected = (c.DRIVE_SCALE, c.DRIVE_KD, c.DRIVE_INTEGRAL_GAIN,
                        c.DRIVE_INTEGRAL_RATE_HZ, c.DRIVE_EFFORT_LIMIT)
        except (KeyError, TypeError) as error:
            raise ValueError("Incomplete saved v2 environment metadata") from error
    try:
        matches = all(math.isclose(float(value), want, rel_tol=0., abs_tol=1e-9)
                      for value, want in zip(values.values(), expected))
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise ValueError(f"Saved policy's rear velocity/PI contract differs from runtime: {values}")
    return {"path": str(config_path), "sha256": hashlib.sha256(text.encode()).hexdigest(), "task": c.TASK,
            "drive_action": "velocity"}
