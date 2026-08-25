"""Bike-agnostic eval hardware/action contract.

Analyzers must not hardcode USD joint names (``steering_joint``, ``Revolute_9``,
…). Resolve names, action slices, scales, clips, and effort limits from the
live environment:

  1. Action manager terms (``steering`` / ``drive`` aliases, else position vs
     velocity action classes).
  2. Matching robot actuators (``effort_limit``, ``velocity_limit``).
  3. Optional ``motor_contract.yaml`` overlay when ``joint_sim`` matches.
  4. Explicit keyword overrides.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

_MOTOR_CONTRACT_PATH = Path(__file__).resolve().parent / "config" / "motor_contract.yaml"

STEER_ACTION_ALIASES = ("steering", "steer")
DRIVE_ACTION_ALIASES = ("drive", "rearwheel", "rear_wheel", "wheel")
VELOCITY_COMMAND_ALIASES = ("velocity", "base_velocity", "lin_vel")
HEADING_COMMAND_ALIASES = ("steering", "heading", "ang_vel")


@dataclass
class BikeEvalContract:
    """Resolved joint / action / motor limits for one bike task."""

    robot_name: str = "robot"
    steer_joint_name: str | None = None
    drive_joint_name: str | None = None
    steer_action_name: str = "steering"
    drive_action_name: str = "drive"
    steer_action_index: int = 0
    drive_action_index: int = 1
    steer_action_scale: float = 1.0
    drive_action_scale: float = 1.0
    drive_vel_clip: float = math.inf
    effort_limit_steering: float = math.inf
    effort_limit_drive: float = math.inf
    velocity_command_name: str = "velocity"
    heading_command_name: str = "steering"
    source: str = "unresolved"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, float) and math.isinf(value):
                out[key] = None
        return out

    def replace(self, **overrides: Any) -> "BikeEvalContract":
        data = asdict(self)
        for key, value in overrides.items():
            if value is None or key not in data:
                continue
            data[key] = value
        return BikeEvalContract(**data)

    def logger_kwargs(self) -> dict[str, Any]:
        """Kwargs accepted by :class:`BicycleMetricsLogger`."""
        return {
            "steer_joint_name": self.steer_joint_name,
            "drive_joint_name": self.drive_joint_name,
            "steer_action_index": self.steer_action_index,
            "drive_action_index": self.drive_action_index,
            "steer_action_scale": self.steer_action_scale,
            "drive_action_scale": self.drive_action_scale,
            "drive_vel_clip": self.drive_vel_clip,
            "effort_limit_steering": self.effort_limit_steering,
            "effort_limit_drive": self.effort_limit_drive,
            "robot_name": self.robot_name,
            "velocity_command_name": self.velocity_command_name,
            "heading_command_name": self.heading_command_name,
        }


def _unwrap_isaac_env(env: Any) -> Any:
    """Walk ``.unwrapped`` until the Isaac Lab env (scene + action manager)."""
    seen: set[int] = set()
    current = env
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "action_manager") and hasattr(current, "scene"):
            return current
        nxt = getattr(current, "unwrapped", None)
        if nxt is None or nxt is current:
            break
        current = nxt
    return env


def resolve_eval_contract(
    env: Any,
    *,
    motor_contract_path: str | Path | None = _MOTOR_CONTRACT_PATH,
    **overrides: Any,
) -> BikeEvalContract:
    """Build a :class:`BikeEvalContract` from a live Isaac Lab env."""
    env = _unwrap_isaac_env(env)
    contract = BikeEvalContract()
    contract = _apply_from_actions(env, contract)
    contract = _apply_from_commands(env, contract)
    contract = _apply_from_actuators(env, contract)
    contract = _overlay_motor_contract(contract, motor_contract_path)
    if overrides:
        contract = contract.replace(**overrides)
    contract.source = "env"
    _validate_joints(env, contract)
    print(
        f"[EvalContract] robot={contract.robot_name} "
        f"steer={contract.steer_joint_name}[{contract.steer_action_index}] "
        f"drive={contract.drive_joint_name}[{contract.drive_action_index}] "
        f"scale=({contract.steer_action_scale:g}, {contract.drive_action_scale:g}) "
        f"τ_lim=({contract.effort_limit_steering:g}, {contract.effort_limit_drive:g}) "
        f"ω_clip={contract.drive_vel_clip:g}"
    )
    return contract


def _iter_named_terms(obj: Any) -> list[tuple[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return [(str(k), v) for k, v in obj.items() if v is not None]
    if hasattr(obj, "items") and callable(obj.items):
        try:
            return [(str(k), v) for k, v in obj.items() if v is not None]
        except Exception:
            pass
    out: list[tuple[str, Any]] = []
    for name in getattr(obj, "__dict__", {}):
        if name.startswith("_"):
            continue
        value = getattr(obj, name, None)
        if value is None or callable(value):
            continue
        out.append((name, value))
    return out


def _first_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return _first_name(value[0])
    return str(value)


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, dict) and value:
        return _as_float(next(iter(value.values())), default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip_abs_max(clip: Any, joint_name: str | None) -> float | None:
    if clip is None:
        return None
    if isinstance(clip, dict):
        pair = None
        if joint_name and joint_name in clip:
            pair = clip[joint_name]
        elif clip:
            pair = next(iter(clip.values()))
        return _clip_abs_max(pair, None)
    if isinstance(clip, (tuple, list)) and len(clip) >= 2:
        return max(abs(float(clip[0])), abs(float(clip[1])))
    return _as_float(clip)


def _scale_for_joint(scale: Any, joint_name: str | None) -> float:
    if isinstance(scale, dict):
        if joint_name and joint_name in scale:
            return float(scale[joint_name])
        if scale:
            return float(next(iter(scale.values())))
        return 1.0
    return _as_float(scale, 1.0) or 1.0


def _class_name(obj: Any) -> str:
    return type(obj).__name__.lower()


def _looks_like_steer_action(name: str, term: Any) -> bool:
    if name.lower() in STEER_ACTION_ALIASES:
        return True
    cfg = getattr(term, "cfg", term)
    return "position" in _class_name(cfg) and "joint" in _class_name(cfg)


def _looks_like_drive_action(name: str, term: Any) -> bool:
    if name.lower() in DRIVE_ACTION_ALIASES:
        return True
    cfg = getattr(term, "cfg", term)
    return "velocity" in _class_name(cfg) and "joint" in _class_name(cfg)


def _action_terms(env: Any) -> list[tuple[str, Any, int]]:
    """``(name, term_or_cfg, action_index)`` in policy-tensor order."""
    mgr = getattr(env, "action_manager", None)
    if mgr is not None:
        names = list(getattr(mgr, "active_terms", []) or getattr(mgr, "_term_names", []))
        dims = list(getattr(mgr, "action_term_dim", []))
        terms = getattr(mgr, "_terms", {})
        out: list[tuple[str, Any, int]] = []
        idx = 0
        for i, name in enumerate(names):
            dim = dims[i] if i < len(dims) else 1
            out.append((name, terms.get(name), idx))
            idx += int(dim)
        if out:
            return out
    cfg = getattr(getattr(env, "cfg", None), "actions", None)
    out = []
    idx = 0
    for name, term in _iter_named_terms(cfg):
        dim = int(getattr(term, "action_dim", 1) or 1)
        out.append((name, term, idx))
        idx += dim
    return out


def _apply_from_actions(env: Any, contract: BikeEvalContract) -> BikeEvalContract:
    terms = _action_terms(env)
    steer = next((t for t in terms if t[0].lower() in STEER_ACTION_ALIASES), None)
    drive = next((t for t in terms if t[0].lower() in DRIVE_ACTION_ALIASES), None)
    if steer is None:
        steer = next((t for t in terms if _looks_like_steer_action(t[0], t[1])), None)
    if drive is None:
        drive = next((t for t in terms if _looks_like_drive_action(t[0], t[1])), None)
    if steer is None and terms:
        steer = terms[0]
    if drive is None and len(terms) > 1:
        drive = terms[1]

    def _cfg(term: Any) -> Any:
        return getattr(term, "cfg", term) if term is not None else None

    if steer is not None:
        name, term, index = steer
        cfg = _cfg(term)
        contract.steer_action_name = name
        contract.steer_action_index = index
        joint = _first_name(getattr(cfg, "joint_names", None))
        if joint:
            contract.steer_joint_name = joint
        contract.steer_action_scale = _scale_for_joint(getattr(cfg, "scale", 1.0), joint)
        asset = getattr(cfg, "asset_name", None)
        if asset:
            contract.robot_name = str(asset)
    if drive is not None:
        name, term, index = drive
        cfg = _cfg(term)
        contract.drive_action_name = name
        contract.drive_action_index = index
        joint = _first_name(getattr(cfg, "joint_names", None))
        if joint:
            contract.drive_joint_name = joint
        contract.drive_action_scale = _scale_for_joint(getattr(cfg, "scale", 1.0), joint)
        clip = _clip_abs_max(getattr(cfg, "clip", None), joint)
        if clip is not None:
            contract.drive_vel_clip = clip
        asset = getattr(cfg, "asset_name", None)
        if asset:
            contract.robot_name = str(asset)
    return contract


def _apply_from_commands(env: Any, contract: BikeEvalContract) -> BikeEvalContract:
    mgr = getattr(env, "command_manager", None)
    names: list[str] = []
    if mgr is not None:
        names = list(getattr(mgr, "active_terms", []) or getattr(mgr, "_term_names", []))
    if not names:
        cfg = getattr(getattr(env, "cfg", None), "commands", None)
        names = [n for n, _ in _iter_named_terms(cfg)]

    def _term(name: str) -> Any:
        if mgr is None:
            cfg = getattr(getattr(env, "cfg", None), "commands", None)
            return getattr(cfg, name, None) if cfg is not None else None
        terms = getattr(mgr, "_terms", {})
        if hasattr(mgr, "get_term"):
            try:
                return mgr.get_term(name)
            except Exception:
                return terms.get(name)
        return terms.get(name)

    vel_name = None
    heading_name = None
    for name in names:
        term = _term(name)
        cfg = getattr(term, "cfg", term)
        if cfg is None:
            continue
        if bool(getattr(cfg, "heading_command", False)) or name.lower() in HEADING_COMMAND_ALIASES:
            if heading_name is None or name.lower() in HEADING_COMMAND_ALIASES:
                heading_name = name
            continue
        if name.lower() in VELOCITY_COMMAND_ALIASES or vel_name is None:
            vel_name = name
    if vel_name:
        contract.velocity_command_name = vel_name
    if heading_name:
        contract.heading_command_name = heading_name
    return contract


def _actuator_matches(act_cfg: Any, joint_name: str) -> bool:
    exprs = getattr(act_cfg, "joint_names_expr", None)
    if exprs is None:
        return False
    if isinstance(exprs, str):
        exprs = [exprs]
    for expr in exprs:
        if fnmatch.fnmatch(joint_name, expr) or joint_name == expr:
            return True
    return False


def _actuator_limit(act_cfg: Any, *keys: str) -> float | None:
    for key in keys:
        value = getattr(act_cfg, key, None)
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _apply_from_actuators(env: Any, contract: BikeEvalContract) -> BikeEvalContract:
    scene = getattr(env, "scene", None)
    robot = None
    if scene is not None:
        try:
            robot = scene[contract.robot_name]
        except Exception:
            robot = getattr(scene, contract.robot_name, None)
    cfg = getattr(robot, "cfg", None) if robot is not None else None
    if cfg is None:
        cfg = getattr(getattr(getattr(env, "cfg", None), "scene", None), "robot", None)
    actuators = getattr(cfg, "actuators", None) or {}
    if isinstance(actuators, dict):
        items = actuators.items()
    else:
        items = _iter_named_terms(actuators)
    for _, act_cfg in items:
        if contract.steer_joint_name and _actuator_matches(act_cfg, contract.steer_joint_name):
            effort = _actuator_limit(act_cfg, "effort_limit_sim", "effort_limit")
            if effort is not None:
                contract.effort_limit_steering = effort
        if contract.drive_joint_name and _actuator_matches(act_cfg, contract.drive_joint_name):
            effort = _actuator_limit(act_cfg, "effort_limit_sim", "effort_limit")
            if effort is not None:
                contract.effort_limit_drive = effort
            if not math.isfinite(contract.drive_vel_clip):
                vel = _actuator_limit(act_cfg, "velocity_limit_sim", "velocity_limit")
                if vel is not None:
                    contract.drive_vel_clip = vel
    return contract


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _overlay_motor_contract(
    contract: BikeEvalContract,
    path: str | Path | None,
) -> BikeEvalContract:
    data = _load_yaml(path)
    if not data:
        return contract
    steering = data.get("steering") or {}
    drive = data.get("drive") or {}
    policy = (data.get("policy") or {}).get("action") or {}
    if steering.get("joint_sim") == contract.steer_joint_name:
        effort = _as_float(steering.get("effort_limit_nm"))
        if effort is not None:
            contract.effort_limit_steering = effort
        scale = _as_float((policy.get("steer") or {}).get("scale"))
        if scale is not None:
            contract.steer_action_scale = scale
    if drive.get("joint_sim") == contract.drive_joint_name:
        effort = _as_float(drive.get("effort_limit_nm"))
        if effort is not None:
            contract.effort_limit_drive = effort
        scale = _as_float((policy.get("drive") or {}).get("scale"))
        if scale is not None:
            contract.drive_action_scale = scale
        clip = _as_float((policy.get("drive") or {}).get("clip_rad_s"))
        if clip is not None:
            contract.drive_vel_clip = clip
    return contract


def _validate_joints(env: Any, contract: BikeEvalContract) -> None:
    scene = getattr(env, "scene", None)
    if scene is None:
        return
    try:
        robot = scene[contract.robot_name]
    except Exception:
        return
    if not hasattr(robot, "find_joints"):
        return
    for role, pattern in (
        ("steer", contract.steer_joint_name),
        ("drive", contract.drive_joint_name),
    ):
        if not pattern:
            raise ValueError(f"Eval contract is missing {role} joint name (no action term found).")
        ids, names = robot.find_joints(pattern)
        if ids is None or len(ids) == 0:
            available = list(getattr(robot, "joint_names", []) or getattr(robot.data, "joint_names", []))
            raise ValueError(
                f"No {role} joint matching {pattern!r} on '{contract.robot_name}'. "
                f"Available joints: {available}"
            )


def contract_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(BikeEvalContract))
