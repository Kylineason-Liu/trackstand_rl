# Ported from ~/bike_test/CycleRL/scripts/mujoco_paper/trackstand_plant.py (imports/paths rewritten for trackstand_rl locomotive).
"""MuJoCo plant for the trackstand CAD the teammate verified.

``trackstand_rl_0916_bike/trackstand_rl_0916.xml`` is the Fusion export of that
CAD. Its ``base_link`` Y axis is world-up, so a Paper observation read from
that body is not a lean. Isaac loads the same inertias and joints from
``trackstand_rl_0916.urdf`` (X forward, Y left, Z up) with r=0.07 m cylinders
of length 1 mm. This builder reproduces that URDF as Isaac simulates it: URDF
joint axes unchanged, PhysX inertia tensors (``isaac_mass_properties_trackstand.json``)
and no CoM shift (the cylinder task disables ``zero_lateral_com``; its CAD CoM sits
~0.6 mm off the wheel-contact line, which the policy trained with).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from tire_from_mesh import TirePlant, _STIFF_SOLIMP, _STIFF_SOLREF

# Byte-identical copies of CycleRL's cyclerl/assets packages already tracked in this repo.
_REPO = Path(__file__).resolve().parents[3]
_CYLINDER_PKG = _REPO / "isaaclab_train/trackstand_rl_0916_description_cylinder"
_URDF = _CYLINDER_PKG / "urdf/trackstand_rl_0916.urdf"
_MESH_DIR = _CYLINDER_PKG / "meshes"
_BIKE_XML = _REPO / "mujoco_sim2sim/models/trackstand_rl_0916_bike/trackstand_rl_0916.xml"

_CYLINDER_RADIUS = 0.07
_CYLINDER_HALF_LENGTH = 0.0005  # URDF length 0.001 m
_VISUAL_MESH = {
    "base_link": "base_link.stl",
    "rear_wheel_link_1": "rear_wheel_link_1.stl",
    "steering_link_1": "steering_link_1.stl",
    "front_wheel_link_1": "front_wheel_link_1.stl",
}


def is_trackstand_xml(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == _BIKE_XML.resolve() or "trackstand_rl_0916" in resolved.name


def _f(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.split()], dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis rotation, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = (float(x) for x in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _mat_to_quat(rot: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.reshape(-1))
    return quat


def _load_urdf() -> tuple[dict, dict]:
    root = ET.parse(_URDF).getroot()
    links: dict[str, dict] = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        origin = inertial.find("origin")
        inertia = inertial.find("inertia")
        collision = link.find("collision")
        visual = link.find("visual")
        cyl = None
        if collision is not None and collision.find("geometry/cylinder") is not None:
            corigin = collision.find("origin")
            cyl_geom = collision.find("geometry/cylinder")
            cyl = {
                "pos": _f(corigin.get("xyz")),
                "rpy": _f(corigin.get("rpy")),
                "radius": float(cyl_geom.get("radius")),
                "length": float(cyl_geom.get("length")),
            }
        vis = None
        if visual is not None and visual.find("geometry/mesh") is not None:
            vorigin = visual.find("origin")
            mesh = visual.find("geometry/mesh")
            scale = mesh.get("scale")
            vis = {
                "pos": _f(vorigin.get("xyz", "0 0 0")),
                "rpy": _f(vorigin.get("rpy", "0 0 0")),
                "scale": _f(scale) if scale else np.array([0.001, 0.001, 0.001]),
            }
        # MuJoCo fullinertia order is ixx iyy izz ixy ixz iyz.
        full = [
            float(inertia.get("ixx")),
            float(inertia.get("iyy")),
            float(inertia.get("izz")),
            float(inertia.get("ixy")),
            float(inertia.get("ixz")),
            float(inertia.get("iyz")),
        ]
        links[link.get("name")] = {
            "mass": float(inertial.find("mass").get("value")),
            "ipos": _f(origin.get("xyz")),
            "fullinertia": full,
            "cylinder": cyl,
            "visual": vis,
        }
    joints: dict[str, dict] = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        limit = joint.find("limit")
        joints[joint.get("name")] = {
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "pos": _f(origin.get("xyz")),
            "axis": _f(joint.find("axis").get("xyz")),
            "range": None
            if limit is None
            else (float(limit.get("lower")), float(limit.get("upper"))),
        }
    return links, joints


_ISAAC_MASS_PROPS = Path(__file__).resolve().parent / "isaac_mass_properties_trackstand.json"


def _apply_isaac_inertia(links: dict) -> None:
    """Replace URDF inertia tensors with the ones PhysX simulates.

    The USD importer gave PhysX different base / steering / front-wheel tensors from
    the URDF (same principal moments, different axes). The policy was trained on the
    PhysX values, so the MuJoCo plant uses them. Masses and CoMs must agree.
    """
    if not _ISAAC_MASS_PROPS.is_file():
        raise FileNotFoundError(
            f"{_ISAAC_MASS_PROPS} missing; run mujoco_sim2sim/scripts/locomotive/dump_isaac_mass_properties.py"
        )
    bodies = json.loads(_ISAAC_MASS_PROPS.read_text())["bodies"]
    for name, link in links.items():
        isaac = bodies.get(name)
        if isaac is None:
            raise KeyError(f"{name} not in {_ISAAC_MASS_PROPS.name}")
        if abs(float(isaac["mass"]) - link["mass"]) > 1e-4 or np.abs(np.array(isaac["com"]) - link["ipos"]).max() > 1e-5:
            raise ValueError(f"{name}: Isaac mass/CoM differ from the URDF; regenerate {_ISAAC_MASS_PROPS.name}")
        inertia = np.array(isaac["inertia"], dtype=np.float64).reshape(3, 3)
        inertia = 0.5 * (inertia + inertia.T)
        # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz.
        link["fullinertia"] = [
            inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2],
        ]


def _fmt(values) -> str:
    return " ".join(f"{float(v):.10g}" for v in values)


def _xml(links: dict, joints: dict, *, steer_sign: float, y_shift: float) -> str:
    solimp = _fmt(_STIFF_SOLIMP)
    solref = _fmt(_STIFF_SOLREF)

    def inertial(link_name: str) -> str:
        link = links[link_name]
        ipos = link["ipos"].copy()
        ipos[1] -= y_shift
        return (
            f'<inertial pos="{_fmt(ipos)}" mass="{link["mass"]:.10g}" '
            f'fullinertia="{_fmt(link["fullinertia"])}"/>'
        )

    def cylinder(link_name: str, geom_name: str) -> str:
        cyl = links[link_name]["cylinder"]
        rot = _rpy_matrix(cyl["rpy"])
        quat = _mat_to_quat(rot)
        radius = float(cyl["radius"])
        half = 0.5 * float(cyl["length"])
        # group=3: collision debug; toggle off in the viewer geomgroups if desired.
        return (
            f'<geom name="{geom_name}" type="cylinder" '
            f'size="{radius:.10g} {half:.10g}" pos="{_fmt(cyl["pos"])}" '
            f'quat="{_fmt(quat)}" density="0" contype="1" conaffinity="1" '
            f'group="3" rgba="1 0.3 0.3 0.25" friction="1 0.005 0.0001" '
            f'priority="1" margin="0" gap="0" condim="3" '
            f'solimp="{solimp}" solref="{solref}"/>'
        )

    def visual(link_name: str, geom_name: str) -> str:
        vis = links[link_name]["visual"]
        mesh = _VISUAL_MESH[link_name]
        rot = _rpy_matrix(vis["rpy"])
        quat = _mat_to_quat(rot)
        return (
            f'<geom name="{geom_name}" type="mesh" mesh="{Path(mesh).stem}" '
            f'pos="{_fmt(vis["pos"])}" quat="{_fmt(quat)}" '
            f'density="0" contype="0" conaffinity="0" group="2" '
            f'rgba="0.75 0.78 0.85 1"/>'
        )

    rear = joints["rear_wheel_joint"]
    steer = joints["steering_joint"]
    front = joints["front_wheel_joint"]
    steer_axis = steer_sign * steer["axis"]
    lo, hi = steer["range"]
    mesh_assets = "\n".join(
        f'    <mesh name="{Path(fname).stem}" file="{fname}" '
        f'scale="{_fmt(links[link]["visual"]["scale"])}"/>'
        for link, fname in _VISUAL_MESH.items()
    )
    return f"""
<mujoco model="trackstand_rl_0916_cylinder">
  <compiler angle="radian" inertiafromgeom="false" meshdir="{_MESH_DIR}"/>
  <asset>
    <texture name="groundplane" type="2d" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="512" height="512"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
{mesh_assets}
  </asset>
  <worldbody>
    <light directional="true" pos="-0.5 0.5 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 0.05" group="2" material="groundplane"
          friction="1 0.005 0.0001"/>
    <body name="base_link">
      <freejoint name="floating_base_joint"/>
      {inertial("base_link")}
      {visual("base_link", "base_link_visual")}
      <body name="rear_wheel_link" pos="{_fmt(rear["pos"])}">
        <joint name="rear_wheel_joint" type="hinge" axis="{_fmt(rear["axis"])}" damping="0.001"/>
        {inertial("rear_wheel_link_1")}
        {visual("rear_wheel_link_1", "rear_wheel_visual")}
        {cylinder("rear_wheel_link_1", "rear_wheel_collision")}
      </body>
      <body name="steering_link" pos="{_fmt(steer["pos"])}">
        <joint name="steering_joint" type="hinge" axis="{_fmt(steer_axis)}"
               range="{lo:.10g} {hi:.10g}" limited="true" damping="0" armature="0"/>
        {inertial("steering_link_1")}
        {visual("steering_link_1", "steering_link_visual")}
        <body name="front_wheel_link" pos="{_fmt(front["pos"])}">
          <joint name="front_wheel_joint" type="hinge" axis="{_fmt(front["axis"])}" damping="0.001"/>
          {inertial("front_wheel_link_1")}
          {visual("front_wheel_link_1", "front_wheel_visual")}
          {cylinder("front_wheel_link_1", "front_wheel_collision")}
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="cmd_steering_f" joint="steering_joint" gear="1"
           ctrllimited="true" ctrlrange="-5.6 5.6" forcelimited="true" forcerange="-5.6 5.6"/>
    <motor name="cmd_rearwheel_f" joint="rear_wheel_joint" gear="1"
           ctrllimited="true" ctrlrange="-6.28 6.28" forcelimited="true" forcerange="-6.28 6.28"/>
  </actuator>
</mujoco>
"""


def _compile(xml: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_string(xml)
    return spec.compile()


def _world_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    total = 0.0
    com = np.zeros(3)
    for i in range(1, model.nbody):
        rot = data.xmat[i].reshape(3, 3)
        point = data.xpos[i] + rot @ model.body_ipos[i]
        com += model.body_mass[i] * point
        total += model.body_mass[i]
    return com / total


def _wheel_bottom(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str) -> np.ndarray:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    rot = data.geom_xmat[gid].reshape(3, 3)
    axis = rot[:, 2]
    axis = axis / np.linalg.norm(axis)
    down = np.array([0.0, 0.0, -1.0])
    perp = down - axis * float(np.dot(axis, down))
    perp = perp / np.linalg.norm(perp)
    radius = float(model.geom_size[gid, 0])
    return data.geom_xpos[gid] + radius * perp


def _wheel_heading(data: mujoco.MjData, gid: int) -> float:
    """Yaw of the wheel's forward direction. +heading is a left turn."""
    axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
    # axis × up lies in the ground plane and is perpendicular to the axle.
    forward = np.array([axis[1], -axis[0], 0.0])
    if forward[0] < 0.0:
        forward = -forward
    return float(np.arctan2(forward[1], forward[0]))


_USD_PHYSICS = _CYLINDER_PKG / "urdf/trackstand_rl_0916/payloads/Physics/physics.usda"


def _isaac_steer_axis() -> np.ndarray:
    """Steering axis in the base frame as Isaac loads it (USD ``localRot0`` · X)."""
    text = _USD_PHYSICS.read_text(encoding="utf-8")
    block = text.split('PhysicsRevoluteJoint "steering_joint"', 1)[1].split("def ", 1)[0]
    axis_token = block.split('physics:axis = "', 1)[1][0]
    quat_text = block.split("physics:localRot0 = (", 1)[1].split(")", 1)[0]
    quat = np.array([float(x) for x in quat_text.split(",")])
    unit = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}[axis_token]
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.array(unit), quat / np.linalg.norm(quat))
    return out


def _steer_moves_left(model: mujoco.MjModel) -> bool:
    """True when positive steer yaws the front wheel toward +Y (diagnostic only)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "front_wheel_collision")
    h0 = _wheel_heading(data, gid)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steering_joint")
    data.qpos[model.jnt_qposadr[jid]] = 0.2
    mujoco.mj_forward(model, data)
    delta = _wheel_heading(data, gid) - h0
    delta = float(np.arctan2(np.sin(delta), np.cos(delta)))
    return delta > 0.0


def _spin_per_vx(axis: np.ndarray, radius: float) -> float:
    """Hub rate (rad/s per m/s) that rolls the upright wheel toward +X."""
    return 1.0 / (float(axis[1]) * radius)


def make_trackstand_spec() -> tuple[mujoco.MjSpec, TirePlant, float]:
    """Return the cylinder plant, its seat pose, and the CoM shift in metres."""
    if not _URDF.is_file():
        raise FileNotFoundError(_URDF)
    links, joints = _load_urdf()
    _apply_isaac_inertia(links)
    for name in ("rear_wheel_link_1", "front_wheel_link_1"):
        cyl = links[name]["cylinder"]
        if abs(cyl["radius"] - _CYLINDER_RADIUS) > 1e-6 or abs(cyl["length"] - 2 * _CYLINDER_HALF_LENGTH) > 1e-6:
            raise ValueError(f"{name} cylinder is {cyl['radius']} x {cyl['length']}, expected 0.07 x 0.001")

    probe = _compile(_xml(links, joints, steer_sign=1.0, y_shift=0.0))
    pdata = mujoco.MjData(probe)
    mujoco.mj_forward(probe, pdata)
    rear_b = _wheel_bottom(probe, pdata, "rear_wheel_collision")
    front_b = _wheel_bottom(probe, pdata, "front_wheel_collision")
    rear_c = pdata.geom_xpos[mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_GEOM, "rear_wheel_collision")].copy()
    front_c = pdata.geom_xpos[mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_GEOM, "front_wheel_collision")].copy()
    contact_y = 0.5 * (float(rear_c[1]) + float(front_c[1]))
    com_y = float(_world_com(probe, pdata)[1])
    y_shift = com_y - contact_y
    # Keep the URDF steering axis: it is the one Isaac trains on, where positive steer
    # turns this CAD to the right. Flipping it here mirrored every policy steer command.
    steer_sign = 1.0
    urdf_axis = joints["steering_joint"]["axis"] / np.linalg.norm(joints["steering_joint"]["axis"])
    if float(np.dot(urdf_axis, _isaac_steer_axis())) < 0.99:
        raise ValueError(
            f"URDF steering axis {urdf_axis} does not match the Isaac USD axis "
            f"{_isaac_steer_axis()}; the MuJoCo plant would steer differently from training."
        )

    # Isaac leaves the ~0.6 mm CoM offset in place; so does this plant. ``y_shift`` is
    # still returned so callers can report how far the CoM sits off the contact line.
    spec = mujoco.MjSpec.from_string(_xml(links, joints, steer_sign=steer_sign, y_shift=0.0))
    pitch = float(np.arctan((rear_b[2] - front_b[2]) / (rear_b[0] - front_b[0])))
    cth, sth = float(np.cos(pitch)), float(np.sin(pitch))
    z0 = -cth * float(front_b[2]) + sth * float(front_b[0])
    plant = TirePlant(
        radius=_CYLINDER_RADIUS,
        half_width=_CYLINDER_HALF_LENGTH,
        rear_center=rear_c,
        front_center=front_c,
        pitch_rad=pitch,
        z0=z0,
        n_hulls=2,
        rear_spin_per_vx=_spin_per_vx(joints["rear_wheel_joint"]["axis"], _CYLINDER_RADIUS),
        front_spin_per_vx=_spin_per_vx(joints["front_wheel_joint"]["axis"], _CYLINDER_RADIUS),
    )
    return spec, plant, float(y_shift)
