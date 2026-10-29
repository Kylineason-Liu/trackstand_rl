# Ported from ~/bike_test/CycleRL/scripts/mujoco_paper/tire_from_mesh.py (imports/paths rewritten for trackstand_rl locomotive).
"""Wheel hitbox matching Isaac PhysX convex decomposition of the CAD meshes.

``new_bike.usda`` enables collision on ``rear_wheel_link_geom`` and
``front_wheel_link_geom`` with ``physics:approximation = convexDecomposition``
and does not override the PhysX schema defaults:

  maxConvexHulls   32
  hullVertexLimit  64
  voxelResolution  500000
  errorPercentage  10
  shrinkWrap       false

MuJoCo has no PhysX cooker. CoACD decomposes the same STEP meshes
(``step_rear_wheel.stl``, ``step_front_wheel.stl``) under that hull cap and
vertex cap. Each convex piece is a mesh geom on the wheel body, in the same
frame as the visual mesh.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_MESH_DIR = Path("/home/lmbike2/bike_models/thubike_new/step_meshes")
_FRONT_STL = _MESH_DIR / "step_front_wheel.stl"
_REAR_STL = _MESH_DIR / "step_rear_wheel.stl"
_CACHE = Path(__file__).resolve().parents[2] / "logs/locomotive/physx_hull_cache.npz"

# PhysX ConvexDecompositionCollisionAPI defaults (schema.usda).
_MAX_HULLS = 32
_MAX_VERTS = 64
# MJCF wheel_collision uses solimp dmin=0 and solref timeconst=20 ms, so the
# tire produces no lateral force until it has sunk in. Isaac is a hard μ=1
# contact (4 position iterations) and does not read those parameters.
# dmin=0.9 is immediate impedance. timeconst=5 ms with dampratio=1 is the
# critically damped pair from the stiff-contact probe (not dampratio=0.5).
_STIFF_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
_STIFF_SOLREF = (0.005, 1.0)


@dataclass
class TirePlant:
    radius: float
    half_width: float
    rear_center: np.ndarray
    front_center: np.ndarray
    pitch_rad: float
    z0: float
    n_hulls: int
    # Hub rate (rad/s per m/s of forward speed). 0 → vx / radius, both wheels.
    rear_spin_per_vx: float = 0.0
    front_spin_per_vx: float = 0.0


def _load_indexed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """STL is millimetres. Return welded vertices in metres and triangle faces."""
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    tris = np.frombuffer(data, dtype=rec, count=n, offset=84)["v"].astype(np.float64) * 0.001
    flat = np.round(tris.reshape(-1, 3), 6)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3).astype(np.int32)
    return uniq, faces


def _decompose(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    import coacd

    verts, faces = _load_indexed(path)
    parts = coacd.run_coacd(
        coacd.Mesh(verts, faces),
        threshold=0.05,
        max_convex_hull=_MAX_HULLS,
        max_ch_vertex=_MAX_VERTS,
        decimate=True,
        # CoACD's remesh grid. 150 inflated a non-manifold STEP mesh to
        # ~570k triangles. 50 keeps the tire surface (outer radius ~70.7 mm)
        # and finishes in about half a minute per wheel.
        preprocess_resolution=50,
        resolution=200,
        merge=True,
        seed=0,
    )
    out = []
    for hv, hf in parts:
        out.append((np.asarray(hv, dtype=np.float64), np.asarray(hf, dtype=np.int32)))
    return out


def _load_hulls() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    if _CACHE.is_file():
        data = np.load(_CACHE, allow_pickle=True)
        hulls = {}
        for prefix in ("rear", "front"):
            vs = list(data[f"{prefix}_v"])
            fs = list(data[f"{prefix}_f"])
            hulls[prefix] = list(zip(vs, fs))
        return hulls
    hulls = {
        "rear": _decompose(_REAR_STL),
        "front": _decompose(_FRONT_STL),
    }
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        _CACHE,
        rear_v=np.array([v for v, _ in hulls["rear"]], dtype=object),
        rear_f=np.array([f for _, f in hulls["rear"]], dtype=object),
        front_v=np.array([v for v, _ in hulls["front"]], dtype=object),
        front_f=np.array([f for _, f in hulls["front"]], dtype=object),
    )
    return hulls


def _lowest(parts: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    best = None
    for verts, _ in parts:
        i = int(np.argmin(verts[:, 2]))
        if best is None or verts[i, 2] < best[2]:
            best = verts[i].copy()
    return best


def _shape(parts: list[tuple[np.ndarray, np.ndarray]], origin: np.ndarray, axis: np.ndarray) -> tuple[float, float, np.ndarray]:
    axis = axis / np.linalg.norm(axis)
    origin = np.asarray(origin, dtype=np.float64)
    pts = np.concatenate([v for v, _ in parts], axis=0)
    rel = pts - origin
    s = rel @ axis
    radial = np.linalg.norm(rel - s[:, None] * axis, axis=1)
    r_max = float(radial.max())
    crown = radial > r_max - 0.002
    half = 0.5 * float(s[crown].max() - s[crown].min()) if np.any(crown) else 0.0
    s_crown = float(s[int(np.argmax(radial))])
    return r_max, half, origin + s_crown * axis


def _copy_contact(src, dst) -> None:
    dst.friction = [float(x) for x in src.friction]
    dst.solref = [float(x) for x in src.solref]
    dst.solimp = [float(x) for x in src.solimp]
    dst.margin = float(src.margin)
    dst.gap = float(src.gap)
    dst.priority = int(src.priority)
    dst.condim = int(src.condim)


def ellipsoid_plant(spec) -> TirePlant:
    """Seat pose for the MJCF ellipsoid tires. Does not replace those geoms."""
    centers = {}
    radii = {}
    halves = []
    for name, key in (
        ("rear_wheel_collision", "rear"),
        ("front_wheel_collision", "front"),
    ):
        geom = spec.geom(name)
        pos = np.array([float(x) for x in geom.pos], dtype=np.float64)
        radii[key] = float(geom.size[0])
        halves.append(float(geom.size[2]))
        centers[key] = pos
    rear_pt = centers["rear"] + np.array([0.0, 0.0, -radii["rear"]])
    front_pt = centers["front"] + np.array([0.0, 0.0, -radii["front"]])
    pitch = float(np.arctan((rear_pt[2] - front_pt[2]) / (rear_pt[0] - front_pt[0])))
    cth, sth = float(np.cos(pitch)), float(np.sin(pitch))
    z0 = -cth * float(front_pt[2]) + sth * float(front_pt[0])
    return TirePlant(
        radius=float(max(radii.values())),
        half_width=float(max(halves)),
        rear_center=centers["rear"],
        front_center=centers["front"],
        pitch_rad=pitch,
        z0=float(z0),
        n_hulls=0,
    )


def install_cad_tires(spec, mesh_dir: Path | None = None) -> TirePlant:
    """Replace MJCF ellipsoid tires with PhysX-capped convex hulls of the CAD wheels."""
    del mesh_dir
    hulls = _load_hulls()
    wheels = (
        ("rear", "rearwheel", "rearwheel_joint", "rear_wheel_collision"),
        ("front", "frontwheel", "frontwheel_joint", "front_wheel_collision"),
    )
    template = spec.geom("rear_wheel_collision")
    centers = {}
    radii = []
    halves = []
    for prefix, body_name, joint_name, old_name in wheels:
        joint = spec.joint(joint_name)
        origin = np.array([float(x) for x in joint.pos], dtype=np.float64)
        axis = np.array([float(x) for x in joint.axis], dtype=np.float64)
        r_max, half, center = _shape(hulls[prefix], origin, axis)
        radii.append(r_max)
        halves.append(half)
        centers[prefix] = center
        old = spec.geom(old_name)
        old.contype = 0
        old.conaffinity = 0
        body = spec.body(body_name)
        for i, (verts, faces) in enumerate(hulls[prefix]):
            mesh = spec.add_mesh()
            mesh.name = f"{prefix}_tire_mesh_{i:02d}"
            mesh.maxhullvert = max(_MAX_VERTS, int(len(verts)))
            mesh.uservert = [float(x) for x in verts.reshape(-1)]
            mesh.userface = [int(x) for x in faces.reshape(-1)]
            geom = body.add_geom()
            geom.name = f"{prefix}_tire_{i:02d}"
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = mesh.name
            geom.density = 0.0
            geom.contype = 1
            geom.conaffinity = 1
            geom.group = 1
            geom.rgba = [0.9, 0.35, 0.2, 0.35]
            _copy_contact(template, geom)
            geom.solimp = list(_STIFF_SOLIMP)
            geom.solref = list(_STIFF_SOLREF)

    rear_pt = _lowest(hulls["rear"])
    front_pt = _lowest(hulls["front"])
    pitch = float(np.arctan((rear_pt[2] - front_pt[2]) / (rear_pt[0] - front_pt[0])))
    cth, sth = float(np.cos(pitch)), float(np.sin(pitch))
    z0 = -cth * float(front_pt[2]) + sth * float(front_pt[0])
    return TirePlant(
        radius=float(max(radii)),
        half_width=float(max(halves)),
        rear_center=centers["rear"],
        front_center=centers["front"],
        pitch_rad=pitch,
        z0=float(z0),
        n_hulls=len(hulls["rear"]) + len(hulls["front"]),
    )
