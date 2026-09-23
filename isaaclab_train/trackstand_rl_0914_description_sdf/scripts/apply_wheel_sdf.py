#!/usr/bin/env python3
"""Set physics:approximation = "sdf" on the two wheel collision meshes of the
imported trackstand_rl_0914 USD, so PhysX collides the wheels against a
signed-distance field of the tire mesh instead of the default convex hull.

Why a post-import patch: the URDF importer cannot emit the sdf token (its
collision_type options are convex hull / decomposition / bounding sphere /
bounding cube, and are only consulted via collision-from-visuals). The token
lives on the USD produced by the import, so we set it there. The wheels' SDF
is built from the tire STLs declared in this package's URDF collision blocks.

Layout of the import (URDF USD Converter v0.1.2): every link becomes two
instanced Xforms -- <link> (visual) and <link>_1 (collision). The collision
instance root references a child mesh that carries PhysicsCollisionAPI and
physics:approximation = "convexHull". Instanced children are invisible to
plain stage traversal, so the roots are found in the traversal and the child
is fetched by name. USD forbids authoring through instance proxies, so the
patch edits the payload layer spec where the converter authored the
approximation (payloads/instances.usda), via Sdf directly.

The PhysxSDFMeshCollisionAPI application is optional (the backend cooks SDF
with defaults from the approximation token alone); it is used to pin the
resolution when the schema plugin is loaded (inside the Kit app).

Usage (with the Isaac Sim python of this build):
    <isaac-sim-root>/python.sh scripts/apply_wheel_sdf.py <path/to/model.usda>
    <isaac-sim-root>/python.sh scripts/apply_wheel_sdf.py <path/to/model.usda> --list
"""

import argparse
import sys

from pxr import Sdf, Usd, UsdPhysics

try:
    from pxr import PhysxSchema
except ImportError:  # schema plugin not loaded: approximation token still works
    PhysxSchema = None


def wheel_collision_specs(stage):
    """The Sdf spec (in the payload layer) that authors each wheel collision
    mesh's physics:approximation. The composed prim is an instance proxy, so
    we locate the owning spec in its prim stack instead of authoring on it."""
    out = []
    for prim in stage.Traverse():
        name = prim.GetName()
        if not name.endswith("_1"):
            continue
        link = name[:-2]
        if "rear_wheel" not in link and "front_wheel" not in link:
            continue
        child = prim.GetChild(link)
        if not child.IsValid():
            continue
        if not (child.HasAPI(UsdPhysics.CollisionAPI)
                or child.GetAttribute("physics:approximation")):
            continue
        for spec in child.GetPrimStack():
            if "physics:approximation" in spec.attributes:
                out.append((child, spec))
                break
    return out


def collision_instance_roots(stage):
    """Every <name>_1 instance root found in the traversal (for --list)."""
    return [prim for prim in stage.Traverse() if prim.GetName().endswith("_1")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usda", help="the imported model's .usda file")
    ap.add_argument("--list", action="store_true",
                    help="print the collision instance roots and exit without editing")
    ap.add_argument("--resolution", type=int, default=256,
                    help="PhysxSDFMeshCollisionAPI:sdfResolution (default 256)")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="PhysxSDFMeshCollisionAPI:sdfMargin (default 0.01)")
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.usda)
    if not stage:
        sys.exit(f"[ERROR] cannot open {args.usda}")

    if args.list:
        roots = collision_instance_roots(stage)
        print(f"[INFO] {len(roots)} <name>_1 collision instance roots on the stage:")
        for prim in roots:
            child = prim.GetChild(prim.GetName()[:-2])
            approx = child.GetAttribute("physics:approximation").Get() if child else None
            print(f"   {prim.GetPath()}  -> child {child.GetPath()}  "
                  f"approx={approx}  collide={bool(child and child.HasAPI(UsdPhysics.CollisionAPI))}")
        return

    wheels = wheel_collision_specs(stage)
    if not wheels:
        print("[WARN] no wheel collision specs found (expected <link>_1 instance roots)")
        print("[WARN] run with --list to inspect the imported layout, then adjust the match")
        return

    for child, spec in wheels:
        spec.attributes["physics:approximation"].default = "sdf"
        # apply PhysxSDFMeshCollisionAPI at the Sdf level (the plugin's SDF
        # branch may be gated on the API presence, not the token alone)
        schemas = spec.GetInfo("apiSchemas")
        if schemas is not None:
            current = (list(schemas.prependedItems or []) + list(schemas.explicitItems or [])
                       + list(schemas.appendedItems or []))
            if "PhysxSDFMeshCollisionAPI" not in current:
                spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(
                    prependedItems=["PhysxSDFMeshCollisionAPI"] + current))
        for attr_name, value, vtype in (
                ("physxSDFMeshCollision:sdfResolution", args.resolution, Sdf.ValueTypeNames.Int),
                ("physxSDFMeshCollision:sdfMargin", args.margin, Sdf.ValueTypeNames.Float)):
            a = spec.attributes.get(attr_name)
            if a is None:
                a = Sdf.AttributeSpec(spec, attr_name, vtype)
            a.default = value
        spec.layer.Save()
        print(f"[INFO] {child.GetPath()}: approximation=sdf + PhysxSDFMeshCollisionAPI"
              f" (edited {spec.layer.realPath}), resolution={args.resolution}, margin={args.margin}")


if __name__ == "__main__":
    main()
