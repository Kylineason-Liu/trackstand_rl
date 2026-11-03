# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/config/triangle_mesh_terrain_importer.py (verbatim; relative imports only).
"""Terrain importer that collides with the authored triangle mesh.

Isaac Lab's ``create_prim_from_mesh`` only enables ``CollisionAPI``. PhysX then
convex-hulls the whole grid. An inverted pyramid is concave, so that hull is a
solid lid at rim height — visually a pit, physically a flat floor the bike
stands on.

``physics:approximation = none`` keeps the height-field triangles, so the pit
is a hole.
"""

from __future__ import annotations

import trimesh

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas import modify_mesh_collision_properties
from isaaclab.sim.schemas.schemas_cfg import MeshCollisionBaseCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.utils import create_prim_from_mesh


class TriangleMeshTerrainImporter(TerrainImporter):
    """Generator terrain importer with triangle-mesh (not convex-hull) collision."""

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        super().import_mesh(name, mesh)
        prim_path = f"{self.cfg.prim_path}/{name}/mesh"
        modify_mesh_collision_properties(
            prim_path, MeshCollisionBaseCfg(mesh_approximation_name="none")
        )
        print(f"[INFO] Terrain '{prim_path}' collision approximation set to triangle mesh.")
        wall = mesh.metadata.get("wall_mesh") if getattr(mesh, "metadata", None) else None
        if wall is None:
            return
        wall_path = f"{self.cfg.prim_path}/border_wall"
        create_prim_from_mesh(
            wall_path,
            wall,
            visual_material=None,
            physics_material=self.cfg.physics_material,
        )
        wall_mesh_path = f"{wall_path}/mesh"
        modify_mesh_collision_properties(
            wall_mesh_path, MeshCollisionBaseCfg(mesh_approximation_name="none")
        )
        wall_prim = sim_utils.get_prim_at_path(wall_path)
        if wall_prim.IsValid():
            sim_utils.set_prim_visibility(wall_prim, False)
        print(f"[INFO] Invisible border wall at '{wall_path}' (triangle-mesh collision).")
