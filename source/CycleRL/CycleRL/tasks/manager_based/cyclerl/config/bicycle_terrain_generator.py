"""Terrain generator with a flat wrap apron and a thin (optionally hidden) wall.

Isaac Lab's stock border is a solid box of thickness ``border_width`` and
height ``|border_height|``. That is a thick visible wall sitting on the outer
cells, so wrap teleports from stairs/waves onto a different cell type.

This generator instead:

1. Adds a **flat z=0 apron** around the curriculum grid (the wrap buffer).
2. Adds a **thin collision wall** on the apron's outer edge so PhysX never
   voids the bike if wrap is a frame late.
3. Keeps the wall out of the coloured terrain mesh and imports it as a
   separate invisible prim (see :class:`TriangleMeshTerrainImporter`).
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.utils import make_border, make_plane
from isaaclab.utils.configclass import configclass


@configclass
class BicycleTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Generator cfg with a flat wrap apron and a thin perimeter wall."""

    apron_width: float = 4.0
    """Flat z=0 buffer (m) between the curriculum grid and the wall."""

    wall_thickness: float = 0.01
    """Collision-wall thickness (m). Keep small; visual is hidden separately."""

    wall_height: float = 2.0
    """Wall height above z=0 (m)."""

    hide_wall_visual: bool = True
    """If True, the wall is not merged into the visible terrain mesh."""


class BicycleTerrainGenerator(TerrainGenerator):
    """Curriculum grid + flat apron; thin wall stored on ``terrain_mesh.metadata``."""

    def __init__(self, cfg: BicycleTerrainGeneratorCfg, device: str = "cpu"):
        super().__init__(cfg, device)
        wall = self._make_perimeter_wall()
        if wall is None:
            return
        transform = np.eye(4)
        transform[:2, -1] = (
            -self.cfg.size[0] * self.cfg.num_rows * 0.5,
            -self.cfg.size[1] * self.cfg.num_cols * 0.5,
        )
        wall.apply_transform(transform)
        if bool(getattr(self.cfg, "hide_wall_visual", True)):
            self.terrain_mesh.metadata["wall_mesh"] = wall
        else:
            self.terrain_mesh = trimesh.util.concatenate([self.terrain_mesh, wall])

    def _add_terrain_border(self):
        """Flat z=0 planes around the grid (same primitive as MeshPlaneTerrain)."""
        apron = float(getattr(self.cfg, "apron_width", 0.0))
        if apron <= 0.0:
            return
        inner_x = self.cfg.num_rows * self.cfg.size[0]
        inner_y = self.cfg.num_cols * self.cfg.size[1]
        overlap = 0.02
        strips = []
        # South / north: full outer width, including corners.
        south = make_plane((inner_x + 2.0 * apron, apron + overlap), 0.0, center_zero=False)
        south.apply_translation((-apron, -apron, 0.0))
        north = make_plane((inner_x + 2.0 * apron, apron + overlap), 0.0, center_zero=False)
        north.apply_translation((-apron, inner_y - overlap, 0.0))
        # West / east: along the grid only (corners already in south/north).
        west = make_plane((apron + overlap, inner_y), 0.0, center_zero=False)
        west.apply_translation((-apron, 0.0, 0.0))
        east = make_plane((apron + overlap, inner_y), 0.0, center_zero=False)
        east.apply_translation((inner_x - overlap, 0.0, 0.0))
        strips.extend([south, north, west, east])
        self.terrain_meshes.append(trimesh.util.concatenate(strips))
        print(
            f"[INFO] Flat wrap apron: {apron:.2f} m of z=0 plane around "
            f"{inner_x:.1f}×{inner_y:.1f} m grid."
        )

    def _make_perimeter_wall(self) -> trimesh.Trimesh | None:
        thickness = float(getattr(self.cfg, "wall_thickness", 0.0))
        height = float(getattr(self.cfg, "wall_height", 0.0))
        apron = float(getattr(self.cfg, "apron_width", 0.0))
        if thickness <= 0.0 or height <= 0.0:
            return None
        inner = (
            self.cfg.num_rows * self.cfg.size[0] + 2.0 * apron,
            self.cfg.num_cols * self.cfg.size[1] + 2.0 * apron,
        )
        outer = (inner[0] + 2.0 * thickness, inner[1] + 2.0 * thickness)
        center = (
            self.cfg.num_rows * self.cfg.size[0] / 2.0,
            self.cfg.num_cols * self.cfg.size[1] / 2.0,
            height / 2.0,
        )
        wall = trimesh.util.concatenate(make_border(outer, inner, height=height, position=center))
        selector = ~(np.asarray(wall.triangles)[:, :, 2] < -0.1).any(1)
        wall.update_faces(selector)
        return wall


# Default class_type so TerrainImporter constructs this generator, not Isaac Lab's.
BicycleTerrainGeneratorCfg.class_type = BicycleTerrainGenerator
