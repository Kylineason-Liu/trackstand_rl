#!/usr/bin/env python
"""Verify the bicycle terrain curriculum visually before training.

Usage:
    ./isaaclab.sh -p scripts/verify_terrain.py
    ./isaaclab.sh -p scripts/verify_terrain.py --terrain stairs
    ./isaaclab.sh -p scripts/verify_terrain.py --color_scheme height
"""

from __future__ import annotations

import argparse
import contextlib
from typing import TYPE_CHECKING

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visual verification of bicycle terrain curriculum.")
parser.add_argument(
    "--terrain",
    type=str,
    default="complex",
    choices=["complex", "stairs"],
    help=(
        "Which generator terrain to spawn for visual verification "
        "('complex' or 'stairs'). Training/play omit --terrain for a flat plane."
    ),
)
parser.add_argument(
    "--color_scheme",
    type=str,
    default="height",
    choices=["height", "random", "none"],
    help="Color scheme for the terrain. 'height' makes slopes and bumps easy to read.",
)
parser.add_argument("--physics", default="physx", choices=["physx"], help="Physics backend.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Start Kit BEFORE importing torch / isaaclab sim modules
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg

from CycleRL.tasks.manager_based.cyclerl.config.terrain_presets import (
    TERRAIN_PRESET_META,
    resolve_terrain_generator,
)
from CycleRL.tasks.manager_based.cyclerl.config.triangle_mesh_terrain_importer import (
    TriangleMeshTerrainImporter,
)

TERRAIN_CFG = resolve_terrain_generator(args_cli.terrain)
_PRESET_META = TERRAIN_PRESET_META[args_cli.terrain]
_COLUMN_LABELS = _PRESET_META["column_labels"]
_ROW_LABELS = _PRESET_META["row_labels"]
_CAMERA_EYE = _PRESET_META["camera_eye"]

if TYPE_CHECKING:
    from isaaclab.assets import AssetBase


def design_scene() -> tuple[dict, torch.Tensor]:
    """Design the scene with the bicycle terrain generator and a dome light."""
    # Dome light
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Clone the terrain config and apply CLI overrides
    terrain_gen_cfg = TERRAIN_CFG.replace(color_scheme=args_cli.color_scheme)

    # Terrain importer — spawns the generated mesh into the USD stage
    terrain_importer_cfg = TerrainImporterCfg(
        class_type=TriangleMeshTerrainImporter,
        num_envs=terrain_gen_cfg.num_rows * terrain_gen_cfg.num_cols,
        env_spacing=terrain_gen_cfg.size[0],  # match sub-terrain width so env origins align
        prim_path="/World/ground",
        max_init_terrain_level=None,          # show all rows/cols
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        # High-friction physics material for bicycle tyres
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=True,                       # show env origin markers
    )
    # Remove the default visual material override so the height color scheme shows
    if args_cli.color_scheme in ["height", "random"]:
        terrain_importer_cfg.visual_material = None

    terrain_importer = terrain_importer_cfg.class_type(terrain_importer_cfg)

    print(f"[INFO]: Terrain preset: {args_cli.terrain}")
    print(f"[INFO]: Terrain grid {terrain_gen_cfg.num_rows} rows × {terrain_gen_cfg.num_cols} cols")
    print(f"[INFO]: Sub-terrain types (columns): {list(terrain_gen_cfg.sub_terrains.keys())}")
    print(f"[INFO]: Sub-terrain size: {terrain_gen_cfg.size}")
    print(f"[INFO]: Apron width: {getattr(terrain_gen_cfg, 'apron_width', 0.0)} m")
    print(
        f"[INFO]: Wall thickness: {getattr(terrain_gen_cfg, 'wall_thickness', 0.0)} m "
        f"(hidden={getattr(terrain_gen_cfg, 'hide_wall_visual', False)})"
    )
    print(f"[INFO]: Curriculum mode: {terrain_gen_cfg.curriculum}")
    print(f"[INFO]: Terrain origins shape: {terrain_importer.terrain_origins.shape}")

    scene_entities = {"terrain": terrain_importer}
    return scene_entities, terrain_importer.env_origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, "AssetBase"], origins: torch.Tensor):
    """Run until the Kit window is closed."""
    while simulation_app.is_running():
        sim.step()


def main():
    """Main function."""
    with contextlib.nullcontext():  # ponytail: keeps indentation; Kit is closed after main()
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=_CAMERA_EYE, target=[0.0, 0.0, 0.0])
        scene_entities, scene_origins = design_scene()
        sim.reset()
        print("[INFO]: Setup complete — inspect the terrain in the Kit window.")
        print(f"[INFO]:   Columns (L→R): {_COLUMN_LABELS}")
        print(f"[INFO]:   Rows    (F→B): {_ROW_LABELS}")
        run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
