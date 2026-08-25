"""Named bicycle terrain presets for CLI selection.

Presets:
  flat    — regular infinite plane (default when --terrain is omitted)
  complex — 5×5 height-field curriculum (flat, rough, slopes, blocks, waves)
  stairs  — 8×3 (2× flat, 2× pyramid up, 2× inverted pyramid, 2× waves)

Eval stages pin every env to one difficulty *row* while still spreading
across columns (terrain types):
  complex_L0 … complex_L4   easiest → hardest height-field
  stairs_L0  … stairs_L2    shallow → steep (steps and waves)
"""

from __future__ import annotations

from typing import Any

TERRAIN_PRESET_NAMES = ("flat", "complex", "stairs")

COMPLEX_NUM_ROWS = 5
STAIRS_NUM_ROWS = 3

TERRAIN_PRESET_HELP = (
    "Bicycle terrain: 'flat' is a regular plane (default when omitted); "
    "'complex' is the 5×5 curriculum (flat/rough/slopes/blocks/waves); "
    "'stairs' is 8×3 planes, up/down pyramid stairs, and waves."
)

TERRAIN_PRESET_META = {
    "flat": {
        "column_labels": "plane",
        "row_labels": "n/a",
        "camera_eye": [5.0, 5.0, 2.0],
        "num_rows": 0,
    },
    "complex": {
        "column_labels": "flat | rough | slopes | blocks | waves",
        "row_labels": "easiest → hardest",
        "camera_eye": [40.0, 40.0, 30.0],
        "num_rows": COMPLEX_NUM_ROWS,
    },
    "stairs": {
        "column_labels": (
            "flat | flat | stairs up | stairs up | "
            "stairs down | stairs down | waves | waves"
        ),
        "row_labels": "shallow → steep (steps and waves)",
        "camera_eye": [28.0, 28.0, 18.0],
        "num_rows": STAIRS_NUM_ROWS,
    },
}


def terrain_stage_count(name: str) -> int:
    """Number of difficulty rows for a named generator."""
    if name == "flat":
        return 0
    if name == "complex":
        return COMPLEX_NUM_ROWS
    if name == "stairs":
        return STAIRS_NUM_ROWS
    raise ValueError(f"Unknown terrain preset {name!r}.")


def resolve_terrain_generator(name: str):
    """Return the TerrainGeneratorCfg for a generator preset name."""
    if name == "complex":
        from .bicycle_terrain_complex import BICYCLE_TERRAINS_CFG

        return BICYCLE_TERRAINS_CFG
    if name == "stairs":
        from .bicycle_terrain_stairs import BICYCLE_STAIRS_TERRAINS_CFG

        return BICYCLE_STAIRS_TERRAINS_CFG
    raise ValueError(
        f"Unknown generator terrain preset {name!r}. Choose one of: complex, stairs"
    )


def _disable_terrain_levels_curriculum(env_cfg: Any) -> None:
    """Drop ``terrain_levels`` — it requires a curriculum generator."""
    curriculum = getattr(env_cfg, "curriculum", None)
    if curriculum is None or not hasattr(curriculum, "terrain_levels"):
        return
    if getattr(curriculum, "terrain_levels", None) is None:
        return
    curriculum.terrain_levels = None
    print("[INFO] Disabled curriculum.terrain_levels (flat plane has no rows).")


def apply_flat_plane(env_cfg: Any) -> None:
    """Replace ``env_cfg.scene.terrain`` with a regular infinite plane."""
    scene = getattr(env_cfg, "scene", None)
    terrain = getattr(scene, "terrain", None) if scene is not None else None
    if terrain is None:
        print("[WARN] flat terrain ignored: env cfg has no scene.terrain")
        return

    from isaaclab.terrains import TerrainImporter

    terrain.terrain_type = "plane"
    terrain.terrain_generator = None
    terrain.use_terrain_origins = False
    terrain.max_init_terrain_level = None
    terrain.class_type = TerrainImporter
    _disable_terrain_levels_curriculum(env_cfg)
    print("[INFO] Terrain preset 'flat': regular plane")


def apply_terrain_preset(
    env_cfg: Any,
    name: str | None,
    *,
    init_level: int | None = None,
) -> None:
    """Swap ``env_cfg.scene.terrain`` to the named preset.

    When ``name`` is None or ``\"flat\"``, uses a regular infinite plane.
    Tasks without a terrain importer are skipped.
    ``init_level`` caps the spawn row (0 = easiest) for generator presets.
    Eval stages pin the row after ``gym.make`` via :func:`pin_terrain_level`.
    """
    if not name or name == "flat":
        apply_flat_plane(env_cfg)
        return

    scene = getattr(env_cfg, "scene", None)
    terrain = getattr(scene, "terrain", None) if scene is not None else None
    if terrain is None:
        print(f"[WARN] --terrain {name} ignored: env cfg has no scene.terrain")
        return

    from .triangle_mesh_terrain_importer import TriangleMeshTerrainImporter

    gen = resolve_terrain_generator(name)
    terrain.terrain_type = "generator"
    terrain.terrain_generator = gen
    terrain.use_terrain_origins = True
    # Convex hull fills inverted-pyramid pits with an invisible rim-height lid.
    terrain.class_type = TriangleMeshTerrainImporter
    n_rows = gen.num_rows
    if init_level is None:
        # Stairs: start on the easiest row and let terrain_levels promote.
        # Complex: sample the full difficulty range (legacy NewBike/THUBike default).
        terrain.max_init_terrain_level = 0 if name == "stairs" else n_rows - 1
    else:
        terrain.max_init_terrain_level = int(max(0, min(init_level, n_rows - 1)))
    types = list(gen.sub_terrains.keys())
    print(
        f"[INFO] Terrain preset '{name}': {gen.num_rows} rows × {gen.num_cols} cols "
        f"types={types} max_init_terrain_level={terrain.max_init_terrain_level}"
    )


def pin_terrain_level(env: Any, level: int) -> dict[str, Any]:
    """Force every parallel env onto one curriculum row after the scene exists.

    Columns (terrain types) stay spread across ``num_envs``. Runtime terrain
    promotion is a no-op when the env curriculum manager is disabled.
    """
    terrain = env.scene["terrain"]
    if not hasattr(terrain, "terrain_levels") or terrain.terrain_origins is None:
        raise ValueError("pin_terrain_level requires a generator terrain with curriculum origins.")
    max_level = int(terrain.max_terrain_level) - 1
    level = int(max(0, min(level, max_level)))
    terrain.terrain_levels[:] = level
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    n_types = int(terrain.terrain_types.max().item()) + 1 if terrain.terrain_types.numel() else 0
    info = {
        "level": level,
        "max_level": max_level,
        "num_types": n_types,
        "n_envs": int(terrain.terrain_levels.shape[0]),
    }
    print(
        f"[INFO] Pinned terrain row L{level}/{max_level} "
        f"({info['n_envs']} envs across {n_types} column types)"
    )
    return info
