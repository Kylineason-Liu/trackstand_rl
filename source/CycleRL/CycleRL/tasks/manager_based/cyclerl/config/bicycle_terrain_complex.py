"""Procedural terrain curriculum for bicycle balancing and velocity tracking.

Grid layout (curriculum mode): 5 columns × 5 rows
  - Columns = terrain TYPE (flat, rough, slopes, blocks, waves)
  - Rows    = DIFFICULTY (0 → 4, easiest → hardest)

A flat z=0 apron around the grid is the wrap buffer. The perimeter wall is
1 cm thick and imported invisible.
"""

import isaaclab.terrains as terrain_gen
from .bicycle_terrain_generator import BicycleTerrainGenerator, BicycleTerrainGeneratorCfg
from .blended_wave_terrain import BlendedHfWaveTerrainCfg

# ---------------------------------------------------------------------------
# Sub-terrain size — large enough for the bike to accelerate and navigate
# ---------------------------------------------------------------------------
SUB_TERRAIN_SIZE = (10.0, 10.0)  # width (x), length (y) in metres

# ---------------------------------------------------------------------------
# Bicycle terrain curriculum
# ---------------------------------------------------------------------------
BICYCLE_TERRAINS_CFG = BicycleTerrainGeneratorCfg(
    class_type=BicycleTerrainGenerator,
    seed=42,
    curriculum=True,  # columns = type, rows = difficulty
    size=SUB_TERRAIN_SIZE,
    num_rows=5,       # 5 difficulty levels per type
    num_cols=5,       # 5 terrain types
    border_width=0.0,  # apron + thin wall come from BicycleTerrainGenerator
    apron_width=8.0,  # almost one cell of flat before wrap / wall
    wall_thickness=0.01,
    wall_height=2.0,
    hide_wall_visual=True,
    color_scheme="height",
    horizontal_scale=0.1,    # 10 cm — fewer HF cells than 5 cm (cheaper PhysX / GPU)
    vertical_scale=0.002,    # 2 mm vertical precision
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        # =====================================================================
        # Column 0 — Flat (baseline balancing & velocity tracking)
        # =====================================================================
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2,
        ),
        # =====================================================================
        # Column 1 — Rough / Uneven (mild bumps; keep soft for GPU + learning)
        # =====================================================================
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            # Heights are absolute z samples in [min, max]. Keep noise_step << range
            # span — Isaac Lab's arange can otherwise jump past max (e.g. step=0.1
            # with max=0.008 produced ~10 cm spikes).
            noise_range=(0.0, 0.03),  # max bump 2.5 cm (< 3 cm recovery limit)
            noise_step=0.005,         # 5 mm steps inside the range
            border_width=0.25,
        ),
        # =====================================================================
        # Column 2 — Sloped (inclines/declines for motor torque & pitch stability)
        # =====================================================================
        "slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.05, 0.43),    # difficulty 0→1:  ~3° → ~30° slope
            platform_width=1.0,
            border_width=0.25,
        ),
        # =====================================================================
        # Column 3 — Discrete Blocks / Speed Bumps (impact recovery)
        # =====================================================================
        "blocks": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.10, 0.50),   # width of individual blocks
            obstacle_height_range=(0.02, 0.12),  # difficulty 0→1: 2 cm → 12 cm
            num_obstacles=25,
            platform_width=2.0,
        ),
        # =====================================================================
        # Column 4 — Waves / Cambered roads (cross-slope balance & lean correction)
        # =====================================================================
        "waves": BlendedHfWaveTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.02, 0.12),  # difficulty 0→1: 2 cm → 12 cm amplitude
            num_waves=3,
            blend_width=0.75,
        ),
    },
)
"""Bicycle-specific terrain curriculum with 5 types × 5 difficulty levels."""
