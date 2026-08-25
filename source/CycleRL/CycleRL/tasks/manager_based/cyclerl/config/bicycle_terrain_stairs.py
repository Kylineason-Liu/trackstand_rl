"""Low-complexity bicycle terrain: planes, up/down stairs, and waves.

Grid layout (curriculum mode): 8 columns × 3 rows
  - Columns = terrain TYPE, two cells each (flat, stairs up, stairs down, waves)
  - Rows    = extremity (0 → 2, shallow → steep)

Cells are 4 m — sized for NewBike (~0.28 m wheelbase). Up-stairs stay as mesh
boxes (cheap, origin on the top platform). Down-stairs are a height-field
inverted pyramid: Isaac Lab's mesh inverted stairs are overlapping solid boxes
whose pit floor is only one step thick, so bike wheels spawn inside that volume
and PhysX depenetration launches them. A height-field is a single surface with
the origin on the pit floor.

Waves are a 2-D sinusoid whose amplitude fades to z=0 over ``blend_width``
at each cell edge so neighboring flats / the wrap apron are not a cliff.
Curriculum interpolates amplitude, which sets the peak slope
(≈ 4° → 25° at three wavelengths per cell, before the edge fade).

The pit (and wave troughs) are concave. Isaac Lab's default mesh collider is a
convex hull, which caps holes with an invisible floor at rim height. Generator
terrains must use :class:`TriangleMeshTerrainImporter` so PhysX keeps the
authored triangles.

Platform stays ≥ 2 m so Isaac Lab's HF origin patch (±1 m around cell center)
sits on the pit floor, not on the first step ring.

Step height is interpolated by curriculum difficulty (1 cm → 4 cm);
effective stair slope ≈ atan(height / width).

Isaac Lab assigns one sub-terrain type per column from the cumulative
proportions via ``col / num_cols``. Equal 1/4 weights with eight columns map to
flat | flat | stairs_up | stairs_up | stairs_down | stairs_down | waves | waves.

The grid is ringed by a flat z=0 apron (wrap buffer) and a 1 cm collision
wall that is imported invisible. Isaac Lab's stock ``border_width`` wall is
not used — that was a thick visible box sitting on the outer cells.
"""

import isaaclab.terrains as terrain_gen
from .bicycle_terrain_generator import BicycleTerrainGenerator, BicycleTerrainGeneratorCfg
from .blended_wave_terrain import BlendedHfWaveTerrainCfg

# ---------------------------------------------------------------------------
# Sub-terrain size — NewBike-scale cells (wheelbase ~0.28 m)
# ---------------------------------------------------------------------------
SUB_TERRAIN_SIZE = (4.0, 4.0)  # width (x), length (y) in metres

# Treads scaled to the bike; wheel radius ~7.3 cm so hard step is 4 cm.
# Easy: 1 cm / 20 cm ≈ 2.9°.  Hard: 4 cm / 20 cm ≈ 11.3°.
_STAIR_STEP_WIDTH = 0.20
_STAIR_STEP_HEIGHT_RANGE = (0.01, 0.02)
_STAIR_PLATFORM_WIDTH = 2.2
_STAIR_BORDER_WIDTH = 0.2

# Three wavelengths in a 4 m cell → λ ≈ 1.8 m after the HF border.
# Peak slope of A(sin kx + cos ky) is A·k·√2 ≈ 4° at 1.5 cm, 25° at 9.5 cm.
_WAVE_AMPLITUDE_RANGE = (0.015, 0.095)
_WAVE_NUM_WAVES = 3

# Equal weights so 8 columns → two of each type (see module doc).
_EQUAL_PROPORTION = 1.0 / 4.0

# ---------------------------------------------------------------------------
# Bicycle stairs curriculum — planes + pyramid stairs + inverted + waves
# ---------------------------------------------------------------------------
BICYCLE_STAIRS_TERRAINS_CFG = BicycleTerrainGeneratorCfg(
    class_type=BicycleTerrainGenerator,
    seed=42,
    curriculum=True,  # columns = type, rows = difficulty
    size=SUB_TERRAIN_SIZE,
    num_rows=3,  # 3 slope extremities per type
    num_cols=8,  # 2× flat | 2× stairs up | 2× stairs down | 2× waves
    border_width=0.0,  # apron + thin wall come from BicycleTerrainGenerator
    apron_width=4.0,  # one cell of flat before wrap / wall
    wall_thickness=0.01,
    wall_height=2.0,
    hide_wall_visual=True,
    color_scheme="height",
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=_EQUAL_PROPORTION,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=_EQUAL_PROPORTION,
            step_height_range=_STAIR_STEP_HEIGHT_RANGE,
            step_width=_STAIR_STEP_WIDTH,
            platform_width=_STAIR_PLATFORM_WIDTH,
            border_width=_STAIR_BORDER_WIDTH,
            holes=False,
        ),
        "stairs_down": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=_EQUAL_PROPORTION,
            step_height_range=_STAIR_STEP_HEIGHT_RANGE,
            step_width=_STAIR_STEP_WIDTH,
            platform_width=_STAIR_PLATFORM_WIDTH,
            border_width=_STAIR_BORDER_WIDTH,
        ),
        "waves": BlendedHfWaveTerrainCfg(
            proportion=_EQUAL_PROPORTION,
            amplitude_range=_WAVE_AMPLITUDE_RANGE,
            num_waves=_WAVE_NUM_WAVES,
            blend_width=0.5,
        ),
    },
)
"""Bicycle stairs terrain: planes, up/down stairs, and waves; 8×3 NewBike-scale grid."""
