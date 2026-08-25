"""Wave height-field whose amplitude fades to z=0 at the cell edge.

Isaac Lab's stock ``HfWaveTerrainCfg`` is ``A (sin kx + cos ky)`` pasted into a
zero pad. At high amplitude the pad is a cliff; ``slope_threshold`` then makes
those faces vertical. This variant multiplies by a raised-cosine window so the
wave meets neighboring flats / the wrap apron at ground height.
"""

from __future__ import annotations

import numpy as np

from isaaclab.terrains.height_field.hf_terrains_cfg import HfWaveTerrainCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils.configclass import configclass


def _edge_taper(n: int, blend_pixels: int) -> np.ndarray:
    """Hann ramp: 0 at both ends, 1 in the interior."""
    w = np.ones(n, dtype=np.float64)
    b = int(max(0, min(blend_pixels, n // 2)))
    if b <= 0:
        return w
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, b, endpoint=True)))
    w[:b] = ramp
    w[-b:] = ramp[::-1]
    return w


@height_field_to_mesh
def blended_wave_terrain(difficulty: float, cfg: "BlendedHfWaveTerrainCfg") -> np.ndarray:
    """Same sinusoid as Isaac Lab ``wave_terrain``, tapered to z=0 at the border."""
    if cfg.num_waves < 0:
        raise ValueError(f"Number of waves must be a positive integer. Got: {cfg.num_waves}.")

    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude_pixels = int(0.5 * amplitude / cfg.vertical_scale)

    wave_length = length_pixels / max(cfg.num_waves, 1)
    wave_number = 2.0 * np.pi / wave_length
    xx = np.arange(0, width_pixels, dtype=np.float64).reshape(width_pixels, 1)
    yy = np.arange(0, length_pixels, dtype=np.float64).reshape(1, length_pixels)

    hf_raw = amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))
    blend_pixels = int(float(cfg.blend_width) / cfg.horizontal_scale)
    window = _edge_taper(width_pixels, blend_pixels)[:, None] * _edge_taper(length_pixels, blend_pixels)[None, :]
    hf_raw *= window
    return np.rint(hf_raw).astype(np.int16)


@configclass
class BlendedHfWaveTerrainCfg(HfWaveTerrainCfg):
    """Wave cell that blends into surrounding z=0 ground at the border."""

    function = blended_wave_terrain
    blend_width: float = 0.5
    """Distance (m) over which amplitude fades to zero at each edge."""
    border_width: float = 0.0
    """Keep Isaac Lab's hard zero-pad at one pixel so the fade can meet z=0."""
