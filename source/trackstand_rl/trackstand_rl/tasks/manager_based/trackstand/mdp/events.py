"""Events used by both stationary and dynamic balancing."""

import torch
from pxr import UsdPhysics

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def randomize_ground_plane_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("terrain"),
    static_friction_range: tuple[float, float] = (0.5, 1.1),
    dynamic_friction_range: tuple[float, float] = (0.5, 1.1),
    restitution_range: tuple[float, float] = (0.0, 0.0),
    num_buckets: int = 1,
    make_consistent: bool = True,
):
    """Randomize the physics material on the shared ground plane.

    The terrain importer uses a shared ground plane prim for all environments.
    This helper updates the material prim bound to that plane instead of trying to
    randomize a non-rigid TerrainImporter asset directly.
    """
    # The terrain importer is not a rigid body asset, so we use its configured prim.
    terrain = env.scene[asset_cfg.name]
    if not hasattr(terrain, "terrain_prim_paths") or len(terrain.terrain_prim_paths) == 0:
        raise ValueError("Expected terrain importer asset with imported terrain prim paths.")

    terrain_prim_path = terrain.terrain_prim_paths[0]
    material_path = f"{terrain_prim_path}/physicsMaterial"
    material_prim = env.sim.stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(
            f"Could not find ground plane material prim at '{material_path}'. "
            "Ensure the terrain importer is configured with a physics material."
        )

    # Sample material values once for the shared ground plane.
    static_friction = float(
        math_utils.sample_uniform(static_friction_range[0], static_friction_range[1], (1,), device="cpu")[0]
    )
    dynamic_friction = float(
        math_utils.sample_uniform(dynamic_friction_range[0], dynamic_friction_range[1], (1,), device="cpu")[0]
    )
    restitution = float(
        math_utils.sample_uniform(restitution_range[0], restitution_range[1], (1,), device="cpu")[0]
    )

    if make_consistent:
        dynamic_friction = min(static_friction, dynamic_friction)

    material_api = UsdPhysics.MaterialAPI(material_prim)
    material_api.CreateStaticFrictionAttr().Set(static_friction)
    material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    material_api.CreateRestitutionAttr().Set(restitution)


def __getattr__(name: str):
    """Resolve locomotion-only events from ``mdp.dynamic.events``."""
    from .dynamic import events as _dynamic_events

    try:
        return getattr(_dynamic_events, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
