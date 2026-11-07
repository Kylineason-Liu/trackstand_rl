# Standalone wheel-mesh URDF verification — 2026-11-06

Base revision: `ee7d1eaa69f4d322071ac23dbf479d9716141dc2`.
New asset bundle: `isaaclab_train/trackstand_rl_0916_description_mesh/`.
Only the new asset directory and this report directory were written.

`build_and_verify.py` clones the cylinder URDF's XML structure, restores wheel
collision elements from `trackstand_rl_0916_description/urdf/trackstand_rl_0916.urdf`,
rewrites every mesh resource to `../meshes/<name>.stl`, copies the four STL files,
inlines the original silver material and removes stale xacro dependencies.
No old generated USD, launch/CMake or obsolete package metadata was copied.

Checks in `verification.json`:

- Four links / three joints; wheel collisions are meshes, no cylinders remain.
- All inertials, joints and visual attributes preserved after normalizing only
  the resource path names; total mass 2.424329412407707 kg.
- All four collisions structurally match the original mesh URDF.
- Rear collision origin: `-0.029465 0.025495 -0.085946`, RPY `0 0 0`.
- Front collision origin: `-0.312612 0.039741 -0.086364`, RPY `0 0 0`.
- Eight mesh references resolve inside the new bundle; four source/copy SHA256
  pairs match exactly. Source license is copied unchanged.
- A temporary isolated copy resolves all resources and loads using MuJoCo
  3.12.0's URDF parser (three joints, four collision geoms). This verifies an
  independent URDF parser can load it, not Isaac importer behaviour or physics.
- All 50 pre-existing files under both original source directories remain
  hash-identical before/after creation.
- No fresh Isaac import, generated USD, policy evaluation or active configuration
  switch was performed. Contact cooking settings remain an import-time choice.

Reproduce from repository root (requires NumPy/MuJoCo already available here):

```bash
python3 isaaclab_train/reports/20261106_mesh_urdf_model/build_and_verify.py
```

The helper recreates only the new URDF/copied meshes/license and verification
JSON; it preserves the original folders and active task assets.
