# Trackstand 0916 with cylinder wheel collisions

Parallel variant of `../trackstand_rl_0916_bike/trackstand_rl_0916.xml`.
Original nominal masses are retained: total **2.4243294124 kg**. This is not the
measured-mass variant. Inertias, centres of mass, joints, visuals, body/fork
collisions and contact defaults are unchanged. All mesh assets are included in
this folder under `meshes/`; the model can be copied and loaded independently
of the original sibling directory.

Both wheel collision geoms are MuJoCo `cylinder` primitives, radius **0.07 m**,
full width **0.001 m** (`size="0.07 0.0005"`; second number is half-width).
They reproduce the collision transforms authored in the Isaac cylinder URDF,
including the rear cylinder's **−0.018 m offset along URDF Y** and the front
cylinder centred on its axle. Those transforms are converted into the distinct
MuJoCo CAD body frames, not copied numerically. The original slight axle/cylinder
orientation discrepancy is preserved; this does not assert identical contact
solving between engines.

Run v2 no-DR with its default no-DR model_26600, from `mujoco_sim2sim/`:

```bash
python3 -u scripts/run_deploy_viewer_v2.py \
  --mjcf models/trackstand_rl_0916_bike_cylinder/trackstand_rl_0916.xml \
  --num_envs 64 --physics_hz 1200 --seed 42 --pure_survival
```

Omit `--pure_survival` to use all six v2 termination conditions. No existing
viewer defaults changed. This model also loads through the v1 evaluator.

[Verification and exact evaluation commands](../../reports/20261106_cylinder_model/README.md)
cover raw compilation, compiled inertial/joint equality, geometric transforms,
and one no-DR episode per v1/v2 policy. They are compatibility checks, not a
statistical comparison or proof of improved transfer.
