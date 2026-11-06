# Cylinder model verification — 2026-11-06

Source revision: `59c3a029a074fd079e239e83817963fa550acd0d`.
New model: `models/trackstand_rl_0916_bike_cylinder/trackstand_rl_0916.xml`.
Original model and all existing simulator code are unchanged.

## Standalone packaging follow-up

At the user's request, the cylinder model now contains its own eight mesh files
under `meshes/`, copied byte-for-byte from the original model. All XML asset
references are local. The reproduction helper also creates this standalone layout.
`standalone_verification.json` verifies a temporary copy with no original sibling:
raw MuJoCo and both production loaders compile it, matching compiled plant/geometry
arrays and producing a finite forward/one-step result. Earlier episode outputs
and hashes describe the initial shared-assets revision; no new policy rollout
was performed for this packaging-only change.

## Geometry and preservation

`build_and_verify.py` reads the original MJCF and the existing Isaac cylinder
URDF, creates the parallel XML and writes `geometry_verification.json`.
At zero joint angles, original MuJoCo axle world positions match URDF link
positions plus the root translation (maximum error < 1.6e-14 m).
For URDF link pose `(pU, RU)`, collision pose `(pc, Rc)` and MuJoCo wheel world
pose `(pB, RB)`, the new local collision transform is:

```
p_local = RB.T @ (root_translation + pU + RU @ pc - pB)
R_local = RB.T @ RU @ Rc
```

Rear local centre is `(0.203363025142151, -0.146890227667027,
0.156240423731154)` m; front local centre is
`(-0.078650359407428, -0.145670689844955, 0.156238644233956)` m.
Cylinder axes reproduce the URDF RPY values; their absolute dot products with
motor axes are 0.999999995979 (rear) and 0.999999999999946 (front).
Both have radius 0.07 m and half-width 0.0005 m. Visual meshes remain separate.

Raw XML compiles directly in MuJoCo 3.12.0, and both production loaders compile
with their own meshdir assignment. Compiled body mass/inertia/inertial pose,
body pose and joint position/axis/range/limited arrays are exactly equal to the
original. Total mass is 2.424329412407707 kg. Existing original XML SHA256 is
`6f839f2c214da9b743cbd310a72d13ed98d383e8ac4c1d1be609eb89cb003476`.
Production reset code uses a fixed root spawn Z of −0.016 m, with no mesh-only
height calculation; reset policy/noise/controller logic was not modified.

## Headless compatibility evaluation

One first episode each, seed 42, no DR, pure survival (timeout/fallen), normal
30-s horizon, original policy/physics/integral clocks and speed-limit mode:

| Pipeline | Policy | Physics / PD | Result |
| --- | --- | --- | --- |
| V1 | no-DR model_75300 | 400 Hz | 1500 steps / 30 s, timeout |
| V2 | no-DR model_26600 | 1200 Hz | 1500 steps / 30 s, timeout |

Both exited 0. Logs and evaluator-native summaries, histograms, torque plots and
NPZ traces are in `v1.log`, `v2.log`, `v1/`, `v2/`. These two seed-specific
completions establish loader/runtime compatibility, not general survival rates
or an improvement over mesh collision. No GUI or hardware test was performed.

Run from repository root:

```bash
python3 mujoco_sim2sim/reports/20261106_cylinder_model/build_and_verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/trackstand_cylinder_mpl \
python3 mujoco_sim2sim/scripts/run_deploy_eval.py \
  --mjcf mujoco_sim2sim/models/trackstand_rl_0916_bike_cylinder/trackstand_rl_0916.xml \
  --num_envs 1 --episodes 1 --seed 42 --pure_survival \
  --output_dir mujoco_sim2sim/reports/20261106_cylinder_model/v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/trackstand_cylinder_mpl \
python3 mujoco_sim2sim/scripts/run_deploy_eval_v2.py \
  --mjcf mujoco_sim2sim/models/trackstand_rl_0916_bike_cylinder/trackstand_rl_0916.xml \
  --num_envs 1 --episodes 1 --seed 42 --pure_survival \
  --output_dir mujoco_sim2sim/reports/20261106_cylinder_model/v2
```

The initial helper attempt imported system SciPy and hit its existing NumPy ABI
mismatch before any model write. The helper now uses direct RzRyRx trigonometry;
no dependency installation or existing environment modification was needed.
