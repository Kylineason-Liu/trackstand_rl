# Cylinder model independent audit

Task `cylinder_model`, iteration 1, single pass. Candidate base revision
`59c3a029a074fd079e239e83817963fa550acd0d`; new XML SHA256
`9ccd397cbf6adabba8ac2858a35df0d91c0af5f18108aaa986783e22a4f67e87`.

All four criteria pass within the scope below. No production edits by inspector.

- **C1 preservation:** original XML byte-identical to HEAD; new sibling directory
  leaves runtime defaults and existing models intact. Mesh references resolve to
  existing assets. Direct XML compilation and both production loaders succeed.
- **C2 plant:** compiled body masses/inertias/COMs/transforms and joint
  positions/axes/ranges/limits exactly equal baseline under v1 and v2 loaders.
  Compared collision masks, friction and contact solver parameters also equal.
  Total mass remains 2.424329412407707 kg.
- **C3 geometry:** both wheel geoms are cylinders, radius 0.07 m and half-length
  0.0005 m. Independent direct-trigonometry calculation from the URDF RPY and
  original MuJoCo wheel transforms matches world centers/axes within 1e-12.
  Rear center uses axle + world (0,-0.018,0) at zero state; front is at axle.
  CAD-local rear center is (0.203363025142,-0.146890227667,0.156240423731) m.
  The inherited slight cylinder-axis/axle mismatch is preserved, not silently
  corrected. Shared visual meshes and body/fork collisions are unchanged.
- **C4 execution:** inspected implementer evaluation metadata/summary files
  under `../20261106_cylinder_model/{v1,v2}`. Each is seed 42, no DR,
  pure survival, one environment/one completed 1500-step (30 s) episode.
  V1 model_75300 at 400 Hz and v2 model_26600 at 1200 Hz both time out.
  These are compatibility/smoke results, not comparative robustness evidence.

Reproduce the independent geometry check from repository root:

```bash
python3 mujoco_sim2sim/reports/20261106_cylinder_model_audit/audit.py
```

`audit.json` records exact array checks and geometry errors. No duplicate policy
rollout, GPU or hardware work was done by inspector. A first optional SciPy
import failed due to the local NumPy ABI; the final checker avoids SciPy and uses
NumPy, MuJoCo and direct trigonometry. No package changes were made.
