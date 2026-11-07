# Independent standalone mesh-wheel URDF audit

Task `urdf_mesh_model`, single pass, iteration 1; source revision
`ee7d1eaa69f4d322071ac23dbf479d9716141dc2`.
Candidate URDF SHA256:
`4e23b41309467f6a72a38ee9d4dd0f889e690715c8012db9e6d8478b5b860511`.

- **C1 PASS:** plain XML has no stale 0914, xacro, package URI or find macro.
  All eight mesh references resolve to four real files inside the new bundle;
  SHA256 values match the existing cylinder bundle meshes. Inline silver
  material and copied license are present. No sibling model dependency.
- **C2 PASS:** all four link inertials, three complete joints and visual
  definitions match the cylinder source after normalizing mesh path names.
  All four collision definitions match the original mesh URDF. In particular,
  rear/front CAD collision origins are (-0.029465,0.025495,-0.085946) and
  (-0.312612,0.039741,-0.086364) m, not the previous cylinder offsets.
- **C3 PASS (scoped):** copied the whole bundle to an isolated temporary folder
  and loaded it through MuJoCo 3.12.0's URDF parser successfully. The active
  training USD asset tree matches HEAD. The new README correctly says no fresh
  Isaac import, generated USD or physics validation was performed.

No material finding remains. This does not establish future Isaac collision
cooking or equivalent physical contact: those require an explicit import and
physics check. No simulation evaluation, GPU or live hardware was used.

Reproduce from repository root:

```bash
python3 isaaclab_train/reports/20261106_mesh_urdf_model_audit/audit.py
```

`audit.json` retains hashes, link/joint counts, path checks and parser result.
Only this new audit directory was written by the inspector.
