# NewBike MuJoCo deployment check (sim-to-sim)

Deploys the CycleRL **NewBike-Balance-v1** policy (`09090950` run, `model_500` —
1500/100% timeouts in Isaac Sim `play_eval`) into a **MuJoCo** simulation of the
same bike, to validate the deployment artifact and I/O contract *before* the
real hardware (Raspberry Pi + ROS2) step.

## Layout

```
mujoco_sim2sim/
├── models/newbike/
│   ├── model.xml     # the robot: free base + 3 hinges, actuators, sensors,
│   │                 #   inertials — all numbers from the Isaac Sim USD payloads
│   └── scene.xml     # environment: flat floor (friction 1.0, DR-free like 0950)
├── policies/
│   ├── model_500.onnx  # the exported deployment artifact
│   └── model_500.pt    # rsl_rl checkpoint (torch fallback backend)
├── src/newbike_mujoco/
│   ├── env.py        # BikeBalanceEnv: reset/obs/step/terminations, contract replica
│   └── policy.py     # DeployedPolicy: ONNX via onnxruntime, .pt fallback
├── scripts/run_deploy_eval.py
├── requirements.txt
└── README.md
```

## Deployment contract (what the Pi will have to do too)

| Piece | Value |
|---|---|
| Control rate | 50 Hz (0.02 s); physics at 0.005 s, 4 substeps per control |
| Observation | 13-dim: `[gravity_y, gravity_z, ang_vel xyz, lin_vel xy, steer_pos, drive_pos, steer_vel, drive_vel, sin_steer, cos_steer]` — rad / rad/s, gravity dimensionless, **no normalization**, drive_pos **wrapped** to ±π (the sim's joint-pos wrap) |
| Action 0 (steering) | `a0 × 1.57` rad **position target**, PD servo kp=10, kv=0.2, ±5.6 N·m, **slew-limited 23.9 rad/s**, joint limits ±90° |
| Action 1 (drive) | `a1 × 6.28` N·m **direct torque** with −0.05·q̇ damping, ±6.28 N·m, speed capped ±61.6 rad/s |
| Reset | roll ~ U(−0.10, 0.10), roll rate ~ U(−0.25, 0.25), joints 0, wheels just off the floor |
| Episode | 1500 steps (30 s) or death: \|roll\|>30° or inverted, \|yaw\|>30°, net wheel > 7.2 rad, drift > 0.5 m |

## Run

```bash
python scripts/run_deploy_eval.py --episodes 64 --seed 42
```

`pip install onnxruntime` to run the exact ONNX; without it the .pt checkpoint
is used (identical mean-action policy, verified against the sim's ground truth:
the torch fallback reproduces the sim's policy outputs to float32 precision).

## Status (2026-09-10)

**Working:** the model loads, the deployment loop runs end-to-end, and the
**contract is verified**: replaying the sim's own action sequence from the
sim's own initial state, the MuJoCo steering/roll trajectories track the sim
closely for several steps, and the closed-loop policy actions match the sim's
step-for-step at first (step 1: `[0.42, 0.28]` vs sim `[0.4141, 0.2701]`).

**Open:** the policy now almost holds the 30 s horizon (64-episode eval,
iteration 5: **mean 1074.8 steps / 21.5 s, 1.6% timeouts** — one full 1500-step
episode; deaths wheel-net 95%, drifted 3%; **roll RMS 0.87 deg** vs the sim's
0.38; max roll 2.46 deg). Iteration-5 findings (the sim's contact model,
reverse-engineered and reproduced):
1. **the sim's wheel contacts are POINTS that slide laterally** (PhysX
   convex-hull-vs-plane gives one deepest vertex that walks as the wheel
   rotates; the bike tips about its CoM like a frictionless-support pendulum,
   theta_dd ~ 0.5*(g/h)*sin(theta), measured). A MuJoCo cylinder makes a LINE
   contact whose lateral span covers the CoM and props the bike upright.
   FIXED: sphere contacts at the hull's effective pivot positions — the rear
   at the right sidewall (y=-0.0502, fitted to the sim's zero-drive fall,
   rms 0.45 deg) and the front at the tire-bottom point.
2. **the steering->roll coupling comes from the TILTED front wheel**: the
   wheel center lies ON the steering axis (the USD front link pos is parallel
   to the rake axis), so steering does not translate the wheel — it tilts the
   disk and swings the ground contact laterally. A sphere at the wheel center
   is rotation-invariant and kills the coupling (the policy's tripod balance
   needs it). The sim's constant-a0 probe fixes the SIGN: a0 up -> roll LEFT,
   so the contact swings RIGHT (the mirrored trail point, fitted magnitude
   ~3.2 mm off the axis — the full 12.9 mm trail overshoots and anti-damps).
3. **the sim's articulation behaves ~x30 heavy**: the rear wheel's step
   response fits I = 0.005-0.0072 (x28-40 of the USD) and the pitch x32.
   FIXED: the wheel diaginertia x35 and the base x3000 (the wheel matches the
   measured response; the base's extra x100 is a closed-loop stability
   calibration — KNOWN COMPROMISE: the uncontrolled zero-drive fall is ~100x
   slower than the sim's; the closed-loop roll dynamics are what it targets).
   The USD-scale fork inertias restored (the x30/x120 hacks slowed the servo
   so the policy's countersteer could not move the fork).
4. **the spawn-drop transient**: the fork's servo swing carries angular
   momentum about the raked axis and kicks the base's roll (+5 deg in 2 steps
   at the USD inertia; the sim stays +-0.7). Damped by the heavy base.
5. **the policy's balance is a static TRIPOD**: the sim's policy pins the
   steering at +1.336 rad (76 deg) — the steered contact geometry gives a
   stable support — with a0 +0.865 and a1 +0.07+-0.21. MuJoCo now matches:
   fork +1.34, a0 +0.85, a1 -0.06, the wheel +-10 rad/s (the sim's +-3.7).
Remaining for iteration 6: the wheel still nets out at ~21 s (mean 0.33 rad/s
drift vs the sim's ~0.01) fed by the residual roll sway (+-2.4 deg vs the
sim's +-0.14), and the bike creeps forward (~2 cm/s) until the drifted term
fires.

Iteration 6 (2026-09-11, two user-observed defects + the pitch finding):
1. **the front wheel looked motorized**: with no ground contact on the front
   wheel body (the balance contact is the steering-body sphere), the drop
   transient's spin never decayed (frictionless hinge). FIXED: a small
   bearing damping (0.001) on frontwheel_joint -- the spin decays in ~0.7 s.
2. **the fall looked like slow motion**: the x3000 base inertia (the
   iteration-5 closed-loop calibration). FIXED: the base is back at the USD
   values -- the zero-drive fall matches the sim's trajectory again
   (roll -1.3/-3.7/-8.1/-16.4 at steps 10/15/20/25 vs the sim's
   -0.9/-2.9/-6.9/-15.4; fallen at step 30). The rear contact also gained
   the sim's fitted rolling resistance (0.04).
3. **the loop's stability vs the fall's realism is a hard trade-off** (the
   iteration-6 core finding): the sim's pitch is stabilized by a contact
   impulse cycle (its drive-only pitch peaks at -2.7 deg with the front
   wheel lifting 2.3 cm and re-impacting at ~0.7 m/s); the effective pitch
   inertia is ~x4 of the rigid USD value, which CANNOT be reproduced: the
   USD inertia sits at the triangle-inequality edge (A+B>=C), so no
   meaningful pitch-only scaling is physically possible, and a uniform
   scale that gives the pitch its x4 makes the roll fall x4 too slow. The
   measured trade-off (base scale: fall@20 / drive-only pitch@4 / mean
   survival): x1: -8.1 / -24.6 / 25; x3: -5.6 / -17.6 / 17; x10: -2.6 /
   -8.9 / 33; x100: -0.5 / -1.3 / 21; x3000: -0.3 / 0.0 / 1075 (the sim:
   -6.9 / -2.7 / 1500). The `--base_inertia` flag on the viewer/eval scripts
   selects the point on this curve; the default is the USD value (the
   full-gravity fall, the honest dynamics), and --base_inertia 3000 recovers
   the long survivals at the cost of the slow-motion fall. Closing the loop
   at the honest inertia needs the sim's contact-impulse pitch dynamics,
   which the rigid-body model cannot express -- the open research item.

Iteration 7 (2026-09-11, the front wheel's friction, user-observed):
1. **the front wheel looked locked and its friction was wrong**: at ~90 deg
   steering the bike still moved front-back freely. The steering-body point
   sphere slid in every direction. FIXED: the front wheel now has its REAL
   rolling contact -- a thin cylinder (r=0.07, half-height 0.001) on the
   front wheel body with its axis along the axle. It rolls in the wheel's
   plane (the spin couples to the ground, no free spin, no artificial
   bearing damping -- the iteration-6 damping is removed) and skids
   (friction 0.5) across the plane, so at ~90 deg steering the front wheel
   blocks the front-back motion like a real wheel (measured: +1 cm vs the
   old free slide under the same drive). The redundant steering-body swing
   sphere is removed (the closed loop is identical with/without it -- the
   tilted-cylinder swing now provides the steering->roll coupling).
2. the honest-base closed loop improved: mean 31.8 steps (was 25.7), max 140.

Iteration 8 (2026-09-11, the Fusion CAD export + the free-body pitch inertia):
1. **the Fusion export (trackstand_rl_bike/) validated the wheel physics**:
   the front tire's outer radius is 70.2 mm (the model's 0.07 m is exact), the
   tire width is ~12 mm, and the wheels' inertia tensors match the USD values
   to float precision. The export's own MJCF cannot load in MuJoCo (its STLs
   exceed the 200k-face decoder cap and contain degenerate triangles) and the
   files are treated read-only -- the export is used as the measurement
   reference only.
2. **the sim's free-body pitch inertia is ~x24 of the USD** -- the first
   contact-free measurement of the sim's base inertia: during the spawn drop
   (no contacts), the sim's drive-only pitch rate after one step is
   -0.39 rad/s (mine at x24: -0.63; at x1: -5.97). The x24 uniform scale is
   triangle-valid and brings the drive-only pitch to -4.6 deg @4 (the sim's
   -2.7). KNOWN RESIDUE: at x24 the zero-drive fall is capped at ~45-70% of
   the sim's rate -- the effective pivot depth is bounded by the sidewall
   contacts (~6 mm) while the sim's vertex-walk reaches ~9 mm -- and the
   closed loop still nets out at ~19 steps (the drive->roll transient of the
   first pulses kicks +2.7 deg vs the sim's +-0.5). The default model keeps
   the USD base (the exact fall, the 31.8-step loop); --base_inertia selects
   the x24 or x3000 points.
3. **the inertia probe (2026-09-12, live sim)**: the sim's articulation
   inertias ARE the USD's authored values (the eigenvalues match to 8
   decimals on all four bodies) -- the x24-35 effective heaviness is NOT
   mass or inertia. The masses differ modestly (total +20%; the base +21%,
   the steering +76%, the front wheel -20% -- the auto-generated collider
   masses, apparently lumped at the CoM). The actuator contract and the joint
   friction/armature are clean. Yet the dynamics remain slow: the drive-only
   wheel spin-up plateaus at 23.11 rad/s (the effective I x61 of the USD) and
   the base's pitch rate plateaus at -0.39 rad/s (x24.5) -- CONSTANT
   velocities under the constant 6.28 Nm torque, which the rigid physics
   cannot produce. Overriding the solver velocity iterations (8) and the env
   decimation (1) changed nothing. The slowdown is therefore a PhysX
   solver/actuator-level behavior that no rigid-body model expresses -- the
   MuJoCo inertia scaling remains the pragmatic reproduction of it, and the
   exact mechanism (the TGS articulation velocity resolution or the implicit
   actuator's internal semantics) is the open question.
   **RESOLVED (iteration 12, 2026-09-13)**: the run's logged env.yaml shows
   the solver is TGS (solver_type 1), and the bike's spawn-time articulation
   props (bicycle_cfg.py NEW_BIKE_BASE_CFG) author solver_position_iteration
   _count=4 / solver_velocity_iteration_count=2 ("0 gave inaccurate
   velocities -> NaN blow-ups"). Raising them to 16/8 BEFORE the spawn makes
   the wheel spin up per the rigid physics (it slams the +-61.6 rad/s speed
   limit and oscillates -- no more 23.11 plateau). The ~x24.5 effective
   heaviness is therefore the TGS's under-resolution of the light bodies at
   4/2 iterations -- the policy was trained on those dynamics, so the model's
   x24.6 wheel / kv 0.2717 / --base_inertia 24 are the correct reproduction
   of the trained-on behavior, not approximations of the authored physics.
   **What x24.5 scales (iteration 13, 2026-09-13)**: the EFFECTIVE
   ROTATIONAL INERTIA OF THE ACTUATED DOFs. The under-iterated TGS delivers
   only ~1/24.5 of the actuator's torque impulse per step (the measured pitch
   rate under the full 6.28 Nm is -0.39 rad/s vs the rigid -9.7), so the
   actuated degrees of freedom -- the wheel's spin, the base's pitch
   reaction, the steering's swing -- accelerate 24.5x slower, exactly as if
   their rotational inertia were 24.5x larger. The gravity/contact-driven
   DOFs (the fall's roll) go through the position iterations (adaptive
   1-255 at the scene level) and stay rigid -- which is the measured
   roll-fast/pitch-slow anisotropy. A torque-scaled variant (the actuators'
   gains /24.5 with the true USD inertias) was tested as the more faithful
   mechanism: the loop regressed (26.0 vs 39.3 steps), so the model keeps
   the inertia-scaled wheel.
7. **iteration 15 (2026-09-13, the tripod equilibrium)**: the trace showed
   the steady state is sim-like (fork 1.28, a0 0.82, a1 -0.03) but the roll
   drifts slowly (+0.06 deg/step) inside the policy's dead-band until the
   sway develops. The rear contact's lateral position trades the drift
   against the yaw: -0.0197 (yaw-dominated), -0.0217 (fallen-dominated),
   -0.0207 the best balance. Eval: mean 74.3 steps / 1.49 s (from 39.3), max
   163, deaths spread (wheel_net 31% / fallen 31% / yaw 20% / drifted 17%).
4. **iteration 9 (2026-09-12, the clean drive measurements)**: the plateau
   analysis of the in-the-air drive response gives the sim's wheel steady
   state at 23.11 rad/s per +1 a1-unit -> the drive's effective damping
   c = 0.2717 (the old 0.16 was a ground-loaded closed-loop fit polluted by
   the traction), and the sub-plateau rise gives the wheel's effective
   inertia x24.6 of the USD -- the SAME factor as the free-body pitch's
   x24.5: the sim's angular dynamics are uniformly ~x24.5 heavy. The model
   now uses kv 0.2717 and the wheel diaginertia x24.6. Eval (the default USD
   base): mean 39.8 steps (from 31.8), the deaths now diversified: wheel_net
   55% / yaw 22% / fallen 22% / drifted 2% (the wheel-net is no longer the
   single dominant killer), roll RMS 4.93 deg. The sim's roll is NOT x24.5
   effective (the fast fall constrains it to ~x1-3), so the base stays at the
   USD value.
5. **iteration 10 (2026-09-12)**: the x3 base (where the wheel-net nearly
   vanishes, 1/16) reveals the YAW as the next killer (12/16) -- the bike
   wanders instead of forming the static tripod, because the fork never pins
   at +1.34 rad. A trail-like x-offset on the front contact (the steering->
   roll authority, swept 3-13 mm) helps modestly; the loop plateaus at ~39-43
   steps across the base-scale and swing knobs. The model keeps the USD base
   (the exact full-gravity fall) with the 3 mm trail offset. Eval: mean 39.3
   steps, deaths split three ways (wheel_net 41% / yaw 30% / fallen 30%) --
   no single dominant killer remains; the balance quality itself (the +-5 deg
   roll sway vs the sim's +-0.14) is now the limiting factor.
6. **iteration 11 (2026-09-12, the obs/action audit) -- verdict**: the
   OBSERVATION space is verified exact (the field order, the body-frame
   transforms, the drive_pos wrap, the 13-entry noise stds -- all match
   balance_imu_state and the cfg). The ACTION space audit initially flagged
   the drive as a velocity servo from the git commit's cfg -- WRONG: the
   run's OWN logged config (params/env.yaml in 2026-09-09_09-50-48) shows
   JointEffortAction(scale=6.28) on the implicit actuator (stiffness 0,
   damping 0.05 authored) -- the torque contract, as originally modeled.
   The git commit's JointVelocityActionCfg(scale=10) is a post-run
   modification. The one real update: the EFFECTIVE drive damping is 0.2717
   (the in-the-air plateau's steady-state fit), not the authored 0.05. The
   velocity-servo experiment and the CoM-rear experiment are reverted; the
   eval is back at the honest 39.3-step state (log deploy_sim_09122237.log
   documents the wrong-experiment numbers and is superseded).

Audit-matched (2026-09-11, full env comparison vs 0950's env.yaml): reset
protocol (fixed spawn state, no roll randomization), spawn height 0.02,
always-on #8 obs noise (0.001 / 0.0012), diverged thresholds (20/50/300),
wheel-ground friction (robot 0.5 / floor 1.0 — measured as not behaviorally
relevant), terminations, drive_pos wrap, self-collisions off.

The MuJoCo-specific mechanics discovered and fixed along the way (all
documented in model.xml / env.py): MuJoCo's body-framequat sensor returns the
INERTIAL frame (env reads the free-joint qpos instead); MuJoCo evaluates
actuator forces explicitly, so the steering servo needs ≤5 ms substeps; the
sim's velocity limits are motor speed caps that need explicit emulation;
self-collisions are off in the sim (excluded pairwise here).

## Model provenance

Masses, CoMs, principal inertias, joint anchors/axes, and wheel geometry were
extracted from `source/CycleRL/CycleRL/tasks/manager_based/cyclerl/assets/
new_bike2/payloads/Physics/physics.usda` (+ `base.usda`, `geometries.usd`); the
actuator contract from `bicycle_cfg.py:NEW_BIKE_HYBRID_CFG` and the
`NewBike-Balance-v1` env cfg (position servo steering + torque drive). The USD
quaternions are authored real-part-first; joint axes were recomputed to body
frames accordingly (rear ≈ +Y, front ≈ −Y, steering ≈ −Z with ~10.6° rake).

## Engine audit (2026-09-13): integrators, solvers, and the full mismatch sweep

Full comparison of NewBike-Balance-v1 (PhysX TGS, per 2026-09-11_01-38-54's
logged env.yaml) vs this MuJoCo model, emphasizing integrators and solver
iterations. Severity-ordered; the sim-side items are mirrored in
CycleRL's logs/rsl_rl/newbike_balance_rsl/cfg_record.txt.

### 1. (known, sim-side root cause) 4/2 solver iterations
NEW_BIKE_BASE_CFG spawns with solver_position_iteration_count=4 /
solver_velocity_iteration_count=2. At 2 velocity sweeps the TGS under-delivers
the effort-actuated drive per step — the ~x24.5 sluggishness the policy
trained on. Probe: 16/8 restores the rigid response. Compensated here by the
rear-wheel diaginertia x24.6 + the drive actuator's effective kv 0.2717.

### 2. (CONFIRMED 2026-09-13, sim-side primary cause) enable_external_forces_every_iteration=False
Actuator forces are external forces in TGS; with the flag off they are
injected once per step and only the 2 velocity sweeps propagate them. Probed
(in-air welded-root, /tmp/sim_extforce_probe.py): at 4/2 + False the wheel
plateaus at 23.11 rad/s (a1>=0.6) — the x24.5 signature; flipping the flag
True at the SAME 4/2 restores the rigid physics (61.60 rad/s, the velocity
limit; effort = the authored 6.28*a1 - 0.05*w exactly). The 16/8 iteration
fix and the flag flip are two independent repairs of the same truncation.

### 3. (FIXED 2026-09-13) wheel friction regime
The sim's tire material is the global physxMaterial 0.5 (the USD carries no
per-wheel physics material), the ground pinned 1.0 — PhysX's default AVERAGE
combine makes the effective pair friction 0.75 (nominal; the per-reset DR
randomizes the tire 0.5-1.2). MuJoCo combines pair friction by MAX (verified
empirically), so all ground-touching surfaces now carry sliding 0.75 with
torsional/rolling 0 (the sim has neither). Note: MuJoCo's friction triple is
(sliding, torsional, rolling) — the earlier audit text misread the orange-bike
"1.0 0.005 0.0001" as longitudinal/lateral anisotropy; it was sliding 1.0 with
a tiny torsional term.

### 4. (standing, measured) +20% collider mass in the sim
The sim's runtime masses run ~+20% above the USD payload masses (default-
density collision geometry; the USD colliders declare no density). This model
carries only the payload masses (2.3996 kg). Un-compensated.

### 5. (standing) spawn/reset mismatch
The sim's initial_state event (09110138): root z=0, roll U(-0.1, 0.1) rad,
roll rate U(-0.25, 0.25) rad/s every reset. env.py resets z=0.02, roll 0,
rate 0 (a leftover assumption from 0950's zeros protocol).

### 6. (by design) DR absent in MuJoCo
The sim's evals randomize mass 0.5-1.5x, CoM, tire friction U(0.5, 1.2), and
actuator gains 0.9-1.1x per reset. MuJoCo runs nominal-only. Compare against
the sim's nominal envelope, not its DR-averaged eval mean.

### 7. (minor, MuJoCo-side) integrator & solver fidelity
Semi-implicit Euler (vs the sim's implicit TGS): slow energy drift on the
undamped free-spinning front wheel (the orange bike's RK4 + tol 1e-10 is the
high-fidelity reference). Pyramidal friction cone (default) vs the orange
bike's elliptic. Contact stiffness (solref 0.005 1.5) is a heuristic for the
sim's hard TGS contacts — never calibrated.

### 8. (minor, emulation) velocity limits + clamps
The sim clamps joint velocity inside the solver; env.py clamps qvel per
0.0025 s substep and slews the steering target. Equivalent at this dt, but
emulation, not the same operator. (Also stale: env.py's docstring "2
substeps" and the DRIVE_KV velocity-servo comments — leftovers of the
reverted contract.)

### 9. (model-level, no fix) GPU patch friction
The sim's GPU patch-based friction (friction_offset_threshold 0.04,
friction_correlation_distance 0.025) has no MuJoCo analog; per-contact
analytic friction is a different model.

## Friction: the full resolution (2026-09-13)

Summary of the friction discussion and its resolution:

**The sim's authored friction (NewBike-Balance-v1, 09110138's env.yaml + USD).**
The robot carries the global physxMaterial 0.5 (isotropic sliding; the USD has
no per-wheel physics material — materials.usda holds only the visual
"silver"). The ground is pinned at 1.0 by the startup event. PhysX's default
combine mode is AVERAGE, so the effective wheel-ground pair = (0.5+1.0)/2 =
**0.75**. The per-reset DR randomizes the tire 0.5-1.2 (pair 0.75-1.1); the
deploy targets the nominal 0.75. PhysX has no torsional/rolling terms.

**MuJoCo's friction semantics.** The friction vector is (sliding, torsional,
rolling), NOT longitudinal/lateral: sliding = Coulomb resistance to
translation in the tangent plane (dimensionless); torsional = resistance to
spinning about the contact normal (torque <= mu*N, length-dimensioned);
rolling = rolling resistance about tangent-plane axes (torque <= mu*N,
length-dimensioned). Pair friction combines by **MAX** (verified empirically:
1.0x0.5->1.0, 0.75x0.5->0.75, 0.3x0.5->0.5), so to reproduce a pair value,
put it on BOTH surfaces.

**The resolution.** Every ground-touching surface carries "0.75 0 0": the
floor, the frame/fork boxes (the <default>), both wheel ellipsoids, and the
compose_scene_xml floor in env.py. Verified against the compiled model: all
live contacts read mu = 0.750. This replaces the orange-bike A/B friction
(1.0 / 0.005 / 0.0001 — sliding 1.0 with a tiny torsional term, earlier
misread as longitudinal/lateral anisotropy) and removes the old hand-tuned
0.04 rolling-friction emulation: zero torsional/rolling is the sim's regime.

## Collision meshes: the pipeline and the representation gap (2026-09-13)

**The pipeline on each side.**
- Sim (NewBike-Balance-v1): the USD's collision meshes (from the URDF export)
  are marked in payloads/instances.usda with PhysicsCollisionAPI +
  MeshCollisionAPI and `token physics:approximation = "convexHull"` — PhysX
  cooks ONE convex hull per mesh at load. The cfg can override this via
  `mesh_approximation_name` (boundingCube | boundingSphere | convexDecomposition
  | convexHull | none | meshSimplification | sdf); NEW_BIKE_BASE_CFG sets no
  override, so the USD's authored hulls stand. PhysX has NO ellipsoid/cylinder
  primitive collider.
- MuJoCo (deploy_sim): the collisions are hand-authored PRIMITIVES in model.xml
  (frame box, fork box, the two wheel ellipsoids). The mesh geoms are
  visual-only (contype 0). No cooking — the geom IS the runtime collider.

**The problem.** The two engines collide DIFFERENT shapes for the same wheel:
the sim's faceted hull (vertex-walk contact) vs MuJoCo's smooth ellipsoid.
That shape difference is the root of the contact gap; the common primitive
menus do not overlap for wheels (PhysX: sphere/capsule/box/hull/mesh/SDF;
MuJoCo: sphere/capsule/ellipsoid/cylinder/box/mesh/SDF).

**The solution paths.**
1. Sphere on both engines: one token (`boundingSphere`) in the sim + one
   `<geom type="sphere">` in MuJoCo — the shape gap closes by construction;
   the point-contact regime changes the physics on both sides -> retrain.
2. SDF on both: `approximation="sdf"` in the sim + the MuJoCo SDF plugin
   (proven by the orange bike's torus) — the true tire geometry on both sides;
   most plumbing, retrain.
3. Status quo: keep the sim's hulls + keep hand-fitting MuJoCo primitives —
   inherently approximate.

**Decision (pending 09131837's outcome)**: bank the solver-fix run first, then
prototype path 1 (cheap) and/or path 2 (fidelity) before the next retrain.

## The three-way deploy comparison (2026-09-14): the compensation era ends

All evals 64 episodes, seed 42, the current contact regime (sim-matched
friction 0.75, the ellipsoid wheel prims):

| Config | mean ep. length | deaths | roll RMS |
|---|---|---|---|
| 0950 policy + x24.6/0.2717 compensation (old) | 54.4 steps | yaw 87.5% / wheel_net 12.5% | 2.89 deg |
| 39750 policy + compensation | 32.4 steps | **fallen 100%** | 10.69 deg |
| 39750 policy + compensation-free | 32.8 steps | wheel_net 64.1% / fallen 21.9% / yaw 14.1% | 7.23 deg |

Findings:
1. **The compensation is now irrelevant.** The 39750 policy scores 32.4 with
   the compensation and 32.8 without — the x24.6/0.2717 stand-in no longer
   helps the new policy (it was trained for the rigid drive; adding the old
   sluggishness back only changes the failure MODE, not the survival).
2. **The drive axis is closed, proven by the death-mode shift.** With the
   compensation, the sluggish wheel cannot rack 7.2 rad -> zero wheel_net
   deaths -> the bike falls outright instead. Without it, the fast wheel
   delivers the policy's corrections but the contact-triggered commands
   over-rotate it -> wheel_net. Same ~32-step survival either way: the drive
   channel no longer explains the gap.
3. **The contact axis owns the remaining gap entirely.** The spread between
   the best config (54.4) and the sim's 1500-step holds is the contact
   representation (the fitted ellipsoids vs the sim's convex hulls, the
   missing steering->roll authority). The compensation-free model is now the
   permanent baseline for the contact-representation experiments (the
   sphere-on-both / SDF-on-both paths in the collision-pipeline section).
