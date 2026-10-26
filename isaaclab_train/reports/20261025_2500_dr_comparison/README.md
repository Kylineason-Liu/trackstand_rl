# Model_2500: Isaac Lab and MuJoCo DR demand/response and observation traces

Fresh headless captures recorded **2026-10-25**. Open [the HTML gallery](index.html).
These evaluate the same DR-trained `2026-10-21_11-18-29/model_2500.pt`, not model_75300.
Checkpoint SHA256: `5389effbd6a0d3e78c9ea9caeac309ceb5082622c8ee125bb701c63d1511fe52`.

## Figures

All main figures have three columns: **Isaac 400 Hz**, **MuJoCo 400 Hz**,
**MuJoCo 800 Hz**. The last is an explicitly labeled timing variant, not the
Isaac training rate. All use DR and a 50 Hz deterministic Beta-mean policy.

| Figure | Content |
| --- | --- |
| [Full demand versus actuation](demand_vs_actuation.png) | Env 0, all 60 seconds; steering position and rear torque/effort |
| [Five-second demand detail](demand_vs_actuation_detail.png) | Central five seconds of env 0's longest completed episode: 12.50–17.50 s in these captures |
| [All 11 actual policy observations](observation_traces.png) | One row per observation, env 0, all 60 seconds |
| [Observation detail](observation_traces_detail.png) | Same 11 inputs, 12.50–17.50 s; independently fitted axes |
| [Motor torque flow](motor_torque_flow.png) | Steering/rear pre-clamp calculation, submitted command and response proxy |
| [Motor torque detail](motor_torque_flow_detail.png) | Same quantities in the five-second detail window |
| [Raw Beta outputs](raw_policy_outputs.png) | Actual bounded normalized actions, not actor concentration parameters |
| [Motor speed traces](motor_speeds.png) | Every recorded physics substep; MuJoCo native speeds precede its hard projection |

Each is also saved as SVG and PDF. For **every environment**, not just env 0:

- [64-page demand/response PDF](all64_demand_vs_actuation.pdf).
- [64-page observation PDF](all64_observation_traces.pdf).

PDF **page 1 = env 0**, page 64 = env 63. Each page compares that environment
index across all three runs. Equal indices/seeds do not imply identical DR draws.
Long PDFs rasterize curve artists for manageable size; numeric source samples
are neither smoothed nor downsampled. Labels and axes remain vector elements.

## Sampled DR values and legend update (2026-10-26)

Every panel's legend is now at the **bottom right**. Below each figure, a block
per column states the actual environment/episode and its interval, body mass
scales, shared CoM XYZ offset (mm), sampled static/dynamic contact friction,
effective steering/rear Kp/Kd, initial roll/rate, gyro XYZ bias (rad/s), and
gravity YZ bias (unit gravity). Restitution is zero. Values use six significant
digits; [sampled_dr_values.json](sampled_dr_values.json) preserves full precision,
body ordering, material-slot ordering and source hashes. MuJoCo's sampled dynamic
coefficient is explicitly labeled metadata: its native contact cone uses the
static coefficient. Isaac's displayed shared CoM offset is the applied base-body
offset; tiny per-body float32 addition differences remain in the JSON.
Kp is in N m/rad, Kd in N m s/rad; mass scales and contact coefficients are
dimensionless. Rear effective Kp/Kd stay zero even when their multipliers are
randomized. These blocks state episode constants, rather than listing the fresh
white-noise sample added to each individual observation.

Full-window plots list **every episode with recorded samples**, including an
unfinished last episode. Detail figures list only the episode(s) intersecting
their window. A reset exactly at 60 s retains its red marker but has no subsequent
recorded samples and therefore no DR block. Both 64-page PDFs include each page's
own environment/episode values; page counts and environment order are unchanged.
The friction-diagnostic figures use the same blocks and lower-right legends.

The original MuJoCo reset snapshots already contained these values. The original
Isaac capture retained effective gains but omitted the other DR draws. A fresh
read-only repeat, [recover_isaac_dr.py](recover_isaac_dr.py), records snapshots
after each reset without adding RNG draws or changing state. All **18 original
capture arrays are byte-identical** in dtype, shape and contents, across all 64
environments and 3000 steps. Only after that check does
[prepare_dr_metadata.py](prepare_dr_metadata.py) associate the recovered values
with the original curves. Its verification and provenance are included in the
sampled-value JSON; the repeat is isolated under `dr_metadata/isaac_repeat/`.

The added blocks extend the canvas downward while retaining existing panels'
physical sizes. Signals, timing, line styles/colors, extrema, Y limits, red reset
markers, detail windows, input captures and evaluation results remain unchanged.
Legend/DR-update validation is separate from the historical verification below:
[legend_dr_verification.json](legend_dr_verification.json) and
[legend_dr_tests.log](legend_dr_tests.log).
The update passes **17 tests**. Direct comparisons of 114 original/updated panels
confirm identical trace artists, numeric axes, extrema and spawn markers. Both
64-page PDFs retain correct environment-specific DR labels on every page, and
179 protected source/capture/ledger/index files remain unchanged.

## Spawns, extrema and readable axes

- Red solid vertical lines mark initial spawn and **every actual automatic reset**:
  `t_spawn = (done_step + 1) × 0.02 s`. A reset at 60 s is included even though
  there are no subsequent samples. Each environment uses its own reset history.
- Lines are broken across resets; no line joins a dying pose to a new spawn.
- Each curve's legend includes **minimum and maximum**, with colored dots at
  their recorded locations. A separate panel label gives overall min/max.
- Every panel fits its own actual values plus 9% padding. Distant nominal caps
  do not flatten small changes. **No plotted sample is excluded by the Y limits.**
- The five-second figures plot only the declared interval with newly fitted
  axes. They are separate views, not clipped substitutes for full-range plots.
  General selection rule: middle ≤5 seconds of the longest completed episode.
- Exact windows, bounds, per-curve extrema/times and spawn lists for every
  plotted environment are retained in [plot_settings.json](plot_settings.json).
  Compare numeric ticks: independently fitted axes are not equal physical scales.

## What the lines actually mean

### Demand versus actuation

Steering: blue **raw action ×1.57 rad**, orange held target after position clip
and the 50 Hz slew limiter, green **actual simulated steering position** at each
physics-step end. Measured initial poses are included at segment starts.

Rear: blue **raw action ×6.28 N m**, orange **motor torque actually submitted**.
The green trace deliberately identifies the different available responses:

- **Isaac:** `get_dof_projected_joint_forces()` active joint-axis effort/wrench
  component, **not isolated motor-current/shaft torque**. Passive friction and
  constraints can make it differ from submitted motor torque. Commanded motor
  effort is recorded separately; a difference is not evidence of dropped commands.
- **MuJoCo:** `data.actuator_force`, the actual motor actuator force. It is
  **not net joint torque after passive losses/contacts**. This ideal torque actuator
  reproduces submitted rear feedforward exactly, so green/orange/blue overlap.

Do not compare these green torque curves as identical sensor definitions. The
torque-flow figures additionally distinguish the pre-clamp PD/feedforward
calculation and post-clamp submission for both motors.

### Observations and timestamps

The eleven rows are gravity Y/Z; body angular velocity X/Y/Z; steering/rear
position; steering/rear velocity; sine/cosine of steering. Gravity/trigonometry
are dimensionless, positions rad, velocities rad/s. These are the **actual noisy
11-channel vectors passed into inference**, not reconstructed ideal inputs.

Policy inputs/actions occur at 20 ms interval starts. Torque commands are
recorded at physics-step starts, positions/response proxies at physics-step ends,
before automatic reset. Offsets of 2.5 ms /1.25 ms reflect recording boundaries,
not experimentally identified real motor delays. MuJoCo speed recording occurs
before its production hard qvel projection; its next policy observation uses
the projected state. No response shifting or numerical saturation is applied
by this analysis.

## Evaluation conditions and results

Each run: **64 environments ×3000 policy steps**, seed 42, 60 simulated seconds
per environment, 30 s horizon, **all six termination terms**, DR enabled and
ordinary observation noise retained. No `--pure_survival`. This differs from
the old training-directory 2500 summary that disabled four termination terms.

| Run | Completed episodes | Timeouts | Timeout rate | Mean completed episode |
| --- | ---: | ---: | ---: | ---: |
| Isaac 400 Hz | 130 | 108 | 83.0769% | 27.409231 s |
| MuJoCo 400 Hz | 270 | 74 | 27.4074% | 9.362889 s |
| MuJoCo 800 Hz | 149 | 92 | 61.74497% | 19.455436 s |

Full-precision means/extrema, every environment's spawn list and source hashes
are in [summary.json](summary.json). Isaac non-timeout episodes terminate on yaw
deviation, not fallen; the evaluator's generic `fall_rate` label also counts
these other non-timeout deaths. Unfinished final episodes are excluded from
completed-episode duration statistics.

Isaac configuration checks match saved actions, observations, rewards,
terminations, actuators, initial state, timing and enabled DR. The current source
spells rear `viscous_friction=0.0` explicitly, while the training YAML leaves it
`None`. The capture passes the **runtime-only Hydra override**
`env.scene.robot.actuators.drive.viscous_friction=null` to retain the exact saved
configuration, without editing source. No 1 kHz training change is introduced.
Rendering is headless. The two engines retain their verified USD/XML geometry,
inertias and axes without retuning.

The MuJoCo instrumentation reproduces **all 13 preexisting archive arrays
byte-for-byte** against the October 23 corresponding full-task captures. Reading
substep state introduced no new RNG/observation calls or trajectory changes.
The engines retain different DR realizations, contact/friction implementations
and speed constraints; this is not a paired plant-parameter causal experiment,
hardware validation or proof of real motor bandwidth.

## Capture provenance and reproducibility

- [Isaac recorder adapter](record_isaac.py) reuses the audited
  `../20261021_75300_actuation/record_eval.py` **in memory**, replacing only the
  checkpoint/run identity, progress labels and `training_dr=True` metadata.
  The runtime configuration override is explicit above. Original files are not
  rewritten. The existing recorder captures pre/post-physics states, actual
  policy inputs, explicit controller output, solver effort and termination flags.
- [MuJoCo recorder](record_mujoco.py) instruments the unchanged production
  evaluator. Native q/qdot, controller gains, computed/submitted/applied motor
  torques and every DR reset are retained at all physics substeps.
- [Analysis](analyze.py), [focused verification tests](test_analysis.py),
  [analysis log](analysis.log), [test log](unit_tests.log).
- Full data: `isaac400/motor_samples.npz`, `mujoco{400,800}/physics_samples.npz`
  and each MuJoCo production `*_evaluation.npz`. Captures are large; do not
  accidentally commit generated NPZs, copied checkpoints or long PDFs.
- Each run has `experiment.json`; Isaac also retains effective env/agent YAML,
  configuration checks, source-integrity results, copied checkpoint and eval
  summary. MuJoCo has every initial/reset DR draw and CLI summary/metadata.
- Three `*_env0_policy_traces.csv` files give actual inputs/actions and native
  start/end position references in SI units. Use NPZs for all physics substeps.

From `isaaclab_train/`, choose new capture directories to preserve these results:

```bash
~/IsaacLab/_isaac_sim/python.sh -B -u reports/20261025_2500_dr_comparison/record_isaac.py \
  --capture-dir /tmp/2500_isaac_dr_fresh --task Trackstand-Balance-v1 \
  --num_envs 64 --num_steps 3000 --seed 42 --dr --headless --device cuda:0 \
  agent.max_iterations=100000
```

From repository root:

```bash
python3 -B -u isaaclab_train/reports/20261025_2500_dr_comparison/record_mujoco.py \
  --physics_hz 400 --capture-dir /tmp/2500_mujoco_dr_fresh
# Repeat with 800 and another fresh output directory for that timing variant.
python3 -B isaaclab_train/reports/20261025_2500_dr_comparison/analyze.py --all-env-pdfs
python3 -B -m unittest discover -s isaaclab_train/reports/20261025_2500_dr_comparison -v
```

The analysis uses the report-local captures named above; offline rerunning
replaces only this new report's plots/derived summaries, not source captures or
older reports. All-environment export uses eight-page subprocess batches and
the installed Poppler `pdfunite` utility to keep PDF image-cache memory bounded.
Batch PDFs/settings remain under `pdf_batches/`. An initial monolithic export
hit a memory limit after the main plots and the demand PDF had been produced;
`analysis_initial_export.log` preserves that attempt. Captures were intact and
were not rerun for this export recovery. Validation covers data/command recurrence, recorded observations,
all environments' red spawn markers and extrema, full-range axes, CSV timebase,
all figure formats, 64-page PDFs and HTML links. No training, hardware/ROS/CAN,
firmware, verified model or deployment-control change was performed.

Final verification: **13 focused tests passed in 6.343 s**, including all 64
environments' recorded observation extrema/spawns, rendered red-marker artists,
full-range axes, original MuJoCo archive equality, CSV precision/timebase,
64-page PDF counts and gallery links. Syntax, Pyflakes and 120-column checks
passed. An initial CSV check caught NumPy float32's shortened text formatting;
the exporter now writes the full stored numerical value, and the final suite
passes exact CSV-versus-capture comparisons. No source capture changed.

The before/after integrity check protects **487 preexisting files**, including
task/evaluator sources, verified models, previous reports, original checkpoint,
training-run evaluation/torque artifacts, progress ledgers and Git index: all
unchanged. See [protected_before.json](protected_before.json) and
[verification.json](verification.json). Only this new report directory was added.
