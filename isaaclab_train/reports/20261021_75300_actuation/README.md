# Isaac Lab model_75300 — systematic motor and policy plots

Fresh evaluations of `2026-10-21_00-41-38/model_75300.pt`, generated 2026-10-21.
This is **Trackstand-Balance-v1, 11 observations, bounded Beta policy, trained
without DR**. The right-hand plots test this same policy under full DR; they do
not use the newer DR training run or claim that 75300 was DR-trained.

Open the [HTML gallery](index.html) for all figures. Every figure is provided
as PNG, SVG and PDF, following the earlier
[91850 demand-versus-response analysis](../../../deploy/experiments/reports/20261015_deployment_failure_analysis/isaac_actuation/README.md).

## Figures and selection

| Figure | Content |
| --- | --- |
| [Full demand versus response](isaac_demand_vs_actuation.png) | Env 0, complete 60 s; red spawn lines, independently fitted vertical axes; all extrema retained |
| [Vertical zoom](isaac_demand_vs_actuation_zoomed.png) | Same samples and timestamps; nominal axes magnify post-startup oscillations; DR retains its full recorded range |
| [Startup detail](isaac_startup_demand_vs_actuation.png) | Same env 0, first 0.6 s |
| [Tracking detail](isaac_tracking_detail.png) | Same env 0, 0.5–5 s |
| [Largest initial rear-demand case](isaac_peak_startup_demand_vs_actuation.png) | First 0.6 s; env 16 nominal / env 0 DR, selected using the largest absolute rear request in that window |
| [Raw policy actions](raw_policy_output.png) | Both normalized actions, env 0, with ±1 references |
| [Motor-limit distributions](limit_distributions.png) | All environments: scaled targets, submitted motor torques, actual speeds; both modes overlaid |
| [Actual policy observations](observation_traces.png) | All 11 noisy input channels, grouped by quantity, env 0 |
| [Motor torque detail](motor_torque_detail.png) | First 5 s, env 0: pre-clamp PD/feedforward, submitted motor torque, solver joint-axis effort |
| [Balance motion](balance_motion.png) | Physical roll, drift, heading deviation, env 0; 400-Hz samples before resets |
| [Episode-length histogram](episode_length_histogram.png) | All completed episodes, ten 3-s bins; final bin includes exactly 30 s |

No smoothing, downsampling, response shifting or numeric clipping is used in
the plotted signals. Demand/response plots use linear axes because the new
policy requests are bounded. Red solid vertical lines mark initial spawn and
every auto-reset in the updated full/vertical-zoom demand plots; older detail
plots retain their dotted reset markers. Lines are not connected across resets.
The 60-s evaluation contains multiple episodes,
not a single 60-s balancing episode. Distribution counts use 50-Hz samples for
targets and 400-Hz samples for torque/speed, explicitly labeled on the plot.

### Demand-plot readability update

The two full-duration demand figures were redrawn **offline from the original
NPZ recordings**, with no new evaluation or change to the signals. Red markers
use `t = (done_step + 1) × 0.02 s`, plus the initial spawn at zero. For nominal
env 0 these are **0, 30 and 60 s**; the boundary reset at 60 s has no subsequent
samples. DR env 0 has 35 markers, including the initial spawn. Both motor panels
use exactly the same spawn times for a given environment.

The updated full-range figure fits each panel to its three actual traces plus
10% padding, rather than expanding the view to distant motor-cap references.
No sample falls outside that figure's vertical range. **Axes are independent
across modes**, so compare tick values rather than apparent heights.

The additional zoom figure uses nominal samples at episode age **≥0.5 s** only
to select vertical bounds: approximately **−59.03 to −55.73°** for steering and
**−0.221 to +0.195 N m** for rear effort. All original samples remain drawn;
188 combined nominal steering samples and 52 combined rear samples extend
beyond this zoomed view, mainly around startup. These counts combine the
50-Hz demand and two 400-Hz traces, not unique time instants. Use the full-range
figure for startup extrema. The DR panels retain all extrema in both versions.
No smoothing, time shift or numerical saturation was introduced.

[Axis limits, spawn times and input hashes](demand_plot_settings.json) are saved
for verification. Recreate only these two figures without rerunning Isaac Lab:

```bash
python3 -B isaaclab_train/reports/20261021_75300_actuation/analyze.py --redraw-demand
```

The command intentionally replaces only the two demand figure sets and their
plot-settings JSON; evaluation summaries, CSVs, checkpoints and capture data
are preserved. Both variants are available in PNG, SVG and PDF.

## What each motor line means

- Blue: actual deterministic Beta action multiplied by **1.57 rad** for steering
  or **6.28 N m** for the rear. Steering plots display degrees; CSV/NPZ retain radians.
- Orange, steering: action-term processed target **entering explicit PD**, after
  position clipping ±1.57 rad and slew limiting 23.9 rad/s at 50 Hz. It is held
  through eight physics substeps. It is **not** a position target submitted to a
  native PhysX position controller: v1 has explicit IdealPD actuators and zero
  native PD gains, so PhysX receives motor torque.
- Green, steering: actual simulated joint position after each physics step.
- Orange, rear: explicit clipped motor torque submitted to PhysX, after the
  actuator calculation. Rear Kp/Kd are both zero; passive friction is separate,
  so this largely overlaps the scaled rear action.
- Green, rear: `get_dof_projected_joint_forces()` after each physics step,
  before episode reset. This is the solver's active joint-axis effort/wrench
  component, **not an independent isolated motor-current/shaft-torque sensor**.
  Do not infer failed torque delivery solely from a difference between this
  green line and the explicit submitted motor torque. The latter is recorded
  separately and verified against the clipped actuator output exactly.

The additional motor-torque figure distinguishes both command stages and the
solver measurement for each motor. Passive steering B/Coulomb friction and
rear moving friction are applied separately in mechanical dynamics, not
mistakenly counted as an extra controller Kd or subtracted again from the command.
The sensor API follows the same boundary as the earlier 91850 analysis and its
saved free-rotor validation; no new rotor calibration or physical-bike test was run.

Command timestamps are physics-step **starts**, responses are physics-step
**ends**, 2.5 ms later. This is correct sampling order, not a fitted motor delay.
Initial joint position is zero; the first drawn response is already after the
first physics step. Closed-loop tracking plots do not identify real motor latency.

## Evaluation conditions and results

Each mode: **64 environments × 3,000 policy steps, seed 42**, deterministic Beta
mean inference, 50-Hz policy / 400-Hz physics, all six ordinary terminations,
no `--pure_survival`. Observation white noise is retained. No-DR configuration
matches the saved training behavior, including rewards, actuators, target limits,
spawn pose and reset events. Full DR intentionally enables the existing reset
randomization as an out-of-distribution robustness test. Complete effective
environment and agent YAMLs are saved for both modes.

| Metric | No DR, training-matched | Full DR, robustness test |
| --- | ---: | ---: |
| Completed episodes | 128 | 2,092 |
| Mean completed episode duration | 30.000 s | 1.481 s |
| Timeout completions | 128, 100% | 53, 2.533% |
| Fallen term activations | 0 | 1,351 |
| Yaw-deviation term activations | 0 | 688 |
| Diverged / drifted / wheel-net activations | 0 / 0 / 0 | 0 / 0 / 0 |
| Standard evaluator roll RMS | 1.20° | 5.00° |
| Largest scaled steering request | 1.031474 rad, 59.099° | 1.299141 rad, 74.435° |
| Largest held steering target | 1.031474 rad | 1.272756 rad |
| Largest absolute rear request | 0.443893 N m | 4.005546 N m |
| Largest pre-clamp steering PD torque | 4.780000 N m | 5.849386 N m |
| Largest submitted steering torque | 4.780000 N m | 5.600000 N m |
| Largest submitted rear torque | 0.443893 N m | 4.005546 N m |
| Steering / rear post-PD clipping samples | 0 / 0 | 7 / 0 |
| Raw-action samples outside ±1 | 0 / 0 | 0 / 0 |
| Submitted motor torque above caps +1e-5 N m | 0 / 0 | 0 / 0 |
| Steering tracking RMSE over all physics samples | 0.730° | 3.358° |
| Same RMSE excluding first 0.5 s of each episode | 0.262° | 1.190° |

Statistics above use **192,000 policy samples and 1,536,000 physics samples per
motor per mode**, not just the plotted env 0. RMSE compares held target and
end-of-physics-step position at their stated sampling boundary; it is not a
causal delay estimate. The histogram's final DR bin contains 54 episodes in
27–30 s, but only 53 are timeouts: bin membership is not termination classification.
Incomplete final episodes are excluded from the histogram and completed-episode mean.

### Physics-substep peaks versus standard evaluator peaks

The new capture retains all eight substeps, including terminal states before
automatic reset. The unchanged evaluator's state metrics are only 50 Hz and
already post-reset on terminal steps, so its maxima can be smaller:

| Actual 400-Hz quantity | No DR | Full DR |
| --- | ---: | ---: |
| Maximum steering speed | 23.850550 rad/s | 29.744759 rad/s |
| Maximum rear speed | 3.059036 rad/s | 108.636383 rad/s |
| Steering samples above 23.9 +1e-5 rad/s | 0 | 387 |
| Rear samples above 61.6 +1e-5 rad/s | 0 | 1,082 |
| Maximum absolute roll before reset | 1.829489° | 38.722757° |
| Maximum solver steering joint-axis effort | 2.712269 N m | 5.390665 N m |
| Maximum solver rear joint-axis effort | 0.338469 N m | 3.901335 N m |

Neither solver joint-axis effort exceeded its respective nominal torque cap by
more than 0.01 N m in these captures. These measurements are not a guarantee of
real motor behavior. Solver velocity constraints can overshoot; explicit torque
clipping is not a physical torque–speed envelope. Terminations evaluated at
50 Hz do not prevent a larger tilt earlier within the last physics interval.

## Conclusions and boundaries

1. **75300 balances reliably in this seed's nominal evaluation**, reproducing
   the previously saved 128 timeouts/30-s mean result. This is not evidence of
   robustness to all seeds, initial conditions or the real bike.
2. **The earlier unbounded-output issue is absent here.** Both actions remain
   inside ±1, target scale/clip/slew processing reconstructs correctly, and
   submitted explicit motor torques never exceed the configured caps. The
   startup figure makes the active steering slew limiter visible.
3. **Full DR exposes a large generalization gap in this no-DR-trained policy.**
   Most completed episodes end early even with bounded actions. This experiment
   does not isolate which DR term causes those failures; that would require
   ablations. It does not evaluate the newer DR-trained policy.
4. The successful nominal behavior uses a steering angle around −57° in env 0,
   rather than keeping the front wheel straight. Straight-ahead steering is
   the reset condition, not a reward requirement throughout balancing.
5. The old 91850 figures use a different v0 policy, 13 observations, implicit
   actuators and different controller/passive-loss handling. These plots match
   their presentation and measurement conventions, **not a controlled
   single-change Beta-versus-Gaussian comparison**. No real-bike readiness or
   independently calibrated motor response is established.

## Verification and artifacts

- [Summary, counts and input hashes](summary.json), [eleven focused tests](unit_tests.log),
  [verification source](test_analysis.py).
- [Recorder](record_eval.py) / [analysis and plotting source](analyze.py).
- Full motor/observation/body/termination data: `no_dr/motor_samples.npz`,
  `dr/motor_samples.npz`; actual policy-input vectors are captured at inference,
  without extra observation or RNG calls.
- [No-DR env 0 CSV](no_dr_env0_traces.csv) / [DR env 0 CSV](dr_env0_traces.csv):
  24,000 rows each, distinct command/response times, episode labels and SI units.
- Each mode contains `experiment.json`, `configuration_checks.json`,
  `effective_env.yaml`, `effective_agent.yaml`, `source_integrity.json`,
  `eval.txt` and a checkpoint copy. Complete evaluator logs are
  [no_dr_run.log](no_dr_run.log) / [dr_run.log](dr_run.log).
- Both 95-file integrity manifests report **zero protected changes**, including
  task/evaluator/model source, original checkpoint/configs, progress records and
  original training-run `eval.txt`/torque plot/data. Only this new report bundle
  was added; no runtime edits, training, export or hardware commands were issued.

Offline checks verify the exact scale/clip/slew recurrence, PD reconstruction
with per-environment gains, submitted==clipped motor torque, finite arrays,
observation-to-native-state correspondence, episode/termination/histogram
counts, CSV values/timebase, input/checkpoint hashes and all eleven PNG/PDF/SVGs.
The redraw checks additionally verify red marker artists, initial/terminal
spawn times, full-range coverage and the explicitly documented vertical zoom.
The full/startup/observation/histogram figures were also visually inspected for
labels and alignment.

Reproduce with fresh capture directories from `isaaclab_train/`:

```bash
~/IsaacLab/_isaac_sim/python.sh -B -u reports/20261021_75300_actuation/record_eval.py \
  --capture-dir /tmp/75300_no_dr_fresh --task Trackstand-Balance-v1 \
  --num_envs 64 --num_steps 3000 --seed 42 --headless --device cuda:0 \
  agent.max_iterations=100000
# Add --dr and choose a different fresh directory for the robustness test.
```

`analyze.py` expects sibling `no_dr/` and `dr/` captures beside the script and
refuses to replace a completed `summary.json`/CSV bundle. Retain the original
data and use a fresh report directory for a new experiment.
