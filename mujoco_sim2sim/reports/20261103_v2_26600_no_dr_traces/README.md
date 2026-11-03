# V2 model_26600: no-DR MuJoCo demand, actuation and observations

Open [the HTML gallery](index.html). This report plots the existing completed
capture; no simulation, policy retraining or production-code edit was performed.

**64 environments, seed 42, one completed 30-second episode per environment,
64/64 timeouts, all six normal terminations enabled.** Physics/PD 1200 Hz,
policy/observations/targets 50 Hz, external rear integral 200 Hz. Ordinary IMU white
noise remains enabled; no domain randomization was sampled.

## Figures and exports

- `mujoco_demand_vs_actuation`: steering raw ×1.57 rad / held request / actual
  steering position; rear raw ×61.6 rad/s / held velocity request / actual rear speed.
- `mujoco_motor_torque_flow`: steering computed/applied torque; rear P, retained I,
  P+I before clamp, and applied motor torque. **Applied motor torque is not solver
  net joint effort after passive friction/contact.** The force trace is recorded
  after each physics step; the computed trace is recorded at its start.
- `observation_traces`: all eleven actual vectors passed into policy inference,
  with the recorded noisy gravity/gyro channels. No linear-velocity channels.
- `raw_policy_outputs`, `motor_speeds`, `all64_response_envelope`: bounded actions,
  available motor-speed samples and env 0 versus the min–max range over all 64 envs.

Main figures have PNG/SVG/PDF versions. The first three have a **full 0–30 s** view,
**startup 0–2 s** view and **settling detail 12.5–17.5 s** view. No smoothing or numeric
downsampling. Independent Y axes include every displayed sample with 9% padding.
Legends are bottom right and give each curve's min/max, with colored extrema dots.
Red vertical lines mark the actual initial spawn at 0 s. Terminal resets at 30 s have
no following recorded trajectory, so no extra episode is invented or plotted.

[64-page demand PDF](all64_demand_vs_actuation.pdf) and
[64-page observation PDF](all64_observation_traces.pdf): page 1 = env 0, page 64 = env 63.
Curves in these large PDFs are rasterized; numeric records remain unchanged.

Numeric exports: [env0 policy](env0_policy_traces.csv),
[env0 physics](env0_physics_traces.csv), [all64 summaries](all64_policy_summary.csv),
[all observation extrema](observation_extrema_all64.csv),
[exact axes/extrema/time windows](plot_settings.json),
[nominal/no-DR values](nominal_configuration.json) and [summary](summary.json).

## Sampling and evidence limits

Steering position/speed are recorded only at 50 Hz policy boundaries. The initial
steering pose is the actual first uncorrupted encoder-style observation. This
capture cannot reveal complete 1200 Hz steering-state oscillations. Rear velocity
is available before every physics step at 1200 Hz; its final 30 s endpoint is also
recorded. There are no recorded solver net-effort or full body-pose histories.
Timestamp offsets label capture boundaries; they do not measure real motor delay.

Rear actions mean velocity, not torque. Rear P is 0.30 × velocity error; I is the
actual retained 200 Hz integral state, with Ki 3.3. P+I is clamped at ±6.28 N m; the
steering PD torque is clamped at ±5.6 N m. Passive friction is separate. These
records never reach either torque clamp. MuJoCo speed projection remains a
documented approximation to PhysX's soft speed constraints. No claim of hardware
performance or of survival beyond 30 s follows from these plots.

`analyze.py` checks the original shapes/counts, action scaling, 50 Hz steering slew,
all 2,304,000 rear P+I calculations and 200 Hz retained integral states, final torque
clipping and uncorrupted joint observation channels before plotting. The report's
separate `verify_plots.py` rechecks CSV values, panel bounds/extrema, PDF pages,
links and protected input hashes. Original raw data and prior reports remain
unchanged. Reproduce with `python3 -B analyze.py` from this directory.
