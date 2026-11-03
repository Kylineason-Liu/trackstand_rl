# Locomotive deployment (CycleRL riding policies)

Branch `locomotive` runs CycleRL's **riding** policies on this bike. The trained
tensor contract is kept exactly as trained; only the deployment motor contract and
hardware mapping are realigned to this repository's audited deploy stack. This is
a different task from Trackstand-Balance (stationary). Nothing here changes the
trackstand launch, nodes or default model.

## Models

**Default (rank 0):** `locomotive_aligned_1900_drive03_2026-10-29_17-06-47.onnx`
(SHA-256 `ec3ccae9be48daa863eb340042a542d4c932f8e4f782b16da2c5ff8ddff91c1d`), CycleRL's
rank-1 recipe retrained in this repo as `Locomotive-Cylinder-Aligned-v0`: same tensor
contract, rewards, commands and DR, but 400 Hz physics, explicit trackstand steering
(Kp 10 / Kd 0.2, measured friction) and rear τ = clip(0.2·(ω* − ω), ±6.28 N·m).
Paper suite (1024 episodes/mode, 2026-10-29): flat BSR Isaac 99.9% / MuJoCo 100%,
flat steering error Isaac 3.51° / MuJoCo 2.38°, MuJoCo gravel 93.2% (original 0%).
Its rear law is MIT Kp 0 / Kd 0.2, so it needs `rear_control_mode: mit`.

The CycleRL originals (ranks 1–5) remain bundled. Source task `CycleRL-Trackstand-Cylinder-Paper-v0`: the same `trackstand_rl_0916`
cylinder-wheel USD as this repo (geometry/mass/joint payloads hash-identical; this
repo's `base.usda` additionally authors tire friction 0.65/min). Ranked by
flat-ground MuJoCo balance success among runs with 100% Isaac success
(CycleRL `logs/skrl/reward_sweeps/cylinder_top5_{rerun,limitfix}/comparison.md`):

| Rank | ONNX (`models/`) | Recipe | Steer | v* range | MuJoCo BSR |
|---|---|---|---|---|---|
| 1 | `locomotive_1900_drive03_2026-10-19_16-30-17.onnx` | 1900_drive03 | ±20° | 0.5–2.5 m/s | 100% |
| 2 | `locomotive_1900_2026-10-19_19-04-32.onnx` | 1900 | ±20° | 0.5–2.5 | 88.2% |
| 3 | `locomotive_1900_drive03_2026-10-20_03-18-29.onnx` | 1900_drive03 | ±20° | 0.5–2.5 | 82.5% |
| 4 | `locomotive_deploy_10deg_2026-10-20_04-32-20.onnx` | deploy_10deg | ±10° | 0.5–2.5 | 68.7% |
| 5 | `locomotive_act_refine_lo_2026-10-19_15-30-18.onnx` | act_refine_lo | ±20° | 0.3–2.5 | 64.7% |

These are paper-suite results (riding: balance success, steering/velocity tracking),
not trackstand metrics. Each file is self-contained (weights embedded) and listed in
`models/locomotive_policy_manifest.json` with checkpoint/ONNX SHA-256, its contract
(read from the run's saved `params/env.yaml`) and 16 reference input/output pairs.
The controller reproduces those pairs at startup and refuses a mismatching file.

Regenerate (Isaac Lab interpreter; deterministic, same hashes):

```bash
cd isaaclab_train
~/IsaacLab/_isaac_sim/python.sh scripts/deployment/export_cyclerl_locomotive_onnx.py \
  --output-dir ../deploy/source/trackstand_deploy/models
```

The exporter embeds skrl 2.1.0's `RunningStandardScaler` exactly
(`clamp((x − mean)/(sqrt(var) + 1e-8), ±5)`), checked against skrl's own class.
CycleRL's older `export_onnx.py` used `1/sqrt(var + eps)` with no clip, which
diverges beyond ±5σ; do not use its exports for these runs.

## Tensor contract (unchanged)

`[512, 256, 128, 64]` ELU MLP, Gaussian **mean** (raw, unbounded), 50 Hz.
The aligned default model trained at `dt = 0.0025`, decimation 8 (400 Hz physics);
the CycleRL originals trained at `dt = 0.01`, decimation 2 (100 Hz).

| # | Observation | This bike |
|---|---|---|
| 0 | `gravity_y` = projected_gravity_b[1] (−sin roll) | IMU quaternion (bike frame, as v1) |
| 1–3 | `ang_vel_x/y/z` (root_ang_vel_b) | IMU bike-frame gyro |
| 4 | `speed` (root_lin_vel_b x) | 0.07 m × rear A2 velocity (no-slip estimate) |
| 5 | `steer`, wrapped to [−π, π] | steering A3 position, fixed straight zero |
| 6 | `vel_cmd` v* | throttle stick → [v_min, min(v_max, `max_speed_cmd_m_s`)], slewed 1 m/s² |
| 7 | `yaw_rate_cmd` ωz* | yaw stick × 0.5 rad/s (stick right → negative/right turn) |

Actions: steering `a0 × scale` clipped to ±scale (20° or 10° per model) → MIT
position target; rear `a1 × 42.857` clipped ±45 rad/s → rear velocity target
(MIT Kp 0 / Kd 0.2 frames by default, or the C1 speed loop with `rear_control_mode: c1`).
No observation LPF, speed floor or host steering slew (none were in training).

## Motor / hardware contract

| | Aligned model (default) training | CycleRL originals training | Bike (`motor_mit_velocity_io`) |
|---|---|---|---|
| Steering | IdealPD Kp 10, Kd 0.2, 5.6 N·m + measured friction | ImplicitActuator Kp 10, Kd 0.2 | firmware MIT Kp 10, Kd 0.2 (unchanged backend) |
| Rear | IdealPD τ = clip(0.2·(ω* − ω), ±6.28) | ImplicitActuator τ = 5.0·(ω* − ω) | `rear_control_mode`: **`mit`** (default) or `c1` |
| Rates | physics 400 Hz, policy 50 Hz | physics 100 Hz, policy 50 Hz | policy 50 Hz, MIT/C1 200 Hz, feedback 100 Hz |

- **`rear_control_mode: mit` (default):** rear MIT frames (CAN 0x501) at 200 Hz with
  position Kp 0, velocity Kd `drive_mit_kd_nm_s_per_rad` (0.2, wire 0.20024) and the
  nominal-zero feedforward residual cancelled through the velocity field. This is the
  aligned model's trained P-only law; no host integral (unlike the §293 PI test). The
  6.28 N·m clip is applied on the host at command time: ω* is bounded to
  ω ± 6.28/Kd using the latest A2 speed, so the predicted torque never exceeds the cap.
  Between 5 ms commands the motor's own loop is not host-clipped; B3 is lowered as a
  backstop but **whether B3 limits MIT-mode current is not verified**. A drive F1
  reporting MIT inactive while driving, or no drive F1 for 50 ms, is a hard fault.
- **Drive F0 (persistent firmware setting):** MIT velocity encoding needs Vel_Max
  ≥ the 45 rad/s action clip; the as-found drive F0 is ±13 rad/s. Startup logs the
  original F0, writes only Vel_Max = `drive_mit_velocity_max_rad_s` (50 rad/s),
  verifies readback, and restores the original after a confirmed shutdown disable.
  After a crash or power loss the widened value stays; it only affects MIT encoding
  (C0/C1 and the trackstand backends do not use the drive's F0). Restore by hand from
  the logged value if needed. `dry_run` only reads F0.
- **`rear_control_mode: c1`:** firmware C1 speed PI behind B3. Its gains (B8/B9) are
  untouched and match neither training law (Kd 5 implicit or Kd 0.2 explicit).
  Airborne hardware tests (§293–294, P-only at Kd 0.30, not 0.2) found MIT P-only
  weak at very small speeds (|ω*| ≤ 0.4 rad/s) and ~3% short at 10 rad/s; riding
  speeds are 14–36 rad/s. Kd 0.2 itself has not been run on hardware.
- **B3:** startup reads B3 (observed default ≈ 82.5 A), writes
  min(existing, 6.28 / Kt) (≈ 20.9 A at Kt 0.30), and accepts acknowledgement and
  readback only ≤ request and at most 1 mA below. Never raised; volatile until
  power cycle; not restored on exit. `dry_run` only reads it.
- **Signs/IDs:** all from `params.yaml` (rear encoder sign −1). Both trainings use
  the same USD joints, so this repo's validated steering/rear signs apply unchanged;
  CycleRL's own bike config (SBUS, flipped IMU, `ros_positive_steer: left`) was for a
  different stack and is not used. C1 uses the A2 native frame, so the encoder sign
  converts both ways (as `drive_speed_sweep.py`).
- **Disable:** inherited unchanged in both modes — steering CF; rear C0 0 A then CF
  (C0 also leaves MIT mode; the wheel coasts).
- **Topics:** commands on `/locomotive/motor_command`
  `[enable, steering_target_rad, rear_velocity_rad_s]`. The trackstand
  `/trackstand/motor_command` (torque) is never read by this backend. Control
  telemetry label `motor_control_mit_rear_velocity_v2` (adds `drive_mit_*` fields): its inherited
  `requested/bounded_drive_target_nm` fields carry rad/s here.
- **Sticks:** `rc_kickstand` publishes `/locomotive/rc_sticks` = `[throttle, yaw]`
  in [−1, 1] only when `throttle_channel`/`yaw_channel` are set (overlay: CH3/CH4,
  AETR — **unverified on this transmitter**).

## Launch → ride sequence

Training episodes start rolling at 1.0–2.5 m/s, so the policy never acts at rest.

1. **DOWN** — RC CH7 down: motors disabled, stands down.
2. **LAUNCH** — RC up and ready (IMU/motor/RC/sticks fresh, |roll| < 30°, fork
   within 5°, not rolling backward): stands stay down, steering held straight,
   rear speed ramps at 0.5 m/s² to max(stick v*, wake) + 0.1 m/s.
3. **RIDING** — measured speed ≥ `wake_speed_m_s` (1.0): stands up, policy steers
   and drives; v* starts at the hand-over speed.
4. **FAULT_LATCHED** — any readiness loss, inference error, motor fault, or speed
   < `min_riding_speed_m_s` (0.3): motors disabled; stands are **not** moved
   (same as trackstand). Only RC down clears it and lowers the stands.

Whether the bike can roll with its stands down during LAUNCH is **not verified**.

## Commissioning (each step needs the previous one)

On the bike's Pi the locomotive stack has its own workspace,
`~/trackstand_rl_locomotive` (branch `locomotive`); `~/trackstand_rl` stays on `main`
for trackstand. Both build a package named `trackstand_deploy`, so source only one
workspace's `install/setup.bash` per shell.

```bash
cd ~/trackstand_rl_locomotive/deploy && colcon build --symlink-install --packages-select trackstand_deploy
source install/setup.bash
P="--params-file source/trackstand_deploy/config/params.yaml --params-file source/trackstand_deploy/config/locomotive.yaml"
```

1. Sticks: `ros2 run trackstand_deploy rc_kickstand --ros-args $P`, then
   `ros2 topic echo /locomotive/rc_sticks`; throttle up → +1, yaw right → +1.
2. Data flow with `-p dry_run:=true` on `motor_mit_velocity_io`, plus IMU, RC,
   kickstand driver and `locomotive_controller`; check `/locomotive/status`,
   `/locomotive/policy_timing` (inference_ms) and the dry-run B3 log line.
3. Rear wheel **airborne**: live `motor_mit_velocity_io`, publish low
   `/locomotive/motor_command` speeds by hand; confirm forward rotation, the B3
   and F0 log lines, `drive_mit_*` telemetry (`drive_c1_*` in `c1` mode) and A2 tracking.
4. Supported rolling launch below wake speed, then a supported hand-over.

Never run another motor backend, `motor_mit_io` included, alongside it.

## Session logging (automatic)

`locomotive.launch.py` records every run into `~/trackstand_logs/locomotive_YYYYmmdd_HHMMSS/`
(`log_dir:=<dir>` to change, `record:=false` to disable):

| Path | Contents |
|---|---|
| `bag/` | `ros2 bag record -a`, split every 300 s: observations, raw actions, motor commands, `/trackstand/motor_{state,control,event,status,fault}`, `/locomotive/{status,policy_timing,policy_event,rc_sticks}`, IMU, kickstands, `/rosout` |
| `ros_logs/` | every node's text log (`ROS_LOG_DIR`), including startup B3 and the original drive F0 value |
| `config/` | `params.yaml` and `locomotive.yaml` exactly as loaded |
| `session.json` | start time, host, git commit/branch/uncommitted files, model + SHA, rear mode/Kd, YAML `dry_run`, record command |

Storage is the distro default (Humble sqlite3 `.db3`, Jazzy MCAP). Stop with Ctrl-C so the
recorder closes its last segment; a power cut loses at most the open 5-minute segment. Nodes
started individually (commissioning steps 1–3) are not recorded automatically.

Plot a session (summary, matplotlib pages, interactive HTML) on the desktop:
`python3 deploy/experiments/plot_locomotive_session.py ~/trackstand_logs/locomotive_<timestamp>`
(see `deploy/experiments/README.md`).
`ros2 launch trackstand_deploy locomotive.launch.py` is **live** (`dry_run: false`).

Switch models by setting `model_filename` **and** its manifest `model_sha256`, and set
`rear_control_mode` to match: `mit` for the aligned model; the CycleRL originals were
trained on implicit Kd 5, which neither mode reproduces (`c1` was their previous default).

## Open items

Rear MIT P-only at Kd 0.2 (only Kd 0.30 tested, airborne); whether B3 bounds
MIT current; rear C1 mode matches no training law; physical MIT steering torque cap
(see `MOTOR_MIT_IO.md`); unverified stick channels and rolling with stands
down; sim success is not a hardware green light (CycleRL originals lost 0–35% BSR
Isaac→MuJoCo; the aligned model transfers well but is still sim-only). The MuJoCo
locomotive mirror is `mujoco_sim2sim/scripts/locomotive/` (`eval_paper_suite.py`); it
has no passive joint friction and no rough/slope/steps terrain.
