# Model 2500 MuJoCo DR evaluation — 2026-10-23

## Conclusion

**Yes: the current v1 / model_2500 system improves substantially over the
historical v0 / model_91850 system in these DR evaluations, including at the
unchanged production 400-Hz physics/PD rate. It is not yet reliably balancing
under all randomized conditions.** This is an end-to-end version comparison,
not proof that Beta, observation removal, or any single change caused the gain.

## Verified policy and conditions

- Checkpoint: `isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_11-18-29/model_2500.pt`.
- SHA256: `5389effbd6a0d3e78c9ea9caeac309ceb5082622c8ee125bb701c63d1511fe52`.
- Saved configuration: DR enabled, 11 observations, Beta actor with 32/32 ELU
  hidden layers, no observation normalization, 50-Hz inference and 400-Hz
  physics, 30-s horizon. Deterministic inference uses the Beta mean, not a sample.
- Actual training `BetaMean` versus the deployed mapping: **1,024 inputs,
  seed42, exactly zero maximum difference**, checked using Isaac Python.
- All five evaluations: **64 independent environments, seeds42–105,
  3,000 policy steps/environment =60 simulated seconds/environment**, DR ON,
  white observation noise retained; MuJoCo **3.12.0**, CPU/headless.
- Same verified `models/trackstand_rl_0916_bike/trackstand_rl_0916.xml` and
  original geometry/inertias/axes. Policies run with their own version's
  observation, motor and DR implementation. No model retuning.
- `current400_*` uses the unchanged current evaluator. `diagnostic800_*`
  changes physics/PD to800Hz and decimation to16 **in process only**; policy,
  target slew and termination checks stay50Hz. This is not installed as a fix.
- Historical v0 is reconstructed read-only from Git commit
  `e5f5d7f8d4652e83e649fbdfc2eea3f9dea8a465`, including its model augmentation,
  environment, group and original 91850 ONNX policy. No checkout/hidden XML write.

## Completed-episode results

| System / physics-PD rate | Terminations | Completed | Timeouts | Mean duration | Timeout rate | Step-weighted roll RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v0 / 91850 /400Hz | timeout + fallen | 594 | 59 | **4.4270s** | **9.9327%** | 4.3270deg |
| v1 / 2500 /400Hz — current default | all six | 270 | 74 | **9.3629s** | **27.4074%** | 2.7164deg |
| v1 / 2500 /400Hz — matched survival criteria | timeout + fallen | 222 | 79 | **11.4870s** | **35.5856%** | 2.7081deg |
| v1 / 2500 /800Hz — diagnostic only | all six | 149 | 92 | **19.4554s** | **61.7450%** | 2.1816deg |
| v1 / 2500 /800Hz — diagnostic only | timeout + fallen | 139 | 104 | **23.3236s** | **74.8201%** | 2.1119deg |

The comparable 400-Hz survival-only comparison gives **2.5948x** mean duration
and **+25.6529 percentage points /3.5827x** timeout rate. Even v1's stricter
full-task evaluation improves over the weaker v0 survival-only baseline.

All-six v1 termination activations can overlap:

| Rate | time_out | fallen | diverged | drifted | yaw_deviation | wheel_net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 400Hz | 74 | 135 | 0 | 7 | 50 | 6 |
| 800Hz diagnostic | 92 | 31 | 0 | 0 | 25 | 1 |

The older saved `mujoco_sim2sim/logs/202610151708_91850_dr_summary.log` reports
6,057 episodes, **6.00s mean /14.7% timeouts**. Its duration/budget is not the
new matched 60s/env budget. It also averages some metrics differently, so its
roll RMS must not be used as a directly comparable step-weighted statistic.
The 400-Hz v1 survival result exceeds this historical aggregate too.

## Independent-environment check and statistical qualification

Taking only the first episode from each of64 environments avoids letting a
frequently failing environment contribute many more trials:

| System | First-episode mean | First episodes reaching30s |
| --- | ---: | ---: |
| v0 /91850 /400Hz, survival | 4.0038s | 6/64 =9.3750% |
| v1 /2500 /400Hz, full task | 12.0741s | 24/64 =37.5000% |
| v1 /2500 /400Hz, survival | 13.6544s | 28/64 =43.7500% |
| v1 /2500 /800Hz, full task | 21.8053s | 46/64 =71.8750% |
| v1 /2500 /800Hz, survival | 25.1319s | 53/64 =82.8125% |

Exploratory **20,000-resample environment-cluster bootstrap**, seed20261023:
400-Hz v1 survival minus v0 survival has95% percentile intervals
**+5.294…+9.120s** for completed-episode mean duration and
**+19.598…+32.843 percentage points** for timeout rate. First-episode differences
also exclude zero: mean **+5.460…+13.763s**, timeout rate **+20.313…+48.438pp**.
Whole environment trajectories, not individual correlated episodes, are
resampled; samples are drawn independently for each version.

These intervals support a clear improvement **within this tested setup**.
They are conditional on the particular checkpoints/material banks, not a
multi-training-seed claim. Equal seed numbers do not mean identical DR draws:
v1 fixes shared material-bucket sampling, CoM-frame conversion, mass-constant
refresh and other runtime behavior. The versions also differ in observations,
action distribution, passive friction and control. Thus this cannot isolate
policy quality or one bug fix as the cause. V1's shared material bank is only
one sampled bank, so the bootstrap does not quantify variability across banks.

## Remaining failure and timing issue

At400Hz the full task still produces135 fall activations and50 yaw activations
among270 completed episodes. Survival-only success is35.6%, not near100%.
The 800-Hz improvement shows the earlier timing sensitivity remains relevant
for this DR-trained policy, although 2500 is not an immediate universal failure
at400Hz like the earlier nominal-trained75300. The 800-Hz samples have different
episode/reset histories and are not identical per-episode DR replays.

This result does not establish faithful PhysX equivalence or real-bike safety.
MuJoCo still has one applied sliding coefficient, not separate static/dynamic
contact friction, and a hard post-step speed projection rather than PhysX's
soft solver speed constraint. Changing PD frequency changes the sampled
controller, not just physics accuracy.

## Evidence and validation

- [Comparison graph](202610230048_2500_dr_v0_comparison.png), also SVG/PDF.
- [400-Hz episode histograms](202610230048_2500_dr_epi_len_comparison.png).
- [Machine-readable comparison/CIs](comparison.json), [episode table](episodes.csv).
- Each v1 folder contains its invocation, console log, summary, ten-bin
  histogram, torque plot/NPZ, complete observation/action/state/physics-torque
  capture and **every initial/reset DR draw**.
- [Historical v0 capture/provenance](historical_v0_91850_survival/metadata.json)
  and DR draws/NPZ permit independent reconstruction of its statistics.
- [Configuration checks](configuration_checks.json) and
  [v0 export verification](v0_policy_validation.json): original91850 PT versus
  evaluated ONNX maximum action difference **1.9073486328125e-6** on1,024 saved
  actual observations, seed42.
- Captures validated: finite observations/actions/states/torques; v1 raw outputs
  within[-1,1]; 50-Hz steering scale/clip/slew reconstructed; all physics torques
  within5.6/6.28Nm; rear actuation equals scaled feedforward; histogram counts,
  reset/episode counts, saved summaries and DR ranges consistent.
- **64 protected source/config/model/artifact/ledger/index hashes unchanged.**
  No production edits, original log/checkpoint overwrite, staging or commit.
  New generated evidence occupies approximately197MiB; do not commit it blindly.

## Reproduction

Current default full-task evaluation from `mujoco_sim2sim/`:

```bash
python3 -u scripts/run_deploy_eval.py \
  --mjcf models/trackstand_rl_0916_bike/trackstand_rl_0916.xml \
  --pt ../isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_11-18-29/model_2500.pt \
  --dr --num_envs 64 --num_steps 3000 --seed 42 \
  --output_dir reports/model_2500_dr
```

Add `--pure_survival` for timeout/fallen only. To reproduce the diagnostic and
historical runs, use `experiment.py <variant>` from repository root; available
variants are `current400_full`, `current400_survival`, `diagnostic800_full`,
`diagnostic800_survival`, and `historical_v0`. These report scripts write their
own evidence directories; `analyze.py` validates captures and rebuilds graphs.
