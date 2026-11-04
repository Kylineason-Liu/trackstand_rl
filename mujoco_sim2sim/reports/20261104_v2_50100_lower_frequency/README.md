# Model 50100: lower physics/PD frequency evaluation

**Reducing the rate does not improve transfer. 800 Hz performs similarly to
1200 Hz overall, while 400 Hz performs substantially worse.**

## Controlled results

Same v2 model_50100 ONNX, 64 environments/one completed episode each, seed 42,
DR on, pure-survival mode and 30-second horizon. All per-environment sampled
DR metadata match exactly across the five rates below. Only physics/explicit
PD rate changes; policy/targets stay 50 Hz and rear integral stays 200 Hz.

| Physics/PD | Mean survival | Median | Reached 30 seconds | Fallen |
| --- | ---: | ---: | ---: | ---: |
| 400 Hz, new | 3.919375 s | 0.71 s | 6/64 (9.375%) | 58 |
| 800 Hz, new | 12.333750 s | 2.17 s | 21/64 (32.8125%) | 43 |
| 1200 Hz, prior baseline | 12.998125 s | 5.51 s | 22/64 (34.375%) | 42 |
| 1600 Hz, prior | 11.912500 s | 3.95 s | 21/64 (32.8125%) | 43 |
| 2400 Hz, prior | 13.719375 s | 3.87 s | 24/64 (37.5000%) | 40 |

Relative to 1200 Hz, 400 Hz rescues one baseline fall but loses 17 baseline
survivors. At 800 Hz, four falls are rescued and five survivors are lost.
The individual trajectories remain sensitive to timestep, even when aggregate
survival rates are close.

## Direct startup evidence at 400 Hz

[Matched startup graph](startup_env6.png) and [exact values](startup_env6.json)
show environment 6's first held 20-ms command. The initial observation, raw
action and 4.550102-rad/s rear velocity target are identical at 400/800/1200 Hz.
At 400 Hz, speed alternates **0 → 9.36 → -0.51 → 10.49 → -1.29 → 12.09 →
-2.56 → 14.42 rad/s**, with torque changing sign every physics step and growing
in magnitude. At 800/1200 Hz, this same startup command rapidly settles toward
the target instead of developing that alternating response. The example is
selected for the largest first-interval torque sign-change count at 400 Hz;
it is not claimed to represent every draw.

Across all 64 environments during the first 0.20 s, rear torque changes sign
between **52.86% /7.48% /1.01%** of adjacent within-target physics samples at
400/800/1200 Hz. Median rear velocity spans within each 20-ms target interval
are **8.57 /3.19 /2.79 rad/s**. These describe response variability, not an
identified oscillation frequency. Together with the matched startup example,
they support additional low-rate numerical/control oscillation at 400 Hz.

The 1200-Hz failure is still not fixed by changing rate: the previous higher-rate
tests did not rescue most episodes, and lowering it is no better. We have not
isolated the remaining transfer cause. Physics/PD/contact/projection effects
change together; one seed is not an exhaustive convergence or robustness study.
Keep the default 1200 Hz pending further matched simulation/hardware diagnostics.

## Evidence and reproduction

- [All five rates: comparison plot](frequency_comparison.png)
- [Verified numeric comparison](comparison.json) and [per-environment lengths](per_environment_lengths.csv)
- [Exact commands](commands.json), [analysis helper](compare_rates.py), [startup plot helper](plot_startup.py)
- Native metadata/full observations/actions/controller traces/histograms/torque plots under `hz400/` and `hz800/`
- Existing 1200/1600/2400-Hz runs were reused, not overwritten or rerun.

Each new run exits 0 and completes all 64 quotas: **128 new episodes** total.
400-Hz active transitions/physics steps: **12,542 /100,336**; 800 Hz:
**39,468 /631,488**. Disabled pure-survival flags remain zero, active actions
are finite/bounded, metadata/masks/terminal counts agree. No native instability
warning lines appeared; this does not mean the 400-Hz response is numerically
accurate. All **117 protected existing source/policy/report files** retain
hashes. No simulator source/default/model/gain, training, GPU, GUI or hardware
change was made. Work completed directly without subagents.

To reproduce, take the saved pure-survival command and set `--physics_hz 400`
or `--physics_hz 800`; retain `--dr --pure_survival --num_envs 64 --episodes 64
--seed 42` and the explicit model_50100 ONNX path. The default output directory
is `logs/v2/`; `commands.json` records the exact report directories used here.
