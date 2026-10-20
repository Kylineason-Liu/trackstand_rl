# V1 timeout bookkeeping and action-rate weighting fixes

Date: 2026-10-21. Both requested fixes are implemented for **Trackstand-Balance-v1**.
V0 retains its historical behavior. This supersedes findings 1–2 of the
[pre-training audit](../20261021_v1_pretraining_audit/README.md); the other audit
findings are not changed by this work.

## Changes

- The v1 registration now selects `TrackstandRLEnvV1`, a small subclass of the
  legacy environment. Its bookkeeping reset mask is
  `reset_terminated | reset_time_outs`. Every automatically reset environment
  clears `wheel_net_angle` and captures its fresh post-reset `spawn_yaw`.
  Continuing environments retain their baseline and integrate wheel motion as
  before. Terminated/truncated signals returned to PPO are not changed; a
  timeout remains a timeout for value bootstrapping.
- V1's independent reward configuration reuses the existing negative L2
  action-change function: `-norm(raw_action - previous_raw_action)`. Both raw
  actions are already dimensionless and normalized; they are no longer divided
  by physical scales `1.57` and `6.28`. The positive weight remains `0.1`, and
  RewardManager still multiplies by `dt=0.02` s. Zero action change now gives
  exactly zero penalty, without the legacy formula's tiny epsilon baseline.
- Survival/drift rewards, termination conditions, resets of physical joints and
  action histories, motor parameters, observations, frequencies and DR settings
  are unchanged. In particular, the existing endpoint-speed wheel integration
  and its one-policy-step termination lag are not redesigned here.

## Actual headless evidence

Isaac Lab GPU physics, seed 42, 64 environments, tested separately with DR off
and full DR on. Three environments are deliberately given old wheel state
`2 rad` and old yaw baseline `0.123 rad`; a fourth continues without resetting.
These are controlled regression sentinels, not naturally observed failures.

| Controlled case | Terminated / truncated | Wheel state after step | Yaw baseline after step |
| --- | --- | --- | --- |
| Timeout only | False / True | 0 | New spawn heading 0 |
| Drift failure only | True / False | 0 | New spawn heading 0 |
| Timeout and failure together | True / True | 0 | New spawn heading 0 |
| Continuing environment | False / False | Old state + current speed × 0.02 | Old baseline 0.123 retained |

Both modes passed the same assertions. Reset joint positions/velocities,
steering slew targets and encoder observation channels were also zero.
Explicit full reset continued to clear all bookkeeping correctly.

| Single-channel raw action change | V0 weighted contribution at 50 Hz | Fixed v1 contribution |
| --- | ---: | ---: |
| Steering +1 | −0.001273885 | −0.002 |
| Rear +1 | −0.000318471 | −0.002 |

The v1 contributions were read from the live RewardManager, not just a direct
function call. All 34 reward computations per mode matched an independent
formula to a maximum absolute error of `2.98e-8`.

Each mode additionally completed **two real PPO updates, 32 policy steps** with
finite observations, rewards, losses and actor parameters. Sampled actions
remained bounded: nominal `[-0.99553919, 0.99843788]`; DR
`[-0.99288177, 0.99201787]`. These are short execution checks, not evidence of
learned balancing ability. No policy checkpoint was saved.

## Tests and reproduction

All **42 focused CPU tests passed**, including seven new episode/reward
regressions, v0 isolation, Beta math/PPO/export, observation and actuator checks.
From `isaaclab_train/`:

```bash
~/IsaacLab/_isaac_sim/python.sh -B -m unittest discover -s tests -p 'test_trackstand_*.py' -v
~/IsaacLab/_isaac_sim/python.sh -B -u reports/20261021_v1_episode_fixes/verify_fixes.py \
  --num_envs 64 --headless --device cuda:0
```

Evidence: [CPU log](unit_tests.log), [GPU summary](validation_summary.json),
[GPU log](headless.log), [diagnostic script](verify_fixes.py).
The [integrity check](source_integrity.json) confirmed that the GPU diagnostic
did not modify any of the 72 monitored runtime/script/test/README/record files.
The earlier audit remains intact as pre-fix evidence.
