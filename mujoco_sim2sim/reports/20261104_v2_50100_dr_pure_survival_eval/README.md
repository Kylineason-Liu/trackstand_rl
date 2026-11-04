# MuJoCo v2 model_50100: DR pure-survival evaluation

Completed directly on 2026-11-04, without subagents. **64 environments, one
completed episode each**, seed 42, DR on, verified model_50100 ONNX/Beta means,
30-second horizon. Policy/targets: 50 Hz; physics/PD: 1200 Hz; rear integral:
200 Hz. Only **time_out** and **fallen** terminate an episode. No runtime,
policy, configuration, geometry or motor gain was changed.

## Result

| Metric | Result |
| --- | ---: |
| Completed episodes | 64/64 |
| Reached 30 seconds | 22/64 (34.375%) |
| Fell | 42/64 (65.625%) |
| Mean survival | 12.998125 s (649.90625 policy steps) |
| Median survival | 5.51 s |
| Minimum / maximum | 0.26 / 30.00 s |
| Standard deviation | 13.459754 s |
| Terminated before 3 seconds | 29/64 (45.3125%) |
| Roll RMS / maximum | 2.407668 / 35.963853 degrees |
| Mean / maximum drift | 0.062466 / 0.645235 m |
| Mean / maximum absolute yaw deviation | 43.531507 / 179.981282 degrees |

**The policy does not consistently remain upright across these DR draws:**
42 episodes still fall when yaw/drift/wheel-net/divergence stops are disabled.
The 22 horizon survivors establish pure survival, not compliance with stationary
heading/drift requirements. One seed and one episode per environment are a
limited DR evaluation, not an exhaustive robustness estimate.

## Comparison with the completed standard-task run

| Same model, seed and initial DR draws | Standard six terms | Pure survival |
| --- | ---: | ---: |
| Mean episode length | 6.44625 s | 12.998125 s |
| Median episode length | 1.10 s | 5.51 s |
| Horizon survivors | 7/64 (10.9375%) | 22/64 (34.375%) |
| Fallen | 21 | 42 |
| Yaw-limit terminations | 36 | Disabled |

Per-environment sampled mass/CoM/friction/IMU residual metadata match exactly
between the runs. The larger fallen count in pure-survival mode includes bikes
that the standard task stopped earlier for yaw; it does not show the controller
changed or became worse. Standard results are preserved separately under
[`../20261104_v2_50100_dr_eval/`](../20261104_v2_50100_dr_eval/).

## Reproduce

```bash
cd ~/trackstand_rl/mujoco_sim2sim
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -u scripts/run_deploy_eval_v2.py \
  --onnx policies/trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx \
  --num_envs 64 --episodes 64 --max_policy_steps 1500 \
  --seed 42 --physics_hz 1200 --dr --pure_survival
```

Default geometry is the verified 0916 XML; default speed mode remains `project`.
MuJoCo's contact/passive-friction treatment and hard speed projection differ
from PhysX. No simulator equivalence or root cause of the failed transfer has
been established by this evaluation.

## Evidence

- [Episode histogram](202611041419_50100_dr_epi_len_histogram.png)
- [Motor torque plot](202611041419_50100_dr_torque.png)
- [Native summary](202611041419_50100_dr_summary.json)
- [Native metadata and actual DR draws](202611041419_50100_dr_metadata.json)
- [All 64 episode results](episodes.csv)
- [Verification](verification.json), [exact command](command.json), [terminal log](run.log)
- [Full observation/action/controller/state archive](202611041419_50100_dr_evaluation.npz)

Exit 0, all 64 fixed quotas fulfilled, **41,594 active environment transitions /
998,256 physics steps /166,376 integral updates**. Disabled flags all zero.
Raw actions are finite and bounded; peak absolute steering/rear motor torque
5.111140/6.28 Nm. Episode lengths, terminal flags, reset events, active masks,
histogram and summary reconcile. All **69 protected inputs** retain their hashes.
No additional training, GPU, GUI, ROS/CAN or hardware operation occurred.
