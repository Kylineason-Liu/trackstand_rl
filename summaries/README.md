# Summary log

One dated Markdown file for every results summary (eval results, ablation findings, run reports).
Files are named `YYYY-MM-DD_HHMM_<topic>.md` (HKT, the time the summary was written).
Newest first. Never rewrite earlier entries; write a new file instead.

| Written (HKT) | Summary | Covers |
|---|---|---|
| 2026-11-04 00:29 | [pi_separate_locomotive_workspace](2026-11-04_0029_pi_separate_locomotive_workspace.md) | Correction to 00:21 §2: Pi `~/trackstand_rl` restored to `main` `f1b1ff1` (650 passed); locomotive in its own clone `~/trackstand_rl_locomotive` @ `9773e1d` (702 passed) |
| 2026-11-04 00:21 | [locomotive_pi_deploy](2026-11-04_0021_locomotive_pi_deploy.md) | Locomotive stack (`9773e1d`) deployed + built on the Pi; alignment re-check passes; on-Pi tests 702 passed; rank-0 eval recap (flat BSR 99.9%/100%, STE 3.51°/2.38°, MuJoCo gravel 93.2%); not launched live |
| 2026-11-04 00:09 | [trackstand_5L_no_gain_report](2026-11-04_0009_trackstand_5L_no_gain_report.md) | Full 5-layer vs [32,32] report: matched pair, all 7 variants, training-vs-transfer (r = −0.48), steering, model size; keep [32,32] |
| 2026-11-03 23:32 | [trackstand_yaw_claim_check](2026-11-03_2332_trackstand_yaw_claim_check.md) | Correction to the 23:03 chat claim: yaw endings likely reward-driven; MuJoCo DR-on cap is mostly falls, not reward |
| 2026-11-03 23:03 | [trackstand_5L_vs_2L](2026-11-03_2303_trackstand_5L_vs_2L.md) | 5-layer vs [32,32] in the DR ablation: no gain from 5-layer; capacity not the bottleneck |
| 2026-11-03 22:36 | [Trackstand DR ablation: figures](2026-11-03_2236_trackstand_dr_ablation_figures.md) | charts (exact values labelled) for the 15:01 / 15:06 results; corrects CycleRL train ep_len 217 → 899 |
| 2026-11-03 15:24 | [TensorBoard noise: RSL-RL vs skrl](2026-11-03_1524_tensorboard_noise_rsl_vs_skrl.md) | why RSL-RL curves look noisier than CycleRL; measured noise; how to view |
| 2026-11-03 15:06 | [Trackstand DR ablation: failure diagnostics](2026-11-03_1506_trackstand_dr_ablation_failure_diagnostics.md) | why each DR policy succeeds/fails in Isaac vs MuJoCo; TensorBoard + report commands |
| 2026-11-03 15:01 | [Trackstand DR training ablation](2026-11-03_1501_trackstand_dr_training_ablation.md) | test_layer DR re-run, 10 runs, trained 2026-11-02 16:48 → 2026-11-03 03:54 |
