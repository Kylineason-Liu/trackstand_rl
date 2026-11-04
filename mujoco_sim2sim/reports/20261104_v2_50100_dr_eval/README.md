# MuJoCo v2 model_50100: standard DR evaluation

Completed 2026-11-04 before the user requested pure-survival mode. Preserved as a
separate baseline; the requested final evaluation is
[the pure-survival report](../20261104_v2_50100_dr_pure_survival_eval/README.md).

64 environments, one episode each, seed 42, DR on, 30-second horizon, all six
standard terminations. ONNX model_50100, verified 0916 geometry, default speed
projection. Policy 50 Hz /physics-PD 1200 Hz /rear I 200 Hz.

| Metric | Result |
| --- | ---: |
| Completed episodes | 64/64 |
| Reached 30 seconds | 7/64 (10.9375%) |
| Fallen | 21/64 (32.8125%) |
| Yaw-deviation termination | 36/64 (56.25%) |
| Other termination activations | 0 |
| Mean / median length | 6.44625 /1.10 s |
| Minimum / maximum length | 0.26 /30.00 s |
| Standard deviation | 9.929338 s |
| Terminated before 3 seconds | 43/64 (67.1875%) |
| Roll RMS / maximum | 2.531569 /35.001223 degrees |

20,628 active transitions, 495,072 physics steps and 82,512 integral updates.
This policy fails the full stationary task on most sampled environments; yaw is
the most frequent termination, which need not mean the bike had fallen yet.
One seed only; no root-cause inference from these statistics alone.

[Histogram](202611041416_50100_dr_epi_len_histogram.png),
[torque plot](202611041416_50100_dr_torque.png),
[summary](202611041416_50100_dr_summary.json),
[episodes](episodes.csv), [verification](verification.json),
[command](command.json), [terminal log](run.log).

Exit 0; metadata/flags/masks/reset counts/histogram/summary agree. All 69
protected inputs unchanged. The first offline verification attempt retained in
verification_initial.log used exact float equality for seconds; it was corrected
to a 1e-12-second comparison tolerance. No simulation rerun or production change
was needed. User's pure-survival rerun is a separate requested evaluation.
