# Model 50100: is MuJoCo failure caused by physics frequency?

**Increasing physics/explicit-PD frequency does not fix the observed failure.**
This controlled comparison does not support insufficient frequency at 1200 Hz
as the main explanation. Some timestep sensitivity remains; other causes have
not yet been isolated.

## Controlled experiment

Same exported v2 model_50100, DR on, pure-survival mode, 64 environments/one
completed episode each, seed 42, 30-second horizon. Policy/observations/targets
remain **50 Hz**, rear integral remains **200 Hz**. Only physics/explicit-PD
frequency changes; default speed projection and controller parameters remain.
Per-environment sampled DR metadata match exactly across all 64 draws.

| Physics and PD | Mean survival | Median | 30-s survivors | Fallen |
| --- | ---: | ---: | ---: | ---: |
| 1200 Hz, existing baseline | 12.998125 s | 5.51 s | 22/64 (34.375%) | 42 |
| 1600 Hz, new evaluation | 11.912500 s | 3.95 s | 21/64 (32.8125%) | 43 |
| 2400 Hz, new evaluation | 13.719375 s | 3.87 s | 24/64 (37.5000%) | 40 |

At 1600 Hz, three baseline falls reach the horizon while four baseline
survivors fall. At 2400 Hz, five are rescued but three previous survivors fall.
Doubling the baseline rate yields only two additional horizon survivors; most
still fall. Env 22 falls at 0.26 s at every tested rate; env 14 falls at
0.32/0.32/0.30 s respectively. This is unlike the prior large improvement from
an under-resolved controller/contact simulation.

## Trace check and interpretation limits

In the common first 0.20 s, rear speed peak-to-peak spans per held 20-ms target
have median 2.79/2.95/2.83 rad/s and p95 12.77/13.22/12.90 rad/s. The largest
1200-Hz span is env 14 at t=0.02–0.04 s: its requested speed is **61.0468 rad/s**,
with actual speed rising from **-10.1525 to 61.6 rad/s**. Its large span is not,
by itself, proof of alternating numerical chatter; the policy requests a large
acceleration. Full sample values are retained in comparison.json.

Physics and explicit PD were changed together, so this experiment cannot
separate servo discretization from contact integration or speed projection.
The results are one seeded set, not exhaustive convergence or proof frequency
has no effect. They do not identify the remaining sim2sim mismatch. A next
useful diagnostic is matched IsaacLab evaluation and controlled initial-state/DR
comparisons rather than assuming that a higher rate alone will restore transfer.
No runtime/default/model/gain change or new hardware operation was made.

## Evidence and reproduction

[Comparison plot](frequency_comparison.png), [numeric comparison](comparison.json),
[per-environment durations](per_environment_lengths.csv),
[exact commands](commands.json), [analysis helper](compare_rates.py).
Each new rate directory contains native histogram, torque graph, summary,
per-episode DR metadata and full masked controller/observation traces.

The existing 1200-Hz pure-survival run was reused, not rerun or overwritten.
New runs each exit 0 with all 64 quotas complete, totaling **128 new episodes**.
Masks, length sums, terminal counts and bounded finite actions reconcile; all
**88 protected existing source/policy/baseline artifacts** retain their hashes.
