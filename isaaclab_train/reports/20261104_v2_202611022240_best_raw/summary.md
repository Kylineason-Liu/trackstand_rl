# V2 DR run 202611022240: raw episode-length checkpoint selection

**Best saved checkpoint: model_50100.pt.** Its raw
`Train/mean_episode_length` is **1296.9000244140625 policy steps**, equivalent
to **25.93800048828125 s** at the saved 50 Hz policy rate. It is the unique
maximum among the 2001 available saved checkpoints; no tie-break was needed.

The absolute raw logged maximum is higher: **1325.760009765625 steps /
26.515200195312502 s**, at iteration **50110**. There is **no checkpoint at
50110**, so that peak cannot identify a saved model. Selection was computed
over exact logged iterations with existing checkpoints, not by choosing a
checkpoint nearest the unsaved peak. The newest checkpoint, `model_99999.pt`,
logs 937.969970703125 steps.

Selected file:
`isaaclab_train/logs/rsl_rl/trackstand_rl_v2_rsl/2026-11-02_22-40-01/model_50100.pt`

SHA256: `32e43762691fcd25a8e72182339557f0f71f43c14cde2106bb71118eb5019af1`.
The checkpoint's internal iteration is also 50100.

| Criterion | Requirement and observed result | Status |
| --- | --- | --- |
| C1 | Correct run and task: saved DR=true, v2 integral velocity action, experiment `trackstand_rl_v2_rsl`, seed 42; physics dt=1/1200 s and decimation 24 give policy dt=0.02 s. | Pass |
| C2 | Exhaustive raw metric: 100000 events at unique iterations 0–99999, no duplicates/regressions, no smoothing or scalar reservoir downsampling. All 2001 saved checkpoints have an exact scalar. EventAccumulator and direct serialized-event parsing agree; coordinator independently rescanned original events with the same result. | Pass |
| C3 | Selected `model_50100.pt` exists, uniquely leads the saved ranking, and its content hash/internal iteration match inspection. The higher global peak at 50110 is explicitly unsaved. | Pass |
| C4 | CPU-only log/checkpoint inspection; no simulator, GPU evaluation or training. All 2006 original run file names/sizes/mtimes remain stable. Content hashes match for the event file, saved configs and selected checkpoint. | Pass |

Evidence: [selection metadata](checkpoint_selection.json),
[saved checkpoint ranking](saved_checkpoint_ranking.csv),
[all raw scalar events](raw_mean_episode_length.csv),
[checkpoint inspection](checkpoint_inspection.json),
[provenance](provenance.json), [preservation](preservation.json), and the
reproducible [selection helper](select_checkpoint.py).

This is a ranking by the raw training metric, not a new frozen-policy evaluation.
Content hashing was not claimed for every unselected checkpoint; those files
were checked through the complete file-list/size/mtime snapshot. Saved
`max_iterations=100000` is supported here by the observed 100000 event iterations,
rather than inferred from the configured budget alone.

Task `v2_dr_202611022240_best_raw`, iteration `i1`, mode `single_pass`.
The expected work-report set is one IsaacLab analysis task, `best_raw_analysis`.
Its artifacts completed before a worker usage-limit interruption prevented final
delivery; the coordinator explicitly relayed the complete artifact set. The
summarizer inspected that evidence and reconciled CSV counts, selected-file hash,
unique saved maximum and absent global-peak checkpoint. Recording is a separate
finalization step and does not determine the technical criterion results above.

User-ready response:

> **Best saved model: model_50100.pt** from the v2 DR run
> `2026-11-02_22-40-01`.
>
> Raw mean episode length: **1296.9000 steps ≈ 25.938 s**. It is the unique
> maximum among **2001 saved checkpoints**, verified against **100000 raw
> TensorBoard events**, without smoothing.
>
> The higher logged peak, **1325.7600 steps (26.5152 s)** at iteration
> **50110**, has **no saved checkpoint**.

Recording finalized by `record_best_raw_dr/i1`: both root progress ledgers were updated
(Markdown §327, line 3437; text line 1748). Recorder coverage, preservation and scope
criteria pass; original ledger bytes/order and protected inputs were preserved.
The accepted technical C1–C4 statuses above are unchanged.
