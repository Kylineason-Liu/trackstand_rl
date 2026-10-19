# Policy → motor response: 3×3 presentation figure

[PNG](policy_to_motor_3x3.png) · [SVG](policy_to_motor_3x3.svg) · [PDF](policy_to_motor_3x3.pdf) · [HTML](index.html)

Columns are the three attempts in `rosbag2_2026_10_12-18_06_07`, model 91850.
Each of the nine panels uses **blue steering radians on the left axis** and
**orange rear torque N·m on the right axis**. Rows are:

1. Recorded raw actions multiplied by 1.57 / 6.28, with no clipping.
2. Final motor-bound requests: `mit_encoded_position_rad` and
   `drive_encoded_torque_nm`, filtered to successful steering **and** rear sends.
   These are decoded payload equivalents after clipping/slew/damping/quantization,
   not the earlier ROS request and not independently captured CAN frames.
3. Measured response from `/trackstand/motor_state`: A3 steering position and
   signed measured Q-axis current × B0 rear torque constant. **Rear feedback is
   current-derived, not independently measured shaft torque.**

All rows in a column use the same recorded clock relative to first inference.
Each column includes the complete attempt and 30 ms of response context after
the last inference; up to 20 ms of measured pre-inference context is retained.
No time shifts, smoothing, resampling or invented feedback are applied. Lines
break across gaps >50 ms; missing values remain NaN. Axis scales match across
columns within each row, and rows 2–3 share identical scales for direct comparison.
Separate unit axes are necessary; equal screen heights do not equate rad and N·m.

This bag predates the removal of command limits. The user reports kickstand
contact within less than one second after lifting; exact contact timestamps are
unknown, and later persistent leaning/oscillation is not free balancing.

The nine CSVs contain every plotted original value before gap-break markers;
[provenance.json](provenance.json) records source fields, counts and hashes.
Only new presentation artifacts are generated; existing reports and deployment
sources are not rewritten. Helper tests use synthetic data only.

Reproduce using a decoded cache containing `raw_action`, `motor_control` and
`motor_state` arrays, plus `details.json` (existing decoder) or `labels.json`:

```bash
MPLCONFIGDIR=/tmp/trackstand_presentation_mpl PYTHONDONTWRITEBYTECODE=1 python3 \
  deploy/experiments/reports/deployment_20261012_180607/presentation_3x3/plot_policy_to_motor.py \
  --cache /path/to/decoded_cache
```

The existing `../deployment_20261012_120632/decode_mcap.py` script, relative to
the parent report directory, can create that cache with its `--bag`/`--output`
arguments; it requires MCAP dependencies and a compatible installed ROS environment.
This plotter uses NumPy and Matplotlib only, with no ROS/CAN runtime initialization.
