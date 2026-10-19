# Motor-specific overlays: middle five seconds

[PNG](motor_overlay_middle_5s.png) · [SVG](motor_overlay_middle_5s.svg) · [PDF](motor_overlay_middle_5s.pdf) · [HTML](index.html)

Six panels: columns are attempts 1–3; rows are steering position (rad) and rear
torque (N·m). Each panel overlays three recorded signals on the **same linear axis**:

- Blue: raw policy output ×1.57 steering / ×6.28 rear, without clipping.
- Orange: final encoded MIT steering position / C0 rear torque equivalent, after
  all host processing, on ticks reporting both motor submissions successful.
- Green: A3 steering position / measured rear Q-axis current × signed B0 torque
  constant. Rear torque is **current-derived**, not independently measured shaft torque.

Each five-second window is centred on the midpoint between first and last
inference in the verified parent report. Time zero is the start of that window;
the corresponding bag times are printed above each column. The same y scale is
used across attempts for each motor. The full raw range is retained, so capped
requests and responses may appear close to zero beside large policy spikes.

**Attempt 1 contains only 1.04 seconds of policy activity.** Its five-second
window shows the available short attempt only; shaded surrounding time is left
blank. No samples are invented, repeated or extrapolated. Available source data
include the earlier figure's ≤20 ms pre-inference context and 30 ms response tail.
Attempts 2–3 use complete middle-five-second slices. No smoothing, lag correction,
resampling or interpolation is applied; gaps >50 ms are broken.

Source: the nine verified CSVs in `../presentation_3x3/`, from
`rosbag2_2026_10_12-18_06_07` with model 91850 and corrected rear signs. This bag
predates removal of command limits. The user reported kickstand contact within
one second; later activity is not unsupported balancing.

Nine cropped CSVs and [provenance.json](provenance.json) preserve the plotted
values, source hashes, exact windows and sample counts. Original reports and
deployment code are unchanged.

Reproduce from the repository root (NumPy, Matplotlib):

```bash
MPLCONFIGDIR=/tmp/trackstand_presentation_mpl PYTHONDONTWRITEBYTECODE=1 python3 \
  deploy/experiments/reports/deployment_20261012_180607/presentation_motors_5s/plot_motor_overlay.py
```
