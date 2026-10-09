# Offline analysis validation (2026-10-07)

- Read all **335,948 rows** of the eight new MIT CSVs. Re-decoded **123,803 motion target/A3/A2/A1/F1 rows** from raw CAN payloads and checked logged units, signs, absolute/relative references, quantization and status bits. Raw F0/B0 calibration/range and CF/status bytes were also checked.
- Verified every filename's frequency and gain pair against measured sine frequency and encoded gains; all requested peak amplitudes are **1 rad**. Encoded Kp is **10.0122100122**, Kd **0.1001221001 / 0.2002442002**. Requested-zero velocity/torque encode to **+0.010989011 native rad/s / +0.004395604 native Nm** at the observed ranges; these residuals are not silently replaced with exact zeros.
- Recomputed the **three ±1 rad C2 baselines** from their original CSVs. Source SHA-256s and gain/phase/RMS statistics match the previous sweep. The generator rechecked all **20 source CSV hashes** and all pre-existing report files after generation; earlier recordings/reports were not modified.
- **130 experiment tests passed**, including **29 C2/MIT numerical-analysis tests**. Tests cover known gain/phase and held-input recovery, finite-record/gap handling, filename parsing, absolute MIT origin and sign conversion, sparse per-reply clocks, raw payload/value disagreements, status bits, missing samples and all eight actual recordings. The real-record test skips if the optional uncommitted CSVs are absent.
- Independent correlation agrees within **0.5 ms** with the fitted A3 phase-equivalent lag for all eight MIT runs. F1 fundamental-gain differences are at most **0.002413** (0.242 percentage points); phase differences at most **2.738°**. These are cross-checks of motor-reported data on host clocks, not independent physical calibration or proof of pure latency.
- All motion command/A3/A2/A1 traces have spacing below **50 ms**. Query late/time-out outcomes remain in the provenance. Each measured reply is plotted once on its own clock; no missing feedback or post-CF position hold is synthesized.
- Generated eight per-condition bundles and one aggregate bundle. Parsed **41 SVGs** as XML; checked **26 PNG** and **10 PDF** signatures; checked five selectable HTML views and all four trace labels per run; resolved **79 local HTML links**; cross-checked **11 comparison CSV rows** with JSON and all individual summaries/analysis files with their source hashes.
- Visually inspected the comparison plot, all-condition three-cycle tracking grid, and representative 10 Hz/Kd=.2 four-panel zoom. Position axes share command/feedback scales; plots retain raw clocks and values with no phase shifting or smoothing.
- Pyflakes and patch whitespace checks pass. Matplotlib emitted the pre-existing Axes3D import warning; all outputs are 2D.

## Scope and reproducibility

Only the new offline analyzer, its tests and report bundles were added. No runtime node, motor configuration, policy, simulation, existing CSV/report or progress record was changed by this task. No CAN socket, ROS node, or hardware command was used.

From the repository root:

```bash
MPLCONFIGDIR=/tmp/trackstand_mit_mpl PYTHONDONTWRITEBYTECODE=1 \
  python3 -m pytest -q -p no:cacheprovider deploy/experiments/tests
python3 -m pyflakes deploy/experiments/analyze_mit_sine.py \
  deploy/experiments/tests/test_mit_sine_analysis.py
MPLCONFIGDIR=/tmp/trackstand_mit_mpl PYTHONDONTWRITEBYTECODE=1 \
  python3 deploy/experiments/analyze_mit_sine.py --reports /tmp/new_mit_reports
```

Use a fresh output directory: the generator refuses to overwrite an existing bundle. Generated SVG/PDF metadata can differ by generation time; numerical results and source hashes are the reproducibility criteria. The manually recorded validation page accompanies the generated bundles; it is not written by the batch generator.

The high-frequency speed/torque findings and the **eight unconfirmed CF shutdowns** remain hardware/firmware validation issues; software tests do not resolve them. See [full conclusions](README.md) and [comparison data](comparison.json).
