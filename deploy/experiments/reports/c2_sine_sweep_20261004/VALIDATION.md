# Offline analysis validation (2026-10-04)

- Analyzed all **12** CSV logs; each filename's frequency and peak amplitude was independently checked against its recorded C2 waveform.
- Generated **11** new per-condition bundles. Each contains five SVG windows, full-record PNG/PDF, middle-second PNG, four-panel HTML, plot provenance, tracking JSON, analysis README, and a three-cycle target/feedback overlay.
- Preserved every pre-existing file in `steering_sine_20261003_1439/` byte-for-byte; its old embedded source path predates the directory move. The sweep index links to the relocated CSV directly.
- Rechecked all source SHA-256 values after generation. No CSV, runtime, firmware configuration, policy, simulation, or progress-record file was modified by this analysis.
- **16 numerical-method unit tests passed**, including known gain/phase recovery, held targets without interpolation/extrapolation, amplitude-label parsing, missing measurements, gap handling, finite-record settling, and recovery of a known first-order model.
- Independent correlation agreed with harmonic-fit phase-equivalent lag within **1 ms** for every record. These calculations use the same receipt-clock data, so this does not validate absolute hardware timestamp accuracy.
- Every selected steady window has maximum C2/A3/A2/A1 receipt spacing below **50 ms**. The 2 Hz/±2 rad record's earlier approximately 342 ms and 106 ms feedback gaps are explicitly marked; no interpolation bridges these outages in the correlation analysis.
- Maximum relative spread of fitted gain across three subwindows of a record is **0.676%**. This is within-record repeatability, not an independent-repeat confidence interval.
- Parsed **55** new SVG files as XML; checked PNG/PDF signatures, all five selectable HTML views, matching plot/analysis hashes, and **119** local HTML links across new reports and the comparison index. Cross-checked all **12** comparison-CSV rows against JSON values.
- Visually inspected the sweep comparison, small-amplitude overlays, 2 Hz/±2 rad full recording with marked gaps, and representative low-amplitude zoom/overlay plots.
- Python syntax checks, Pyflakes, and `git diff --check` passed. Matplotlib issued its existing Axes3D availability warning; all generated plots are 2D.

## Reproduce the numerical tests

From the repository root:

```bash
MPLCONFIGDIR=/tmp/trackstand_c2_mpl python3 -m unittest discover -s deploy/experiments/tests -v
python3 -m pyflakes deploy/experiments/analyze_c2_sine.py deploy/experiments/plot_motor_feedback.py deploy/experiments/tests/test_c2_sine_analysis.py
```

The batch generator rechecks hashes/conversions and refuses to overwrite existing report directories. Use a fresh output directory for reproduction, as documented in [README.md](README.md).

No hardware was commanded during report generation or validation. The descriptive model's approximately 90 ms time constant and approximately 1.76 Hz corner are effective fit parameters, not measured firmware registers, a pure delay, or a directly measured cutoff frequency.
