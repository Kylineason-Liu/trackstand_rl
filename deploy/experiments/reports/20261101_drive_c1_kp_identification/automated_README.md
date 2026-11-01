# C1 effective velocity-loop gains

B8/B9 were read only; P and I are fitted together.

Provisional initial velocity Kp: **0.2738932596479153 Nm/(rad/s)**. `None` means evidence screens failed.

This is the velocity-P/MIT-Kd/IdealPD damping coefficient, not passive mechanical damping.

## Estimates

| Method | P A/(rad/s) | P Nm/(rad/s) | I Nm/rad |
|---|---:|---:|---:|
| Time domain | 1.01417 | 0.304251 | 0.0739058 |
| Frequency domain | 0.912977 | 0.273893 | 2.9488 |

## Evidence and limits

- Unknown units of the firmware registers are not inferred from their numeric values.
- Steady torque/error is not used as P: integral action makes that quotient misleading.
- Each time-domain trial has a nuisance offset for unknown prior integral state.
- Whole-trial validation/bootstraps preserve within-trial sample correlation; systematic bias remains.
- Frequency fits use exact actually-sent held targets and separate asynchronous harmonic fits.
- Current-derived torque is electromagnetic torque estimate, not independent shaft torque.
- Passive friction/inertia remain separate plant properties; these gains alone do not reproduce C1.

[Exact statistics](statistics.json) · [Figure gallery](index.html)
