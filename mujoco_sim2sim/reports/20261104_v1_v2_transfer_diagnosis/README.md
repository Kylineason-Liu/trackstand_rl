# Why v2 model_50100 transfers worse than v1 model_2500

**The evidence points to a less transferable learned v2 policy/controller
combination, especially under lateral CoM offsets and startup transients.**
This is a real transfer gap, not just an unsatisfactory training metric. A
specific incorrect MuJoCo controller equation has not been established. No
production fix, model retuning, checkpoint/default change or hardware operation
was performed during this analysis. Work was direct, without subagents.

## 1. Confirmed transfer gap and fair v1 comparison

| Evaluation, pure survival and DR | Mean duration | 30-s survivors |
| --- | ---: | ---: |
| v2 model_50100, new frozen IsaacLab evaluation, 1200 Hz | 27.292813 s | 58/64 (90.625%) |
| v2 model_50100, MuJoCo, 1200 Hz | 12.998125 s | 22/64 (34.375%) |
| v1 model_2500, matched MuJoCo plant, 1200 Hz | 24.894688 s | 53/64 (82.8125%) |
| v2 model_50100, MuJoCo, 800 Hz | 12.333750 s | 21/64 (32.8125%) |
| v1 model_2500, matched MuJoCo plant, 800 Hz | 25.379063 s | 54/64 (84.375%) |

The new Isaac run uses the actual deterministic trained actor, not the
TensorBoard training average. All 64 first episodes complete, 58 survive and
six fall. Its DR draws differ from MuJoCo's; equal seeds do not make the two
physics engines sample identical parameters. See the
[Isaac evaluation](../../../isaaclab_train/reports/20261104_v2_50100_dr_diagnosis/README.md).

For the fair MuJoCo comparison, v1 runs on **exact clones of the randomized v2
plants**: masses, inertias, CoMs, contact coefficients, passive losses, initial
qpos/qvel, IMU residuals and future white-noise RNG streams match. All 64 initial
policy observation vectors match exactly. Steering Kp/Kd are fixed 10/0.2 for
both. V1's own gain randomization is disabled for this controlled comparison;
its rear action remains direct torque. Policy/targets stay 50 Hz, all horizons
are 30 seconds, and only timeout/fallen terminate. Each environment contributes
one completed episode, avoiding overcounting rapidly failing streams.

This comparison changes **learned policy and rear action/controller together**;
it does not prove PI velocity control is inherently worse than torque control.
The historical v1 6,000-episode result was 78.4% timeouts at 800 Hz, but it used
a different sampling budget. The matched experiments above confirm the gap
without relying on that historical aggregate.

## 2. Lateral CoM sensitivity is the strongest isolated DR factor

All cases below use model_50100, 1200 Hz, seed 42, 64 one-episode environments,
DR and pure survival, changing only the stated factor. Unchanged DR fields
match the baseline draw exactly in every environment.

| Condition | Mean duration | 30-s survivors |
| --- | ---: | ---: |
| Full DR baseline | 12.998125 s | 22/64 |
| Remove all CoM offsets | 19.757500 s | 40/64 |
| Remove only lateral CoM offset; retain X/Z offsets | 18.977188 s | 36/64 |
| Keep lateral CoM offset; remove X/Z offsets | 11.437813 s | 20/64 |
| Nominal environment, all DR off | 26.716250 s | 47/64 |
| Remove initial roll/roll-rate variation only | 13.110313 s | 22/64 |
| Remove mass variation only | 12.631563 s | 23/64 |
| Fix contact friction at 0.65 only | 12.095000 s | 20/64 |
| Remove IMU turn-on residuals only | 11.755000 s | 19/64 |
| Remove hard velocity projection only | 12.842188 s | 20/64 |
| Remove rear integral term only | 14.776563 s | 27/64 |

[Diagnostic bar chart](ablation_results.png), [numeric table](ablation_results.csv).
Removing lateral CoM variation rescues many cases; removing initial lean does
not. A lateral CoM shift creates a continuing roll moment, requiring sustained
coordinated steering/rear action. The learned v2 policy handles this less
robustly in MuJoCo. This is a demonstrated sensitivity, **not proof that the CoM
conversion is incorrect** or that turning DR off solves the sim2sim problem.

The installed Isaac event samples one XYZ offset per environment and applies
it to the selected bodies. MuJoCo similarly shares an offset and converts it
into the rotated CAD-local frames. Configured ranges agree. Solver/contact and
passive-friction response still differ across engines. The remaining failure
under zero-CoM/nominal cases requires further isolation; ablation effects must
not be added as independent percentages.

## 3. V2 learned much more aggressive startup demands

The matched first observations are identical; the two policies request:

| Initial demand, absolute median across 64 environments | v1 | v2 |
| --- | ---: | ---: |
| Raw steering action before scale/slew | 0.241947 | 0.948356 |
| First rear torque | 0.106124 Nm | 1.206024 Nm |
| First rear torque, p95 | 0.619091 Nm | 3.894636 Nm |

[Initial demand comparison](initial_demands.png).
V1 torque here is reconstructed from its native `6.28 * raw_action` law; v2
uses the actual first physics-step applied motor torque archive. Neither is
claimed to be solver net joint effort or a hardware shaft measurement.

V2's first rear torque is about **11.36 times larger at the median**. Its raw
steering median is close to the Beta endpoint, although target slew still
limits the first position update to 0.478 rad. The integrator starts at zero,
so this initial difference is not accumulated-I history. These commands have
valid units and remain inside their configured limits; they are aggressive
learned behavior, not evidence of an unbounded-action or missing-scale bug.

The v2 action means target speed, so rear torque depends on the simulated
velocity response through `clip(0.30*(v_target-v)+I, +/-6.28 Nm)`. V1 submits
its chosen torque directly. With v2, contact/inertia/friction differences alter
not only the bike motion but the subsequent inner-loop torque. A more abrupt
learned startup strategy gives those differences more opportunity to matter.
This is the leading mechanism supported by the matched comparison and demand
data, not a separately identified contact-versus-servo causal contribution.

## 4. Integrator history is a plausible secondary contributor

I is retained between target changes and updated at 200 Hz, but it is absent
from the eleven policy observations. Identical observed pose/velocity can
therefore coexist with different I and different applied rear torque. That
additional hidden controller state is new relative to v1 rear feedforward.
It is not intrinsically a programming error; both simulators implement retained
history, reset, trapezoidal integration, anti-windup and final torque clipping.
Existing independent checks found identical integral equations on identical
input histories. In actual engine rollouts the histories diverge.

Removing I increases survival only from 22 to 27/64 and changes the controller
away from training. It does not identify I as the main cause or justify deploying
the altered controller with this policy. The v2 training also removed v1 steering
gain DR; reduced response-uncertainty coverage is an additional unisolated
training difference, not a demonstrated cause.

## 5. Weaker explanations and what remains unknown

- **Frequency alone:** 800/1200/1600/2400 Hz gives 21/22/21/24 timeouts; 400 Hz
  worsens it to six. Higher rates do not broadly rescue transfer. Physics and
  PD vary together, so some timestep sensitivity remains.
- **Yaw termination alone:** standard task stops 36 cases for yaw and gives
  seven timeouts, but pure survival still has 42 falls. The fair comparisons use
  pure survival throughout.
- **New observation ordering/normalization:** v1/v2 use the same eleven terms;
  the MuJoCo observation function is unchanged and matched first inputs are
  byte-identical. ONNX/PT fidelity and saved v2 action metadata were verified
  previously. This does not independently prove every later cross-engine state
  matches; it makes a simple newly swapped feature/scale error unlikely.
- **Wrong bicycle geometry:** no geometry/inertia/axis change was introduced;
  the matched tests literally clone the same verified plants. V1 succeeds much
  more often on them.
- **Hard speed cap or one other DR term:** the controlled removals above did
  not restore performance in this sample. They remain engine differences but
  are not supported as the main sole cause by these tests.

## Recommended next work

1. Capture the same short v2 target sequence plus **initial state, sampled
   CoMs/inertias/friction and I state** in Isaac, then replay it in MuJoCo.
   Compare steering position, rear velocity, electromagnetic torque and body
   roll/contact response. The current same-draw comparisons are between two
   MuJoCo controller/policy combinations, not identical sampled Isaac/MuJoCo
   plants. This next test separates controller response from policy feedback.
2. Validate that v2 velocity-PI plant response under loaded, small-action and
   reversal conditions matches the intended hardware. Calibrate uncertainty
   in response rather than retuning verified CAD geometry.
3. If retraining is needed, test less aggressive startup behavior/velocity action
   scale, and include I/controller history or use recurrence. Mirror the resulting
   contract across simulators; changing the scale or gains requires retraining
   and verification rather than silently altering this existing policy.

## Evidence, scope and reproduction

[Verification](verification.json) checks **768 new completed MuJoCo episodes**
(ten 64-env counterfactuals plus two 64-env matched-v1 tests), all 64 first Isaac
completions, quotas/masks/finite bounded actions, unchanged DR fields and
**154 protected source/config/asset/policy inputs**. No production files changed.
The native full-DR and frequency results are reused from prior reports, not
counted as new episodes. One seed/material bank; this is not a universal ranking
of controller families or a multi-training-seed study.

- [`run_ablation.py`](run_ablation.py) applies only process-local diagnostic
  overrides around the unchanged v2 evaluator; each case's `case.json` records
  its exact native argv and counterfactual. `zero_integral` is explicitly marked
  as an override distinct from the checkpoint's trained PI contract.
- [`matched_v1.py`](matched_v1.py) clones sampled v2 plants and invokes the
  unchanged v1 environment step and policy; it retains initial states, DR draws,
  masked policy inputs/actions and per-episode outcomes.
- [`commands.json`](commands.json) records invocations; [`analyze.py`](analyze.py)
  verifies the artifacts and redraws summary figures without physics evaluation.
- Original checkpoint hashes, native logs, DR metadata and full native traces
  are preserved. Isaac's initial sandbox CUDA failure is retained separately;
  its authorized GPU retry completed successfully. No training or live hardware
  operations were performed.
