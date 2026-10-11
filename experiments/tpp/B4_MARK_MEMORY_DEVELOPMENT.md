# B4 mark-memory development after numerical repair

The frozen seven-dataset B4 record is complete and unchanged. Mark prediction
causes the total-loss gap on Github and LastFM despite strong timing scores.
The guarded Github fit also skipped a padded batch containing most TRAIN
events. The log-domain repair completes75 passes with zero skipped updates and
improves selected DEV loss by2.607 nats/sequence versus the guarded record. This comparison isolates the next mark-path change against
that repaired model, rather than attributing numerical recovery to mark memory.

## Design and retained mechanisms

Reuse the established v20 per-event temporal mark memory: the MOOC windowed
development gain is about3.9 mark nats/sequence; LastFM's common-window mark
loss improves722.8->707.6 with memory plus statistics, while statistics alone
does not robustly help. Select the simpler one-memory-layer/no-statistics
variant for this Github development comparison; no grid is admitted.

The new branch retains event history with per-event decay instead of erasing
it during elapsed silence. Its zero-initialized read enters each conditional
mark distribution, including the state clock. Physical-time computation,
log-normal/exponential races, persistent addressed mark slots, original state
clock and silence compensation remain intact. Clock spreads use the passing
log-domain algebra. This is an auxiliary dense-updated mark pathway; it is not
a sparse-route or full learned-counterfactual-credit demonstration. Base core
parameter initialization is preserved. Added modules change subsequent RNG
consumption, so there is no trajectory-identity claim for independently fitted
arms.

At fixed shared parameters, timing loss has no gradient into the mark branch;
joint learning can still change timing through shared embeddings and core
updates. A mark-only gain must therefore also improve total development loss.
No hazard clipping or padded-state-update change is silently bundled into this
arm. Those changes need their own numerical/quality contracts.

## Completed contract and admitted comparison

`aws_b4_marks_contract_20261010T2257Z` passes on two short TRAIN sequences:
zero-initialized branch score difference0; score difference from v20's established
mark pathway7.11e-15; mark-layer timing-gradient norm0, mark-gradient norm108.36;
full gradient finite. Capacity21,768->29,057 parameters (+33.5%). No claim about
unchanged training outcome or benchmark quality follows from the contract.

Slot1 fit `aws_b4_marks_develop_20261010T2300Z`: Github split1/seed0 TRAIN/DEV,
original d32/modes16/two-layer temporal core, one added mark-memory layer,
no mark statistics, AdamW0.003,100-pass ceiling and30-pass total-DEV patience.
Same data and core recipe as slot2 repaired-baseline fit
`aws_b4_logspread_develop_20261010T2112Z`. Any nonfinite update stops the run
after that pass. Best checkpoint selected on total DEV loss; parameter/Adam/RNG
and batch-order resume checkpoints saved each pass. No TEST interface.

Result-only `aws_b4_marks_decision_20261010T2301Z` compares completed files.
Decision rule recorded before the first new DEV result: require at least1
nat/sequence improvement in both marks and total at the selected checkpoints
and in the mean last20 common epochs, with zero skipped updates. The common
window reduces best-epoch selection noise; this is a one-seed development gate,
not statistical confirmation or a family verdict. Pass -> replicate another
DEV split/seed before choosing a developed protocol; failure -> diagnose the
time/mark and window differences before changing the model. No automatic fit
follows the decision.

## Complete resource boundary

Both arms record passes, scored TRAIN targets, fitting and evaluation wall
time, peak RSS and selected-checkpoint DEV likelihood-forward latency. Summary
columns use whole-fit seconds and seconds/scored-target for both arms. Extra
mark capacity/activity is charged; this is not an iso-compute comparison.
Full fitting FLOPs are unmeasured. The timing races score all available clocks;
the auxiliary memory updates densely, while the addressed mark slot is selected
by the observed event. Sparse-only accounting is not admissible.

Mark fit: timeout10,800s, between-pass model wall budget7,200s, RSS watchdog6GB,
VMS8GB, existing one-thread AWS slot scheduler and host floor8GiB available.
The repaired full-backward diagnostic measured2.95GB; the extra mark layer
motivates the larger cap. At admission the host had over26GiB available, with
the repaired fit and Amazon confirmation in the other slots. No active sources
or prior queues were replaced. All result/checkpoint names are unique.

## Current completed result and diagnostic next step

The joint-memory fit completes64 passes with zero skips; its selected mark/total
loss is65.417/-116.337 versus67.237/-114.144 for the repaired baseline. The
registered gate fails: last20 common-epoch mark gain0.687 is below1.000; total
gain1.061 passes. The gate is not relaxed and this variant is not adopted.

Frozen selected-checkpoint mark-read ablation finds direct mark gain13.822
nats/sequence, helps28/34 DEV sequences, and changes timing by0. The disabled
branch checkpoint has mark loss79.239 versus67.237 in the independently fitted
repaired baseline. That cross-fit difference includes joint-learning effects
and changed RNG; it is not an identified causal interference effect.

This supports testing a fresh residual branch on the repaired checkpoint with
all existing parameters bitwise frozen. Core dropout stays in evaluation mode;
only mark layers/read train with the existing0.003 AdamW recipe. Time loss is
asserted unchanged after every pass; core parameters/buffers are checked exactly.
Admitted aws_b4_frozen_marks_20261011T0307Z:100-pass cap,30-pass total-DEV patience,
4GB RSS/8GB VMS,10,800s queue timeout,7,200s between-pass budget,one-thread AWS
slot1. Gate: selected and last20 mean mark/total gain>=1nat/sequence,zero skips,
unchanged timing/core. Warm-start3345.70s and all residual work are charged.
No TEST or automatic next fit. Labels/models remain first-stage development.

## Current residual and transfer diagnosis

Frozen-core residual fit completes43 passes with zero skips, unchanged timing
and bitwise frozen existing parameters. Best mark gain0.169, last20 mean−0.536:
the registered gate fails. Additional fitting646.58s; cumulative fitting with
repaired warm start3992.29s. Late residual TRAIN mark loss141.49->140.33 improves
while DEV67.07->68.41 worsens. No replication or larger fit follows this failure.

Saved-checkpoint factorial evaluation shows joint branch transfers poorly:
repaired core plus joint branch DEV marks71.615 versus baseline67.237. Keeping
joint input embedding/gap privately changes71.615->71.592 only. This rules out
a large direct input-weight correction in this swap diagnostic, not all
representation hypotheses. The matching joint core/branch stays65.417.

The next measured control fits one global confidence coefficient on TRAIN,
using two complete conditional-mark distributions rather than transferring a
relative logit correction into another core. With p_a=(1-a)p0+a p1, the sum of
negative log probabilities is convex in a: derivative−sum((p1-p0)/p_a), second
derivative sum((p1-p0)^2/p_a^2)>=0. Existing predictor forward values provide
both terms; one scalar primitive solve closes this local fitting problem.
It is not a future-learning acquisition objective. Coefficient a is fixed
before DEV scoring and is identical for every candidate mark and history.
Repaired timing density times p_a remains a normalized marked distribution.
No target-dependent or word-keyed inference mixture is restored.

Completed aws_b4_mark_confidence_20261011T0619Z:TRAIN coefficient0.613305;
DEV marks65.203,total−116.178,timing unchanged,baseline marks67.237. Mark gain
2.034; incremental mark gain0.214 versus joint predictor alone. The one-seed
>=1 mark-nat diagnostic gate passes; this is not adoption or a benchmark win.
Scalar fit18.53ms after predictor evaluation2.941s. Both native predictors
(50,826 parameters including coefficient) and prerequisite fits7040.78s are
charged. The calibrated total is worse than the joint fit's selected total
−116.337 because it retains baseline timing. Do not replace that leading
selected DEV number with the calibration diagnostic.

Next decision: independently confirm scalar calibration before engineering
one integrated mark readout. No blind extra residual/private-input fit is
admitted. All completed failures/checkpoints remain evidence; TEST untouched.

## Fixed-recipe calibration confirmation pipeline — 11 October

After completing the saved-curve scale audit and preserving Amazon seed2, AWS admits two independent training seeds (1,2), not a tuning grid. Github split1, existing repaired and joint-mark architectures, original learning rate/dropout/patience, maximum100 passes; CPU one thread, 3GB RSS cap per fit,9000s queue timeout,8GiB available floor. Each seed trains its two prerequisite predictors and then fits one global coefficient using TRAIN only. No TEST loading. Seed0 remains exploratory selection evidence; seeds1/2 prospectively check the selected recipe on the same DEV split, not independent datasets.

Registered confirmation gate: mean DEV mark improvement at least1nat/sequence across seeds0/1/2 and strictly positive improvement in each seed, with repaired time density unchanged. Initial seed0 single-run gate and failed joint/residual stability gates stay distinct; no gate is relaxed. Report every completed seed and charged predictor fits, including failures. A passing calibration gate is reproducibility of this diagnostic, not a benchmark win or an integrated sparse-learning result.

Immutable queued tags: `aws_b4_repair_confirm_20261011_s{1,2}`, `aws_b4_marks_confirm_20261011_s{1,2}`, `aws_b4_calibration_confirm_20261011_s{1,2}`. Versioned drivers add only explicit seed/input selection, preserving original drivers. Slots1/2 remain governed by the existing AWS scheduler. Checkpoints preserve Adam/RNG and epoch history.

`tmux aws_calibration_confirmation` runs `scripts/continue_aws_calibration_confirmation.py`: await completed prerequisites, submit `aws_b4_calibration_confirmation_20261011_decision` through run_safe, then three saved-predictor DEV serving/per-sequence diagnostic queues. Both passing and failing calibration gates receive these diagnostics; no additional fitting is automatic. Failures or the bounded continuation deadline stop with state in `.git/aws_b4_calibration_confirmation_20261011.continuation.json`. The next engineering choice uses that measured evidence: integrate a compact readout if the benefit is reproducible, otherwise diagnose before spending more compute.

The continuation also has a publication watcher: `tmux aws_calibration_publish` runs `scripts/publish_aws_calibration_confirmation.py`. After the decision and all three serving/error records complete, it waits for a clean tracked tree, takes the publication lock, verifies bound inputs, rebuilds the combined report and split PDFs, commits the current result and pushes. State/log remain in `.git/aws_b4_calibration_confirmation_20261011.publication.*`; failed validation or a concurrent rejected push is recorded explicitly. The report helper emits confirmation blocks only after all records are completed; no pending metric is rendered.
