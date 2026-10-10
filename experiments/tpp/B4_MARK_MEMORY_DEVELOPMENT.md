# B4 mark-memory development after numerical repair

The frozen seven-dataset B4 record is complete and unchanged. Mark prediction
causes the total-loss gap on Github and LastFM despite strong timing scores.
The guarded Github fit also skipped a padded batch containing most TRAIN
events. The log-domain repair recovers those gradients; its full development
fit is running. This comparison isolates the next mark-path change against
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
