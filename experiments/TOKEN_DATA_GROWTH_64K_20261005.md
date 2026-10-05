# Reserved 64K integrated data-exposure test

5 October 2026 · Development only · CPU only · Source-pinned bounded continuation.

## Question

Does the replicated8Krecipe acquire more useful contextual/persistent capacity
with greater training exposure, without changing its mechanisms? At8Kboth seeds
beat initialization and use context. Neither full-width tails nor a doubled
unit message gain improves the selected recipe consistently. Do not redesign
living temporal memory after those results. Test the next data point.

64Kprovides10,914distinct training tokens, compared with2,696at8K. Per pass,
13,343targets use the first adaptive tail and913use the second, compared with
695first-tail targets at8K. This gives more lexical/core learning exposure. It
is a hypothesis about statistical support, not a predicted quality or memory gain.

## Frozen first cell and protocol

- Packed integrated P16/D2/H2/U4, balanced GPT-2token input, frequency-initialized
  exactly normalized adaptive decoder, default tail ranks, original unit gain.
- Persistent numerical state, separate keys/values, temporal rotation/decay,
  computational races, sparse addressed writes and uniform-event K4actual
  alternative-write utility. Batch64per lane, credit16,8lanes, same optimizer.
- Admit65,536TRAINtokens from the same immutable shard,8contiguous lanes.
  Two passes produce131,056scored fitting target presentations,256updates.
  TRAINfrequency ordering/prior uses all65,536admitted tokens at initialization;
  this preprocessing is explicit and charged separately from the fitting trace.
- Preserve the existing development interval at offset20,971,520,2,048tokens,
 8lanes/2,040targets. Initial weights eligible, four checkpoints every64updates.
  Public validation remains untouched. No new LSTM/Transformer training.
- RSS1.2GB/VMS5GB, at least8GiBMemAvailable, one thread/one host job through
  run_safe.sh in tmux. Timeout900s follows ordinary8K395–422targets/s and the
  larger adaptive-head exposure. Log checkpoints and retain every trajectory.

## Admission and continuation

Admit only after the8Kcredit16whole-fit ledger completes with full arithmetic
formula coverage and exact trajectory parity, and both8Kseeds pass the0.02NLL
initial-inclusive learning gate with positive context contributions. Active
jobs and existing AWSpackets are preserved. No parent result is overwritten.

The first64Kcell must improve initialization by0.02NLL to support the next
learning stage. Compare constant TRAIN-mean features and state erasure on its
selected checkpoint. A memory gain alone does not replace held-out quality.
If this cell fails, diagnose its completed trajectory before widening/scaling.
If it passes, measure neighboring widths24/32at the same2passes/protocol, repeat
the selected promising width in seed7, then the reserved256Kpoint. These are
crossed width/data measurements, not an all-knob tuning grid. Pool growth is
separate, keeping selected activity fixed and charging all discovery/credit.

The8Kand64Kdata slices have different frequency priors:report within-stage
learning gains and absolute held-out quality separately. Same2040devtargets
permit direct loss comparison; a lower initialized prior is not itself a
learned contextual improvement. FLOP and wall-time curves have separate axes.
Whole-fit arithmetic, special functions, preprocessing/evaluation, optimizer
and unquantified random sampling remain explicit. No pending point is filled
with an extrapolated score; no scaling exponent is inferred from two cells.

Reserved larger points:256K, then1M;4Mheld back for a predicted extrapolation
check if those stages produce a viable scaling surface. The eventual primary
comparison remains the selected strong Transformer in the region where it leads LSTM references.
