# Sparse observable targets and a calibrated reconstruction test

16 October 2026. Existing three-arm learning sources and schedules stay frozen.

## Decoder learning is already a substantial confound

The frozen native encoder's64-update control finishes with DEV NLL0.686605 and
exactly unchanged native features. Geometry decoder losses decrease markedly
at every level. Those decreases cannot be credited to new native representations.
The joint arm must beat decoder fitting,not its own untrained auxiliary head.

Add a per-level TRAIN-only constant reference. Fit the mean of each scene's mean
target,then average squared reconstruction error over objects within scenes and
scenes within each level,matching the native objective. This prevents different
numbers of local objects from changing the reference distribution. No DEV value
fits the constant. Reconstruction skill1-L/L_constant gives the fraction of
this baseline error removed;negative skill means worse than a constant. It is
not a semantic or causal contribution score. A near-zero baseline needs an
explicit undefined/null result instead of a huge ratio. Both quality endpoints
and reconstruction skill are needed before admitting further learning.

## Exact static teacher without raw-row replay per memory

The existing implementation visits the observation table once for every local
memory (7616row tests in the admitted32-event/two-scene gate). That is a teacher
scaling obstacle,not an identified dominant cost of the native32-event fit.
For this static geometric target,one pass can accumulate positive/negative
counts in persistent world-pixel records. A pixel-relative signature has only
its count cosine channels;its sine channels are zero. Parent signatures follow
from role rotation and addition. Each level needs only four rotation tables,
shared across its active parents. Compute trigonometric tables once per level
per construction rather than per observation/parent target. Normalize once
per output record by sqrt(observed count).

If N observations occupy P pixels and A objects across L levels,construction
uses O(N)observation visits,O(A)group/merge operations,and O(LF)trigonometric
work for F Fourier directions,plus O(AF)rotation/addition work and stored output
signatures. The old code performs O(NA)row membership tests plus O(NLF)phases
when each level partitions the observations. Count products,phases,trig calls,
merges and logical tensor bytes separately. These are teacher-only boundaries;
no full-model FLOP,CPU speed,hardware energy or latency claim is made.

An exact child-to-parent identity underlies the replacement;double-precision
contracts compare every local target against direct observed encoding to1e-12
on a96-event/eight-level fixture,including negative world coordinates. Repeating
the same pixel observations100times changes counts/signatures as required while
leaving merge and trig counts unchanged. Integrated native predictions are
bitwise equal;auxiliary loss and all tested native parameter derivatives agree
to1e-11. Static signed counts deliberately ignore timestamps;do not substitute
this teacher or its sufficient statistics for a temporal model.

SparseGroundedNativeComposition is a separate implementation with the same
state dictionary and inference. It is not installed into active source-bound
fits. Reuse it only after an integrated resource gate if the quality result
selects local grounding. A TRAIN/DEV target-variance audit requires no actor
fit and also checks finite-fixture opposite-label separation of root targets.
Separation is not universal sufficiency or evidence of semantic compression.
