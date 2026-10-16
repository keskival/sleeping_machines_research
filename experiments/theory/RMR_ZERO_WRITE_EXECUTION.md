# Remove zero old-write state without changing the model

16 October 2026, R1.

The eight-level current-slot residual-memory fit completed all 4,096 updates.
Final evaluation exceeded the 3,000,000 KiB process-group RSS limit while scoring
the long development windows. Its final checkpoint is preserved. This is an
execution failure, not a prediction score or a reason to repeat training.

For this admitted variant every level has `top_m=0`. Therefore its old-slot
write matrix W is identically zero. The original implementation nevertheless
retains a T by T matrix per level and computes Q K(dold)^T W and a W^T dold.
Both products are exactly zero. Delete those terms and their matrices. Keep
the current-slot refinements, chunk-lagged own reads, nonlinear joint map,
causal score mask and all trained parameters. A zero-valued scalar dependency
on the old-write output preserves zero gradients rather than changing them to
None, so optimizer weight-decay behavior is preserved if used for training.
Actual old-slot-writing variants retain the original implementation.

`experiments/r1_rmr_compact.py` implements this algebra as a separate subclass;
active loaded trainers and their source files are unchanged. Persistent retained
old-write storage changes from O(K T²) to zero for this variant. The current-slot
state remains O(K T d) during differentiation; soft-read discovery still costs
quadratic arithmetic and chunked attention still needs O(chunk T) temporary
scores. This does not establish sparse execution or improve prediction by itself.

Before admission to a fit, verify complete NLL plus exploration-objective
gradients for every parameter, dense and sparse selected reads, causal prefix
and row-chunk parity, and old-write fallback. The bounded gate runs only through
the existing AWS coordinator. Then an evaluation-only job reconstructs the exact
historical trained source from its immutable admission, loads the finished
checkpoint and scores the original flagship and long development windows.
No training, context reduction or TEST evaluation is introduced.

The predictive priority remains task credit through useful deep compositions.
This repair removes a measured obstacle to evaluating and scaling those models;
it is not a substitute for improving representation learning.
