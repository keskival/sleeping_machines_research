# Diagnose stored memory before changing its transport

The completed fixed-batch2K models improve development quality in two seeds.
Their frozen erasure interventions show a useful recurrent-message history
path (0.046–0.067NLL), while addressed-memory erasure has small mixed effects.
This identifies the selected recipe's addressed bank as the next diagnostic
coordinate. It does not identify a retention failure by itself.

For a selected receiver, the existing write primitive is

    m_new = R(age * frequency) D(age, rate, forget(x)) m_old + write(x) W_in x.

Rotation preserves Euclidean norm; the diagonal exponential factors determine
retention at that update. The delivered value then applies W_out, normalization
and a gated residual. Nonzero retained norm therefore does not guarantee useful
prediction, and a nonzero value VJP does not guarantee that training improves
that path. Read keys use key + W_key m, so old memory also changes race scores;
that information path is distinct from the selected delivered-value path.
Stored keys are rebuilt from current parameters at credit-window boundaries
and refreshed only at writes inside a window. Clock ages, private state and
all-key scoring remain part of the measured recipe.

The observer calls the unchanged primitive, returning its actual tensors.
It records previously seen selected-slot age, decay factors, retained/write
norms and eight bounded selected-value VJPs per checkpoint. Synthetic double
finite differences contract the memory-to-value and memory-to-score paths.
Unselected memories must receive zero gradient from the fixed-choice value
primitive; score gradients can reach every candidate memory. These contracts
are queued, not claimed passed. Eight stdlib admission checks pass without
executing model work.

The paired8K selected checkpoints must first reproduce original development
NLL. One-token partition and the observer must preserve the original RNG and
target count. Matched-noise memory/message erasures then repeat the2K test.
Frozen interventions change the state distribution; they are not retrained
ablations or independent additive component attributions.

No core substitution is proposed before these measurements. If selected
retention is weak, the next comparison changes a declared timescale treatment
while preserving clocks and temporal transport. If retention is substantial
but read/value sensitivity is weak, test the measured access path. If both are
live but predictive benefit is absent, test learning/exposure on the same
small data and full work boundary. A repair requires its own numerical
contracts and integrated comparison. Useful bank capacity is earned through
held-out prediction, not occupancy or gradient presence.

Immutable slot1 packet:
`queue/aws_model_improvement_repair_20261005T161000Z/addenda/zzz_aws_memory_path_audit_20261005T192000Z.json`.
It follows all four8K fixed-batch fits, preserving their weights and producer
records and every original AWS job. Numerical execution and selected-model
results await the existing scheduler's safe boundary. Broader capacity,
all-key discovery, finite credit and complete fitting work remain separate
engineering priorities. No quality or work gain is predicted from inspection.

## Updated completed2K interpretation

The concurrent source-bound memory-path audit now measures median retention
0.7545–0.7584, retained norms3.61–3.85 versus write norms2.65–2.81, and
same-receiver value effects0.246–0.263. Stored memory survives and changes
computation. This evidence selects learning/readout access as the next2K
coordinate; it does not support a dead-transport repair. Preserve its producer
`token_memory_path_audit.py`; our independently queued8K observer is
`aws_selected_memory_path_audit.py`. Repeat at8K and add bounded derivative
measurements rather than retraining the already diagnosed retention question.
The concurrent complete2K fitting/inference ledger is also preserved; the
remaining work requirement here refers to the selected8K/larger recipe.
