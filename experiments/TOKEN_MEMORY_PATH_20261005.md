# Stored memory is active; predictive coupling is the next target

Frozen selected GPT-2 FineWeb2K fits, seeds6/7, credit16/64,1,016 development
targets. Actual causal factual trajectories are instrumented without changing
outputs; all four selected losses reproduce within2e-6. No training or public
validation. Selected-seen statistics condition on revisiting an existing slot.

| Seed | Credit | Selected previously seen | Median revisit age | Median per-mode retention | Mean retained norm | Mean new-write norm | Mean same-winner value difference |
|---|---:|---:|---:|---:|---:|---:|---:|
|6|16|0.956939|3.000000|0.754514|3.712738|2.648101|0.258950|
|6|64|0.957185|3.000000|0.755845|3.770240|2.707131|0.263050|
|7|16|0.957431|2.998705|0.754645|3.851711|2.814969|0.249928|
|7|64|0.957431|2.999716|0.758360|3.609333|2.699348|0.245856|

Retention is exp(−age × rate × forget), before norm-preserving temporal rotation.
The value difference compares actual and zero-memory proposals for the SAME
selected unit with identical input, parameters and time. It isolates immediate
value sensitivity, not changed routing or suffix utility. Memory-to-key score
contributions are also nonzero; their full pre-clamp distribution is saved.
All-key scoring, winner-only actual writes and continuous temporal computation
are retained. Instrumentation work is diagnostic overhead, not model FLOPs.

Together with the frozen erasure audit, these numbers rule out uniformly dead
memory or absent value/key connections as an explanation for weak addressed
prediction benefit in these four checkpoints. They do not establish that all
stored modes or slots are useful. Stored state survives and affects computation;
its learned contribution to development prediction is the issue to improve.

## Decision

Do not replace the temporal transport or simply lengthen all time constants
on this evidence. Preserve the existing paired credit-window and full-width
adaptive-decoder AWS comparisons. The full-width decoder tests a concrete
projection restriction: width32 features currently enter rank16/8/4 tail maps.
It retains clocks, depth, sparse state, separate keys/values and actual future
write credit, with additional charged decoder work. The already prepared
integrated default/full-width comparison selects whether that change helps.

Next local diagnostic: measure readout NLL sensitivity and differentiable
credit delivered to stored memory versus recurrent messages at trained states,
using identical factual histories. Then test the indicated learning or output
coupling change in a small integrated fit. Actual-write teachers already handle
both key and transported-value consequences; no immediate surrogate is silently
substituted for those consequences. Broader horizon and more data remain tested
axes, not predicted fixes. Repeat utility audits on selected AWS8K models.

Evidence: `results/diagnostics/curie_token_memory_path_audit_20261005_v1.json`.
Guarded one-job queue:700000KiB RSS,8192MiB available reserve,180s timeout,
one CPU thread. Four forward-parity contracts passed.
