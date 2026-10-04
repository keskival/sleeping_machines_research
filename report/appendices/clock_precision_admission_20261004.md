# Clock precision and current public confirmation — 4 October2026

Completed owner Mackey–Glass development evidence now separates three inference
policies from the **same fitted p16/D2/pool2 weights**, on three tau18 repeats:

| Policy | Development sMAPE | Core streams per forecast |
| --- | ---: | ---: |
| Sampled races | 25.3096 | 1 |
| Greedy races | 18.4612 | 1 |
| Averaged sampled streams | 17.1638 | 8 |

These are completed exploratory values from
`experiments/results/neurobench_mg/curie_mg5_t18_p16_20261004T052000Z.json`.
Eight streams consume additional inference computation/state. This is neither
an iso-work improvement nor a public tau17 score. Enlarging to pool4 gave
mix8=23.0434 on the same DEV protocol; that negative result remains retained.

The greedy comparison changes both winner and clock. Conditional exponential
race factorization permits a separate clock-noise control. The new prototype
keeps the original sampled winner, changes the common first clock using
T_nu=(ZT)^nu/[Gamma(1+nu)Z], and preserves E[T_nu]=1/Z. Clock coefficient of
variation is1 at nu1,0.5227 at nu0.5 and0 at nu0. These are algebraic facts at a
fixed entering state, not measured prediction gains. Learned scores still
control time, and changed clocks can change later messages, memory and routes.
The native bounded delay's mean is not invariant under this normalization.

| New stage | Data and work boundary | Status |
| --- | --- | --- |
| Native numerical admission | FP32/64 logits/every-gradient, actual compilation, two-step Adam and serialization checks | Prepared, unrun |
| Integrated precision control | tau19 DEV repeat4; three serial arms;32updates ×8lanes ×64positions per arm | Prepared, unrun |
| Public benchmark advantage | Official frozen protocols and matched quality/resource accounting | Not established |

The integration model retains temporal computation, two layers/two heads/two
private receivers per head, separate keys/values, hard addressed writes and
counterfactual message credit:8available/scored receivers and4selected writes
per event. Each arm fits14,336 loss-bearing target presentations after8-position
warmup, while executing16,384 fitting event positions. Full fitting and inference
arithmetic are pending; the prototype evaluates all losing proposals and cannot
claim winner-only work. Compilation/setup, evaluation, memory/traffic and energy
remain separate costs. No pending quality cell receives a forecast.

13 standard-library algebra/guard/protocol checks pass. Frozen numerical
contracts and the prerequisite-gated pilot are in
`experiments/queue/native_clock_noise_20261004T065700Z/manifest.json` (53source
hashes). No model was executed in the Docker workspace. Existing six-session
primate and30-repeat tau17 AWS confirmation queues have allocation priority,
along with owner curie and replay work. These official confirmation results
remain pending; exploratory primate R²0.7409 cannot replace a six-session mean.

See [the new law and numerical scope](../../experiments/theory/151_normalized_clock_noise_and_precision_credit.md)
and [the preceding route/clock factorization](../../experiments/theory/aws_20261004_route_clock_precision.md).
