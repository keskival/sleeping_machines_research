# Exact public-reference target plan

`aws_reference_target_plan.py` implements the audited Jan26 modded-nanoGPT
validation population with40independent262,144-target sequences, grouped as
five batches of eight rank sequences. Inputs cover0–10,485,759; targets
cover1–10,485,760. Every execution chunk includes its lookahead token. A
chunk boundary preserves numerical state; each rank sequence starts fresh.
Observed EOS resets its own lane before consuming EOS, as in the existing
native token path. No target or loss decides whether a position is scored.

Nine stdlib contracts pass: exact denominator/reset count across five chunk
sizes, required final lookahead, final target inclusion, contiguous chunk
coverage, and rejection of missing/duplicate windows, internal resets,
boundary crossings and invalid chunk sizes. The source audit's embedded
published log hash is verified. Result:
`results/diagnostics/aws_reference_target_plan_20261005T192500Z.json`.
No model ran and no public validation was scored.

This is execution engineering for the selected published-reference protocol,
not a new model restriction. Native persistent state may use the full allowed
sequence/document history; its internal computation need not imitate attention
blocks. Larger history tests have their separately declared target/history
boundary. Random route state and inference work must be recorded by the eventual
scorer. The current training diagnostic's lane interval omits a final lookahead;
that denominator convention is preserved in its historical results and is not
inherited by this public plan.

The next public scorer must consume these exact ranges from the pinned shard,
load a development-selected benchmark-ready checkpoint, perform causal EOS
resets and report its target count, checkpoint/data/source hashes, route RNG,
quality and complete inference work. Public scoring follows selected native
quality/scaling and resource gates; these planning contracts supply no quality
prediction. Reuse the published3.2774NLL rather than training its Transformer.

## Implemented bounded native scorer

`aws_reference_stream_score.py` now implements the native stream over these
ranges with exact next-token likelihood, per-sequence state resets, causal EOS
handling and one recorded route generator. Losses are summed by actual target
count; shorter final chunks receive their correct weight. It verifies selected
checkpoint/data/producer provenance, records CPU wall/RSS and reports the
published-reference quality verdict only after all10,485,760targets complete.
It rejects pending fits, initialized selections and incomplete benchmark
selection records before public scoring. No public run is admitted yet.

Eleven stdlib scorer tests pass: target-weighted chunk invariance, correct
state continuity/reset groups, rejection of target skips, nonfinite losses,
incomplete/reordered populations and incomplete selection evidence. Native
model parity is still pending, not proved by those tests. The guarded numerical
prerequisite `zy_aws_reference_score_contracts_20261005T200000Z.json` runs the
actual same `native_window` on synthetic GPT-2 IDs and observed EOS. It compares
numeric states, route RNG and NLL across two execution partitions, without
fitting or accessing public data. The immutable packet follows slot1's existing
allocation packet at its safe admission boundary.
