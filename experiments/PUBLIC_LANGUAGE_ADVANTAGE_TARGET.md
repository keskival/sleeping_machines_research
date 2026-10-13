# R1: concrete public-language target and acquisition ladder

13 October 2026. Own-model development; no external-model fitting. The purpose is a useful quality/work advantage on properly tokenized sequential language. Persistent history, temporal races, key/value separation, hard selected writes and actual counterfactual credit stay integrated.

## Fixed external record, not an ecosystem average

The first pinned public reference is KellerJordan/modded-nanogpt's 29 October 2024 optimizer record, commit `074a19767e46cc0c3d4bc7bd83ea70e8bc60b123`, run `95a9fd44-7c13-49c7-b324-3e7d9e23a499`. Final published validation NLL is **3.3406**, at 5100 updates with global batch512 sequences and sequence length1024: **2,673,868,800 training targets**. The record's training clock is705525ms on its GPU setup; this is not a CPU or ASIC work denominator. Full training FLOPs and measured energy are not inferred from that clock. Raw authors' log and SHA256 binding are under `experiments/results/public_reference/`; this is source evidence, not a trained local control.

[Immutable source and run](https://github.com/KellerJordan/modded-nanogpt/blob/074a19767e46cc0c3d4bc7bd83ea70e8bc60b123/records/track_1_short/2024-10-29_Optimizers/95a9fd44-7c13-49c7-b324-3e7d9e23a499.txt).

The broader modded-NanoGPT benchmark target is3.28 on10485760 validation targets. The pinned historical record is an initial strong-Transformer milestone, not the current state-of-the-art record. Passing3.3406 does not imply current leaderboard supremacy. The next published-record comparison must pin its own completed source and scoring protocol, rather than substitute a changing headline.

## Evaluation contract to implement before opening public validation

Read the immutable reference loader and evaluation source. Bind the exact GPT2-token shard bytes, header interpretation, target offsets, lane/window boundaries, context resets and target count. The public metric is a mean normalized next-token log probability, with every counted target accounted for once. Data identity, denominator and score indices must match. A valid probability model may have a different architecture; context/history differences must be named rather than quietly treated as an exact-history comparison.

Our development scores use2040 targets from offset20971520, eight lane streams and a different context policy. They cannot be placed in the public-reference column. Keep public validation sealed during repair and selection. Verify the scorer on synthetic token streams and development-only data before the selected checkpoint is scored once under the declared contract. A quality win can be reported without an invented external training-compute record; matched-compute claims require that record.

## Registered larger-data promotion

The probability0.25 importance-weighted teacher pilot improves final development NLL by0.076848 at131056 targets, with3.60% less complete CPU training time. Before promotion, complete unchanged seeds6/7/8 at131056 and524272 targets. Each quality/work gate requires final DEV degradation≤0.01nats and less complete training time. Mean final quality must improve at both sizes.

Only then admit the same retained P24D2H2U4/credit16 member at2097088 targets (4096 updates), seed6, against the saved full-teaching2M result. Admit8388352 targets (16384 updates) only after that larger gate passes. Use immutable `bernoulli_teacher_token_language.py`, probability0.25, identical fixed TRAIN prior and fresh-data blocks, proper GPT2 tokens, the same evaluation cadence and the same final-checkpoint rule. Preserve all raw results; record actual teacher calls, replayed tokens and suffix scored targets alongside whole-fit training time. Full factual windowBP and denseAdam remain charged.

This tests whether the new quality/work improvement survives increased acquisition. It is not a substitute for the public score. If the gate fails, diagnose teacher variance, clipping and optimizer response before another fit; do not blindly enlarge data or tune on public validation. Controller `scripts/continue_bernoulli_scaling.py`, one R1 CPU job in AWS slot2, run_safe, one thread, RSS1.5GB,≥8GiB MemAvailable, 5400s2M/14400s8M timeout, bounded10h continuation. No new dense controls.
