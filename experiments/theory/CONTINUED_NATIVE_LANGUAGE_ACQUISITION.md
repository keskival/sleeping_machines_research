# Continue acquisition without discarding the acquired learner

14 October 2026 · R1 scaling engineering.

The current driver correctly rejects a changed update budget during ordinary exact recovery. That guard also prevents extending a completed fit without a deliberate migration. Starting again would discard the trained actor, optimizer moments, numerical memories, route-audit statistics and random trajectory. Large language acquisition needs a separate, auditable operation rather than weakening the recovery check.

## Budget-only migration

For the current allocated-depth driver, the declared total steps determine the number of available fresh data blocks, the endpoint and a logged midpoint fingerprint. They do not set a learning-rate schedule or alter an update before the endpoint. Extending the block plan retains its original token prefix and initial TRAIN counts/rank map. If the complete learner state at step K and all generators are preserved, the existing next-update function continues on block/cursor K rather than restarting.

This is conditional on this source-bound implementation. An endpoint-dependent schedule, different tokenizer, count prior, block plan, credit rule or architecture would require a different migration and comparison. Continuation does not make those changes identical training.

The implemented migration permits only `settings.steps` to increase. It requires a completed packet with the same identity and all original settings, verifies source and data hashes, clones the checkpoint and compares the full retained trajectory fingerprint after removing settings. Actor parameters, Adam, persistent numerical state, allocator, all random generators, cursor/block, cumulative targets/timers, curves and selected weights remain unchanged. Exact recovery still requires matching settings; the deliberately migrated checkpoint supplies the new endpoint explicitly.

The existing data shard contains100,000,000GPT-2 tokens. The64M plan admits67,108,864TRAIN tokens and scores67,106,816targets over131,072updates; lane-boundary targets are excluded. Parent32M scores33,553,408targets. This uses the next distinct blocks of the same pinned stream with the existing block/EOS reset semantics; it is not a claim of document-unbounded memory.

## Acquisition and evidence sequence

The admitted sequence is currentD8/8M→D16/8M→D8/32M, then the gated longer-horizonD8/8M comparison, then a budget-only continuation of the completed D8/32M actor to64M. It measures a larger-data point without another cold start. There is no small-data quality veto or extra configuration grid. The64M trajectory retains the baseline credit rule; changing it to the horizon candidate would conflate acquisition with a learning-rule migration and needs a separate decision.

Admission uses completed parent throughput to bound incremental work, with at most7days per fit. The one-thread CPU/run_safe/RSS3GB/host8GiB/cgroup2GiB guards remain. Existing locks are respected. Completed checkpoint, result and selected-weight files are preserved; migration and child artifacts have unique names.

Parent process work, migration preparation and child process work are added once. The successful-trajectory training timer already carries the parent acquisition and must not be added to it again. Any unknown interruption tail remains a lower bound in downstream per-target reporting. Post-fit scoring retains final and fixed short-DEV-selected long-history results, spectra, content cuts, teacher work and capacities. Public validation stays sealed; the larger point is development evidence until the selected public protocol is scored.

## Verification

Focused tests reject incomplete parents, mismatched identities/settings, unchanged budgets and repeat-data arms; they verify independent parent preservation, retained optimizer/RNG tensors, identical prior counts/data prefix and admission of the next fresh block. The real parent migration performs full source/data/trajectory validation inside the R1 continuation pipeline before launch. No migrated64M quality result exists yet.
