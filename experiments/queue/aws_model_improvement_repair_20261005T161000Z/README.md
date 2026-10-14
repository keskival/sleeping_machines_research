# AWS improvement recovery, v3 accounting

V2 numerical contracts PASSED; pool4 k1/k2 full-shape smokes both rejected
unsupported aten.logsumexp.default. Failed logs and original manifests remain.
V3 adds the stable reduction: for n inputs and r outputs, 2n arithmetic FLOPs,
n+r special evaluations and n-r max comparisons. Forward/backward operator
coverage and a 3x4 exact ledger check join the numerical contracts. No forward
architecture changed in this repair. Fresh tags and source-bound predecessors
are used throughout. Scope/hypotheses are in the original improvement README.

Successful WD smoke preserved: its saved source fields bind admission, with
1.5x measured-RSS margin. This smoke is not a quality result. Slot3 v3
contracts/smokes -> WD.01 -> sparse pool4k1/k2 comparison -> WD.1 -> capacity
smoke/fit pairs. Slot2 retries early C LSTM at identical settings with a fresh
output tag, then original C Transformer and both D Transformer controls.
Slot1 resumes p96 checkpoint without changing its model/settings.

The C LSTM original provenance/stdout remains interrupted evidence, not a
completed fit. early_control_interruption.json binds its old metadata and
transition.json retains the native discarded-work bound. Original control
result path now maps to the same-tag `_recovery_20261005T161000Z` directory;
future tuned-budget/scoreboard audits must read the recovered path explicitly.

Scheduler accepts immutable addenda/<name>.json with host,
parent_manifest_sha256 and slots of new unique job definitions. It freezes
each packet itself plus source/queue hashes, so free slots can be filled
without interrupting healthy trainers. Idle workers wait 30s between reads;
the ordinary host reservation remains held. Do not bypass it.

V2 contract publication briefly failed pull due to tracked edits being in
progress; the contract commit40947335 was subsequently pushed with ece1e643.
This was a publication error, not a failed numerical contract. No result
is silently promoted. Complete results/weights publish automatically.

## Current B10 market work order (14 October)

The founder explicitly authorizes LSTM and Transformer market controls at equal
compute. Implement the source-bound work order in
`../aws_b10_matched_controls_20261014_v1/work_order.json` under
`experiments/market/B10_AWS_MATCHED_CONTROLS.md`. This is queued engineering work,
not a runtime addendum: common likelihood/causality, exact data, full-work ledger
and resource gates precede executable addenda. Fresh fits require no Curie parent
checkpoint. Preserve active language jobs; keep September TEST sealed.
