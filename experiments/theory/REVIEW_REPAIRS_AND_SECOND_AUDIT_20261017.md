# Implemented review repairs and second audit

17 October 2026, R1. These are separately implemented, gated constructions.
Existing source-bound competitive runs finish unchanged. No new accuracy,
leaderboard, training-efficiency or hardware win is inferred from these gates.

## Repaired information and learning paths

1. **Version-bound direct producer edges.** `DirectInputProducerCredit` records
   the actual immediate lower publication separately from queried children.
   Binding-store part edges retain its version through GC. Sampled replay
   covers direct and query uses with their actual inclusion probabilities;
   aliases are distinct uses. Old acquired publications get no invented edges.
2. **Issuing-event credit, not only later-read credit.** The second audit found
   that retaining direct lineage alone repairs later reads but does not repair
   the detached direct input of the current event. A zero-forward local replay
   now attaches that producer VJP before the upper query/receiver computation.
3. **Current receiver write input derivative.** The second audit also found the
   new selected numerical memory detached from its incoming write path. A
   zero-forward input VJP now replays that local write with frozen write
   coefficients; receiver-parameter eligibility remains separate. This avoids
   double-counting the write-parameter derivative. A deterministic one-receiver
   current-event witness matches finite differences through the lowest mix.
4. **Actual nonlinear query consequences.** `OutcomeCompositionalCredit`
   replaces the query's tangent alternative with an actual replay of the
   current-event native suffix and normalized task NLL after outcome revelation.
   The alternative is a legal ordered pair, using the original prefix, separate
   branch state and common unit-hazard clocks. One sampled query depth receives
   inverse-depth-probability credit. The issued trajectory is never rolled back.
5. **Stable critic feature epochs.** `StableNativeFutureCredit` leaves its
   encoder parameters fixed while fitting recursive response statistics.
   Native numerical state still evolves. Feature parameter changes require an
   explicit epoch reset/recalibration; covariance is never silently reused in
   new coordinates. Explicit forgetting is separate. Actual zero-effect steps
   have zero features/values, no-op wins zero-cost ties, and invalid action norms
   are rejected before state mutation. This critic is not promoted into service.
6. **Writer-input first-order VJP.** The separate RMR writer variant restores
   the derivative through old-slot values consumed by per-target writers.
   Calling this only a second-order path was incorrect. The naive live graph
   exceeded the job envelope; exact non-reentrant checkpoint recomputation
   retains the same gradient without retaining the large gather graph.
7. **Semantic native optimizer groups.** Packed clocks and biases have bank
   dimensions; tensor rank alone misclassifies them as decayable matrices.
   The new policy explicitly protects clocks, transport, biases, gates/scalars
   and address/concept keys while preserving all existing Adam moments and
   group-specific learning rates. Other dormant receiver matrices still incur
   ordinary optimizer work/decay; selective optimizer execution is not claimed.

## Combined integration

`enable_reviewed_learning` installs direct/current write input credit, nonlinear
query teaching and the existing deferred episodic producer credit in the same
native actor. No parameter objects are replaced. The second audit caught two
integration failures before promotion: episodic snapshots needed the new direct
edge map, and prefix cloning had to remove the live writer context rather than
copy a model/gradient context into the audit branch. Both are repaired. The
branch callback recomputes addressed discovery from its own numerical branch;
only the factual state is committed. Factual receiver/word-law predictions and
factual RNGs are preserved.

The acquired learner uses the saved ordinary eight-layer actor at cursor286128,
including its parameters, Adam moments and persistent state. The legacy native
head law is checked against the exact recorded old source; default constructor
migration and fixed-weight forecasts match. Optimizer regrouping preserves
moments. Four new targets followed by four more are real training updates; two
discarded four-target replays test in-place and **fresh-constructor** recovery.
A resumable reviewed checkpoint is retained, with the original parent path and
hash. `restore_reviewed_actor` reconstructs installed methods and optimizer
schema, not merely tensor values.

## Completed evidence

- `aws_r1_review_repairs_20261017_v12`: 16 numerical contracts PASS, 3.139s
  complete process,382960KiB peakRSS. Finite-DAG expectations, nonlinear query
  categorical gradient, factual parity, GC, native episodic integration,
  optimizer moments, epoch behavior and checkpoint recomputation are covered.
- `aws_r1_review_acquired_20261017_v3`: acquired-model PASS,110.862s,
  2463768KiB peakRSS; exact legacy head/factual forecasts and full
  actor/Adam/state/RNG recovery including fresh construction. Eight actual
  TRAIN targets and eight discarded recovery targets; eight nonlinear audits.
- `aws_r1_writer_input_resource_20261017_v2`: PASS,78.544s,
  2746720KiB peakRSS. Sixteen integrated updates,16320 scored TRAIN targets,
  per-target writing, address probes, carry256,256-token lanes. Learning
  gradients and reported gate metrics finite; no history/model reduction.
- The original full-graph resource attempt v1 was stopped by its watchdog at
  process-group RSS3461152KiB, above3000000KiB. Its queue/log remain evidence.
  The earlier v3 numerical failure was a test fixture missing its required
  acquired cause bank; it was corrected, with the failed log retained.

The final numerical gate includes fixed-feature critic recovery through a fresh
constructor, preserving the fitted response, covariance, numerical state and RNGs.

## Second-review boundaries and scaling decisions

- The sampled VJP is unbiased for the recorded frozen-parameter conditional
  content DAG before clipping. Adding a third input use can multiply inverse
  path weights by up to3 per level:3^7 at8layers. This is not a low-variance
  guarantee, and clipping/Adam do not preserve raw-gradient unbiasedness.
  Measure root norms, path weights, clipping and producer use in the next
  bounded predictive continuation; do not infer a depth win from contracts.
- Nonlinear query credit covers one actual event, not all later consequences.
  Historical query routing, native receiver choice/timing and inbox/token
  provenance keep their explicitly limited existing teachers. The repaired
  interface permits longer legal suffix evaluators; it does not fabricate them.
- Audit prefixes and alternative execution have real copying, traffic and
  arithmetic costs. Copying an entire episodic bank is currently an execution
  bottleneck; immutable metadata sharing or a measured copy-on-write bank is
  the next engineering option. No constant-activity learning claim follows.
- Full writer-input checkpointing pays recomputation. Its measured integration
  envelope is passed at the ordered gate; long-history/depth runtime and
  learning benefit still require their own resource/prediction decisions.
- Fixed-feature critic epochs repair coordinate consistency, not proof of
  useful action values. The previous learned-encoder trial loses; no repeat
  policy fit is admitted just by changing temperature or exploration.
- The concurrent theory173 corrections remain: representation depth can add
  useful computation on the same observations; a failed argmax-attribution
  stream does not imply race-produced values/events cannot compose.

Next promotion decision: use the repaired acquired native checkpoint for a
bounded, source-bound predictive continuation with complete online/state/work
admission. Consult the current higher-priority product orders before allocating
new fitting work. Existing competitive jobs are not superseded by these gates.

## Post-integration review

Curie's production RMR writer now computes the full first-order input VJP with
an O(Td) accumulated adjoint. The fifteenth contract compares both values and
gradients with the explicit writer reference, including per-target writes. Use
this production path; the checkpointed implementation remains a resource-tested
reference rather than the preferred scaling construction.

Sparse slot `event_forward` returned correct one-call reads but did not export
its final persistent state. It now materializes lazy timestamps into the same
canonical `last_state` as chunked execution. This costs B*S row materializations
once per lane, reported separately from per-token sparse read/write traffic.
Gate v12 checks split-lane reads and final state against uninterrupted chunked
execution for both ordinary and product-key addressing. This is a persistence
repair; dense key/history discovery and expected-versus-realized firing work
remain explicitly accounted implementation boundaries.

RMR result provenance now includes `native_slot_memory.py`; previously the loaded-source packet omitted that dependency. Earlier raw packets are preserved with that limitation.

## Curie follow-through

The duplicate experimental writer model and its admission script are retired.
Explicit one-hot/checkpoint algebra survives only in tests/rmr_writer_reference.py
as a numerical oracle, not a training candidate. The real resource gate now uses
production RMR directly. Native e2e v3 completed final scoring with peak RSS
1455940KiB; carried diagnostic NLL8.34138 versus reset8.37077 on8192 targets.
This tiny fit is an integration result, not a competitive score. The production
integrated16-update gate is admitted in slot3; the sparse16M rung runs on Curie,
and AWS does not duplicate it. Curie's product-key resume-default fix is merged.
