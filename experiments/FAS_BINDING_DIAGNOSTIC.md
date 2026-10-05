# FAS binding and truncated-write diagnosis after R8

5 October 2026 · Bounded diagnostic, deferred behind language scaling · Physical owner: curie.

The later FIFO de-interleaver matches the oracle and beats native on FAS v1
(FAS_BENCHMARK.md). This diagnostic can inform a transferable binding/credit
repair; it no longer belongs to a required v1 home-field-win campaign. Keep
already admitted jobs and choose any further admission under the revised
LANGUAGE_CROSSOVER_PLAN.md priority.

R8 keeps slower modes (median base half-life 24s, p90 about 350s) but reaches .573
AUROC at 256 events, close to R1 .567 and R0 .578. Longer horizons alone did not
close the gap. Binding and original-write credit are hypotheses to diagnose,
not consequences already proved by these scores.

Use the actual selected R8 checkpoint, recording its SHA256, result/source
bindings and data-manifest hash. Pair with the tied R1 checkpoint at the same
pool, epochs and training protocol if it remains available. Recover item IDs
through the existing oracle instrumentation; verify regenerated event IDs/times
against the original clean TRAIN runs. Use a fixed 150-run TRAIN diagnostic
cohort, not new test-driven configuration selection. Hidden item IDs never
enter model inputs, keys, timing or training targets.

Record the model's actual selected write addresses (layer/head/slot) and event
times on these sequences. The recording adapter must reproduce logits, states,
arrival times and winners of the selected inference policy on a bounded prefix
before a longer trace. Do not infer a winner solely from rounded probabilities
or changed timestamps, which can alias simultaneous events. Charge recording
and any duplicated computation as diagnostic work.

Report:

- Slot occupancy and write counts per layer/head.
- Per-item route consistency and per-slot item composition/purity.
- Item–slot association conditioned on event type, alongside a shuffled-item
  null that preserves event-type frequencies. This separates item binding from
  routing by process stage. Pool 8 need not store 30 items exclusively; low raw
  purity alone does not prove representational failure.
- Fractions of an item's consecutive events that cross the actual 128-event
  detach boundaries. Distinguish retention of values from a live gradient to
  the original writer. Boundary crossings identify missing direct BPTT support,
  not the accuracy of every route-credit estimator.

If binding is weak beyond the event-type null, nominate one addressing change
and an otherwise matched integrated comparison. If binding is adequate but
write credit is truncated, nominate one credit-horizon intervention and charge
its complete learning work. Preserve races, sparse addressed state and separate
keys/values. Validate each proposed change before a long fit.

Publish to
`experiments/results/diagnostics/curie_fas_binding_r8_20261005T104500Z.json`,
with `task_id: fas_binding_r8`, the specification hash, actual checkpoint hash,
`recorded_policy_parity_passed`, cohort and all measurements. No queue is
fabricated here: physical owner preparation/admission remains required because
the checkpoint and physical lock are inaccessible in this review container.
