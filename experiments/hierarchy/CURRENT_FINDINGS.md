# G-HIER current findings — 15 October 2026

## Native multi-memory composition learns a usable FIT signal

Eight abstraction levels, four native race stages per level, 169,675 parameters.
Persistent indexed objects integrate child events and query responses from all
active lower learned memories in their support. The final round has no raw-input
shortcut. Reads preserve child state; root publication is generated at a native
race time. Query selection, ownership and arming policies remain explicit.

On 12 distinct 32-event prefixes from one official FIT recording, 64 batch-one
updates reduce paired stochastic NLL 2.547975 → 2.482746. Two common noise tapes
are shared across all episodes, with bitwise train/evaluation forward parity.
Label association rises −0.013494 → +0.037650; it is positive on both tapes.
Raw accuracy reaches the majority predictor's 16.67%; raw NLL remains above the
empirical constant-prior 2.369382. This prompted a frozen readout diagnosis,
not another unchanged backbone fit.

| Frozen construction | Raw NLL | Prior-centred FIT NLL | FIT accuracy |
| --- | ---: | ---: | ---: |
| Initial core and head | 2.547975 | 2.369382 | 16.67% |
| Trained core and head | 2.482746 | **1.925817** | **37.50%** |
| Trained, query contents zeroed | 2.452741 | 2.369382 | 16.67% |
| Initial core, trained head | 2.541522 | 2.369382 | 16.67% |
| Trained core, initial head | 2.461518 | **1.926302** | **37.50%** |

Scope: one FIT recording, two stochastic tapes, no DEV/TEST access. Prior,
logit centre and one nonnegative scalar gain are fitted and scored on these same
FIT logits. These are optimistic learnability results, not generalization or a
semantic hierarchy win. Swaps measure these fixed decoders, not a basis-invariant
ranking. They locate a useful learned signal in the native core for this probe;
removing lower-memory content removes that gain. No per-item noise identifier is
provided. The calibrated artifact changes only the external linear readout.

The loss identity L = H(p) + mean KL(p || q) − A explains why erasing useful
query content can improve raw loss: content adds 0.051704 label association and
0.081709 label-independent penalty. Prior-centred logits expose the signal.
For q(g)=softmax(log p + g(z−mean z)), convex FIT loss has derivative −A at g=0.
This guarantees a positive-gain FIT improvement when A>0, not held-out transfer.

## Native learning and execution costs

64 training presentations contain 2,048 raw event presentations; unique input
contains 384 events. Fitting plus all scheduled scoring took 961.276 CPU seconds,
peak RSS 2,074,196 KiB. Frozen five-way diagnosis took 164.565 seconds and
297,376 KiB RSS. These are implementation diagnostics, not hardware estimates.
For the measured first episode: 79 objects, 78 lower-memory queries, 334 delivered
messages/publications, 10,688 candidate scores and 5,344 selected native updates.
Learning evaluates 10,688 counterfactual values; selected-only evaluation does
not. Two timer candidates are scored. Full fitting FLOPs, per-target arithmetic,
traffic and hardware energy remain unaudited. State tensor counters exclude
graphs, input records, queues, aliases, parameters and Python metadata.

## Formal and implementation advances

State-ownership contracts protect lower published states from a changed private
upper hypothesis. This is inference-state independence, not independence from
shared parameter learning. Native coupled value/time credit passes an integrated
expectation identity for affine selected-value losses; nonlinear value losses
retain a local surrogate. Do not modify its timing derivative in isolation.

`VarianceReducedNativeComposition` preserves native parameters, random draws,
forward outputs, publication times and event work. Only pure emission-timer
credit changes: average the winner indicator over its conditional race law,
removing winner variance from the interior timing gradient. Receipt/value races
retain their coupled teacher. This does not repair event-order discontinuities
or establish a training-quality improvement. See the linked mathematical notes.

## Completed synchronous diagnostic and interpretation

On12 FIT episodes,64updates at batch2 (128 sampled training presentations), the older cached synchronous model's argmax-route
NLL rises2.494155→2.520648; accuracy stays1/12. A frozen diagnosis using three
fixed stochastic race seeds finds average NLL improves by0.010006 instead;
argmax worsens by0.026493. Therefore some stochastic-objective learning occurred,
but it did not yield useful classification or transfer to argmax execution.
Do not describe this as zero learning, successful hierarchy or a family loss.
Root feature standard deviation rises0.10513→0.14717 under argmax; total feature
collapse is not supported by this statistic. It does not measure class separation.

Removing the final model's top raw-read source gives2.520060, an improvement of
0.000588. Of11 top raw reads,9 hit empty input. Static geometry explains the
restricted access: at level7 on128x128 input, source0 stencil offsets are-128,
0,+128 around the origin; only(0,0) is valid. Thus retaining memory was not enough:
this candidate-discovery construction failed to offer broad raw detail at the
highest level. This does not identify the sole cause of the learning failure;
other source reads and credit variance require separate diagnosis.
Sources: `results/hierarchy/aws_g_hier_cached_20261015_v2_probe.json` and
`results/hierarchy/aws_g_hier_cached_frozen_diagnosis_20261015_v1.json`.

## Decision and next decisive test

Use the native multi-memory core, the calibrated readout as an acquired artifact,
and the forward-identical pure-timer improvement for subsequent development.
Do not scale the failed synchronous stencil or repeat the unchanged FIT fit.
The next gate is held-out part-combination transfer, with calibration frozen on
training data and memory-content/pose interventions. Its design must distinguish
part reuse from instance memorization and forbid raw-input shortcuts. See
`theory/HELD_OUT_NATIVE_COMPOSITION.md`. Native benchmark promotion requires
that evidence; a tiny prefix fit does not establish gesture recognition quality.

Evidence: `results/hierarchy/aws_g_hier_native_composition_20261015_v1_{data,learn}.json`,
`aws_g_hier_native_query_attribution_20261015_v1.json`,
`aws_g_hier_native_frozen_readout_20261015_v1.json`.
Theory: `theory/COMPOSITION_STATE_OWNERSHIP.md`,
`theory/NATIVE_RACE_CREDIT_AND_PURE_TIMERS.md`,
`theory/PRIOR_CENTRED_COMPOSITION_READOUT.md`.
