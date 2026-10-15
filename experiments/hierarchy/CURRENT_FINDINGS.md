# G-HIER current findings —15 October2026

The current native construction is an indexed temporal object: persistent state,
lazy indexed reads, native receipt/routing races and race-generated emissions.
Eight abstraction levels each use four native stages. Parent receivers integrate
many child messages and may receive explicit queries over multiple lower memories.
Arming/query/coalescing/ownership policies are not yet learned. The native local
counterfactual teacher is not exact credit through all queue reorderings.

## Completed raw-native mechanism/resource gate

32 raw FIT events,169,675 parameters,79 addressed objects. Native state tensors
use78,520bytes; graphs, queues, published aliases, input records and Python
metadata are separate.8448 candidate scores,4224 selected updates,8448 native
counterfactual value evaluations and2 timer scores were charged. Indexed query
and an armed race produced further parent computation without new sensor
observations; the queue drained and gradients were finite. Peak process RSS was
1,078,092KiB including runtime/graphs, process time7.579s. These are software
resource checks, not hardware latency/energy or quality results. Source:
`results/hierarchy/aws_g_hier_native_objects_preflight_20261015_v1.json`.

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

## Decision

Do not deepen or scale the unchanged synchronous stencil construction. Keep its
completed evidence and cached suffix implementation. Native event delivery
exposes actual active child updates to parent memory, avoiding an all-silent
fixed-stencil assumption. Preserve stochastic race semantics between learning
and evaluation; replacing a race with argmax is a separate policy change.
Next native learning admission must test this train/evaluation parity and input
identifiability before a bounded FIT learning gate, then held-out composition.
No raw-native trained-quality result or learned-part transfer is claimed.
