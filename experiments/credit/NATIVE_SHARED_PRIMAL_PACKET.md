# One native primal packet for teaching and credit

B1/R1 exact execution project, AWS 11 October. Preserve the forward architecture and active fits. Concrete failure: native tangent probes repeat primal operations through generic autodiff, while exhaustive dense cut reconstruction stores a quadratic map. Retain the same native temporal recurrence, gates, normalization and local derivatives; cache the issued primal intermediates instead. This changes execution, not the prediction/learning objective.

`native_cached_jvp.py` captures an actual eval-mode TemporalMemoryLayer forward and supplies exact input-direction responses at fixed elapsed times. Its differentiated memory recurrence carries tangent state through the same complex decay/rotation. Gates use sigmoid derivative g(1−g); exact GELU derivatives and biased-variance LayerNorm derivatives are cached. LayerNorm JVP is invstd*(v−mean(v)−normalized*mean(normalized*v))*weight. The adjoint multiplies by the weight before the symmetric centering projection. `native_cached_linearization.py` supplies that matching reverse map, reversing the temporal recurrence without a dense Jacobian.

Both maps bind context/version and captured parameter tensor versions. Queries after a parameter change or training-mode switch are rejected. A cache cannot silently become credit for a different prefix or elapsed-time law. Cached numerical activations, input/query/output storage and model state are distinct costs; retaining fixed elapsed times here is a declared target, not a claim that timing gradients are unavailable to the model family.

## Completed execution measurements

Five-repeat warmed CPU medians,16 directions/cotangents per issued native layer:

| Native shape | Capture + JVP vs batched autodiff | Capture + VJP vs batched autodiff | Cached floats |
|---|---:|---:|---:|
| d8/n4/2 events | 5.49x | 1.14x | 124 |
| d32/n16/8 events | 8.72x | 1.15x | 1,936 |
| d32/n16/32 events | 13.14x | 1.26x | 7,744 |

Results `aws_native_cached_jvp_20261011_v2` and `aws_native_cached_linearization_20261011_v1`: input tangent/cotangent errors≤2.23e−15; reciprocity error≤2.85e−14; stale-weight rejection passes. Capture includes the real forward call, hook collection and derivative statistics. The forward reference disables unnecessary reverse-graph recording; the initial v1 measurement is preserved as data, and current numbers use this optimized v2 reference. Query buffers and shared parameters are excluded from cached-float counts. These are exact native execution points, not whole-fitting or quality wins. The hypothetical dense1024×1024 cut matrix stores1,048,576 floats; autograd memory itself was not measured.

## Integrated actor parity

`native_training_linearization.py` additionally retains the original forward parameter graph while emitting the same numerical packet. `check_native_packet_actor_update.py` detaches only the input producer boundary, then reconnects it with the packet VJP after the observed loss backward. Layer/head parameter derivatives remain exact autodiff. One actual forward call supplies both the retained graph and packet; there is no dense teaching sweep.

Completed `aws_native_packet_actor_update_20261011_v1`, one real four-event Taxi TRAIN prefix/four Adam updates: normalized every-parameter gradient difference≤4.51e−17, parameter difference≤6.94e−18, Adam difference≤1.05e−17. Each arm scores12 target presentations; gate wall0.648s,peak RSS387.2MiB. Full fitting FLOPs, per-target fitting work and inference work are unmeasured. The cache holds248 floats here plus the retained graph/shared model/query storage. This is a dense-write native diagnostic, not hard-route/counterfactual or arbitrary-depth learned credit. No TEST or DEV quality claim.

The construction provides cheap exact observations and transport inside a finite forward/credit cycle. It is not a new learned teacher: local primitive derivatives anchor the packet. Next use it as an exact reference/teaching interface for the owned compact-credit integration, measuring all acquisition/audit costs and later learning benefit. Do not scale the small fixture into a long fit without an owned decision and measured workload.

## Complete learning-step cost

Completed `aws_packet_actor_step_20261011_v1` counts forward/loss/backward/producer reconnect and Adam on real Taxi TRAIN prefixes,5 timed steps after3 warmups with alternating arm order. B1/d8/n4/L4:reference2.432ms vs packet3.063ms; B4/d32/n16/L32:7.030ms vs9.136ms. The packet integration is26–30% slower, despite parameter parity≤5.56e−17. It retains the parameter graph and performs an additional producer transport, so teaching/query gains do not imply cheaper ordinary fitting. Keep native autodiff as the ordinary learning reference. Use the packet for extra teaching/audit queries; a fused all-parameter reverse backend would need its own contracts and complete-step measurement before adoption.

## Actor plus forward-teaching cost

Completed aws_packet_actor_teaching_step_20261011_v1 includes the full actor update and16 matching input-direction responses: reference/packet7.687/3.402ms at B1,L4 and39.895/12.716ms at B4,L32,2.26/3.14x faster. Response parity<=3.56e-15; parameter parity<=5.56e-17. Single native dense-write layer, real Taxi TRAIN, five timed updates after three warmups. Critic fitting and downstream use are excluded. Ordinary updates without teaching remain26–30% slower: this supports economical teaching-label production, not whole-fit learning efficiency.

## Teaching-query crossover and connected actor

Design: the detached-input actor is needed to demonstrate packet-delivered producer credit. Teaching-only execution need not detach its producer graph and then manually reconnect it. `connected_teaching_actor.py` captures the same native forward packet from a connected input. Native backward supplies ordinary actor gradients; the issued packet supplies extra forward teaching responses. Temporal recurrence, native writes, clock inputs and all parameter derivatives are retained. This is an execution choice for teaching, not learned-credit substitution.

The registered workload sweep uses1/2/4/8/16/32 directions, each a complete forward/loss/backward/Adam step with acquisition charged and matched response buffers. Real Taxi TRAIN, one layer, shapes B1/L4/d8/n4 and B4/L32/d32/n16; five warmed timings. No DEV/TEST, critic fitting or future-learning benefit endpoint. The connected arm additionally records all parameter-gradient/Adam parity; symmetric gradient clones are charged. Sources and timing records are preserved separately from the earlier measurements.

Completed aws_packet_teaching_crossover_20261011_v1: detached packet1–32-query speed2.14–3.52x. Completed aws_connected_teaching_crossover_20261011_v1: connected speed2.42–4.15x, all parameter gradients and Adam moments exactly equal in every measured case. At least one teaching query pays capture cost in these shapes. Zero-query overhead is measured separately below. These are teaching-label-production gains, excluding critic fitting and future utility.

Completed aws_connected_teaching_zero_query_20261011_v1: zero-query native/packet speed0.931/0.951x (capture5–7% slower); one-query2.398/3.790x and16-query2.466/3.886x for small/large shapes. Gradients and Adam remain exactly equal. Use native execution when no teaching query is needed; enable connected capture when supplying teaching evidence. This scheduling choice does not decide the future learning value of acquiring a query. Next queued depth2/4/8 composition retains all native layers and measures full actor/update plus teaching cost and parity.

Completed aws_connected_deep_teaching_20261011_v1: composing packets through depths2/4/8 yields one-query4.79/5.21/5.29x and16-query4.35/4.63/4.72x complete-step speed. All parameter gradients and Adam moments exactly match; tangent responses agree with native forward autodiff. Actual B4/L32/d32/n16 TRAIN shapes. This supports depth scaling of exact teaching-label production, not learned-credit/generalization/whole-fit scaling. Depth16/32 follow-up admitted with the same bounded CPU pipeline.
