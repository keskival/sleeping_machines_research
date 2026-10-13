# Credit reuse should follow operator change, not receipt age alone

R1, 13 October 2026. Formal continuation of anchored depth credit and AWS's
[future-learning memory value](FUTURE_LEARNING_VALUE_OF_ROUTE_MEMORY.md).
The D8/8M,D16/8M,D8/32M program continues unchanged. No new fit or memory-policy
implementation is admitted by this note.

## Three different kinds of staleness

A credit receipt has an observed-prefix cut, actor/optimizer version, critic
version, native state and a declared future-loss query. It can become unsuitable
for the current decision in three distinct ways:

1. The actor/optimizer response changes, even under the same external data law.
2. The conditional future data law or relevant inductive structure changes.
3. The critic parameters change before a delayed issued gradient is applied.

The first changes the credit target's geometry; the second changes what future
consequences should be predicted; the third is stale-gradient optimization.
A single time-decay factor does not distinguish them. Old receipts can remain
useful when their conditional relation still applies. A recent receipt can be
incompatible immediately after a large actor update or routing transition.

## Exact transport drift at depth on a common path

Fix a declared differentiable path/cut with the same coordinate meanings.
Under actor versions A and B let their response maps be A_j and B_j and their
terminal anchors be lambda_D^A and lambda_D^B. Define

    P_A = A_1^T ... A_D^T,   P_B = B_1^T ... B_D^T.

Then

    lambda_0^B-lambda_0^A
      = (P_B-P_A) lambda_D^A + P_B (lambda_D^B-lambda_D^A),

and the product difference telescopes exactly:

    P_B-P_A = sum_j (B_1^T ... B_(j-1)^T)
                       (B_j^T-A_j^T) (A_(j+1)^T ... A_D^T).

Thus a reuse discrepancy is bounded by the sum of changed local operators,
weighted by their actual prefix/suffix amplification, plus terminal-query drift.
Counting optimizer steps or total parameter displacement alone does not give
this bound: equally sized weight changes can have very different consequences
at an issued state. Forward signed-response sketches can measure relevant
projections without materializing all Jacobians, but incomplete projections
are incomplete evidence. Their uncertainty and acquisition cost remain explicit.

For useful deep learning, retain true upstream signal rather than shrinking
all transport to make an absolute bound small. Reuse decisions should assess
response change relative to the target/query and the actor update it controls.
Projected update-space geometry from CROSS_AUDITED_CONNECTED_DEPTH_RISK applies
the same identity to the declared local/optimizer response, not just an input
cotangent. Actual clippedAdam has shared norm and moment couplings to charge.

## Hard-route changes require a second term

The common-path identity does not cover switching a hard winner or changing
which memory cell is read/written. A small continuous parameter change can
cross a race boundary; do not assume a global Lipschitz credit target there.

For a finite horizon, suppose the two conditional route laws differ by at most
delta_t in total variation at each step, uniformly on histories where the
coupled paths have not yet diverged. Sequential maximal coupling bounds the
probability of a route-path mismatch by sum_t delta_t, capped at1. If a scalar
credit-effect observable is bounded in absolute value by M and its same-path
continuous difference is bounded by b, then a coupling yields

    |E effect_B - E effect_A| <= b + 2M min(1,sum_t delta_t).

This is a conditional sufficient bound, not a bound for unrestricted cotangents
or unbounded likelihood losses. Common exponential race tapes are a useful
implemented coupling, but are not automatically maximal; their observed
mismatch frequency is not interchangeable with total variation. If neither a
valid law bound nor a bounded-observable premise is available, use measured
paired interventions and label the resulting compatibility score empirical.

Arrival/state changes and missing memory coordinates can invalidate the cut
comparison too. A matching route address does not certify matching behavior;
the current state, signed operator response and downstream query matter.
This connects learned route similarity to transfer: similar parts are those
with compatible conditional consequences, including the shared route legs,
not merely similar address strings or close embeddings.

## Delayed learning remains causal without rewriting the past

When feedback arrives, an old label can still train the credit function on its
stored old input/version packet. It is not a label for the current actor unless
response/query compatibility is established or the target is re-acquired.
Future data teaches the conditional forecast only after it is observed.
Recomputing a present prediction on that packet is ordinary supervised replay;
applying a retained old critic gradient is a different delayed-update method.

For the latter, if the issued gradient field is L-Lipschitz in critic parameters,
its difference from the corresponding current-packet gradient is bounded by

    ||g(psi_now;x)-g(psi_issued;x)|| <= L ||psi_now-psi_issued||.

This says nothing by itself about actor/target drift and does not establish
convergence of the coupled nonlinear learner. A version number identifies the
snapshot; it does not supply L or the response discrepancy. Keep the retained
activation graph separate from detached counterfactual replay copies, and
record which version receives an update. Receipt expiry should follow a paid
compatibility/uncertainty policy and memory budget, rather than silently
discarding every old observation or treating every old label as current truth.

## Native state and the next implementation decision

Use native critic memory to separate reusable conditional effects, uncertainty,
operator/state compatibility and forecast demand. These can share temporal
features and addressed banks while retaining distinct semantics. Lazy local
transport can advance estimates on elapsed time; observed innovations update
drift only when evidence arrives. Both the learned relation and its inductive
bias may change causally. Learned weights need not be the only adaptive state,
and fixed weights do not imply a stationary effective credit function.

Current language allocation learns only TRAIN layer audit statistics. It does
not implement this operator-conditioned connected critic, nor does DEV update
critic weights. One batch's nonzero layer gradients are not a quality or
deep-scaling win. Finish the registered acquisition and content cuts, then
choose a native shared cut-query repair based on actual quality/response errors.
Report old-packet fit, current-packet transfer and prequential adaptation
separately. Stronger investor evidence is a completed useful quality/data/work
advantage on the exact language protocol, not the number of formal notes.
