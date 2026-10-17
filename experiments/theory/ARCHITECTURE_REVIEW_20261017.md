Current repair status: [implemented repairs and second audit](REVIEW_REPAIRS_AND_SECOND_AUDIT_20261017.md). The findings below refer to the earlier source-bound learners; repaired variants are separate and gated, with no predictive win inferred.

# Architecture and learning review — 17 October 2026

Battle R1. Review of the actual integrated native actor, historical producer
credit, reciprocal update critic and the carrier-backed residual-memory model.
Existing source-bound trainers remain unchanged. This is a repair priority list,
not a family loss or a reason to discard supported positive results.

## P1: direct lower-message producer credit is absent

The packed queryable kernel detaches x and arrival at every layer boundary
(`packed_queryable_local_deep_race.py`, consume). This deliberately bounds
current graphs. However, producer replay reconstructs `producer.base_input`
and `producer.inbox` as saved numerical values. Full-depth sampled replay
adds derivatives only through `lineage.sources`, the selected query children.
It does not reconnect the immediate lower-layer message that enters the next
layer's residual stream. Full-depth query credit is not full input-path credit.

Consequence: a lower representation can materially change the upper output
while upper-layer task loss supplies no derivative through that direct edge.
Direct per-depth task heads and queried-child credit provide other learning
signals; the integrated actor is not entirely disconnected. Nevertheless,
calling this complete compositional producer credit would be wrong. It can
favor individually predictive lower features over features useful to the
actual upper composition.

A native-kernel witness fixes race RNGs and winners: the deepest loss has
nonzero finite-difference sensitivity to the lowest mix matrix, while its
backward derivative into that matrix is zero through the detached chain.

Repair first: record the immediate lower publication/version as a distinct
producer-input edge. Preserve that publication while its parent is live.
Include both queried-child and direct-message edges in a declared local VJP
estimator; count distinct uses separately even when they refer to the same
publication. Preserve factual values and race RNGs with zero-forward derivative
attachments. Retain the depth-descending DAG and charge lineage retention,
inclusion weights, replay and gradient variance. Inbox provenance remains a
separate missing temporal path; do not silently invent it. Require exact small
DAG derivative expectations and acquired-actor/Adam/state recovery before a
predictive continuation. Do not replace this with unbounded temporal BPTT.

## P1: linearized alternative credit can rank a harmful route as beneficial

`coupled_native_query_credit.py` adds a zero-forward score term proportional
to `(alternative - assembled).detach()`. Backward multiplies it by the factual
suffix cotangent. This is a conditional linearized assembly teacher, as its
source correctly declares. It does not evaluate the alternative's actual
nonlinear downstream loss.

Counterexample: L(z)=(z^2-1)^2, factual z=0.2, alternative z=2.
The tangent predicts a loss change of -1.3824, while the actual change is
+8.0784. A two-action expected-loss gradient therefore opposes the tangent
teacher. The test proves possible sign reversal, not that the actual actor's
routing gradient is usually wrong. It makes tangent correctness inadequate
as a composition-learning certificate.

Repair: use bounded actual suffix/task-loss differences on sampled legal joint
alternatives where affordable. Preserve the factual issued trajectory, shared
initial numerical state and revealed-data boundary; count both alternative
execution and learning work. Existing actual-loss primitives in the analytical
audit are a starting point, not an already integrated fix. Keep the tangent
teacher labelled as an approximation until actual nonlinear returns demonstrate
its usefulness. Full-history exactness is not a prerequisite for a useful,
properly scoped bounded alternative teacher.

## P1: the calibrated reciprocal learner drifts its feature coordinates

`CalibratedNativeFutureCredit.learn_audit` updates the native encoder with
Adam and then updates response/covariance with the feature captured before
that update. Subsequent observations use the changed encoder, while prior
normal-equation statistics remain in their original coordinates. Even a
pure orthogonal encoder rotation requires transforming the response and
covariance to preserve predictions. General nonlinear feature changes cannot
be repaired by such a simple basis transform.

This does not prove divergence: slow joint adaptation can still be useful.
It does mean the current covariance is not a coherent fixed-feature ridge
information matrix across encoder updates, and its service penalty is not a
calibrated confidence bound. Changing actor parameters and sparse audit
coverage add further nonstationarity. The 130 response coordinates have only
66 audit observations in this gate-plus-trial sequence; extrapolation matters.

Completed prospective evidence now supports withholding promotion:

| Endpoint | Measured value |
| --- | ---: |
| Candidate prequential NLL | 6.332144585 |
| Ordinary native learning | 6.331350446 |
| Immediate-greedy native policy | 6.331610873 |
| Candidate gain vs ordinary | -0.000794139 |
| Candidate gain vs immediate | -0.000533712 |
| Audit MSE | 0.0000719752 |
| Predict-zero audit MSE | 0.0000693515 |
| Committed nonzero service actions | 6 / 64 |

One parent/seed, 2304 fresh chronological TRAIN targets, unequal paid learning
work. Lower NLL/MSE is better. Calibration's scale failure is much smaller than
the earlier v6 pilot, but this comparison still loses. The different NLL level
from v6 reflects a different data interval; compare only within each trial.
Source: `experiments/results/diagnostics/aws_r1_calibrated_reciprocal_continue_20261017_v1.json`.

Repair before another policy trial: hold encoder parameters fixed during each
value-calibration epoch, then explicitly refresh/recalibrate the response when
advancing the encoder. Persistent numerical native state can still evolve.
Use a deliberate forgetting/window policy for actor nonstationarity; this
alone does not fix moving feature coordinates. Require prospective prediction
of audit effects better than zero, then useful service interventions. A
no-op-dominated policy with expensive audits is safe exploration, not a learning
efficiency win. Ordinary native adaptation remains the default.

## P2: residual-memory stability and execution need explicit bounds

Learned positive decay makes each old-write age factor decay, but positivity
alone does not bound beta away from zero. For bounded per-step writes c, the
accumulation factor is at most c/(1-exp(-beta)); it grows without bound as beta
approaches zero. This is not a proof of whole-model stability: write magnitudes,
attention feedback and multiple levels also matter. The observed K4 flagship
win (4.9901) and long-lane regression (5.0874) must stay together.

The running K8 decay experiment is appropriate. Inspect learned rates,
correction norms and the long-lane endpoint before promoting it. A lower rate
floor or state-dependent overwrite/normalization needs evidence that this
specific accumulation remains a defect; do not add arbitrary regularization.

The native query teacher also materializes all n^2 ordered-pair probabilities
to sample one alternative. With the current bounded ring this is manageable,
but it is not sparse total discovery work. An exact conditional sampler can
factor the first-choice marginal conditioned on excluding the factual pair,
then the legal second choice, avoiding the square table. This is an execution
opportunity, not an explanation for the present quality gap.

## What survived this review

Persistent versioned memories and higher-layer multi-memory access are coherent
information paths. Separate keys/values, native temporal computation, explicit
nonlinear composition and residual identity paths remain worth retaining.
Nonzero relational operators and nonzero internal proposals avoid a stack of
simultaneously zero gates; zero residual output scales alone are not a dead
architecture when their own gradients can start learning. Existing numerical
contracts do not establish prediction gains, but neither do the defects above
erase the completed native-race and residual-memory accuracy results.

## Validation and next action

Three witnesses PASS in 0.78s through AWS slot 1, source-bound result:
`experiments/results/diagnostics/aws_r1_architecture_review_gate_20261017_v1.json`.
No new training or benchmark scoring was used for the review.

Priority: close the direct producer-input credit gap, then validate bounded
nonlinear alternative returns. Keep the existing source-bound native and decay
runs intact; extract their completed evidence. Do not launch another reciprocal
policy trial by merely changing a temperature or exploration rate. This review
has identified structural credit and coordinate issues that require an actual
repair and an integrated correctness gate first.

## Integrated concurrent review and formal corrections

Curie's concurrently merged second review has implemented grouped weight decay,
nonidentical concept initialization and balance, longer/carrying training lanes,
per-target old-slot corrections, separate write-address races and zero-forward
write probes. These are implemented capabilities with numerical contracts;
new prediction gains from their combination remain to be measured. They do not
repair the native direct-input or moving-coordinate critic findings above.

The new per-target writer reads old refinements under `no_grad()` in
`RMR.slot_values`. This omits a first-order input derivative through those
refinements; calling it only a second-order path is incorrect. Credit through
other read paths survives. Test a bounded input VJP before treating this as
complete writer/producer composition credit. Do not remove the memory-saving
stop blindly on long lanes.

Theory173 also needed scope corrections. Depth can compute a useful hierarchy
from exactly the same raw observations: two bilinear levels compute
(x1*x2)*(x3*x4), whereas one bilinear level has degree at most two. This is a
formal counterexample to requiring each level to read new information. New
information is neither necessary nor sufficient for a useful descent direction.
Likewise, failed argmax-attribution composition does not imply races cannot
compose through useful values, posterior features or generated events. These
claims are corrected in173; identity initialization and Jacobian conditioning
remain design tools, not a universal convergence theorem.
