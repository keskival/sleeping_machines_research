## Current result and calibrated repair — 17 October 2026

The completed v6 prospective trial loses: NLL 4.822277 versus 4.808329
for ordinary native learning and 4.821158 for immediate-greedy updates over
1,152 chronological TRAIN targets. Action-value MSE is 0.014223 versus
0.000710 for predicting zero. Random initial values had a much larger scale
than the measured action effects, yet the policy assigned them high confidence.

The new calibrated construction retains the acquired eight-layer native credit
core and its persistent state, but replaces the random response with a neutral,
action-scaled regularized least-squares response. A signed feature captures
first-order response; an absolute-sign feature captures effects shared by both
step directions. A separate uniform audit policy covers all nonzero actions.
The service policy minimizes estimated future cost plus an uncertainty penalty;
no-op has exactly zero cost and uncertainty. This penalty is a heuristic under
changing representations, not a statistical coverage guarantee.

Reuse the best saved ordinary actor at cursor 283800 with its Adam moments,
numerical memory and RNGs. Reset only the changed credit optimizer/readout.
Two acquired integration episodes precede 64 prospective episodes of four
support plus 32 future targets, with continuously adapting ordinary and
immediate-greedy native references. Score before adapting, commit the issued
trajectory regardless of audit outcome, and checkpoint all actors and RNGs.
Numerical gates and the future comparison are conditional jobs in slot 3;
no prediction benefit is reported before their completed results.

# Native reciprocal learning with measured future targets

16 October 2026, R1. Founder requests integration and repair of the earlier
backward/forward future-oriented learner. This builds on notes 160,
RECIPROCAL_EVENT_LEARNING, CONNECTED_CREDIT_CRITIC, CREDIT_LEARNING_CERTIFICATES
and the acquired eight-layer native language model. Numerical and acquired
gates precede prospective learning; implementation is not a performance claim.

## What the earlier evidence actually says

The early future-trained update rule improved query NLL by 0.010945 relative
to its immediate-objective counterpart but remained 0.008624 behind the selected
Adam reference (`aws_reciprocal_update1_20261009T2140Z_analysis.json`). Coarse
coordinate predictors did not reliably transfer credit. Connected future-value
critics initially failed a replicated calibration/utility gate. Averaged
continuation labels subsequently exposed useful conditional signal; the report
records positive action utility at all three seeds, with one uncertainty interval
including zero. Co-adapting actor/credit pilots also exposed clock instability.
None of these is a replicated deep language learning win.

The diagnosis is specific: noisy targets, insufficient state/action interfaces,
and a mismatch between predicting a credit label and improving subsequent
prediction. Repeating coarse-coordinate regression is not the integration path.
Existing exact temporal transport, local eligibility and counterfactual credit
remain useful; learn the unresolved conditional consequence around them.

## Restore the missing episodic producer path

The native indexed prediction head reads numerical stored event features with
an explicit detach. It learns current read-time maps, but that later use cannot
teach the original producer through this boundary. Keep the numerical bank
unchanged and expose its feature cotangent as a learning-only channel.

For each available event/depth pair j let g_j be the actual task cotangent of
the stored feature and J_j the declared conditional producer derivative. The
desired available contribution is sum_j J_j^T g_j. After g is known, sample

    p_j = (1-epsilon) ||g_j|| / sum_i ||g_i|| + epsilon / N,
    estimate = J_J^T g_J / p_J,       epsilon > 0.

Conditional expectation equals the available sum. Choosing learning work after
the observed outcome does not alter the issued forecast. Cotangent norms are
a proxy for complete parameter-credit norms; no optimal-variance claim follows.
When every g is zero, do no replay. Use a separate serialized credit RNG.

Version references identify the actual producer; the numerical bank is not
cleared and unavailable history is not invented. Preserve immutable binding
references for the current credit batch so collection during later events cannot
invalidate its deferred replay. Those snapshots, references, extra reads and
own-write eligibility projections are charged. No stored stream autograd tape
is introduced. Gradients are conditional on the recorded provenance and write
parameters; parameter drift and missing historical paths remain approximate.
Raw conditional expectation is not a statement about clipped or Adam updates.

`deferred_episodic_writer_credit.py` implements this path. The head law is
versioned separately so active jobs remain source-bound. Contracts compare its
prediction and all head derivatives, enumerate the actual producer pullback
mean, require finite delivered native credit, and recover weights, Adam,
numerical memory and every RNG exactly.

## Make the backward learner native and future-directed

`NativeFutureUpdateCredit` is a separate eight-layer persistent native race
machine with receiver-local eligibility, versioned lower-memory queries and full
recorded-depth sampled producer credit. It consumes the actual published state,
producer input, effect of a proposed update at that interface, and its address
and magnitude. Per-interface elapsed observation time is retained explicitly.
It does not reduce parameter coordinates to one coarse mean.
It quotes action advantages through a shared nonlinear readout; no-op is exactly
zero. The credit machine's own state persists across decisions.

The initial action family changes one native mix block, using either sign of
an addressed RMS-normalized known-gradient direction or no-op. Ordinary actor
AdamW still updates every eligible parameter. The action family is a bounded
integration interface, not the full set of possible writes, routes or optimizers.
Actor numerical memory and optimizer moments are retained when the intervention
is applied. The action changes weights, not past stored numerical states.

Let X include actor parameters, numerical memory, Adam, RNGs and versioned
records. For the declared continuation policy pi (ordinary causal learning
within the horizon), supervise

    A_H(X,a) = mean future loss under do(a), pi
               - mean future loss under do(no-op), pi.

Start both branches at the same X and use the same external continuation and
coupled initial randomness. Each branch recomputes its own states, gradients
and subsequent ordinary updates. Do not clamp the changed trajectory to factual
routes. This measures the actual finite nonlinear learning consequence rather
than a tangent substitute. It is one realized target for a conditional expected
advantage, not a noiseless future oracle or an exact whole-history meta-gradient.

Action selection uses only the observed prefix. The first integration episode
has a predeclared bootstrap action; later selections use the native critic with
positive-support exploration. Only after the continuation is observed does its
actual advantage train the critic. Commit the issued branch even if the audit
branch later does better. Choosing the better branch retrospectively would not
be the declared prospective learner.

This is the reciprocal loop: the actor's trajectories teach the native credit
machine; the credit machine selects interventions in the actor's future learning.
The local derivatives and finite audits have different jobs. The episodic
estimator has a conditional inverse-probability identity. The future action-value
learner is supervised by actual intervention returns; it is not silently called
an unbiased gradient estimator. Audit, critic and optimizer work are not free.

## Decisive admitted sequence

1. Numerical contracts, including the available-producer pullback expectation,
   no-op/support and complete actor/credit recovery.
2. Acquired-model gate: reuse the live native eight-layer checkpoint, state,
   Adam and RNGs. Two chronological TRAIN episodes; four support targets then
   eight future targets. Count 24 actual targets and 16 additional audit targets.
   Check finite actor/credit changes and exact actor reconstruction. Keep the
   acquired actor and trained credit machine for reuse.
3. Conditional prospective continuation: 32 episodes, four support targets then
   32 future targets. The candidate and a continuously adapting complete-native
   reference start at the same acquired actor/Adam/numerical state. Both receive
   1,152 actual chronological targets. The candidate pays for 1,024 extra local
   no-action audit targets and credit training. The reference retains the same
   architecture and ordinary learning, without extra future-policy actions.

Report prequential NLL gain, local audited action advantage, critic MSE versus
the zero-advantage predictor, the full target counts and measured resource work.
Calibration and action utility are separate endpoints. The continuous reference
is distinct from the local no-action audit. No TEST or public benchmark is used;
one parent/seed is development evidence. Positive results do not establish
matched-work or hardware supremacy. A negative result triggers diagnosis of
calibration, action magnitude/interface, state sufficiency and horizon noise.

The continuation checkpoints both causal learners, actor and credit optimizers,
all numerical state, rows and all RNGs in one atomic package at every completed
episode. Source identities are captured at launch, not re-read after training.
An interruption may discard the unfinished episode, never a successful epoch
under changed settings. Unsaved process tails are excluded and disclosed.

All three stages run through the existing slot-3 coordinator after its active
source-bound fit. No active job or established mechanism is removed. The
future-value scheme must earn promotion through actual prediction, not just
through more contracts or a better fit to credit labels.
