# 162 — Forward propagation predicts credit; sparse replays correct its surprise

9 October 2026. B1/R1 shared credit enabler. Builds on notes 86, 88, 110,
127, 140, 145, 160 and 161. Target: close a measured learning gap at useful
total fitting cost while retaining temporal computation, persistent addressed
writes, separate keys/values and credit to unrealized alternatives.

## The advance and the objective it applies to

The forward pass can supply more than an activation: a causal description of
the write, its sensitivity and a compact feature from which its future utility
can be predicted. A learned credit model reads that packet when an outcome
arrives. The proposal here uses that prediction as a control variate and pays
for exact counterfactuals only to correct the residual. Prediction errors
then affect variance rather than the mean of the specified route update.

This complements note 160's pairwise posterior/MH construction. Its target is
the **conditional expected-loss route component** of the actual-write teacher
in note 86. It is not a finite-budget exact estimator of a normalized
marginal-likelihood posterior. Expected negative log likelihood over sampled
routes and negative log marginal likelihood are different objectives. Do not
exchange them when comparing estimators or claiming unbiasedness.

Freeze a legal causal prefix, first-arrival time, forward parameter version
and independent future draws. Let π_i be route probabilities, F_i the complete
forced-route future loss, and h_i = ∇θ log π_i the declared parameter-space
score vector. Then Σ_i π_i h_i = 0, and the route-credit target is

    G = Σ_i π_i F_i h_i.

The remaining content, common-clock, timing and exposure terms are retained
separately. Forced returns include persistent writes, timestamps, seen flags
and subsequent hard decisions. Their replay cost is paid. Detached returns
alone provide no direct losing-content gradient. If the score Jacobian uses
the native custom backward, G describes that declared teacher; it is not a
new proof of the whole physical fixed-noise derivative.

## An unbiased paired residual estimator

Before this audit, fix predictions b_i of F_i from forward packets and the
learning-time outcome. The baseline route vector is

    G₀ = Σ_i π_i b_i h_i.

Observe factual W ~ π and its actual F_W. Draw one supported alternative
I ≠ W from q(I | W), and replay its full state/continuation. Define

    D_IW = (F_I − F_W) − (b_I − b_W).
    Ĝ = G₀ + [π_I / q(I | W)] D_IW h_I.

Write e_i = F_i − b_i. Conditional on W,

    E_I[Ĝ − G₀ | W] = Σ_i π_i (e_i − e_W) h_i
                    = Σ_i π_i e_i h_i,

because the omitted i = W term is zero and Σ_i π_i h_i = 0. Thus the
conditional mean is G for every W. Averaging over factual routes and future
draws preserves the original teacher's mean. The credit predictor may be
wrong without biasing this component.

Useful consequences:

- A common loss offset cancels from every pair. Predictions need only learn
  relative utilities; b_i = F_i + constant gives zero audit variance.
- At two routes, the sole alternative recovers G in every draw, including
  with an arbitrary prediction. This nests the exact two-route teacher.
- If b improves toward F by b_new = αF + (1−α)b + constant, fixed-q audit
  variance is multiplied by (1−α)². This is an identity, not a convergence
  claim for a neural predictor.

The predictor, proposal and probabilities must be fixed before the current
audit draw. Learning from that draw and recomputing its baseline can break
the identity. Update the predictor for the next audit, or use independent
cross-fitting. Detach the predictor from the actor's route-credit objective;
its learning has its own explicit objective.

## The forward pass can amortize the baseline too

Suppose a causal forward packet supplies f_i, and a later learning-time
outcome supplies u(y), with b_i(y) = f_iᵀu(y). Then

    G₀(y) = K u(y),     K = Σ_i (π_i h_i) f_iᵀ.

K can be prepared before the outcome is known. For features transported by
the temporal memory f_i(t) = A(t,τ_i) f_i(τ_i), a stored aggregate advances by

    K(t_next) = K(t) A(t_next,t)ᵀ,

with additions for newly issued packets. Real decay/rotation coordinates
give the same reciprocal transport as note 160. More general packet dynamics
need their actual Jacobian; this linear identity does not cover arbitrary
state-dependent routing or newly changing weights.

This is a concrete interface between forward computation and later credit:
the forward side packages a prediction and its parameter influence; the
backward side supplies an outcome feature and audited surprises. A simple
implementation can instead form G₀ at the outcome using one ordinary mixed
score VJP. Precomputing K earns its cost only when reuse outweighs preparation.

K has P×r entries for P affected parameters and r packet features. That can
be expensive. Dense exact eligibility matrices have the same issue. Local
parameter blocks, lazy time advancement and compressed packets are engineering
options whose storage, arithmetic and approximation error must be measured.
Candidate discovery, forming b_i, all scored keys and optimizer visits remain
charged; a sparse correction does not make the baseline free.

The complete delayed-query horizon is retained in the audited F_i. A short
credit predictor can therefore be corrected by a long replay. Shortening the
replay instead changes the target, as notes 110 and 161 establish.

## A quantitative budget for surprise

With q(I | W) = (1−ζ)π_I/(1−π_W) + ζ/(M−1), the importance multiplier is
at most 1/(1−ζ). If residual utilities share a center with |e_i − c| ≤ ε
and ||h_i|| ≤ H, then

    trace Cov(Ĝ | prefix, time, future draws)
        ≤ 4 ε² H² / (1−ζ)².

The prediction's error controls variance quadratically. The norm H includes
the real shared parameter Jacobian, not just a softmax coordinate norm.
Note 127's parameter-space projections provide one measured approximation
to that geometry; their own probe work and uncertainty are paid.

For variable audit budgets, independently audit each alternative with
Bernoulli probability p_iW and use correction a_iW/p_iW, where

    a_iW = π_i (e_i − e_W) h_i.

The mean remains G. Conditional covariance trace is exactly

    Σ_(i≠W) (1/p_iW − 1) ||a_iW||².

At fixed expected replay budget Σ p_iW c_iW ≤ B, minimizing this quadratic
variance or a certified upper bound gives

    p_iW = clip_[p_floor,1]( ||a_iW|| / sqrt(λ c_iW) ),

with λ chosen to meet the budget. A learned residual norm or forward bound
can replace the unavailable exact norm for allocation; unbiasedness still
holds with positive, recorded propensities. Such an estimate is not a
certificate for dropping an alternative. Preserve exploration unless zero
residual is proved. Shared replay-prefix costs require a different cost model.

This square-root allocation applies to independent inclusion audits. For
one categorical alternative with Σ q_i = 1 and a cost constraint, stationarity
instead gives q_i ∝ ||a_iW||/sqrt(λ c_iW + μ). The normalization multiplier
μ cannot generally be discarded.

## Train the backward predictor for the credit error that matters

For fixed π, q, score geometry and forward targets, consider

    E_W,I [ (π_I/q(I|W))² ||h_I||² D_IW² ]

The covariance trace is this second moment minus ||G−G₀||². Optimizing
the second moment alone is a proxy when G₀ moves; the exact variance
objective includes the derivative of the subtracted term. With two routes
the estimator is exact for every predictor, even when that moment is positive.
For independent inclusion audits the exact variance objective is the sum of
(1/p_iW−1)||a_iW||², averaged over W; this is directly a weighted pairwise
residual regression objective with the probabilities held fixed.

The distinction matters: utility MSE can improve without improving actual
parameter-space credit variance. Forward message parameters can be trained
by this credit objective while keeping the actor's predictive objective
separate. Initially use a credit-only packet head with detached actor features.
Sharing those features later introduces an explicit auxiliary objective and
requires a predictive-quality comparison.

No artificial write Jacobian is inserted recursively into every state update.
This avoids assuming that note 145's unstable surrogate feedback is repaired
by a small coefficient. The genuine factual Jacobians and delayed returns
still require their existing fidelity contracts.

For changing weights, retain parameter versions with packets and replays.
A delayed old-version gradient is a stale gradient, not an exact gradient
at the new weights. Fixed-weight correctness, tracking error and update
stability must be measured separately. Unbiased raw gradients also do not
imply unbiased clipped or Adam-transformed updates (note 139).

## Evidence, attribution and the benchmark decision

`tests/test_forward_residual_credit.py`: eight finite mathematical tests pass.
They enumerate all paired draws at 2/3/8 routes with a shared, nonorthogonal
parameter Jacobian; verify arbitrary-predictor mean, two-route exact nesting,
quadratic variance reduction and perfect-relative-prediction cancellation;
enumerate independent inclusion covariance; check budget/KKT allocation; and
retain the equal-message/different-write counterexample; verify temporal packet
transport; and demonstrate bias from refitting on the current audit. These are constructed
identities with no fitting, benchmark data or integrated native-gradient claim.

Learned control variates and unbiased corrections are established methods:
[Grathwohl et al., Backpropagation through the Void](https://arxiv.org/abs/1711.00123).
Forward sensitivity compression has established precedents:
[Tallec and Ollivier, UORO](https://arxiv.org/abs/1702.05043), with its variance
analyzed by [Cooijmans and Martens](https://arxiv.org/abs/1902.02405).
The contribution developed here is the paired full-write residual identity,
its parameter/cost allocation, and its forward temporal-packet interface
for this substrate. No claim to invent the general control-variate principle.

Next B1/R1 decision: compare exact forced-write credit, the existing local
teacher, and this frozen-predictor-plus-audit teacher on the SAME integrated
recall model, credit horizon and fitting examples. Include a delayed-query
case, a poor predictor, a calibrated predictor and an equal-cost audit budget.
Measure every-parameter mean/variance, clock/content preservation, optimizer
update distributions, posterior-versus-expected-loss objective identity,
packet storage, complete work and recall. The existing 77.4% local-credit
versus 97.5% full-differentiation recall gap supplies the benchmark-relevant
target. Admit a bounded fit after native contracts and measured smoke costs;
use the existing queue discipline and no new external architecture.
