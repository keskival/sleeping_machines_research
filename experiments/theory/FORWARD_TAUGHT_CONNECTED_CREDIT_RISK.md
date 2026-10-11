# Corrected future-risk training for a connected credit function

R1/B1, 11 October 2026. Current failure: approximate transported credit is
accurate on fresh language prefixes, but32 policy updates do not produce
confirmed future utility. A trainable prefix-conditioned consequence function
is a missing part of the reciprocal learner, not a claim of current success.
Retain temporal computation, hard-route alternatives, sparse addressed state,
key/value separation and full optimizer state in the integrated construction.
No native architectural substitution is proposed here.

At issued history X, common route set and optimizer snapshot Omega, let a
connected predictor h_psi(a,X,Omega) forecast a48-target route contrast C_a.
X contains only observations available at issuance. Later forward feedback
supplies cheap shared transport labels t_a and one exact audited contrast C_A.
Use positive sampling probabilities q_a and a fixed declared risk weighting w_a.
Labels and weighting are detached from psi for this training objective.

    R(psi) = sum_a w_a E[(h_psi(a)-C_a)^2 | X].

An unbiased loss estimator for the finite observed future is

    R_hat = sum_a w_a(h_a-t_a)^2
            + (w_A/q_A)[(h_A-C_A)^2-(h_A-t_A)^2].

Conditional on the future and all common labels, E_A[R_hat] is the exact
all-route squared loss. Its psi-gradient is

    2 sum_a w_a(h_a-t_a) grad_psi h_a
       + 2(w_A/q_A)(t_A-C_A) grad_psi h_A.

Thus shared approximate future labels teach every route, and the sparse exact
residual corrects their systematic label bias. The corrected loss can be
negative; clipping it or discarding its correction changes the objective.
Use gradient-scale controls only when their bias/objective effect is declared.
One independent exact no-op reference supplies the same contrast denominator.
Keep the common transport/reference calculation independent of sampled A.

This trains a data-dependent predictor of future consequences, not a predictor
of conventional backpropagated gradients. Its shared connected parameters can
learn reusable route/feature/learner-state relationships. Include parameters,
moments, retained issued activations and feature versions in its work/state
budget. A scalar output is compatible with a large connected predictor.

Use the issued critic value in actor residual credit and retain its activation
until feedback arrives. Train psi after the feedback; do not recompute the
issued critic with future inputs and call it a prefix prediction. If actor
learning uses h trained on its sampled action's return before forming the same
correction, action independence is broken. The first implementation keeps
versions fixed during the finite return and separates feedback updates.

For depth, replace isolated local predictors with connected cut-to-cut credit
functions and add multiscale composition/long-range anchors. This corrected
risk supplies observed-consequence anchors for optional route decisions; it
does not by itself give exact continuous native-parameter cotangents. Those
require their declared derivative objective and positive-support residual audit.

Next engineering gate: prefix-only future-risk prediction on fresh private
language prefixes, versus zero and a capacity-matched uncorrected predictor,
with exact validation labels, complete acquisition/fitting/serving work and
no held-out selection. Then integrate issued predictions into online route
learning and measure held-out prediction per total work. Tiny32-update utility
fits are not a depth-scaling result; the new objective is a defined route
consequence learner that can subsequently be tested across data and depth.
