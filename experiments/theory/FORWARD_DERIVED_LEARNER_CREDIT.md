# Exact forward structure plus a causal reference learner

B1/R1,11 October2026. After the failed [revealed-future pilot](REVEALED_FUTURE_CREDIT.md), retain known native computation instead of fitting its entire realized response from62 noisy packets. The14,054-parameter future critic fits TRAIN MSE0.000005 but fresh MSE0.001110, versus frozen prefix prior0.000385; its matched zero-future control is0.001123. The information-path gate fails. This diagnoses a first representation/fit, not the conditional-gradient identity or the family.

## What the forward model already knows

In this fixed native carrier, an optional addressed key write changes output-key logits. It does not change deep temporal carrier h, value slots, clock hazards or incoming exogenous observations. Linear elapsed-time key transport gives the exact per-target bonus for every specified write. One causal carrier pass can therefore serve all counterfactual worlds. Each independently adapting readout still has its own288 parameters and Adam moments; those costs scale with worlds. Shared replay matches independent-world scores/gradients/final weights/moments within1.8e-15 at depths2/4/8, including72 worlds at depth4. At72 worlds supported arithmetic2.917→0.331MFLOPs; median82→76ms against the already cached per-world carrier. This is kernel scope, not general internal routing, changing-carrier learning or hardware measurement.

When raw future U has arrived, the frozen-readout response D_F(U,a,b) is known by this forward algebra. For actual branch-local learner response D_A,

    D_A = D_F + (D_A-D_F).

The original paired-effect attempt used a noisy learned baseline. Here D_F is the exact realized frozen response, so its baseline approximation error is removed. Measure the remaining learning-effect residual before fitting it; do not assume it is small for every learner.

## Let one reference world learn causally

Issue b(H) before U, independently of sampled A. Let w^b_j be the reference's readout/Adam state before target j, updated only after scoring that target in world b. Project an alternative a at the same learned reference readout, giving L(U,a;w^b). Define

    h_B(U,a) = L(U,a;w^b) - L(U,b;w^b).

The second term is the real reference-learner loss. The first is a projected alternative using its own key bonus and the reference learner's weight trajectory. It is not the true independently adapting alternative learner. Therefore

    D_A(U,a,b)-h_B(U,a)
       = L(U,a;w^a)-L(U,a;w^b).

All reference learning and temporal forward information already enters h_B. Only route-induced learner differences remain in its residual. The reference head learns from dense supervised observations, not from backpropagated gradient labels for a credit regressor. This is a trainable reference prediction component mapped through exact native intervention algebra; it is not persistent cross-episode learning of a new credit network.

Because b and its learner depend on H,U and not sampled A, h_B is a legal learning-side predictor in the revealed-future raw policy-gradient estimator. Positive supported audits repair its residual for any prediction accuracy. A factual A-trained readout cannot be substituted as a route-independent common reference without changing that proof. Forecasters still choose routes and predict before seeing future U. No time-symmetry assumption or future-dependent inference is introduced.

## Decision and next integration

First contract the reference row's scores/gradients/weights/Adam moments against its independent learner, zero-lr against the exact frozen response, and future-suffix causality. Compare static and reference-learning contrasts against the same256 paid fresh per-future labels retained by the completed future pilot. Charge all72 projected route values, reference head updates, copy/extraction/storage and inherited verification cost. This is read-only model evidence reuse plus a native numerical comparison, not a new data benchmark.

Completed diagnostic: reference-learning projection MSE0.000000252 versus exact frozen-forward0.000005317 and prefix prior0.000385219 on the same256 retained DEV labels. The reference learner reduces error95.26% beyond the frozen response; paired32-prefix bootstrap improvement interval is positive. This admits coupled route learning, not a forward-quality claim.

For this integration use full residual correction, p=1. The predictor already executes the reference learner and the factual learner supplies its own loss, so the true paired contrast is available without another world. Thinning audits would discard paid feedback. Count both actual learner worlds (reuse one only for physically identical interventions), all projected route outputs, independent Adam states and inherited acquisition. Compare prefix, static-forward and reference-learning credit from identical acquired controller/optimizer states and data. No persistent cross-episode critic is fitted in this first comparison. A persistent connected residual critic can subsequently learn route-induced learner differences from paid contrasts, with TRAIN-only trust and full acquisition/reuse accounting. Preserve temporal/keyed state, hard writes, independent factual learners and unbiased raw correction; do not replace these mechanisms with an isolated small gradient predictor. Extension to writes affecting the carrier needs a new causal factorization and contracts.
