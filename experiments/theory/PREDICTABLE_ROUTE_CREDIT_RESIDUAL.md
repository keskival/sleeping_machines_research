# Predictable route-credit residuals: closing teaching with fewer counterfactuals

13 October 2026; R1 development following fixed-probability teacher admission. This is the next integrated estimator design, not a fitted-result claim. Preserve the current confirmation and larger-acquisition queues unchanged.

## Construction at a factual event

Condition on a complete factual event frame F: observed inputs, numerical state, current operator/version, chosen site and its factual winner w. Let pi_j be differentiable route probability and Delta_j the actual forced-j suffix loss minus factual suffix loss, on the same continuation/noise and declared scoring horizon. Delta_w=0. Current sampled route credit targets

    g = sum_(j != w) Delta_j grad_theta pi_j.

This is the declared conditional route-credit target. It is not the complete derivative through moving discontinuous winner boundaries, future parameter adaptation or the full learner state. Existing site and suffix-position sampling have their own weights and remain accounted for.

Let a predictor b_psi(F,j) estimate Delta_j before observing the current counterfactual outcome. Evaluate it for admitted candidates, anchor b_w=0, and detach its outputs in the actor-credit surrogate. Draw teacher admission A with probability p(F)>0, then alternative J with the existing positive proposal q_j(F). Neither choice may depend on the current unseen Delta outcome except through a recorded valid conditional propensity. Use

    ghat = sum_j b_j grad_theta pi_j
           + A/(p q_J) (Delta_J−b_J) grad_theta pi_J.

Conditional expectation equals g for every predictor b. A poor predictor increases variance but does not alter the raw-gradient target. Sharing actor/critic parameters requires stopping the actor surrogate's derivative through b; otherwise additional grad_theta b terms change this target. If actor and predictor losses share leaves, their intentionally selected gradients sum at a common parameter version; sequential Adam updates are not equivalent to that sum.

For implementation, use the zero-forward surrogate

    L_credit = sum_j (pi_j−stop(pi_j)) stop(b_j)
               + A/(p q_J) (pi_J−stop(pi_J)) stop(Delta_J−b_J).

The predicted and residual terms fit the existing backward call. Training the prediction head can contribute its own supervised loss to that same call; no separate global actor replay/backward pass is mathematically required. Its compute and paths still cost work. The label's actual counterfactual continuation remains a paid teacher.

## The training loop closes at observed consequences

For the frozen frame, define critic risk sum_j E[(b_j−Delta_j)^2 | F]. An admitted alternative supplies the importance-weighted loss

    L_pred = A/(p q_J) (b_J−stop(Delta_J))^2.

Its conditional expectation is the complete alternative risk. Stop derivatives through p and q for this supervised estimator update. When p or q are learned, train their decision rule against a separately specified causal utility/residual-cost target; differentiating this importance factor naively is not the intended supervised estimator gradient. Factual forecasts can replace factual replay only when parameter/state/proposal identity actually matches, as in Curie's factual-feedback contract.

The actor predicts the world; factual/forced forward continuations label route effects; the predictor turns those effects into actor credit; actual consequences correct both. There is no infinite stack of learned graders: conditional unbiased residual correction anchors the credit target, and the observed suffix loss anchors the critic label. Future-learning credit uses the existing future-update utility instead of suffix loss, with the larger learner-state conditioning and paid labels that objective requires.

## Residual variance gives a concrete admission rule

Let v_j=grad_theta pi_j, and M be a predictable positive semidefinite update metric. Treat b and the differentiable route geometry as fixed under F. For conditional deterministic Delta, the route-sampling/admission variance is

    E ||ghat−g||_M^2
      = (1/p) sum_j ((Delta_j−b_j)^2 ||v_j||_M^2 / q_j)
        − ||sum_j (Delta_j−b_j) v_j||_M^2.

For stochastic continuation/position labels, replace squared residuals by their conditional expectations and retain their sampling variance. The formula assumes q is outcome-independent under F. Denote the first sum by R_b. Minimizing paid teacher cost p c plus lambda times this variance gives p*=clip(sqrt(lambda R_b/c),p_min,1). At fixed p, q*_j is proportional to sqrt(E[(Delta_j−b_j)^2 ||v_j||_M^2 | F]), subject to support floors; minimizing sum a_j/q_j under sum q_j=1 yields that rule by Cauchy–Schwarz. Separate site/suffix sampling requires the appropriate joint conditioning, not multiplying heuristic propensities after observing outcomes.

This replaces the raw teacher moment with a learned residual moment. One admitted label can train the mean b and its second residual moment. Gradient geometry norms are not free: use charged local coordinate metrics or already-available projections initially, and distinguish them from full parameter/update norms.

## Why unbiasedness is not enough for efficient learning

Global clipping C_R is nonexpansive. For a fixed factual gradient g0 and conditionally unbiased ghat,

    || E C_R(g0+ghat) − C_R(g0+g) ||
       <= sqrt(E ||ghat−g||^2).

Thus residual variance controls a useful upper bound on clipping-induced mean-update distortion. Adam's adaptive moments introduce additional sequence dependence; this inequality alone does not guarantee identical Adam trajectories or improved quality. The current admission pilot's quality improvement is empirical, not explained merely by unbiasedness. Measure critic accuracy, clipping frequency and total work to matched quality before assigning the improvement to a mechanism.

## Break-even and integration gate

Let k_b be predictor work per update, c the incremental actual teacher work, and k_train admitted critic-fitting work. Against teacher-every-update, expected saved work per update is

    (1−p)c − k_b − p k_train.

Charge any moment predictor, proposal/discovery, extra stored contexts and optimizer work too. A small teacher does not pay for an arbitrarily large critic. With the present all-key scorer, candidate credit predictions may share current route/context summaries, but they still count as evaluated capacity; no zero-cost dormant learning claim follows.

Next integration should compare fixed probability0.25 against predicted-credit plus corrected residual at the same acquisition/evaluation budget and teacher cap, after the unchanged confirmation/scaling ladder. Numerical gates: b=0 recovers the present admitted estimator; p=1 and exact b agree with the full conditional route target; an exhaustive tiny-route expectation agrees for arbitrary b; shared-parameter stop rules exclude unintended terms. A small integrated fit must show enough quality or teacher savings to pay for prediction/training before scaling. This note admits no new fit or standalone diagnostic queue.
