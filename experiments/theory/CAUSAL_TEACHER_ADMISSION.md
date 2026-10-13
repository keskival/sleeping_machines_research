# Causal counterfactual-teacher admission (R1)

## Failure addressed and retained mechanisms

The integrated proper-token learner pays for an actual alternative-write suffix teacher on every optimizer update. Exact replay caching and ordered content gradients remove redundant execution; they do not decide which teaching events are worth buying. We now test stochastic teacher admission while retaining temporal races, persistent numerical state, separate keys/values, hard selected writes, factual window differentiation, the same alternative proposal and the same causal suffix-loss objective. All-key scoring, readout, backward, clipping and dense Adam remain charged. No substitution of a synchronous carrier is involved.

The history interventions show useful state beyond one, four and sixteen tokens, with increased signed value after more acquisition. They do not justify eliminating counterfactual credit or replacing memory with current-token prediction. The next intervention reduces teacher frequency, not retained history or the credit horizon.

## Conditional unbiased teaching and its cost

Let F_t be information available before teacher admission. Let G_t be the existing sampled alternative teacher gradient, including its site, route-proposal and suffix-position importance factors. Choose an admission probability p_t measurable with respect to F_t, draw A_t~Bernoulli(p_t) independently of the current sampled teacher outcome, and use

    Ghat_t = (A_t/p_t) G_t.

Then E[Ghat_t | F_t]=E[G_t | F_t]. If p_t is chosen after seeing a current teacher outcome, this equality requires the actual conditional propensity and positivity; guessing a probability retrospectively is invalid. A separate admission RNG avoids altering factual route-noise draws.

For a positive semidefinite update metric M_t and m_t=E[||G_t||_M² | F_t], the added conditional variance from admission is (1/p_t−1)m_t. If the incremental teacher work is c_t, expected paid teacher work is p_t c_t. Minimizing expected work plus lambda times added variance gives the pointwise rule

    p*_t = clip(sqrt(lambda m_t/c_t), p_min, 1).

This follows by differentiating p c + lambda m/p; it is an admission frontier, not a claim that raw gradient variance is the final future-learning utility. A predictor for m_t can be taught by admitted receipts A_t ||G_t||_M²/p_t. Because matrix norm evaluation itself costs work, start with cheap packet norms or projected update-response summaries. The optimal useful-learning policy ultimately depends on future loss and nonstationary regime, as in the existing future-value construction; variance/cost is a tractable supporting statistic.

A learned pre-admission control variate b_t permits

    Ghat_t = b_t + (A_t/p_t)(G_t−b_t).

Its added variance depends on the residual second moment rather than on ||G_t||². The teacher trains b_t from the same admitted gradients; no third credit trainer is required for this supervised target. All prediction and receipt costs must be included. Cheap forward response labels and factual-feedback reuse can reduce the cost of these targets. Counterfactual outcome and future-update utility remain distinct labels.

## What does and does not stay equal

The present integrated uniform-site mode disables the local sampled-value surrogate in its factual pass. Its suppress-site marker therefore does not change factual numerical prediction or the factual loss gradient. Admission affects only the alternative teacher. This matters: dropping a teacher while also changing another factual surrogate would change the objective, not merely its sampling.

The raw gradient equality does not imply identical Adam updates: clipping and normalization are nonlinear, and moments depend on the realized sequence. Higher variance may reduce achieved quality. The practical criterion is complete work to reach useful quality, including teacher generation, optimizer work and admission overhead.

## Bounded implementation decision

1. Probability1 must reproduce an existing16-update integrated learning trace exactly, excluding wall time and nondifferentiable SVD rounding.
2. Fit probability0.25 for131056 fresh targets, seed6, against the saved ordered execution reference. Keep the same data, optimizer budget, evaluation schedule and frozen protocol. Record actual teacher calls, replayed tokens and scored suffix targets, and complete training time.
3. Admit524272-target confirmation only if final DEV degradation is at most0.01nats and complete training time is lower. This is a development promotion tolerance, not iso-quality supremacy. Stop and diagnose a failed gate; do not tune on public validation.

A successful result licenses a learned residual/admission integration inside R1, with a paired quality/work test. It does not license an unmeasured architectural or ASIC-energy claim.
