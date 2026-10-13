# Depth-first language: allocate useful credit across temporal stages

R1, 13 October 2026. Implementation and fixed acquisition plan, not a quality win.
The objective is normalized next-token loss under the pinned Transformer
benchmark's GPT2/FineWeb scoring protocol. Reused development data selects the
model; public validation stays sealed. AWS owns the existing depth8/16 uniform
teacher fits. Curie owns this distinct deep learning implementation, not another
two-layer comparison or synthetic learning task.

## Failure, retained architecture and intervention

The existing teacher samples one token/layer/head site and applies inverse site
probability plus suffix weighting. At length L, depth D and H heads, uniform
site weight is LDH. As D grows, a finite number of teacher calls is spread across
more layers. Different layers may have very different utility-gradient second
moments, so equal sampling need not buy equal learning progress. This is an
estimator-allocation problem; it is not proof that uniform allocation caused
the language quality gap.

Keep actual8/16temporal stages, final-stage prediction, hard races, persistent
private addressed state, separate keys/values, nonlinear temporal transport,
counterfactual write replay, GPT2 tokens, fixed TRAIN frequency prior and
optimizer/acquisition protocol. No early-layer prediction bypass is added.
Change only TRAIN teacher-site allocation, keeping tokens/heads uniform and
learning layer probabilities from previously matured TRAIN audit signals.
Inference computation is unchanged. Factual window backpropagation and dense
AdamW are still used and charged: this is not a compact-credit/BPTT replacement.

For layer probability p_d, site probability is p_d/(LH). Use inverse weight
LH/p_d exactly where LDH was used. Conditional on the issued learner and past
allocation statistics, summing the weighted site estimator has the same mean
as the original raw teacher. Clipping and Adam do not inherit this identity.
The factual random generator and teacher admission probability remain separate
from the site generator; observed future outcomes do not select their own site.

## Analytical reason and its limits

Let G_d include the within-layer random token/head/alternative utility gradient
and let V_d=E||G_d||^2 for the declared raw estimator. One layer sample gives
second moment sum_d V_d/p_d; its minimum at fixed sample count is attained by
p_d proportional to sqrt(V_d). Equal sampling is optimal only when V_d match.
With p_d >= alpha/D, the second moment is at most1/alpha times the uniform
allocation second moment for those same conditional G_d. Alpha=.25 preserves
all-layer support, not a guarantee that learning improves. Variable replay cost
requires an additional cost constraint; dividing by cost without that derivation
would not be the fixed-count optimum.

The implemented statistic is cheaper: squared gradient in the site's probability
logits of suffix_weight * mean(pi_alternative * loss_difference/q). It includes
the conditional alternative weight and lane averaging, but not the upstream
parameter Jacobian or factual-gradient covariance. Its EMA estimates a local
score proxy, not V_d. The learned distribution mixes normalized square-root
proxy with25%uniform exploration. Its effectiveness must be measured by actual
deep-model prediction and complete work, not by allocation entropy or a good
toy fit. Initial unseen-layer moments use a common value; observed layers get
their first measured moment, then a per-layer EMA. Allocation state and its
generator are checkpointed for exact restart.

## Owned deep acquisition

One integrated64update smoke at each of D8/D16 verifies exact finite-support
weighting, analytic proxy versus autograd, allocator checkpoint replay, finite
updates and input gradients at every actual stage. These are implementation
gates, not model-selection experiments. Measured throughput/RSS determines the
timeout and memory envelope; failure requires an engineering repair without
shrinking depth.

Then reserve D8/8M and D16/8M under the same fixed prior/data, seed6, payload24,
heads2, pool4, creditwindow16, teacher probability.25 and final checkpoint rule
as the AWS deep reference. D8/32M follows as acquisition scaling. No small-data
quality veto and no architecture grid. Compare completed same-depth/same-budget
uniform references when available; earlier2layer results are supporting rows,
not substitutes for deep comparisons. Whole training time, target count,
parameters, selected writes, key scores, teacher replays and allocator work are
reported. CPU time is not ASIC energy or a complete FLOP ledger.

The next quality result must include the same8192target/1024history development
score used by AWS and the pinned public scorer after final model selection.
Internal DEV NLL cannot be compared numerically with the published3.3406 record
until the exact external scoring boundary is used. The historical reference is
a first Transformer milestone; it is not a current state-of-the-art claim.
Existing AWS acquisitions and completed market evidence stay intact. The queued
market query follow-up is withdrawn, freeing Curie for this deep language path.
