# Observation-anchored audit allocation without a trainer hierarchy

B1/R1, AWS implementation companion to CREDIT_AUDIT_WORK_FRONTIER.md. No active Curie actor/critic sources are changed. This implements reusable risk/budget primitives; native integration and learned-risk quality are separate next experiments.

## Risk learning closes on observed feedback

For an issued history/action let r=(D−h)s be the actual gradient correction, with credit prediction h and policy score s frozen before feedback. A small conditional estimator v_w(x)=softplus(wᵀx) predicts E[||r||²|x]. An audit coin I has recorded propensity p>0 and is independent of unrevealed feedback conditional on the issued coordinates. The sampled objective and gradient are

    L_w = I/(2p) [v_w(x)−||r||²]²,
    ∇_w L_w = I/p [v_w(x)−||r||²] sigmoid(wᵀx) x.

Their conditional means equal those of the full observed-residual objective. Thus sparse audits teach a risk estimator using a scalar already available from correction feedback; no learned teacher of that risk estimator is needed. The risk estimator guides which credit errors to observe next. Targets and gradients carry issued features/predictions/propensity, so delayed feedback remains an old-version training observation. It is not silently asserted to be a current-version gradient. A changing actor can change the target conditional moment; drift and sparse-data coverage still require measurement.

`ResidualRiskLearner` implements this local update. No residual is accepted for an unaudited packet. A poor risk estimate changes sampling economy, but the positive recorded audit propensity still preserves the declared raw gradient estimator's mean. Clipping, Adam and changing causal objectives need their separate treatment. Risk fitting costs O(feature dimension) per audited update and queries; features, their construction, actor scores, critic queries and correction work are additional costs. There is no claim that useful features are free.

## Global reference allocation

Given route probabilities π, positive audit costs c, nonnegative risks v and a budget B, solve

    minimize Σ_a π_a v_a/p_a
    subject to Σ_a π_a c_a p_a ≤ B, p_min ≤ p_a ≤ 1.

The KKT solution is p_a=clip(sqrt(v_a/(λ c_a)),p_min,1). π cancels from each interior derivative, but determines λ through the global budget. The allocation is optimal for supplied risks/costs, not for unknown future values. The covariance subtraction term is independent of p for fixed critic and policy.

`allocate` provides a full-array correctness reference. Version2 solves in log coordinates to avoid underflow/overflow of v/c at extreme finite values. Zero risks retain the support floor and need not exhaust the budget. Infeasible budgets are rejected. This reference scans all supplied routes; it is not capacity-independent discovery or sparse scoring.

## Selected-route controller

The entire route array need not be enumerated to *apply* allocation. Maintain one scalar log price η=log λ and issue, for the already-selected route,

    p=clip(exp((log v−log c−η)/2),p_min,1).

After the audit coin, update η by a positive step times (I c−B), assuming c is the known incremental audit cost in the declared units. Its conditional drift is E[c p]−B. Overspending raises the price and lowers audit rates; underspending does the reverse. With oracle risks and the optimal price, the local rates equal the global constrained solution. A known cost may be an audited operation count; predicted or conditional-mean costs require checking actual expenditure separately. Do not call predicted cost measured wall time.

`AuditBudget` implements the issued scalar controller. It has constant-sized state and does not enumerate candidates. This is an average-work controller, not a hard event/resource cap. The host watchdog, memory limits and queue timeout remain authoritative. A budget below the expected floor cost is infeasible and must be detected. Nonstationary risks/costs, critic acquisition cost, step-size noise, delayed feedback and score sensitivity can prevent equilibrium. The finite drift contract is not a convergence proof.

Together, credit errors teach risk; audit expenditure teaches price; risk/price choose future observations. Both learned auxiliary quantities terminate on actual correction/cost observations. The forward–credit loop is finite rather than an infinite sequence of learned trainers. This closes the allocation subproblem operationally, not the larger problem of optimal future exploration or all-carrier learning.

## Evidence and integration boundary

`aws_predictable_audit_allocation_20261011_v1` and log-domain `...v2r1` are completed exact finite contracts through AWS slot3/run_safe. Four oracle risks/costs, budget1.2: second moment16.064853 versus uniform19.127500, a16.0117% reduction; conditional gradient mean error0 and risk-gradient mean error≤2.78e−17. Version2 additionally handles risks/costs spanning1e−300 to1e300. Units are supplied synthetic incremental costs; controller/query/feature cost is not included in that budget. This is not a native variance, quality, hardware or scaling win.

`aws_selected_route_audit_budget_20261011_v1` also completes: local/global propensity difference1.11e−16; expected budget drift1.11e−16 at the oracle price. Underpricing/overpricing gives expected work1.9785/0.7282 around budget1.2, the corrective direction. This tests issued-coordinate algebra, not online learned-risk convergence.

Next: integrate with *completed* native audit histories under the existing compact-credit owner. Fit risk on causally earlier packets and assess later sensitivity-weighted residual/work; do not use future residuals to choose the audit that reveals them. Compare learned risk against constant risk at the same measured expenditure, including controller/feature/fitting costs. No new dense references or TEST scoring. Preserve active actor/critic sources and the B4 confirmation pipeline.
