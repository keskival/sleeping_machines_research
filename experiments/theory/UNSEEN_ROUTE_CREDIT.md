# Credit for inactive and unqueried routes

Curie,10 October2026; B1/R1 shared credit enabler. The critic can represent credit beyond the realized route and counterfactual pool. The current real-market critic emits persistent-state cotangents, not route-indexed intervention values; this note defines the next route interface rather than claiming it already exists.

## What can generalize

Let U(H) be the route universe at a causal prefix, A the realized active routes, P the paid counterfactual pool, and Q the routes queried from a critic. These sets differ. A shared connected function C_psi(c,H,d_i,version,horizon) can be queried for any describable route i in U, including i outside A and P. The descriptor d_i may include its key/address, local timing law, input/message, receiver state and a compact description of its function. Exactness depends on information sufficiency: merely knowing an address does not reveal what an arbitrary unseen receiver computes. Reading descriptions/state is work.

The learned target may be finite-horizon intervention advantage V_i=E[future loss under do(route i)-future loss under reference|H], or an addressed cotangent/parameter contribution g_i under a declared objective. Advantages and gradients are different targets. Force a route and replay its downstream state/learning policy to measure total intervention utility; a fixed-path local decoder is not automatically exact across intervened online updates.

Sharing across descriptors lets observed routes teach unseen ones. For an admitted symmetry T, useful equivariance is C(T H,T d_i)=T_* C(H,d_i), with the appropriate gradient/value transformation and versioned state. This is an inductive bias to learn or encode and test; inactive routes need not be equivalent, and shifts can change the useful symmetry. A route that was never observed can receive a learned prediction, but that prediction is neither an exact label nor an identifiable consequence without assumptions tying its behavior to observed evidence.

No query means no individually emitted credit: a critic may represent a function over U without evaluating it everywhere. Hierarchical/shared low-rank fields can propose a promising region or aggregate class credit; their discovery, decoding and update costs are charged. A dense list of scores for every dormant route is dense work even if only one value is delivered.

## Economical supported exploration without summing every prediction

For the additive target G=sum_i g_i, select I from a known proposal q_i>0 over every admissible route. Compute only its prediction h_I, then independently audit its exact contribution with probability a_I>0. Predictions, proposal and audit propensity may use already-revealed forward feedback, but are fixed before the relevant draw and before the exact label is exposed. Return

    G_hat = [h_I + B_I/a_I * (g_I-h_I)] / q_I.

Conditional on I, averaging the audit gives g_I/q_I; averaging the proposal gives sum_i g_i. Exhaustive finite enumeration can check the estimator without a fit. The address may be delivered lazily as a sparse contribution; shared parameters require their actual contribution map. This formula removes the need to evaluate sum_i h_i, at the price of route-sampling variance. A better predictor reduces residual-audit variance, not all variance. Shared network inference, route proposal/normalization, descriptor reads, exact alternatives, optimizer work and memory remain in the ledger.

For independent audit draws and fixed g,h, the trace variance is

    sum_i ||g_i||^2/q_i - ||sum_i g_i||^2
      + sum_i (1/a_i-1)||g_i-h_i||^2/q_i.

The first term is route-sampling variance; the second is residual-audit variance. Exact predictions do not remove the first. This is a raw-gradient/contribution identity before clipping or Adam, not unbiased nonlinear updates or full-history credit.

If q_i=0, the missing route cannot be corrected by this estimator. Restricting to the chosen counterfactual pool is a different target unless an exact aggregate for the omitted routes or a separately justified approximation is available. A causal mixture q=(1-epsilon)q_learned+epsilon q_explore, with q_explore full support, preserves discoverability. Uniform-index or normalized hierarchy sampling avoids a mandatory dense score scan; it does not make inspecting arbitrary receiver functions free. In very large spaces, positive support does not guarantee that a finite run visits every important route.

## Integration and decision

`credit/sampled_route_credit.py` implements the two-stage correction for one selected route; its contract enumerates routes inside/outside a two-route pool, audit outcomes and variance. This is arithmetic/support validation, not a trained route critic result. The next native interface uses the connected structured-response critic's arbitrary intervention queries together with a lazy route descriptor and explicit universe/exploration sampler, then measures held-out route utility against restricted-pool and zero/frozen critics. Preserve hard temporal races, unrealized-route replay, addressed values, state/version contracts and complete work. No new external model fit or sealed TEST is involved.


## Allocate audit work where learned credit is uncertain

For a fixed supported proposal q, write e_i=g_i-h_i and let c_i>0 be the measured incremental work of an exact audit. Under expected audit budget C, choose probabilities a_i to minimize the residual variance term subject to sum_i q_i*a_i*c_i<=C. For fixed realized residuals, the constrained optimum is

    a_i = clip(||e_i|| / [q_i*sqrt(lambda*c_i)], a_min, 1).

The multiplier lambda enforces the budget (when binding). Derivation: the varying objective is sum_i ||e_i||^2/(q_i*a_i); its derivative plus the cost multiplier is -||e_i||^2/(q_i*a_i^2)+lambda*q_i*c_i. Clipping gives the box-constrained optimum. A positive floor a_min preserves correction support, including apparently easy routes. The budget must exceed the floor's cost; zero residuals need no extra allocation beyond that floor. This optimum assumes the actual residuals are known; implementation must estimate conditional squared residual norms from prior independent revealed audits, before the current draw. Estimated allocation keeps the estimator unbiased when actual propensities are used, but does not inherit oracle optimality.

For fixed audit probabilities, define v_i=||g_i||^2+(1/a_i-1)||e_i||^2. The variance-minimizing proposal under normalization alone is q_i proportional to sqrt(v_i), with exploration for zero or uncertain entries. This follows from Cauchy–Schwarz: sum_i v_i/q_i >= (sum_i sqrt(v_i))^2. The joint work-constrained problem also depends on route discovery/descriptor and predictor costs, so this separate optimum is not a complete hardware policy. A hierarchy trained on previously revealed contribution/error moments can approximate the proposal without a full score scan; its sampling probability must remain known and supported.

This connects learning quality directly to economical execution: lower conditional residual error permits fewer paid exact audits at fixed residual variance. It does not eliminate route-sampling variance, local factual derivatives, or descriptor/optimizer work. The next integration should compare measured end-to-end quality/work under fixed sampling budgets, with audit allocation frozen causally before each sample and all versions retained.
