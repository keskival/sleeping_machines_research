# Event-local credit and versioned update banks

13 October 2026, B10/B1/R1. Integrate native temporal eligibility with the other agent's LocalCreditBanks. Removing reverse-time history graphs and removing synchronized optimizer application are distinct steps. Our own/projected market kernels provide the former within each fixed-parameter window. Versioned leaf banks provide reordered application of already-final gradients. Neither step eliminates current-window loss/readout differentiation, shared-parameter aggregation, local arrival joins or the existing global clipping rule.

For a declared differentiable risk R and an L-smooth neighborhood, an actual bank update d supported on parameter block b satisfies

    R(theta-eta*d) <= R(theta)-eta*<grad R(theta),d>
                     +(L*eta^2/2)*||d||^2.

This inequality does not require a periodic global clock. It requires a well-defined parameter snapshot, the actual delivered update, and bounded local curvature. For d=P_b*(g_b+e_b) with symmetric positive P_b, define ||x||_P^2=x^T*P_b*x. If ||e_b||_P <= rho*||g_b||_P, rho<1, then

    <g_b,d> >= (1-rho)*||g_b||_P^2.

The improvement condition compares this alignment to the quadratic update term. It is not a guarantee from a small raw-gradient MSE: error must be assessed in the update-relevant metric. Actual Adam/clipping changes d; use its measured proposal or justify a fixed preconditioner approximation before applying the metric formula. Training on delayed actual-update utility addresses that difference directly.

An old packet can be wrong for three separate reasons: credit approximation, parameter movement and a changed future-risk distribution. Under explicitly assumed parameter-gradient Lipschitz bound H and measured/assumed distribution-gradient variation V, its error can be bounded by

    ||e_current|| <= ||e_issued|| + H*||theta_current-theta_issued|| + V.

Version/event lineage and local state displacement are therefore useful acceptance information; wall-clock age alone is not a sufficient trust criterion. The current banks reject stale packets strictly. The inequality motivates a future bounded-staleness policy, not automatic permission to reuse an old gradient.

Window-synchronous parity is a useful starting contract. With final gradients issued at a common version, matched global norm clipping, identical fresh moments and AdamW weight_decay=0, updates to disjoint parameter leaves commute: each leaf has its own moments and step counter. Reversing bank delivery therefore preserves the endpoint ordinary-Adam update. Predictions still wait until that declared parameter version is complete. Canonical shared leaves need one owning bank and aggregation of all contributions before a final packet; duplicated ownership breaks the contract.

Breaking the window barrier changes the learning rule. Each trace's derivative target must name a parameter version; a current fixed-version sensitivity cannot silently include writes made with changing weights. A bank update needs explicit trace invalidation/reinitialization for its affected producer dependencies, or a validated version-aware transport rule. Global norm clipping cannot be relabelled local clipping: changing it changes proposals and future utility. The next integration should measure local credit packets, dependency invalidations, accumulation work and residual global boundaries together on an unchanged prediction task.

Hardware accounting follows the same decomposition: actor and trace arithmetic, all-key discovery, native value proposals, gradient packet bytes, moments/parameters, local readiness joins and shared reductions. Count logical accesses separately from storage placement and chip energy. This formalism targets economical asynchronous execution without treating either a familiar optimizer or a clockless circuit as proof of useful learning.
