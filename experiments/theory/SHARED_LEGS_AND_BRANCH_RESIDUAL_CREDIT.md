# Shared legs and branch residuals in route credit

12 October 2026, user direction: route similarity should promote transfer through shared computation within the same data item, then diverge when data requires different handling. Prefix sharing is one case; shared subpaths and recurrent reuse also count. This extends PREDICTIVE_LAW_GEOMETRY_FOR_ROUTE_CREDIT.md. Conditional projections and Jacobian chain rules are known tools; no new field-level theorem is claimed.

## The object is a context-labelled computation leg

Represent a realized or candidate route as an ordered collection of legs on a causal computation graph. A leg record includes operator/address, incoming content/state, elapsed time, actor version, parent identity and alternative continuation. Two legs using the same node under different incoming states are not automatically equivalent. A shared prefix supplies a particularly clear match, because both routes have the same upstream computation. Shared later legs require checking their incoming context, not just their addresses.

Transfer has two ingredients: actual shared computation, and similar predictive effects of distinct computations. Retain both in the representation. A whole-route embedding alone erases where similarities occur. Route-local embeddings without context incorrectly equate different operations. Prefix trees are economical when routes form trees; general sparse/recurrent graphs require shared fragment identities plus context and occurrence records.

## Sharing the producer does not force identical credit

For shared producer P and route-specific continuation loss L_r, the exact contribution is

    grad_theta L_r = J_P^T c_r,  c_r=grad_P L_r.

J_P is shared, but c_r can differ because branches have different future consequences. Therefore share the representation and its learned credit machinery, not an unconditional numeric credit target. Aggregating branch credit uses the relevant conditional branch weights and includes interactions; merely assigning the same error to every shared leg is incorrect.

A hierarchical critic can use shared features F0, then branch information F1, and so on. For a square-integrable credit/actual-update-utility target T, define

    m0=E[T|F0],  delta_j=E[T|Fj]-E[T|F(j-1)],

where feature sigma-algebras are nested. The best prediction telescopes as m0+sum(delta_j), and the residual increments are orthogonal in expectation. This provides a principled shared trunk plus increasingly specific residuals. Weakly informative branches should receive small residuals; changed conditional handling can receive large ones. The fitted neural implementation must learn these expectations; the identity does not guarantee that it will.

For overlapping non-prefix fragments, arbitrary summed residuals need not be orthogonal and can double count. Use an ordered conditional refinement or an explicitly accounted graph message model. Keep join identities, not separate copies of the same upstream teacher. Missing alternatives are predicted through the learned function with uncertainty; observational data alone does not identify every unrealized intervention.

## Effect geometry must retain interactions

If admitted infinitesimal route response d_r=sum_j d_rj, then predictive score s_r=sum_j s_rj and its kernel against route e is

    K(r,e)=sum_(j,k) E_Q[s_rj*s_ek].

Shared legs provide repeated components that can reuse statistics and learning. Cross-leg terms still matter: effects can reinforce or cancel. An overlap count or prefix length is a useful structural feature, not the complete credit metric. Finite nonlinear writes need actual counterfactual responses or a measured approximation; this linear identity cannot be silently applied to arbitrary route edits.

## Native implementation direction

Use context-labelled fragment tokens for the critic, with addressed memory shared by equivalent operator/parent contexts and branch-specific residual state. Receipt routing follows actual graph joins and parent references. Forward activity supplies incoming state and factual/alternative response information; delayed forward loss teaches credit after the relevant outcome arrives. The critic's races decide which shared statistics and branch residuals to read/update. Avoid scanning every available fragment: selected admission, uncertainty-triggered exploration and exact audits must earn their work.

The present B10 market actor has dense temporal layers and mixture clocks; its new packet represents whole forecast responses, not an internal route-fragment learner. Its critic has native hard races, but its external activation history is still addressed by observed mark. Do not claim the shared-leg construction is already demonstrated there. The immediate implementation priority is the measured replacement budget; a future internal hard-route integration must preserve fragment ancestry and explicitly compare shared versus isolated branch learning under the same work/data protocol.
