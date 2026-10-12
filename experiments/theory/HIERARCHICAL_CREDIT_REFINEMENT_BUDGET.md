# Hierarchical credit refinement under a work budget

12 October 2026. B10/shared learning enabler; analytical consequence of SHARED_LEGS_AND_BRANCH_RESIDUAL_CREDIT.md and measured native critic costs. This uses finite martingale projection identities, not a claim that conditional expectation is new. No deployed hierarchical-credit quality or speed result is attached.

Let T be a square-integrable actual-update utility target in a fixed admitted control basis. Context-labelled shared legs and increasingly specific branch observations define nested information F0 subset ... subset FK. Set m_h=E[T|Fh], Delta_h=m_h-m_(h-1). These are refinements of the same global target, not products of independently estimated local gradient maps.

For a fixed refinement depth k,

    E||T-m_k||² = E||T-m_K||² + sum_(h=k+1..K) E||Delta_h||².

Conditional projections are orthogonal, so the remaining approximation error adds as refinement energy. This identity does not multiply estimator errors down the route. Actual neural estimators need to learn consistent conditional predictions; arbitrary residual heads do not automatically have the orthogonality property. Producer Jacobians, admitted directions and numerical stability remain separate costs and sources of error.

The same identity permits adaptive sparse admission. Let tau<=K be a stopping time determined by information already acquired, and expose no later branch observation before choosing to stop. For finite K,

    E||T-m_tau||² = E||T-m_K||²
                     + E sum_(h=1..K) 1{tau<h} ||Delta_h||².

Proof: T-m_tau=(T-m_K)+sum_h 1{tau<h}Delta_h. The terminal residual is orthogonal to FK. For i<j, 1{tau<i}Delta_i and 1{tau<j} are measurable before Delta_j, whose conditional mean is zero. Hence all cross terms vanish. The stopping decision cannot peek at the withheld refinement and then pretend the same accounting holds.

Thus a shared-leg predictor can stop after a useful coarse credit prediction and pay for branch refinements selectively. If unresolved branch refinement energy stays small as available model depth/capacity grows, useful credit work can grow more slowly than exhaustive transport. This is an explicit condition to measure, not an assumption that every domain has compressible credit.

With per-refinement work C_h, minimize prediction error plus lambda*expected admitted work. The value of acquiring one more refinement includes both its conditional expected energy and its option to reach later informative branches. A myopic energy/cost gate is not globally optimal when branches reveal complementary information. Learn admission from delayed actual utility receipts and preserve exploration/audits with known positive support. Temporal races can implement sparse competing refinement requests, but delay, queue work and stale state must also be charged.

Engineering decision from the measured B10 profile: exact boundary transport/proposals0.780s, response/inference2.944s, future teaching4.015s per512forecast cohort. A32-observation credit sweep for every actor update is too expensive here. First prototype budgeted context/fragment receipt admission and shared prediction reuse, then measure response/inference and rare teaching against the available transport budget. Quality must remain above the common-acquisition masked/frozen controls before a scaling claim. Do not reduce audits alone or describe the current full-BP-plus-critic pipeline as efficient.
