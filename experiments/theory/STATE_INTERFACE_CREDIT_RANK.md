# Learn credit across state interfaces, not unrestricted inverse maps

13October2026. B1/R1 native learning enabler. Builds on closed-loop exploration/credit, shared producer legs, projected clock credit and versioned banks. The prospective objective already exists; this note specifies the representation and cost of its transport.

## State-mediated delayed credit

For a fixed parameter version and a declared differentiable factual route, suppose

    s_t = F_t(s_(t-1),x_t),   y_t = H_t(s_t,x_t).

Let A_t=partial F_t/partial s_(t-1), B_t=partial F_t/partial x_t, and C_t=partial H_t/partial s_t. If every influence from x_i to y_j, j>i, passes through this state interface, then

    J_(y_j,x_i) = C_j A_j ... A_(i+1) B_i.

Its rank is at most the narrowest interface dimension along the path, independently of the elapsed history length. The adjoint credit is

    g_(x_i) = B_i^T A_(i+1)^T ... A_j^T C_j^T c_j.

This explains the forward/backward similarity: forward sensitivity traverses the interface; credit traverses its adjoint. It is not an inverse of inference. Preserve exact inexpensive local B/C and clock propagation when available; learn context-dependent cross-interface cotangents or residuals instead of an unrestricted activation-to-activation dense matrix. Factor storage/work must be charged and can exceed one exact reverse pass on a short episode; horizon reuse is the proposed benefit, not an automatic saving.

Parallel memory paths widen the cut. Persistent message values, timing variables and every other state variable transmitting the effect belong to the interface. Hard route changes require actual alternative-route consequences; a fixed factual Jacobian does not label unrealized interventions. Shared producer factors may be reused, but each downstream branch supplies its own cotangent.

## Future-facing value supplies the cotangent

Let z_t include numerical state, weights, optimizer moments, credit memories and a causal belief about the changing stream. For an update/exploration policy pi, finite-horizon value is

    V_t(z) = E[sum_(k=0)^(H-1) gamma^k ell_(t+k) | z_t=z, pi].

A factual differentiable learner transition z_next=G_t(z,observation,action) transports

    lambda_t = partial_z ell_t + gamma J_(G_t,z)^T lambda_(t+1),

with additional action-policy derivatives when actions depend on z. A detached/randomized intervention estimates its declared alternative value; it does not silently supply those missing policy derivatives. Actual-update advantage compares the paired future risk under the update with a no-update/other-action continuation and charges resources separately.

The interface rank bound must now apply to the complete learner-state cut, not merely neuronal memory: changing parameters/moments may carry delayed effects too. Sparse local ownership and admitted cross-bank packets determine its useful width. This is precisely why dense Adam/global clipping boundaries and packet dependency lineage are part of the research accounting.

Finite horizon closes at the observed terminal loss or an explicitly learned terminal value. A discounted Bellman value operator contracts under its usual bounded-value/Markov-state assumptions; derivative transport additionally requires gamma times the relevant Jacobian norm<1 for a simple gradient contraction. Do not infer the latter from the former. Outcome labels, forward probes, local eligibility and paid residual audits supply finite supervision; the future objective does not introduce a mandatory hierarchy of trainers.

## Train rank where future updates need it

For an issued conditional transport J^T and cotangent covariance S, let P be a positive update-relevant metric. Approximation B has quadratic risk

    R(B) = E_c{[(B-J^T)c]^T P [(B-J^T)c]}
         = ||P^(1/2)(B-J^T)S^(1/2)||_F^2.

For fixed positive-definite P/S, the best rank-r approximation in this metric is the truncated SVD of M=P^(1/2)J^T S^(1/2), mapped back through the inverse square roots. Its minimum risk is the sum of squared omitted singular values. On singular support use the appropriate restricted/pseudoinverse construction. This ranks credit directions by their future-useful error, not by activation variance alone. It is a conditional local result, not a global actor convergence theorem; actual Adam/clipping proposals require actual-outcome validation.

Forward probe v with covariance P yields supervision q=c^T Jv and loss(v^T Bc-q)^2. Its expectation is the same risk when v/c are independently sampled conditional on the issued context. Existing paired responses may correlate: retain their joint sampling law or correct it; do not claim the covariance identity for arbitrary reused pairs. A local linear transport update uses residual times v c^T, with no third reverse teacher. Low-rank/context-conditioned factors need their own local factor updates and paid coverage audits. Historical activation/cotangent distributions, effective counts and versions can inform the metric and route similarity while actual responses teach which reuse is useful.

## Native gate

check_state_interface_credit.py audits a two-event TemporalMemoryLayer, width8/two complex modes. The only delayed path has four real state coordinates. Separate current-event write/read derivatives and exact decay/rotation reproduce the complete delayed8x8 Jacobian and its cotangent; rank<=4. Exact full reverse/JVP is an independent audit and is charged. This is a native temporal-layer mathematical/implementation contract, not full hard-routed actor replacement, learned prospective cotangent quality or an ASIC efficiency win. It provides the structural target for the next integrated estimator: learn/reuse the smallest sufficient cross-leg credit representation and prove full cost to future quality.
