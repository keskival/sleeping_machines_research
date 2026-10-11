# Whole learner-state credit for an update function

R1 development, 11 October 2026. The local one-step update-function test failed its primary utility gate. Sixteen-feedback aggregation gives a positive but inconclusive four-episode mean; an unchanged-rule fresh extension is running. This note prepares a distinct objective test, rather than another aggregation grid.

## Failure and proposed change

A current update changes numerical parameters, momentum and bias-correction state. It also changes the next error gradient, forward-conditioned geometry and the context which selects later updates. Local teaching detaches those incoming dependencies. Averaging local derivatives does not restore them.

For complete learner state S=(z,m,r) and fixed exogenous native forward packets x_t, define the actual update F_ψ(S_t,x_t,y_t). Within a bounded window keep ψ fixed and differentiate all paths: current feedback, step-size/damping/momentum context, block curvature, solve, norm cap and bias-correction product. Retain the source native temporal/addressed inference, proper tokenization and forward-derived block geometry. Native structural parameters stay frozen in this bridge; it adds no optional routes. The departure is restoring lexical/optimizer-state dependence inside the window, not replacing the substrate with a dense conventional model.

Let P_t=dS_t/dψ, J_t=∂F_ψ/∂S_t, B_t=∂F_ψ/∂ψ. Starting from detached S_0,

    P_0=0,  P_{t+1}=J_t P_t+B_t,
    d[H^-1 Σ_t ℓ(S_t,x_t,y_t)]/dψ = H^-1 Σ_t (∂ℓ_t/∂S_t) P_t.

Reverse differentiation through the same bounded rollout gives the identical derivative. Finite differences of the actual numerical rollout are an additional check: comparing two stop-gradient implementations alone can hide omitted dependencies. Each target must be scored before its own update. At the commit, consume the window gradient, update ψ, then detach/reset derivative state for the next window. Unused final updates contribute no direct target loss; flush a short window with its actual target count.

## Work and sparse extension

Explicit forward eligibility has dimensions |S|×|ψ|. It is not automatically economical: the dense J_t P_t action scales with both state and credit capacity. A bounded reverse window stores the realized graph instead; its memory and derivative wall time are measured. The proposed implementation is a diagnostic upper-fidelity teacher, not a hardware-efficiency result.

Connected approximate eligibility or cotangent functions can later predict the transported influence of numerical parameters and optimizer state. Native forward JVP probes can teach those projections, and positive-support residual audits can repair first-order raw-gradient means under their declared sampling law. Sparse support, candidate discovery, losing alternatives, precise parameter versions and all teacher/audit costs must be charged. A compressed predictor is promoted only after downstream quality per total work improves.

## Contracts before a fit

A four-target nonlinear fixture checks the forward-eligibility recurrence against the complete reverse gradient and parameter finite differences. It includes curvature and feature sensitivity, momentum/bias correction and capped updates. Normalization must clamp variance before square root to keep zero-momentum derivatives finite. This establishes the implemented derivative, not useful task learning or depth scaling. A real-vocabulary streaming smoke is required before a registered fit; changes to objective horizon and feedback aggregation are reported separately.
