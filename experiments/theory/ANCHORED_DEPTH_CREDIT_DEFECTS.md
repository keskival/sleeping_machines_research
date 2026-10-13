# Anchored connected credit: depth error as an auditable defect sum

R1, 13 October 2026. Formal extension of
[forward-probe depth credit](DEPTH_CREDIT_FROM_FORWARD_PROBES.md), tied to the
running D8/D16 language program. Conditional identities and design consequences;
not a claim of a new mathematical theorem, completed compact-credit integration
or a Transformer quality win.

## Distinguish the actor's geometry from the credit model's errors

At a fixed learner version and declared differentiable factual path, let
h_d=f_d(h_(d-1)), J_d=D f_d and lambda_(d-1)=J_d^T lambda_d. The state at a cut
must include all declared content, elapsed-time and persistent-state coordinates
that cross it. Current language training truncates temporal gradients at16tokens;
this derivation does not silently extend that horizon.

A chain of learned maps introduces errors into products of approximate response
operators. Instead let one connected native credit system query each cut:

    lambda_hat_d = C_psi(cut_d, observed_state, future_loss_query, learner_version).

Queries share native temporal functions, route-address structure and memory, but
each output is a direct cut prediction. Define the local defect and terminal
error

    r_d = lambda_hat_(d-1) - J_d^T lambda_hat_d,
    e_D = lambda_hat_D - lambda_D,
    P_0 = I,  P_k = J_1^T ... J_k^T.

Then the exact telescoping identity is

    e_0 = P_D e_D + sum_(d=1..D) P_(d-1) r_d.

This uses the actor's actual response maps, not products of learned backward
maps. A terminal anchor makes e_D=0. It removes the free boundary solution in
which every predicted credit is zero and every consistency defect is zero.
For multiple supervised losses, their anchors and source contributions must be
added once; the same elapsed-time cotangent is not a new independent error at
every layer.

The unconditional norm bound is

    ||e_0|| <= ||P_D|| ||e_D|| + sum_d ||P_(d-1)|| ||r_d||.

If ||P_k|| <= C rho^k with rho<1 and ||r_d||<=epsilon, the defect contribution
is at most C epsilon/(1-rho). If ||P_k||<=C without decay, it is at most CD epsilon.
No depth-independent bound follows without a conditioning or residual-decay
premise. Normalization alone does not supply one: its Jacobian can amplify
directions with small variance. Hard races and nonlinear persistent transport
must be assessed under the actual declared differentiation/credit rule.
This is an absolute-error bound. A contractive actor may also make the true
upstream signal decay like rho^D, leaving a small absolute error larger than
the signal. Driving transport toward zero is not a solution to useful deep
learning. Measure relative error, direction agreement and actual update utility
where the signal is nonzero; norm-preserving transport can retain signal while
requiring summable or explicitly audited credit defects.

Conditional zero-mean, pairwise uncorrelated transported defects give additive
squared error E||sum v_d||^2=sum E||v_d||^2, where v_d=P_(d-1)r_d. Correlated
systematic defects can add coherently and produce squared-depth growth. Training
global cut-to-output probes, rather than only isolated local fits, is how the
learning objective can expose that shared bias. Uncorrelated errors are an
assumption to measure, not a benefit guaranteed by parameter sharing.

## Exact audits and the cost that does not disappear

With an exact terminal anchor and issued predictions fixed before sampling,
choose one depth K with positive causal probability p_K. The estimator

    lambda_corrected_0 = lambda_hat_0 - v_K/p_K

has conditional mean lambda_0. Its correction variance is

    sum_d ||v_d||^2/p_d - ||sum_d v_d||^2.

At fixed one-audit count the oracle optimum is p_d proportional to ||v_d||.
Our running allocator measures a local route-logit proxy, not these transported
cotangent defects; its learned probabilities cannot be called this optimum.
Predicted uncertainty can propose audits with positive support and recorded
weights. Learning from an audit before issuing its same-sample correction
changes the conditional argument; retain issued predictions/versions.

An exact v_K needs local J_K^T and prefix transport P_(K-1). That can cost K
reverse stages, plus discovery and retained forward state. Sparse depth
sampling does not make this transport free or abolish its sequential dependency.
For a scalar source-direction query w, use

    <w,v_K> = <J_(K-1) ... J_1 w, r_K>.

This permits a forward response probe instead of materializing P, but still
pays prefix response work and has the projection variance/identifiability
limits in the earlier note. A trained global transport prediction plus a
supported exact residual audit can reduce average work only when its residual
variance is small enough. Correcting one set of cut cotangents also does not
automatically correct every actor parameter gradient or optimizer update.

For independent Bernoulli audits A_d with rates a_d, the sum correction
sum A_d v_d/a_d has variance sum(1/a_d-1)||v_d||^2, under independent admissions
conditional on issued quantities. With costs c_d and expected budget B,
minimization subject to sum c_d a_d<=B gives

    a_d = min(1, ||v_d||/sqrt(mu*c_d)),

with mu chosen for the budget. This is the cost-constrained rate rule, distinct
from fixed-count categorical sampling. Real prefix audits share caches/work;
their nonadditive costs need a different accounting model. Random routes may
also make transport stochastic; those draws and propensities belong in the
declared estimator rather than being treated as deterministic cotangents.

## Hard-route losses and sparse asynchronous execution

The telescoping identity applies to the differentiable path that is actually
declared. Expected hard-route choice gradients require their supported score or
counterfactual terms separately. Existing first-time surrogates are declared
credit constructions, not classical derivatives of the hard winner. Neither
local consistency nor a terminal cotangent alone certifies credit to an
unrealized route, delayed write or omitted state coordinate.

A future-loss query carries loss identity, causal cut, state/parameter version
and the projected credit being requested. Native critic reads/writes can process
these as sparse temporal events. The actor retains bounded local activations
until their feedback matures; no global backward wave is needed for a prediction
to arrive. The audit may still launch a dependent transport chain. Bound receipt
lifetimes, stale-version policy, audit frequency, candidate key scoring, message
bytes and optimizer visits explicitly. Current global clipping and dense AdamW
remain synchronous implementation work; this note does not rename them clockless.

Finite future utility is a separate target. An action-value contrast after an
actual optimizer step can outperform following a short-horizon gradient, but
its conditional credit is not the lambda of this fixed-path identity. It can
share the same native query representation while keeping its anchor, units,
horizon and teacher separate. This avoids training useful update preferences
against BPTT as the only acceptable answer.

## Immediate consequence for the owned language program

Finish the unchanged D8/8M,D16/8M,D8/32M acquisitions and content-path diagnostics.
Use their actual quality, per-layer gradient geometry, routing and complete work
to identify the limiting mechanism. Nonzero layer gradients alone are not a
depth win; dependence under a frozen content cut is not trained-depth superiority.

The next compact-credit integration should use a shared native cut-query model
with a real terminal anchor, local defects and occasional composed-response
audits. Its decisive measurements are upstream transported residual error and
work to language quality, including the initial teacher investment. If upstream
conditioning is poor, change the temporal actor's transport with a retained-core
contract; adding critic capacity alone cannot make an ill-conditioned actor
well-conditioned. No additional fit or small standalone campaign is admitted
by this note.
