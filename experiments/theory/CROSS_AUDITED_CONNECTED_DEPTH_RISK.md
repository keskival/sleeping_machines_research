# Train connected depth credit on the cross-layer error

R1, 13 October 2026. Formal and executable objective, not an integrated critic
fit. Extends [anchored depth defects](ANCHORED_DEPTH_CREDIT_DEFECTS.md) while the
actual D8/8M,D16/8M,D8/32M acquisition program continues unchanged.

## Why independently fitting local errors is insufficient

With a fixed terminal anchor, upstream credit error is E=sum_d v_d, where
v_d is a local consistency defect transported through the actor to the declared
source cut. Its squared risk is

    ||E||^2 = sum_d ||v_d||^2 + sum_(d!=j) <v_d,v_j>.

Two equal defects reinforce and give four times one-defect risk; opposite
defects cancel at this endpoint even when each local defect is large. Local
squared-error objectives omit both effects. Minimizing their sum can still be
useful, but it is not the same connected objective. Cancellation at the input
cut also need not cancel errors in other actor parameter blocks.

Condition on the same issued actor, critic, realized path and loss query. Select
K and L independently with positive probabilities p and q, issued without
using their unseen audit outcomes. They may coincide. Set

    Z = v_K/p_K,  Z' = v_L/q_L,  loss_pair = <Z,Z'>.

Then E_(K,L)[loss_pair]=||sum v_d||^2. If actor response maps and propensities are
stopped while differentiating the retained critic graph,

    E_(K,L)[gradient_psi loss_pair] = gradient_psi ||sum v_d||^2.

Both audited contributions receive parameter credit through the same connected
critic. The identity follows by expanding the finite double sum; it is not
a new theorem or a guarantee of optimization success. A single squared audit
instead estimates sum ||v_d||^2/p_d, a probability-dependent second moment.
Different proposals for the two draws are permitted with their own weights.
Sampling without replacement needs joint inclusion probabilities and a diagonal
term; treating such draws as independent would be wrong.

Samples of loss_pair can be negative although its expectation is nonnegative.
Clamping sample loss at zero changes the objective. Critic gradient clipping,
adaptive optimizer moments and finite audit noise also separate the raw-gradient
identity from actual optimization. Probe variance and learning stability must
be measured before a large fit. The optional implementation only computes the
scalar objective; it does not conceal these obligations.

## The same world and the same version matter

For stochastic paths, the conditional double-sum identity requires both audits
to refer to the same realized path/query. Shared issued randomness followed by
paired replay can provide that coupling without using future information to
select sites. Averaging shared-world pairs estimates E_world||E_world||^2.
Independent future worlds instead can target ||mean_world E_world||^2 (the squared
mean error) under appropriate conditional independence; these are different
risks. Record which one is trained, rather than mixing their labels.

Keep actor and critic versions, terminal anchor and observed-state availability
fixed until the paired receipt matures. Updating the critic after the first
audit and evaluating the second with new weights breaks the issued-graph
objective. Future loss is revealed after prediction. The numerical graph for
the issued critic activations is retained separately from detached counterfactual
world copies; live memory is not overwritten by either replay.

## Use the actual learning geometry, not only the root cotangent

Let each cut error e_k map through a fixed linear response W_k to the declared
gradient, optimizer-step or forecast projection. With a terminal anchor,

    E_update = sum_k W_k e_k = sum_j V_j,
    V_j = sum_(k<j) W_k (J_(k+1)^T ... J_(j-1)^T) r_j.

An empty product is identity. W_k can inject local parameter-credit responses
into disjoint blocks, followed by a declared actual-optimizer Jacobian and a
forecast trust-metric factor. The same cross-audit objective applies to V_j.
This handles error covariance across parameter blocks and the optimizer's
couplings; using v_j at the root alone does not do so. Building dense W matrices
would negate sparse economy. Use validated local response contractions and
charge the shared clipping reduction, Adam state, all transport and projections.
This is first-order geometry at an issued update, not a full nonlinear future
utility objective or permission to train solely against BPTT labels.

For state-sized economy, independently issued directional projections w can
replace full V vectors: train (<w,V_K>/p_K)(<w,V_L>/q_L). Covariance Sigma of w
gives risk E_update^T Sigma E_update. Positive-definite covariance identifies
the declared full cut; low-rank probes identify only their supported subspace.
Directions chosen using the current audit outcomes change the identity.
Forecaster/update-aligned probing is valuable only if its missing directions,
acquisition work and variance are accounted for.

## Native architecture and evidence boundary

The critic should query layer/route/clock targets through shared native temporal
functions and private addressed memory, not disconnected tiny gradient heads.
The paired objective gives a concrete reason to train those targets jointly:
their transported errors interact in the same future-loss/update geometry.
It does not prove that a dense global critic or a particular native size is
optimal, and it does not supply an invariant route embedding by itself.

Two site audits can require two prefix transports; cache reuse and parallel
local messages are implementation choices to price, not free computation.
Sparse asynchronous receipts avoid a mandatory prediction barrier, while audit
chains and the current actor optimizer may still synchronize. Retained state,
key scoring, critic acquisition and optimizer visits belong in complete work.

`sleeping_machines/connected_depth_risk.py` implements the scalar objective and
an exact depth8/16 finite-sum/gradient contract, including cancellation and
negative-sample witnesses. The contract is attached to the already-admitted
completed-fit evaluation, not another standalone training campaign. Native
cut-response extraction and integrated critic training are not implemented by
this helper. The running language fits remain the immediate quality evidence;
the next integration is chosen from their measured limiting mechanism.
