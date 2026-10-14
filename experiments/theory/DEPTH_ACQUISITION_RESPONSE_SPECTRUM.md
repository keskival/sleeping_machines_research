# Depth, acquisition and the spectrum of learning response

13 October 2026. R1 theory for selecting a deep model family by eventual quality and complete work. Supports the admitted D8/D16 and8M/32M program. This is a derivation in a specified local learning model; it is not a fitted scaling law for our native family, nor a new benchmark admission.

## 1. What smaller experiments should estimate

A short-run loss conflates representational error, learning timescale and update noise. A deeper member can reduce representational error while adding useful directions that need more acquisition. Selecting only the earliest loss favors fast directions and can discard the member with the better later curve.

Separate these quantities rather than assigning a universal power-law exponent to an architecture:

    attainable approximation error;
    task-relevant response strengths and their learning timescales;
    noise/error from finite data and credit estimation;
    cost of inference, discovery, teaching, optimizer and memory;
    evolution of these quantities as representations and distributions change.

Parameter count and layer count alone do not determine them. More temporal stages provide computational composition, but their useful directions must remain observable and trainable.

## 2. Exact conditional quadratic calculation

Fix a representation, parameter/update geometry and stationary local objective

    F(w) = F_floor + (1/2) sum_i lambda_i (w_i-w_i^*)^2,
    lambda_i > 0.

Use SGD with step eta, B fresh targets per update and gradient noise xi_(i,k) having conditional zero mean, variance sigma_i^2/B and no correlation with the current error. In these diagonal coordinates,

    e_(i,k+1) = q_i e_(i,k) - eta xi_(i,k),
    q_i = 1-eta lambda_i,   |q_i| < 1.

The exact expected loss after k updates is

    E F_k = F_floor + (1/2) sum_i lambda_i [
        q_i^(2k) e_(i,0)^2
        + eta^2 sigma_i^2/B * (1-q_i^(2k))/(1-q_i^2)].

Proof: expand the error square, use the conditional zero cross term, and sum the geometric recurrence. Cross-coordinate noise does not alter this diagonal quadratic's expected loss, though it matters under other metrics. The stationary excess in direction i is eta sigma_i^2/[2B(2-eta lambda_i)]. Fixed-step noise floors are part of the comparison, not approximation error.

For small eta lambda_i, the transient is approximately a_i exp(-2 eta lambda_i k), where a_i=lambda_i e_(i,0)^2/2. The response timescale is set by eta lambda_i, weighted by task error a_i. With fixed B, target count n=Bk converts the exponent to -2 eta lambda_i n/B. Changing update grouping, learning rate or noise changes this curve even at equal target count.

This calculation is exact for the declared quadratic SGD system. Native hard routes, changing representation, global clipping and Adam are not automatically covered. A local quadratic approximation can diagnose a frozen region only when its validity is checked; arbitrary depth gradients are not response eigenvalues.

## 3. A conditional space of scaling curves

In the continuous-time zero-noise approximation, represent task-error mass by a positive measure mu over response rates r. Then

    L(n) = L_floor + integral exp(-r n) d mu(r).

These are mixtures of learning timescales. They include exponential convergence, broad shoulders, delayed improvements and apparent power laws. For a density dmu/dr approximately c r^(alpha-1) near zero, with alpha>0 and suitable integrability,

    L(n)-L_floor approximately c Gamma(alpha) n^(-alpha)

at large n. This follows by substituting v=rn in the Laplace integral. The power law describes the task-weighted slow-response density, not simply the count of small eigenvalues. If response rates are bounded away from zero, this fixed-model transient decays exponentially instead.

Within this conditional class the excess curve has alternating derivative signs: (-1)^j d^j(L-L_floor)/dn^j >= 0. Strong departures can signal representation acquisition, distribution change, optimizer changes or an invalid fixed-response approximation. They are not proof of a particular mechanism.

A nonlinear model family can create new features or routes as acquisition proceeds, so mu and the attainable floor can evolve. That learned change is central to our thesis. A spectrum at initialization alone can miss later specialization; repeated causal measurements at saved acquisition stages are more informative than an initial-kernel verdict.

## 4. Why short data cannot identify the winner by itself

Consider two conditional curves

    L_1(n) = L_base(n) + b + A,
    L_2(n) = L_base(n) + b + A exp(-n/tau),

with A>0. For n<=N,

    |L_1(n)-L_2(n)| <= A N/tau.

Choose tau>=AN/delta and the curves agree within delta throughout the observed range, but their limiting losses differ by A. This is a constructive non-identifiability witness: short-run accuracy alone cannot reveal a sufficiently slow useful mode. It does not assert that our model has such a mode.

The engineering response is to measure evidence about timescales and attainable representation, preserve registered larger acquisition, and distinguish a measured slow direction from a hoped-for one. Data volume cannot rescue a direction that is absent from the representation or unobservable through its credit interface.

## 5. A concrete delayed crossover certificate

Suppose a deep member has a justified upper bound

    L_D(n) <= b_D + v_D + A_D exp(-n/tau_D),

and a comparison class has a justified attainable lower bound L_S(n)>=b_S. If b_D+v_D<b_S, then the deep member is certified better once

    n > tau_D log[A_D/(b_S-b_D-v_D)],

with n>=0 and an already-negative right side meaning no additional delay is required. The result follows by solving the exponential inequality. Multi-rate bounds use the corresponding sum, and drift/noise must be included in their own terms.

Observed early losses do not supply these floor bounds. A flexible fit of three points is not such a certificate. The useful heuristic is nevertheless specific: seek members that reduce approximation error without making task-relevant response timescales or noise costs uneconomical. A transient deficit can be justified by evidence of that trade-off, not by depth alone.

## 6. Learned credit can change the response spectrum

Under a fixed linear positive-definite preconditioner P and quadratic Hessian H, balanced coordinates have response operator

    H_eff = P^(1/2) H P^(1/2).

Its eigenvalues determine local deterministic learning rates for a stable step. A learned credit/update policy can increase weak task-relevant response rates, but it can also alter noise, bias and the attainable learning objective. Unbiased gradient residual correction before a nonlinear optimizer does not imply unbiased updates.

A useful credit estimator should preserve signed producer demands and optimize actual future utility. Update-space covariance, clipping/moment transport and forward consequence teachers from the preceding notes specify that objective. Merely minimizing raw credit MSE or making the backward path contractive can suppress useful modes. Stability should control error amplification while preserving the modes that encode long-term dependencies.

Temporal decay/rotation gives known local transports; gates, normalization, persistent-state edges and cross-layer maps complete the response. Evaluate their task-conditioned effect rather than declaring depth stable from one transport factor. Direct cut prediction and terminal anchoring offer ways to avoid multiplying noisy learned maps while retaining deep demand.

## 7. What to measure without a brute-force family sweep

The admitted acquisition stages already supply fixed-data-order loss trajectories, per-layer gradient norms, occupancy and memory rank. Use them to distinguish numerical gradient access from actual prediction progress. Frozen long-history and stage-content interventions test whether deeper computation carries useful information; intervention loss is not a separately trained shallow comparison.

For a later compact-credit integration, small paid forward probes can estimate responses along a few task-relevant directions and their noise, without constructing a full parameter-square matrix. Native temporal descriptors and learned direction proposals can concentrate those probes. Discovery, JVP/VJP, future-loss evaluation and omitted directions remain explicit. Raw gradient norms alone cannot identify Hessian rates or attainable floors. Adaptive probe selection must retain the requisite support and causal receipts.

The decision record should show: depth and available capacity; selected activity and discovery; acquisition; quality trajectory and long-history score; any measured layer-information dependence; full teacher/optimizer/work resources. Label empirical slopes as finite differences. Do not compare an observed native CPU curve with a published Transformer GPU clock as if they shared a hardware denominator.

The next step remains actual D8/D16 acquisition and32M scaling. This response-spectrum construction tells us how to interpret those results and which conditioning/credit failure to repair. It does not substitute theoretical optimism for the selected checkpoint's public language score.

## Completed D8 acquisition: gradient coverage is not useful information

14 October2026, source: `results/diagnostics/aws_native_deep_d8_8m_quality_20261013_s6_v1.json`.
At8,388,352 acquired targets, paired8192DEV/1024history: D2 NLL6.160183,
D8 NLL6.275280, gap+0.115097. Complete training7734.249/26115.095seconds,
ratio3.376552. One seed, same initial TRAIN prior and adaptive final-stage
readout size; evaluation cadence differs256/1024. All four frequency bins
worsen. D8 is not promoted at this point; D16/8M and reservedD8/32M continue.
This first member result is not a developed-attempt family verdict.

The exact scored intervention decomposition writes L=Z-G, where Z is this
fitted readout's zero-feature NLL and G=Z-L is factual feature-conditioned gain.
D2 gives Z7.962873/G1.802690; D8 gives Z7.952018/G1.676738. Therefore

    L8-L2 = (Z8-Z2) - (G8-G2)
           = -0.010855 + 0.125952 = 0.115097.

The decoder offsets improve slightly while factual feature-conditioned gain
falls. This intervention includes current content and persistent history;
it is not mutual information, a pure memory measurement, or a proof that a
particular transport/readout/credit mechanism caused the deficit.

For population logarithmic loss with representation Z_d and normalized decoder
q_d, the exact information-theoretic identity is

    R_d = H(Y|Z_d) + E[KL(P(Y|Z_d) || q_d(Y|Z_d))].

Hence the depth gap separates representation information from decoder fitting
error. This follows by adding and subtracting log P(Y|Z_d). The finite DEV
scores do not identify either term individually. If an augmented representation
retains a parent Z and adds U, its optimal prediction improvement is
H(Y|Z)-H(Y|Z,U)=I(Y;U|Z)>=0. Our D8 final-stage representation is not declared
to contain the trained D2 parent, so that nested-information guarantee does not
apply. More layers and nonzero parameter gradients alone cannot supply it.

A constructive witness makes the gradient point explicit. Let every stage
multiply a scalar input by a nonzero trainable factor. An endpoint squared
loss can have nonzero derivatives for every factor, while every intermediate
representation is an invertible scalar rescaling of the same input. Depth
adds no predictive information and can change optimization conditioning.
This witness does not describe the native model; it disproves the inference
that every-layer gradient coverage alone establishes useful acquired depth.

The engineering priority is to measure predictive consequence at stage cuts,
retain acquisition long enough to test slow learning, and implement faithful
learning-policy replay for later credit. Forward depth, available route capacity,
selected activity, information retained and learning response are separate
quantities. The current completed result makes their separation necessary;
it does not replace temporal computation or counterfactual learning with a
dense architectural substitute. No new fit or public selection is admitted.
