# Structured response credit and derivative generalization

R1/B1, user-directed extension, 10 October 2026. This is a proposed credit representation with completed local numerical contracts, not a demonstrated efficient online learner.

## Failure addressed and retained mechanisms

A generic critic must learn known local intervention algebra as well as uncertain future context. Averaged labels improve calibration, but depth-4 critics fail the three-seed action-utility gate; at depth 8 the fixed action is optimal on 87.5%/98.4% of evaluated contexts in two seeds. Low value MSE alone does not guarantee correct action ranking or useful derivatives. Preserve native temporal blocks, separate keys/values, predecessor messages, persistent addressed memory and silence-aware race likelihood. The substitution changes only the auxiliary critic's response head. Inference remains unchanged except when a selected bounded key-write adjustment is applied. All candidate scoring and learning work remain charged.

## Exact local response

Consider a key-write intervention a at event j. In this validated case it changes mark logits without changing future hidden states, queries or timing hazards on the same external continuation. Let u_i be the elapsed-time-transported query divided by sqrt(key dimension), beta_i=u_i^T a, p_ic the source-mark probability in clock c, w_ic the normalized posterior clock responsibility for the realized future event, and b_i indicate that its mark equals the source mark. The exact future event loss difference is

    D_i(a) = -b_i beta_i - log sum_c w_ic/[1+p_ic(exp(beta_i)-1)].

The survival term cancels. Its numerical implementation adds logsumexp(log w) to anchor the no-op exactly despite floating-point normalization error. This identity does not hold automatically for a route/write that changes future hidden-state dynamics; that case needs intervened transition prediction or replay.

At zero intervention,

    grad_a D_i(0) = (sum_c w_ic p_ic - b_i) u_i.

Thus the forward pass supplies exact local response structure, while the critic learns how future response evidence depends on the causal prefix. It need not learn a tiny disconnected gradient approximation. The v5 representation uses a shared connected prefix encoder and eight learned response components, each with bounded u, valid mark probabilities, normalized clock weights, b in [0,1] and positive mixture weights. Component meanings are not identified by value supervision alone: this is a learned response-mixture representation, not a claim of calibrated generative future simulation.

## Bounded sensitivity and curvature

For one component, define p_c(beta)=p_c exp(beta)/[1+p_c(exp(beta)-1)] and posterior weights proportional to w_c/[1+p_c(exp(beta)-1)]. Differentiation gives

    D'(beta) = E[p_c(beta)] - b,
    D''(beta) = E[p_c(beta)(1-p_c(beta))] - Var[p_c(beta)].

Consequently |D'| <= 1 and |D''| <= 1/4: the first expectation and the variance both lie in [0,1/4]. If ||u|| <= U, then |D(a)| <= U||a||, ||grad D(a)|| <= U and ||Hessian D(a)||op <= U^2/4. Positive mixtures and conditional continuation averages retain these bounds because component/mixture parameters depend on prefix history, not on the candidate action.

For the current unnormalized native query, a global frozen-actor bound is

    U = (||W_query||op (||gamma||infinity sqrt(d)+||bias_LN||) + ||bias_query||)/sqrt(d_key).

Positive normalization epsilon bounds the normalized hidden-vector norm by sqrt(d); physical key decay is contractive. Recompute/version this bound when forward parameters change. Other query normalization or transport mechanisms require their own bound.

## From value error to derivative error

Let true conditional future response A and learned response C both have Hessian norm at most U^2/4. Suppose |C-A| <= epsilon uniformly on a neighborhood containing a+h v for every unit direction v. Comparing directional finite differences gives

    |v^T(grad C-grad A)| <= 2 epsilon/h + (U^2/4) h.

If the minimizing h=sqrt(8 epsilon)/U fits inside that neighborhood, then

    ||grad C-grad A|| <= U sqrt(2 epsilon).

This is a conditional derivative certificate, not a bound inferred from held-out average MSE. At boundaries choose a permitted h and keep the original expression. A finite action set needs a covering radius rho before sampled errors imply uniform error: the shared U-Lipschitz property gives epsilon_uniform <= epsilon_grid + 2U rho. Sixteen random unseen directions do not supply a verified covering radius; the present pilots measure transfer rather than certifying uniform accuracy.

Together with the existing smooth future-descent bound, this provides a concrete route from learned future values to useful bounded updates. It does not prove learnability, grokking, deployment counterfactual identification or economical depth scaling.

## Implementation and evidence gates

`credit/structured_future_credit_v5.py` extracts exact future response packets and implements the constrained mixture decoder. Value and action-gradient parity pass at native depths 2/4/8; no-op anchoring and bounded outputs pass. The initial unanchored implementation stopped on roundoff at no-op; v5b has fresh queue identities and no changed successful runs.

Three depth-4 pilots compare connected free-form and structured critics with identical prefix encoder initialization, basis labels, training steps and nearly identical head parameter counts (integer rounding). Only nine basis actions supervise critic fitting; sixteen fixed additional directions test transfer. Their TRAIN-mean reference uses the same TRAIN continuations at those directions, giving the reference additional action information. Four selected native writes per critic are replayed to verify actual state adjustment and unchanged past loss. Actor and critic checkpoints are retained for reuse, with hashes in result files.

Charge all teacher response extraction, fitting, candidate evaluation, selected-write replay and optimizer work. A learned component pool's capacity and scored activity are separate. Future gates: reproducible unseen-direction calibration and utility, a meaningful intervention headroom measurement, then versioned integrated online learning with sparse residual audits. No asynchronous scheduling, sparse hardware energy or full actor-learning advantage is claimed by these pilots.

## Credit activations and a bounded state-update policy

The response head emits reusable activations (u_r,p_rc,w_rc,b_r,mixture_r). Its predicted future cotangent at zero is sum_r mixture_r (sum_c w_rc p_rc-b_r) u_r. Thus a connected high-capacity predictor supplies a state-sized credit vector through known local algebra, without action-gradient autograd or parameter-sized temporal sensitivity storage. Head inference, all component contractions, storage and training remain paid. This is credit to one four-dimensional key write, not a replacement for full producer-weight credit. Actor/critic versions accompany activations; a mismatch requires recomputation in the prototype.

With L=U^2/4, use eta=min(1/L, radius/||predicted_cotangent||). A valid external current-gradient error bound epsilon_g gives future loss change <= -eta(1-L eta/2)||g_hat||^2+eta epsilon_g||g_hat||. Emit only when this upper bound plus incremental emission cost is negative. Without a supplied valid error certificate the prototype returns a bounded proposal explicitly marked uncertified. Measured point gains and mean value MSE are not accepted as certificates. Optimizer gains may later be learned under the same radius/version/error constraints; this first implementation uses the derived trust bound.
