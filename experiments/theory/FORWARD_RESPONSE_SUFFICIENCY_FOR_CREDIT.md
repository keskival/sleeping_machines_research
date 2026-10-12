# Forward response information can be necessary for credit

Curie,12October2026. B1/R1. A consequence of route-response geometry that directly guides the native world-credit interface. Derived during the registered fixed comparison; no fit or target selection is changed here.

## A calibrated world prediction can still give biased credit

Let H be the available pre-target history, S=S(H) the critic's retained representation, and V(H) an issued route-response feature such as a supplied logit gap. The best binary world predictor using only S is q_S=E[Y|S]. Calibration gives E[Y-q_S|S]=0. It does not imply

    E[(Y-q_S)*V | S] = 0.

For a BCE pair, V=z_a-z_b and estimated-versus-realized contrast error is (Y-q_S)*V. Its conditional mean is

    E[(Y-q_S)*V|S] = Cov(Y,V|S).

Thus ordinary posterior calibration is sufficient for this credit only if the representation also removes the relevant conditional outcome/response covariance. This is a precise input-sufficiency condition, distinct from estimator capacity. For parameter updates, replace V with the complete issued producer response; the covariance becomes vector-valued and must combine shared channels.

## A causal counterexample

An observed pre-target bit X is equally likely -1/+1. The future target is Y=(X+1)/2. A critic summary erases X, so its Bayes-optimal q_S is0.5. Two issued route logits are z_a=X/2 and z_b=-X/2. Both are functions of already observed X; no future target is used to issue them.

Using softplus(z)-softplus(-z)=z, the actual BCE contrast is

    Delta(Y)=(0.5-Y)*X=-0.5.

The perfectly calibrated summary-only world projects Delta(q_S)=0 for every example. It assigns zero mean credit to an alternative that improves actual loss by0.5. Supplying V=X lets the posterior recover Y and the correct credit. This proves that an optimal forecast under the critic's restricted information need not supply optimal teaching for routes whose responses contain more information.

The example is an information obstruction, not proof that the present native world's pooled interface exhibits this exact alias. Its context/memory may preserve the relevant facts, or its training may be insufficient. Those possibilities require separate diagnosis.

## The forward pass is a source of learned sufficient information

An issued actor forecast P_F is itself a learned compression of history. Enlarging the critic's information from S to (S,P_F) cannot increase Bayes Brier risk:

    E[(Y-E[Y|S])^2] - E[(Y-E[Y|S,P_F])^2]
      = E[(E[Y|S,P_F]-E[Y|S])^2] >= 0.

This is the orthogonal conditional-expectation identity. A finite trained network need not realize that improvement. Actor predictions do not become truth; they are available side-information supervised against actual later outcomes. Counterfactual output gaps, clocks, current producer/version and aligned response directions can similarly expose credit-relevant history absent from a pooled token.

For current logits treated as supplied observables, the shared posterior should condition on the response information used in its projections. A general learned world representation can instead retain a response-sufficient history, avoiding a new query for every alternative. Which compression is economical is an architectural question, with candidate discovery and response computation charged.

## A practical native construction to evaluate after diagnosis

Use the native world's independently chosen reads/writes and persistent state. Supply already computed issued predictions and sampled counterfactual responses as typed input to that model; do not add a dense recurrent controller. These features must be computed before the outcome, detached as teacher input where needed, and bind the current actor version. An unseen uncomputed route still needs a supported response estimate or exploration; no free counterfactual oracle appears.

A conditional residual output can start from an issued forward reference b(H):

    q_hat=sigmoid(b(H)+r_phi(H,b,response_information)).

Proper BCE gives partial loss/partial r_phi=q_hat-Y. Zero residual preserves the reference forecast; native state learns where and how to correct it from actual outcomes. This is a logit residual parameterization, not a literal independent-evidence likelihood ratio unless that stronger probabilistic factorization holds. Do not double-count a common feature as independent evidence.

For a coherent shared world belief, choose one declared reference/conditioning set for all route projections. Choosing a separate best posterior for each route without action-conditioning consistency can reintroduce unrelated credit heads. If actions alter the external or learner transition, learn the intervention-conditioned outcome law rather than assume the same exogenous future.

## Decision rule for the current pipeline

The v1 implementation queries pooled native context and relationships but omits the already computed actual route logits. Its completed source/reuse traces record issued actor and world probabilities. Compare their proper losses on identical targets and measure response-weighted error before declaring the world representation adequate. Actor forecast superiority is a concrete signal to investigate the missing side-information; it does not identify whether the cause is representation loss, optimization, conditional target drift or capacity.

If the fixed comparison wins, confirm unchanged before redesigning it. If it fails and the world loses useful forward predictive information, the next owned intervention is a native response-conditioned world query with a declared forward reference, matched against frozen forward-belief credit and exact observed-label teaching. Preserve source acquisition, shared memory controls, causal timing and full work. Only if the information test supports it should a joint factual/route posterior projection from COHERENT_WORLD_CREDIT_AND_SHARED_PRODUCER_NOISE.md follow. This connects intuition to a discriminating implementation choice rather than an unbounded architecture search.
