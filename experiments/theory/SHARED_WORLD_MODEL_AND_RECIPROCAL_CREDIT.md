# One predictive world includes its own learner

User-directed Curie formalization,10 October2026. B1/R1. A shared world model can be an architectural consolidation of forward prediction, policy and credit. Mathematical contraction is a separate property requiring explicit conditions. This note links [reciprocal actor–critic](RECIPROCAL_ACTOR_CRITIC.md), [information/depth economics](CREDIT_INFORMATION_AND_DEPTH_ECONOMICS.md) and [optimizer-aware route effects](OPTIMIZER_AWARE_ROUTE_CREDIT.md).

## What the shared model must predict

Let X=(environment belief,forward state,credit state,theta,psi,optimizer moments,retained evidence,versions,time). A sparse event action U specifies a route/message, acquisition and learning decision. The joint generative law predicts observations, losses and the next augmented state conditional on do(U), not merely the observational frequency of U. A policy head proposes U; an expected future-loss head V predicts task/resource cost; a credit head predicts the cotangent or finite intervention effect of a change to U or X. They share latent structure, while retaining different typed targets and their causal timing.

The learner belongs in the world. If a route changes an update to theta or psi, subsequent observations/actions can change. A model predicting only the external process omits that pathway. Known update algebra, physical delay laws and local derivatives can remain exact primitives inside a learned transition. There is no requirement to learn every deterministic operation again. Unknown continuation, credit transport, acquisition and useful compression are trainable functions, grounded by actual future errors and supported interventions.

Known connections: [DreamerV3](https://arxiv.org/abs/2301.04104) learns environment models and improves behavior through imagined futures. [Meta-gradient RL](https://proceedings.neurips.cc/paper/2018/hash/2715518c875999308842e3455eda2fe3-Abstract.html) adapts learning-return choices online; [the2020 extension](https://proceedings.neurips.cc/paper/2020/hash/ae3d525daf92cee0003a7f2d92c34ea3-Abstract.html) learns an objective from interaction. These establish related principles. The proposed event-native construction jointly accounts for sparse temporal routes, persistent learner state, producer-weighted credit and paid discovery/audits; its complete novelty and practical advantage require implementation comparisons.

## Policy, loss and credit must describe a consistent causal world

For a fixed joint policy and discount0<gamma<1, write

    V(X)=E[cost(X,U)+gamma V(X') | X],
    Q(X,u)=E[cost(X,u)+gamma V(X') | X,do(u)].

The cost includes observed supervised error plus declared work/delay prices. The critic's task-valued teaching term is the effect of an action on later V through changed critic/actor/optimizer state. Thus a policy can learn to make its own future credit easier and more useful; credit can teach that policy to expose informative alternatives. Lower regression error alone is not the objective.

A coherent loss projection uses E[ell(prediction,Y)|X,u], integrating the predicted distribution of Y. It generally differs from ell(prediction,E[Y|X,u]). For squared loss, the omitted term is the conditional variance of Y. A world model that collapses uncertainty can therefore misvalue both task risk and evidence acquisition, even when it predicts the conditional mean accurately.

Shared latent state should preserve conditional future consequences relevant to both acting and teaching. Prediction sufficiency alone need not imply credit sufficiency; the information theorem quantifies that loss. A latent derivative grad_z V supplies the original-coordinate credit only through the appropriate encoder/producer Jacobian, with its actual computation charged. Changing or truncating that map changes the target. Shared weights do not eliminate these derivatives.

## Architectural consolidation does not itself prove contraction

The fixed-policy Bellman operator for bounded costs contracts in sup norm by gamma. Let P be the real augmented transition law and M a learned one, under the same fixed policy. Assume conditional expected cost error<=delta_c uniformly and W1(P(.|X,u),M(.|X,u))<=delta_T uniformly in a declared state metric. If the real V_P is L_V-Lipschitz and both discounted value functions are bounded, then

    ||V_M-V_P||infinity <= (delta_c+gamma L_V delta_T)/(1-gamma).

Proof: subtract V_M=B_M V_M and V_P=B_P V_P, insert B_M V_P, use Bellman contraction for the first difference, and the Wasserstein/Lipschitz bound for the second. This is a conditional model-error bound, not measured constants for our unbounded-NLL models. A finite horizon can use accumulated per-step errors instead. Nonstationarity fits the augmented time/belief state if the model supplies sufficient causal information and the error premises hold.

Credit derivatives need stronger control than value contraction. Smooth fixed-policy transport has lambda=b+K lambda; sufficient contraction is gamma*sup||T_X||<1, including learner and optimizer coordinates and necessary policy/transition score terms. Uniform value fit or a Bellman residual alone does not certify derivative accuracy. Hard winner selection requires expected-return score terms or declared forced-route effects, not a smooth derivative through a discontinuous winner.

Physical elapsed-time discount is not automatically a strict contraction per event: zero-gap events can have discount one. Finite-horizon analysis or a verified block contraction handles such sequences. Clockless asynchronous execution is compatible with causal transition/credit equations; it does not grant these stability properties by itself.

## A shared world model gives a useful decomposition to learn

For the same continuation randomness, define D_F as frozen-learner intervention response and D_A as the response including the declared learning policy. Then

    E[D_A|H] = E[D_F|H] + E[D_A-D_F|H].

A calibrated established response model can be retained while a learning-state-conditioned model learns the paired difference. The paired-label variance is

    Var(D_A-D_F|H)=Var(D_A|H)+Var(D_F|H)-2 Cov(D_A,D_F|H).

Shared futures reduce noise when that covariance is positive. They do not do so universally, and do not fix error already in the baseline prediction. Paired effects target the newly introduced learning pathway rather than forcing a new small dataset to relearn every old response. The difference channel must include learning-policy/optimizer/version information when it varies; a head trained at one fixed policy is scoped to that policy.

Our first optimizer-aware pilot gives the concrete engineering diagnosis: two2817-parameter whole-response corrections fit32 prefixes with4 continuations each. Learning-aware labels improve calibration against the fixed-label correction, but both hurt selected future utility versus the already trained response model. The next comparison preserves the model and learns only D_A-D_F from the same paid paired packets. It is development, with the assessment set already examined; any positive revision needs fresh confirmation before a generalization claim.

The eventual shared model can emit expected effects and uncertainty for candidate routes, parameter updates and information queries using a common persistent representation. Evaluate its task forecast, intervention calibration, actual teaching utility, version tracking and total resources separately. The intended consolidation is a coherent trainable model of *acting and learning*, implemented through time and sparse events, not a claim that sharing one tensor makes all predictions true.
