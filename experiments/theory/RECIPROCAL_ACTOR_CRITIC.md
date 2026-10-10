# A reciprocal actor–critic that learns how to make learning useful

User-directed Curie formalization, 10 October 2026; B1/R1. This develops the user's description: the actor acts and generates useful teaching evidence, while the critic teaches the actor how to improve both prediction and future teaching. Read with [closed-loop exploration and credit](CLOSED_LOOP_EXPLORATION_AND_CREDIT.md) and [credit information economics](CREDIT_INFORMATION_AND_DEPTH_ECONOMICS.md). No new fit or benchmark claim.

## Shared task objective, reciprocal teaching

Let X=(forward state, credit state, theta, psi, optimizer state, retained packets, regime belief, versions). An action u chooses a forward route/message, packet representation, probe or counterfactual acquisition, and optional learning/update decision. The declared transition X'=T(X,u,xi) includes both actor and critic updates. Minimize expected future supervised error plus resource and delay cost:

    V(X) = E[loss(X,u) + work_price*work(X,u) + gamma*V(X')].

For nonstationary online environments, include causal regime information or full history in X. The actor affects what is observed and what the critic can learn. The critic estimates task-relevant future effects and teaches actions, packets and learning choices. Dense supervised feedback supplies many observations, but does not reveal every unchosen route or future regime. Sparse acquisition and delayed credit remain genuine problems.

This is related to actor–critic and dual control, not a claim that those fields lack information-seeking actions. [Sutton et al. (1999)](https://proceedings.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) derive policy gradients aided by approximate values; [Klenske and Hennig (2016)](https://www.jmlr.org/papers/v17/15-162.html) study controls that simultaneously affect a system and learning about it. Our proposed construction applies that joint decision principle to trainable credit, forward-produced counterfactual packets, sparse temporal routes, producer-weighted audits and persistent asynchronous state. Novelty of the complete construction needs implementation and a literature comparison; these connections are established primitives.

## The actor's teaching term is an actual derivative

For differentiable u and a declared fixed-noise transition, define Q(X,u)=cost(X,u)+gamma E[V(T(X,u,xi))]. Under derivative interchange,

    D_u Q = D_u cost + gamma E[(D_u T)^T grad V(X')].

Partition X'. One term is (D_u theta')^T grad_theta' V, another is (D_u psi')^T grad_psi' V. The second measures the downstream task value of teaching the critic; packet, state, optimizer, acquisition and time effects supply additional terms. Dropping the psi' term freezes the critic's future learning in the actor's objective. It is a modeling choice, not a consequence of the critic having a different network name.

For a simple local critic update

    psi' = psi - beta grad_psi L_credit(psi; packet(u)),

the teaching term is

    -gamma*beta [D_u grad_psi L_credit]^T grad_psi' V(X').

This requires a mixed derivative if evaluated directly. The existing first-order CPU scan does not supply that derivative for free. Supported intervention returns can train its value instead, or local tangent/reciprocity probes can approximate declared pieces with their costs and errors. A hard routing policy uses a supported stochastic score estimator or exact forced-route value; differentiating through a discontinuous winner as if it were smooth omits its selection effect.

For elapsed time Delta(u) and continuous discount exp(-rho*Delta), differentiating the future term adds -rho*D_u Delta*exp(-rho*Delta)*E[V(X')], as well as the discounted transition derivative. Delay can also change what the environment reveals, so its physical transition effects must remain in T. Clockless execution changes schedules, not the need to define causal dependence.

## Teaching loss is a surrogate, not the shared reward

Making the critic's regression loss small is not by itself useful. The actor could stop exposing difficult routes, emit constant uninformative packets or visit only easy states. Self-consistency can then improve while task performance worsens. The task-valued teaching term above values *changes in the critic by their downstream consequences*. Properly normalized audits, supported exploration and held-out later losses ground those consequences.

A credit-relevant representation objective is conditional producer-risk plus complete packet/predictor cost, as derived in the information note. Its population target cannot be manipulated by changing the assessment distribution silently: compare on a declared task stream and account for policy-induced changes. At a causal boundary, old audit evidence may teach a new packet, but the resulting prediction cannot be scored as an independent pre-audit prediction on that same packet.

The critic can teach the actor to preserve useful causal distinctions, choose informative alternatives and expose uncertainty where audits matter. The actor can teach the critic with generated causes, local tangent probes, future observed errors and audited intervention effects. These signals have distinct targets: generated-cause attribution is not observed-data responsibility; local derivatives are not future intervention values; current gradient matching is not future task utility. Versioned packets can feed all relevant losses without conflating them.

## A finite example where teaching pays, and where it fails

Let a hidden regime z be equally likely +/-1. A critic's scalar state starts at mu=0. A paid probe returns s in +/-1 with P(s=z)=p>=1/2. It updates mu'=beta*s; the actor later predicts theta=mu'. Terminal task loss is (theta-z)^2. Without a probe, theta=0 gives loss1. A probe costs c, including its declared work/delay penalty. Its expected task gain is

    gain = 2*beta*(2*p-1) - beta^2 - c.

Proof: E[s*z]=2*p-1 and s^2=z^2=1. For unconstrained beta, the best beta is 2*p-1; gain is (2*p-1)^2-c. An actor should acquire the probe when that value is positive, not simply because it changes the critic or fits the signal. A noisy probe with an oversized critic update harms task performance even though it supplies a training example. A finite learner's acquisition value therefore depends on its actual learning rule, not only the theoretical information content of an observation.

The same mechanism applies to optional counterfactual routes: probe quality, receiver-generalization accuracy, update strength, delay and work jointly determine later utility. This scalar example proves the mechanism is possible, not that a neural implementation obtains the optimal beta or knows p.

## Implementation gates

First retain the native receiver-generalization contracts and completed three-seed results. Then measure the four causal effects separately: actor teaching evidence enabled/disabled; critic state persistent/reset; forward access to credit state enabled/disabled; supported exact correction retained. Match available observations, task stream and total work. Expose an explicit task-valued acquisition head rather than rewarding critic training error reduction alone.

Changing-weight counterfactual continuations must include their own subsequent actor/critic/optimizer updates. Fixed-path native response algebra remains an exact local diagnostic, with that scope. A next online implementation can use bounded replay to train task-valued route/acquisition responses, causal risk estimates for audit allocation, and current-version residual correction. Useful cooperative teaching must improve later task loss under a declared resource boundary. No additional unbounded hierarchy of trainer networks is required by the formulation.
