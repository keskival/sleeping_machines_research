# Whole-family primitives, integration and capability bounds

4 October 2026. A systematic synthesis, not a new architecture or training
admission. Read with [the family design rationale](../../report/model_family_design.md),
[the visual review](../../report/architecture_review.md),
[the complete family inventory](../../report/model_family_inventory.md),
and the earlier notes linked below. Each inclusion requires its stated local
operators, information, precision, state and execution budget. No inference of
efficient learnability follows merely from computational universality.

## 1. Review axes and semantic levels

**Computational capacity** is the set of operations and causal programs a
construction can execute, together with work, latency, communication and
storage. **Representational capacity** is the input distinctions and predictive
functions that survive encoding, state evolution, routing and readout at a
specified precision. **Trainability** is the availability, correctness, alignment,
variance, horizon and cost of useful updates to those functions. None is a
substitute for the other two.

Use all three axes at the event, primitive, receiver, module, interaction,
layer, stack, memory composition and serving/learning-system levels. A larger
module inventory is not proof all modules are composed in one trained model.
The family includes logic/timing, statistics, carrier encoders, native selected
receiver stacks, historical retrieval, protected outcomes, richer temporal
reception and learning/execution variants. Native p64/D4 is one member.

## 2. Event and input representation

The semantic event is e=(observed address a, arrival t, content x). Its
address is input information or an explicitly learned route, never a hidden
target. A finite-precision event has a bandwidth budget: content numbers,
address bits, timestamp bits and scheduled-event multiplicity all count.

For any deterministic adapter A of the observed history E, and a predictor q
under log loss, the exact decomposition from149 is

    E[-log q(Y|A(E))] - H(Y|E)
      = I(Y;E|A(E)) + E KL(p(Y|A(E)) || q(Y|A(E))).

Proof: add/subtract H(Y|A(E)); the entropy difference is the conditional
mutual information because A is a function of E, and the remaining term is
conditional KL. A real packet adapter that retains only counts/centroids
cannot generally retain within-packet order. A stronger native learner cannot
reconstruct discarded information, but adapter loss does not identify whether
the remaining representation is trained successfully. Preserve raw/packet
information boundaries across every reference comparison.

## 3. Algebraic primitive capacity

First-of computes min; serial delays add; latest-arrival joins compute max.
Coincidence/hold/veto logic supplies temporal predicates. Fixed exponential
flow F_delta=e^(A delta) has semigroup composition; diagonal complex modes
become real damped rotations. Delay transforms weights and phase, allowing
interval features and deterministic delay-coded aggregation (§105).

Timing shifts/order only add usable information if later receivers observe
and preserve them. A common shift can be a nuisance under a time-equivariant
objective; an interval can be sufficient under another objective. Counting
possible orders is an encoding-cardinality statement, not a statistical
generalization bound. Precision, jitter and deadlines restrict those orders.

For a transverse smooth threshold crossing g(z(T),T)=0,
implicit differentiation gives dT/dtheta=-D_theta g / D_t g. This requires
nonzero total slope and the same crossing type. An arrival-triggered crossing
instead has T=t_arrival on its history cell and includes the state jump in
the emitted content. EventProp and notes155ff establish the prior/contract
boundary. Support changes, resets and cancellations need separate terms.

The two-counter timing construction (§59) is an ideal emulation argument
requiring sufficient reference timing and unbounded precision/storage. A
bounded clock interval, bounded queue and finitely many finite-precision
states cannot constitute an unbounded machine. This narrows the resource
interpretation without removing the finite synthetic construction evidence.

## 4. Identity and computational time of a race

At a fixed entering state, s_i are logits, lambda_i=exp(s_i), Z=sum lambda_i.
For independent unit exponentials E_i, I=argmin E_i/lambda_i and T=min E_i/lambda_i.

    p(I=i,T=t)=lambda_i exp(-Zt)
    pi_i=lambda_i/Z; T~Exp(Z); I independent of T.

Relative logits control identity; their common shift controls speed. The
observed joint score h_i=1[I=i]-lambda_i*T decomposes as

    h=(onehot(I)-pi) + pi*(1-ZT).
    F_choice=diag(pi)-pi*pi^T; F_clock=pi*pi^T;
    F_joint=diag(pi).

This is an exact conditional statistical-information identity (102), not a
guarantee that a useful label depends on either coordinate. With shared model
parameters and score Jacobian J, F_theta=J^T diag(pi) J can be singular.
Parameter coupling can therefore remain despite orthogonal local likelihood
coordinates. Hard score clipping gives zero raw-score derivatives outside its
range; more counterfactual samples cannot restore an identically missing
parameter direction. Bounded-score bridges are separate tested/proposed paths.

The normalized clock family151, T_nu=(ZT)^nu/[Gamma(1+nu) Z], preserves
conditional identity and E[T_nu]=1/Z, with

    Var(T_nu)=[Gamma(1+2nu)/Gamma(1+nu)^2-1]/Z^2.

At fixed independent W=ZT, dT_nu/ds_i=-T_nu*pi_i. Native bounded arrival
g(T)=.001+.010*T/(1+T) changes its mean under this transformation; later
states/routes can change. This control retains learned content/time coupling.
Its native numerical/quality admission is pending, not an integrated advance.

## 5. Unit, module and interaction capacity

For the current native unit, score reads raw stored memory before proposal:

    k_eff=k+K*m; s=clip(q(x)^T*k_eff/sqrt(P)+b,-12,12).
    f(x)=softplus(control_forget(LN(x)))/log(2).
    w(x)=2*sigmoid(control_write(LN(x))).
    m_prop=Rotate(omega*age)[exp(-rho*f(x)*age)*m]+w(x)*B*x.
    y=LN(C*m_prop+x).
    v=x+gain*y*sigmoid(G*GELU(y)+g).

Only the winner commits m_prop and the incoming timestamp. Input-conditioned
f is applied over past age; this program is not identical to an autonomous
fixed A over that interval. Memory damping is locally contracting at fixed x,
but the complete input/state Jacobian need not contract: gates, key reads,
clock sensitivities, residuals, LayerNorm and routing all contribute.

A module selects among programs, not merely among output vectors. A route can
change the content, persistent memory, timestamp, seen bit, next discovery
scores and future emission time. Separate key/value maps add degrees of freedom
but do not erase this shared-parameter/state interaction. Head mixing and a
latest-arrival join let several private programs interact through the next
layer. Cross-source runtime interaction must be explicitly connected; sharing
maps alone transfers rules, not the contents of private memories.

For any smooth fixed-history output z(theta), g=J^T e and
||g||^2=e^T(JJ^T)e. Available states or large Jacobian norms do not ensure a
nonzero component aligned with the current task residual. Likewise, for an
H-smooth local loss and proposed update -eta*g_hat,

    L(theta-eta*g_hat) <= L(theta)-eta<g,g_hat>
                          +H*eta^2*||g_hat||^2/2.

This diagnoses bias, variance and finite update size. It does not extend
smoothness across an unmodeled route/schedule boundary or through a different
optimizer transform. Clipping scalar gradients and Adam coordinate updates
must be inspected at the actual parameter/function step (120–139).

Adding a zero-nested new parameter block expands the available local Jacobian
columns without removing the old columns. Its reachable output perturbation
space therefore cannot shrink at that parent point. This is a precise local
reason offsets, separate keys/values or residual branches can help interference.
It does not ensure the new columns align with the task residual, that the new
optimizer finds them, or that later nonlinear routes preserve the inclusion.
Alternating route/content updates changes optimizer blocks, not the forward
coupling: changing content can still change scores/times through a frozen route
function. Conditional score-clock orthogonality is not parameter orthogonality.

## 6. Exact conditional credit and its incomplete substitutes

Fix the entering state and independent common-clock noise. Let Q_i be the
complete downstream loss for each selected timed/stateful branch. Holding
branch programs and the common first time fixed, the categorical derivative is

    dR/ds_i = pi_i*(Q_i - sum_j pi_j Q_j).

Actual common-clock and continuous branch-program derivatives are additional
terms, not competing alternative definitions. The raw joint likelihood score
can cover score-dependent winner/time distributions when used with a valid
return and baseline; reparameterized common time can instead supply the clock
derivative. Do not add two estimators of the same clock term inadvertently.
The broad precedent is stochastic computation graphs, not a novel score trick.

The default language teacher replaces Q_i differences by the realized
message cotangent dotted with alternative values. It is exact for that local
linear objective, not for arbitrary nonlinear expected sequence risk. Example
140: two routes emit the same v but one writes useful future memory. Message
credit is zero; exact future-state utility is nonzero. Enumerating cheap
statistic-delivery losses is exact conditional credit for those current losses;
it still does not credit all consequences of later count writes.

Full sequence route histories can grow exponentially. Complete continuation
replay pays every evaluated suffix; candidate sampling needs full support on
required contributions and known propensities. If q_i=0 for a useful missing
contribution, inverse weighting cannot recover it. If one useful contribution
has inclusion probability q, an unbiased Bernoulli inclusion estimator has
variance (1/q-1)*g_i*g_i^T. More samples can reduce this variance, not omitted
horizon bias or incorrect utility. Prefix reuse/caching changes paid work,
not the objective by itself.

## 7. Memory capacity, retrieval and learned exposure

M private P-vectors at b-bit precision give MPb content bits, plus timestamps,
metadata, key/parameter storage and queued events. Distinct state layouts or
routes add at most their explicitly stored information. To answer every exact
coordinate query about n arbitrary independent symbols from an alphabet V,
the accessible machine state must distinguish V^n histories, requiring at
least n*log2(V) bits. Proof: if two assignments map to the same state, a query
at a differing coordinate forces the same answer for both. This assumes no
external replay oracle; external memory must be included if used.

Counts compress repeated exchangeable evidence; learned summaries compress
according to task structure. Neither is a lossless arbitrary-history oracle.
Exact addressed counts take O(K) context-address operations per symbol for K
fixed suffix levels, plus their count-vector/readout cost. State grows with
distinct contexts. Bayes consistency requires occupancy-aware rates or an
equivalent represented statistic; the fixed-gate variance-floor result58 does
not rule out a learned recurrent gate that represents occupancy.

With N visits and pi_a stable on average, selected program exposure is roughly
N*pi_a, not N. An untied bank can dilute the examples teaching each processing
map. Shared maps aggregate rule exposure while preserving private evidence
and key exposure. This is the capacity/exposure reasoning398, not a universal
sample-complexity formula or proof balanced routing is always best.

For a fixed predictive expert set with positive initial weights w_k summing
to1, an exact sequential Bayesian/Hedge log-loss mixture has sequence
probability sum_k w_k*product_t p_k(y_t|past). Its cumulative loss obeys

    L_mix <= min_k (L_k - log(w_k)).

Proof: the positive sum is at least any one term; take minus log. Uniform
weights give a log(number_of_experts) regret bound. Every expert evaluation or
exact equivalent sufficient statistic must still be computed, and this bound
does not apply to an arbitrary gradient-trained or geometric mixer. For an
additive escape/base predictive p_y=a_y+e*q_y, the exact base logit gradient
is its usual supervised gradient multiplied by responsibility e*q_y/p_y.
Very low responsibility explains possible learning starvation without proving
the base lacks expressive capacity. Counts need not replace learned features;
use the right statistical/learned information path and exposure (58–62).

Available capacity, retention, candidate recall, delivery bandwidth, decoder
interaction and training credit are separate gates. A detached state retains
values but can supply zero derivative to an earlier writer (110). Under a
proved contraction bound rho<1 and bounded local loss cotangent C, an omitted
geometric credit tail after H has bound C*rho^H/(1-rho). Such global
contraction has not been proved for the native learned state stack.

## 8. Compositional expressivity and stable depth

For fixed history, residual maps x_next=x+alpha_l F_l(x) with
||DF_l||<=c_l and alpha_l*c_l<1 satisfy

    product_l(1-alpha_l*c_l) <= sigma_min(J_stack),
    sigma_max(J_stack) <= product_l(1+alpha_l*c_l).

Proof: each I+alpha DF has singular values between 1-alpha c and1+alpha c;
compose the lower/upper bounds. Full sequence/state Jacobians, shared-key
fan-out and event support still matter (§§113–114). A per-message certificate
does not control a deep recurrent system. Near-identity initialization can
preserve information and expose teachers; it cannot guarantee useful depth.

Noncommuting state updates represent order-sensitive functions. For linear
event maps A and B, AB!=BA distinguishes two arrival orders that an unordered
count summary does not. This is a strict separation from that restricted
summary, not from all statistical models, which may count ordered histories.
Combining separately retrieved values with a bilinear decoder can expose a
relation invisible to an affine decoder; that is a readout/function-class
fact, not proof all native states already retain the required evidence.

Unbounded optional branches can create b^D histories. One required continuation
through D layers costs D selected transformations/input; bounded branches,
coalescing and dynamic programs need explicit conditions. Associative affine
packet descriptors can coalesce exactly when they retain the full closed
state/query operator, but arbitrary nonlinear selected state updates do not
inherit that scan identity automatically (30–31).

## 9. Which model families are contained, and at what cost?

**Statistics:** suffix counts, continuation statistics, escape races and
proper predictive mixtures reproduce supported smoothing methods with their
actual statistic definitions. Learned statistic pools already exist as small
integrated implementations. Hard writes plus a dense predictive mixture are
not winner-only inference. Log-loss mixture algorithms can have expert-regret
guarantees, but not every neural mixture/mixer implements those algorithms.

**SSMs/convolution:** supported diagonal complex exponential filters are
exact event-mode special cases. Finite delay taps with weighted accumulation
give finite convolutions. Other matrix families, nonlinear blocks, pooling and
normalization must be reproduced before claiming complete EventSSM/Mamba
containment. Existing selective carriers are useful diagnostic members, but
their all-layer dense execution is not sparse race retrieval evidence.

**Attention:** for I~Categorical(pi), E[v_I]=mu=sum pi_i*v_i. R independent
deliveries have mean-square error tr(Cov(v_I))/R. Even with identical expected
head outputs, nonlinear layers/losses need not commute with expectation.
For the older time-normalized o=T*sum lambda_i*v_i, o=(ZT)*mu and
Cov(o)=mu*mu^T (45); all terms share the clock. Do not use independent-value
variance formulas for that different estimator.

The deterministic delay-coded construction105 sends value and unit-count
channels after delta_i=kappa*(s_i-s_cut), reads after every delivered arrival,
and divides the accumulated channels. Common exponential damping cancels and
gives mu over the delivered support. It pays candidate score generation,
every delivered value, waiting/logit range, precision and the ratio. Adding
the required projections, nonlinear blocks, residuals, positional convention
and normalization gives an in-principle finite Transformer emulation path.
The native receiver default is not that complete implementation.

If a fixed shortlist C retains mass1-epsilon, mu=(1-epsilon)*mu_C+epsilon*mu_O.
Consequently ||mu-mu_C||<=epsilon*diameter(values). Every forward statement
needs measured or certified retained mass. Gradient/depth bounds additionally
need fixed-support and fan-out conditions; a clock cutoff is not a mass proof.
Unstructured arbitrary full-bank exact queries cannot guarantee sublinear
candidate inspection: an unread arbitrary key could be the best match, and an
unread arbitrary value could change a dense weighted sum. Index structure,
approximation, distributional assumptions or preprocessing change the setting.

**RNN/MoE/retrieval:** the broad local-map family can host a specified gated
RNN cell, select expert programs, or retrieve historical learned keys/values.
Exact standard LSTM/selected expert-mixture inclusion requires its actual
gates and aggregation, not merely a claim of universality. MoE and product-key
memory already establish capacity/activity precedents. Here the additional
question is coupling to computational time, private evolving state, and paid
counterfactual future utility. Mature competitors may already exploit caching,
sparse kernels and recurrent decoding; compare actual implementations.

## 10. Computational bounds of integrated execution

For one source and D layers/H heads/U candidates/P payload numbers, cached
winner-only proposal work is approximately

    O(D*H*(U*P + K*P^2) + D*(H*P)^2), K=1,

plus task input/output projections, state/mask traffic, rate/clock operations,
packing and search. Refreshing a winner's key read costs its local P^2 map.
The conventional batched learner has K=U proposal-map work and additionally
backward, clipping and optimizer work. Sampled local credit can use K=2,
but retains U-key discovery and has different gradient variance. Untied
optimizer work scales with the updated parameter population and update cadence,
not just with the number of committed memories.

In physical races, numerical normalization can emerge from clock competition;
setting rates, precise timing, RNG, routing and circuit energy are not free.
In software, explicit exp/div/min still execute. FlashAttention demonstrates
that exact attention IO can improve independently of arithmetic; logical
value-read comparisons are not whole-system traffic/energy measurements.
The conditional ~2x attention-aggregation arithmetic opportunity is not a
complete-model or equal-quality Transformer speedup. One winner can change
the function unless approximation/precision is adequate.

## 11. Silence, windows and optionality

Smooth compact temporal windows with zero value/slope at boundaries can
differentiate membership for a fixed finite event set, with an expiry queue
and moment summaries. Popcorn reception resets a silence deadline after each
arrival; finite timeout policies require complete causal merge/split outcomes.
Neither reference by itself installs deep optional-message learning.

For stationary Poisson rate lambda and timeout H, a burst ends at a gap>H,
with probability exp(-lambda H). Hence mean burst size exp(lambda H), mean
duration (exp(lambda H)-1)/lambda, emission rate lambda*exp(-lambda H).
Continuous arrivals can postpone a pure silence emission indefinitely. A
maximum-burst deadline bounds latency but changes the operator; document it.

Counterfactual optionality is task-visible alternative correction capacity,
not route entropy alone. Many highly correlated timing alternatives can add
little rank; nearby temporal modes can share almost identical Krylov orbits
(32). More alternatives also add bias/variance/search costs. Existing evidence
does not establish an uncertainty/optionality policy beating the best immediate
utility policy across benchmarks. Retain reachable useful alternatives with
prospectively measured work and prediction consequences.

## 12. Review corrections and evidence discipline

Historical sections96–103 include useful constructions but stronger learning
language needs its later45 corrections: expected outputs/Jacobians do not
establish deterministic nonlinear-objective equality; sparse cutoff needs a
mass bound; finite noisy approximation needs composition regularity.

The route-detector mistake-bound sketch97 is about a restricted realizable
basis with specified proposal/credit and positive conditional-drift assumptions.
It is not a bound for learned intermediate vector features. The statement98
that Q candidate hypotheses automatically imply Littlestone dimension>=log2Q
is not generally valid: singleton indicator functions on Q isolated inputs
have dimension1 for Q>=2. A lower bound needs an actual shattered mistake
tree for the specified route class. Preserve historical experiments, qualify
this lower-bound assertion here, and do not generalize it to the native learner.

No general theorem proves this complete family trains efficiently on every
computable function, exceeds every incumbent, or has a lower whole-system
energy requirement. No reviewed theorem forbids useful advantage either.
Constructive expressivity opens a direction; contracts and honest benchmarks
determine which parts become practical. The prioritized owned integrated
queues remain the official primate/MG confirmation, curie native continuation,
AWS replay owners and completed-weight sparse validation; this review launches
no job and substitutes no core mechanism.

Primary attribution: [Transformer](https://arxiv.org/abs/1706.03762),
[stochastic computation graphs](https://arxiv.org/abs/1506.05254),
[EventProp](https://arxiv.org/abs/2009.08378),
[learned delays](https://arxiv.org/abs/2306.17670),
[EventSSM](https://arxiv.org/abs/2404.18508),
[Mamba](https://arxiv.org/abs/2312.00752),
[MoE](https://arxiv.org/abs/1701.06538),
[product-key memory](https://arxiv.org/abs/1907.05242),
[kNN-LM](https://arxiv.org/abs/1911.00172),
[FlashAttention](https://arxiv.org/abs/2205.14135).

## 13. Local expansion, containment and asynchronous composition

Let a causal reference layer be G and an event-interface construction be F.
An exact containment claim requires an admissible parameter/state/schedule
mapping such that F(e,h)=G(e,h) on the declared domain, including subsequent
state updates and outputs. Matching only the current mean output is weaker.
If every layer has this contract and their interfaces preserve the required
information and numerical semantics, induction gives a complete composed
predictor contract. Otherwise a per-operator identity is not a stack proof.
Copying a reference computation establishes an available prediction point;
it does not prove a new optimizer reaches that point or pays less work.

For a nested residual expansion f_new(theta,eta)=f_parent(theta)+B(theta,eta)
with B(theta,0)=0, the parent is preserved at eta=0. The local output Jacobian
has the form [J_parent, J_eta]; its column span includes the parent's, provided
the equality holds throughout the stated theta neighbourhood. Thus extra
coordinates cannot reduce infinitesimal reachable directions there. They may
be useless for the target or poorly conditioned; finite-step training and
generalization remain separate. Enlarging a normalized candidate race is not
automatically such a nesting: positive new rates change winner probabilities
and first-time distributions. Head addition changes the join unless controlled.

For per-layer expansion, place additional message/state bandwidth or access
before the lossy bottleneck. The adapter information-loss decomposition in2
applies equally to an intermediate deterministic representation. Widening
only downstream layers cannot recover an erased distinction. Stochastic
bottlenecks similarly require an adequate sufficient representation of the
history, or a deliberate return to that history. A credit bottleneck instead
requires useful utility, support, conditioning and horizon; more values alone
do not fix it. Full future-write counterfactuals remain distinct from the
successful local delivered-value teacher.

An event graph is scheduled by its causal dependencies, not necessarily by
global t=k*Delta updates. Local delays, races, deadlines and joins are allowed;
lazy analytic flow is exact only for the chosen evolution law and its controls.
There can be global reference coordinates without a global periodic update
barrier. Dense and sparse local subgraphs can compose if adapters preserve
their causal interfaces and query cutoffs. Physical asynchronous circuits,
message delivery, learned-time precision and optimizer coordination require
separate contracts and resource measurements. Current episode/GPU emulators
and Adam training are not evidence of a clockless chip or fully asynchronous
online learning. A single shared multimodal state requires explicit shared
paths; source-private memories with shared weights alone are insufficient.

## 14. Constructive reach and synchronous/dense endpoints

Conditional finite-execution containment: take a reference predictor's finite
executed dependency graph, including state reads/writes, reductions, branches
and declared arithmetic. Assume the chosen local operator library can reproduce
each primitive, the event interface retains all required operands, and state,
precision and order match the reference. Schedule each operator after its
dependencies, preserve state lifetime, and impose any required snapshot/barrier.
Induction over the execution graph proves equal intermediate values, writes and
outputs. A recurrent finite unrolling follows the same argument. With update
operators included, a specified finite learning execution is also reproducible.
No use of future information beyond the reference's declared observation is
permitted. This is an explicit conditional construction, not proof every
implemented native variant contains all reference operators or matches cost.

A synchronous step m_next=G(m_old,x) is an allowed event schedule: every
participating update reads m_old, and its result commits only after required
computations complete. Events sharing a timestamp but committing immediately
can yield another function. For old state(a,b)=(1,2), updates a_next=b and
b_next=a+b give synchronous(2,3); immediate a-then-b gives(2,4).

Configurable support M in{0,1}^n includes the dense mask(1,...,1). Variable
k-selection admits full delivery when k=n, provided the required aggregation
and normalization also exist. A fixed k<n restriction does not itself include
arbitrary dense pooling. These inclusion relationships concern admissible
models/schedules, not an assertion of dense behavior at strictly sparse cost.
At sufficient compatible resources they remove any representational ceiling
caused merely by event scheduling or configurable support. Broad computability
does not strictly exceed all computable predictors; practical superiority is a
quality/learning/resource question about specific constructions and tasks.

## 15. Learning in the same structure: online, TTT and batches

Write a deployed unit's variables as persistent evidence m, learned parameters
theta, optimizer/eligibility state o and versions v. A causal learning event can
apply (theta,o)_next=Update(theta,o,credit) to that same unit/program. This
allows gradient-based online adaptation and native TTT without an obligatory
separate dense teacher. Statistical increments and ordinary forward memory
writes are distinct updates, not automatically TTT. A TTT inner learner may
place trainable fast parameters in local state; exact containment requires its
actual objective, inner update, query and outer derivatives where claimed.

In predictive online evaluation, produce the scored forecast before observing
its label; after observing it, update and predict subsequent events. A query
may also use self-supervised adaptation on already observed content. Persistence,
reset, labels, update budget and pre/post-adaptation scoring are protocol terms.
Current static test numbers are not TTT evidence. Relevant precedents are
[test-time adaptation](https://arxiv.org/abs/1909.13231),
[TTT layers](https://arxiv.org/abs/2407.04620) and
[Titans](https://arxiv.org/abs/2501.00663); none implies a project TTT result.

Sparse learning needs an explicit work/credit contract. Sampling alternatives
can reduce expensive proposal evaluations; exact normalized score gradients may
still touch all scored keys. Shared maps/optimizer state may coordinate many
units. Delayed credit computed under theta_old cannot silently be advertised
as a derivative under theta_new; identify replay, retained eligibility or stale
gradient scope. Cached keys/values must be invalidated/refreshed or versioned
when their producing maps change. Fully asynchronous physical training is an
additional consistency/resource problem, not a consequence of a sparse winner.

Independent state lanes may be grouped while preserving their forward causal
semantics. Parameter update batching changes learning unless its own contract
matches the reference. For loss .5*(theta-y)^2, theta0=0, eta=.1 and successive
labels1,2, sequential SGD gives theta2=.29. A single summed-gradient update
on both labels at theta0 gives .30. Local/asynchronous update order likewise
need not commute. Declaring the schedule, optimizer, horizon and shared-version
policy makes streamed/batched/mixed learning meaningful within one family;
it does not assert those trajectories are identical or equally good.

## 16. Endogenous architecture choice and state migration

Let a local architectural policy choose a from admissible operators/connection/
support/reception actions A, and let each action expose a compatible event/state
contract F_a. With a fixed candidate supergraph, include policy state in the
machine state. Constant policies recover the included static constructions;
data-dependent policies give dynamic execution, while slower allocation actions
can alter model structure. This extends learned weights and event-level routes
to some family-level choices. It does not yet describe a completed general
learned-morphing implementation in this repository.

For L=E[causal task loss+beta*C], a discrete policy's choice contribution is
pi_a*(Q_a-sum_b pi_b Q_b), where Q includes future loss and the declared full
cost. Pathwise changes in content/state/time and direct cost derivatives also
contribute where applicable. Evaluating every structural option is expensive;
sampling preserves only its stated support/estimator, not automatic full
credit. A local message-linearized Q can miss a structural action's persistent
future benefit. Resource C includes scored options, losing proposals/replay,
migration, optimizer and actual executed maps, not merely nonzero gate count.

Growth/morphing beyond a fixed supergraph needs a state embedding/migration
S_old→S_new, parameter/optimizer mapping and time/version contract. Exact parent
preservation requires the new computation commute with that embedding on the
declared histories, before deliberately learned deviations. A lossy migration
cannot generally be inverted: erased counts/facts need retained history or
another source. Zero-initialized residual coordinates can preserve a function;
new race rates or head joins can change it immediately, as in13.

Relevant precedents include [DARTS](https://arxiv.org/abs/1806.09055),
[Net2Net](https://arxiv.org/abs/1511.05641),
[Once-for-All](https://arxiv.org/abs/1908.09791) and
[ProxylessNAS](https://arxiv.org/abs/1812.00332).
They motivate search, controlled growth, elastic settings and task/hardware
costs; they do not prove temporal state migration or future-write credit here.
The additional family ambition is economical automatic design within the
causal temporal/stateful interface. Static searched architecture, conditional
event policy and online structural reallocation must be separately evaluated.

## 17. Operational composition and finite event execution

A member must specify its admissible graph/operator library, state ownership,
routing/reception, schedule, query/objective, learning/version policy and
resource limits. The observed-information boundary, modeled event-time
coordinate and measured physical runtime are separate semantic roles; they
need not be separate tensors. Finishing computation after a query cutoff does
not authorize access to later observations. Interface compatibility is about
information, state, time, credit and resources, not just matching vector shape.

For state S=S_A×S_B, fixed parameters and local operations
T_A(a,b)=(f_A(a),b), T_B(a,b)=(a,f_B(b)), T_A T_B=T_B T_A. The same holds for
their emitted event multiset when emissions depend only on the local input/
state and random draws are keyed independently of processing order. Observable
query equivalence additionally needs the same accessible snapshot: a query
between updates can distinguish their order. Shared reads/writes, global
parameter changes, shared RNG streams and finite-precision reductions need
explicit dependency/order/aggregation contracts. This identifies legitimate
asynchronous freedom without claiming arbitrary reorderings are equivalent.

Positive delays alone do not preclude infinitely many internally generated
events in finite modeled time: delay_n=2^(-n) has a finite sum. One sufficient
finite-horizon contract is N external/initial pending seed events in the
observation interval, zero-delay work grouped into terminating blocks of at
most c local events, and each block emitting at most b continuation seeds.
Every internally generated continuation must arrive at least delta_min>0
after its parent block's triggering time. Crucially, b is the effective fan-out
of the block, not raw connection degree inside a zero-delay graph. Descendants
have chain depth at most floor(H/delta_min) in an interval of length H, giving
at most

    c*N*sum_{k=0}^{floor(H/delta_min)} b^k

local events. This can still be exponentially expensive; a practical execution
budget or tighter graph structure is necessary. Alternatively a finite DAG,
explicit event/depth cap or another proved terminating schedule supplies the
contract. Current finite-segment native stacks already have explicit bounded
loops; this note imposes no source change or new numerical admission.

## 18. State sufficiency and three preservation contracts

The [composition guide](../../report/model_family_composition.md) treats the
family envelope, individual member and integrated research target separately.
The envelope permits different compatible local programs; an individual
member has only its chosen operators, accessible evidence and learning rules.
An event wrapper for a known program gives no automatic advantage. Exact
containment is a transition/observation construction with stated work.

### Operational sufficiency at a boundary

Let a history h determine complete operational state S(h), including relevant
memory, time metadata, pending work, and versions. Suppose a compressed state
A(S) is the sole information available to all subsequent computation. If
A(S(h1))=A(S(h2)), deterministic continuation from the same future events and
queries cannot distinguish h1 from h2. Therefore, if an admissible future
query must distinguish them, this compression cannot exactly implement that
task. For stochastic programs, equality of the compressed state with the same
future input and random law gives the same conditional output distribution;
hidden history cannot be recovered by extra random samples. A future input
may itself reveal the missing fact, but that is additional evidence.

This simple obstruction includes counts losing order, payloads losing timing,
and a currently silent receiver losing its pending deadline. It is not a
claim that every task requires full history: a sufficient statistic can be
much smaller. Storage, timestamp/address bits and queued evidence all enter
the actual boundary state, rather than only the advertised vector width.

### Forward continuation preservation

For fixed parameters, define an admitted macrostep T(S,e)=(S_next,trace),
including a finite internal event execution and boundary observations. Queries
and timers are admissible actions too. Let E embed reachable parent states
into child states. A sufficient emulation condition for every admitted step is

    T_child(E(S),e) = (E(S_next), trace)
        whenever T_parent(S,e) = (S_next, trace).

Both programs use the same observation cutoff, time units, boundary addresses,
tie/stop semantics and query snapshot. Hidden steps may differ. For stochastic
programs, a coupling that makes this equality hold almost surely is sufficient;
it must preserve each program's specified random law. Trace timing here is
modeled time, not a requirement of identical physical execution time.

Induction over any finite admitted sequence proves equal boundary traces and
embedded future state. Resuming from a nonempty pending queue is included.
Termination/admission must also hold in the child; an equivalent output after
an infinite hidden loop is not an executable emulation. These sufficient
conditions are stronger than comparing one readout on a saved state.

Example: at q=2 the scalar popcorn receiver has m=2, deadline2.6. Mapping
(m,deadline) to (m,0,deadline) with an isolated new coordinate commutes with
parent additions, resets and timer firings. Mapping to (m,0,no_deadline) can
preserve the current “not emitted” query but loses the subsequent emission.
Changing race rates or completion joins can similarly violate preservation
even with zero new values. A wider state is safe only under its actual update
and scheduling contract.

### Learning transition preservation

Extend the state to include parameters, optimizer variables, retained credit,
random/update ownership and producing versions. If a learning embedding M
commutes with each specified observation/credit/update operation and preserves
its observations, the same induction preserves the declared learning
trajectory. Forward emulation alone does not supply this condition.

For theta>0, y=theta*x and y=phi^2*x with phi=sqrt(theta) represent the same
function at every mapped parameter. The derivative pulls back correctly:
dy/dphi times dphi/dtheta = 2*phi*x/(2*sqrt(theta)) = x. But for x=1, target0,
loss y^2/2, theta=phi=1 and SGD eta=.1, parent theta_next=.9 while child
phi_next=.8 gives y_next=.64. The same Euclidean step size does not commute
with this coordinate change. A matching update needs a specified transformed
optimizer, not merely copied forward weights. Detached eligibility or changed
counterfactual returns can likewise break learning with equal forward outputs.

Even parent-preserving growth need not immediately offer a useful tangent:
y=y_parent+u*v*x at u=v=0 has both new first derivatives zero, despite its
expanded finite-change function class. y=y_parent+w*x at w=0 has derivative x.
Whether the loss supplies a useful nonzero cotangent is an additional
condition. Nested function classes and non-shrinking old Jacobian span do not
guarantee newly available directions are exposed to first-order optimization.

### Resource refinement and practical selection

A resource refinement adds a declared work, storage, traffic or latency bound
to a functional/learning contract. Forward trace preservation says nothing
about that bound: the child can execute costly ignored work. For a realized
finite trace, component work plus adapters/discovery/joins/communication is
additive across invoked branches; ideal independent parallel latency follows
the critical path, with actual hardware contention charged separately.
Counterfactual replay and optimizer work belong to learning cost. Approximate
refinements must instead state their error, task protocol and resource scope.
Shared/cached executed work is counted once, with storage, refresh and validity
costs; views of the same computation are not independent additional work.
An advantage additionally requires improving the declared resource/quality
comparison, rather than merely meeting some budget.

These three contracts separate safe semantic substitution, consistent learning
and economic improvement. Automatic design should choose which contract a
structural action requires, retain a successful parent, and evaluate future
task/resource consequences after its state/version migration. This is a
family-level design rule and derived condition, not a completed general
structural learner or a new benchmark result.
