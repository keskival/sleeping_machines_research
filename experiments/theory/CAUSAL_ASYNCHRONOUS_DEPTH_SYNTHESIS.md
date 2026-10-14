# Causal asynchronous depth: representation, residuals and learning

14 October 2026. AWS implementation guide for R1/B10 and later event-native
hierarchies. This synthesizes measured results, established mathematical tools
and new design deductions. It is not a convergence theorem, novelty claim,
benchmark result or training admission. **Events need only be causal; no common
clock, shared timestep, level barrier or periodic update is required.**

## 1. What our experiments actually establish

| Evidence | Measured result | Architectural lesson and scope |
|---|---|---|
| E118 SHD, matched data/order/updates, seed6 | D1 65/256 vs D8 104/256; NLL2.2544 vs1.7496 | Useful temporal depth exists in this carrier member; D8 has more parameters/work. Smaller128-example comparison did not reproduce a consistent depth gain. |
| E117/E118 head-only intervention | Same frozen deepest features: conditioning changes39/128 correct vs7/128; NLL2.2408 vs2.9728 | Poor optimization can hide accessible information. Input-feature probe was slightly better than deepest-feature probe: this does not prove added information from depth. |
| E83 sparse spike stack | D4 silent layers3/4, exactly zero hidden main-loss gradients on audited batches; D8 alternates extinction/activity cascades | Activity support and stability are real failure modes. More shadow credit alone did not solve recognition. Original invalid shared-grid objectives are excluded. |
| E83 all-depth readout intervention | All21 matched deepest-only L1 spike effects zero although fused-head effects often improve | A shallow supervised bypass can look like useful hierarchical credit while no causal event reaches the deepest objective. |
| Integrated episodic character fit, six/eight blocks | 3.632968→3.541515 DEV bpc, about32.6% more work | Small single-seed positive depth evidence; capacity/initialization/gains also change. |
| text8 hard-route credit, D8 | 2.456→2.326 TEST bpc with route credit | Counterfactual credit helps a deep fitted model; this contrast isolates credit at D8, not depth alone. |
| E151 six→twelve event blocks | Exact initial untimed outputs/old teachers; live new output-map teachers | Functional containment and first learning paths can coexist. Six added6ms delays change latency36ms; no deadline preservation. Internal feature teachers start later. |
| E156 actual Adam continuation | Same batch CE0.622628→1.867725 atLR0.000325, but→0.333181 at1/16 rate | A good infinitesimal direction and gradient clipping do not certify a safe realized probability update. |
| Current Curie token language,8M | D8 final6.289904 vs D2 final6.160183; richerD2 6.099337 | This deep member loses at this acquisition. Teacher allocation/gain/head/horizon confounds prevent a single-cause verdict. |
| AWS Taxi temporal-depth successor, known seed2 | LL0.523773 vs ordinary continuation0.523577; parent0.521049 | Small positive added-depth contrast beyond continued fitting, at17.9% more incremental time; secondary metrics slightly worse than continuation. Exploratory known-seed revisit. |
| Native CPU exact recurrence execution | Synthetic128-event full-step speedup2.47x atD2→5.95x atD8 | Engineering execution scales better; this is not learning-to-quality scaling or physical distributed asynchrony. |

Sources: FINDINGS.md sections E118/E151/E156/E83/integrated episodic;
results/e118/matched_summary_20260929.json;
results/tpp/aws_taxi_successor_curiosity_20261014_s2_v1.json;
HANDOFF.md current D8 record; report/I_SCIENCE.md sections4.2 and credit development.
Original completed JSONs govern exact numbers; table rounding is intentional.
Carrier, spiking, modal-event and native sparse-token implementations are different
members. Do not transfer an isolated member's success or failure to every member.
The Taxi result is not an independent confirmatory extension of the24/25 record.

## 2. Five meanings of depth that must be recorded separately

1. Architectural levels with distinct parameter/state ownership.
2. Actual causal path length from an observation/write to a prediction.
3. Number of events/state transitions across which information persists.
4. Differentiated or counterfactually audited credit path/horizon.
5. Physical scheduling latency and sequential work.

A large level count can have short paths through skips. A single recurrent level
can have a long temporal path. Sparse writes need not mean sparse candidate search.
Serial CPU replay of irregular events is event-time modeling, not a measurement
of distributed asynchronous hardware. Persistent numbers do not imply persistent
gradients. Reports and implementations must name all five quantities.

## 3. A model is a causal event transducer on a partial order

An event e has identifier, causal parents, observation/semantic time s_e, local
availability/arrival time a_e, destination/address, payload, source version and
optional validity interval. Distinguish modeled physical elapsed time from
computation/communication delay. A source may observe an old physical event late.
At prediction query q, permit only the history H_q of events actually available
in its causal past. No backdating a correction into an already issued forecast.

Each receiver owns state S_r and updates by

    S_r^+ = U_r(S_r^-, e; theta_version),
    S_r^- = Phi_r(delta_available_or_declared_physical_time) S_r.

The time convention is explicit per state. Gaps must be nonnegative in that
convention; late physical observations need assimilation, versioned replay, or a
model of observation delay, not a silent negative-time step. Output heads query
available state at their own causal service events. Different levels may have
unrelated arrival streams and can react immediately to important events.

A topological execution is sufficient for a deterministic causal graph. It is
independent of the chosen topological order only when unordered updates on shared
state commute, or have a specified conflict order/confluent merge. Nonlinear
writes/races generally do not commute. Assign a receiver owner and deterministic
local tie/conflict rule; independent receivers need no global barrier. Random
streams are owned/checkpointed so a harmless scheduling change cannot silently
reassign race noise. Physical arrival differences can legitimately change the
model's observations and predictions; do not claim schedule invariance then.

## 4. Asynchronous residual algebra: compare like state at like reference time

A residual is an increment in a declared vector space relative to a reference;
it is not inherently a whole extra synchronous layer evaluation. For a linear
modal state let T(t,s) be its causal transition, satisfying T(t,u)T(u,s)=T(t,s).
A timestamped increment delta_e at state time s_e contributes at query time t:

    z(t) = T(t,s0) z0 + sum_(e available before q) T(t,s_e) delta_e.

This exact expression assumes additive writes and fixed linear transitions on
the interval. State-dependent gates, changing parameters or nonlinear downstream
responses need the ordered transition composition instead. Every increment has
an exactly-once event ID; duplicate delivery must not duplicate its contribution.
Maintain a separate accumulator from the base, or assimilate each increment once.

For packets closed at t1<=t2 with transported increments b1,b2, merge as

    (t1,b1) merge (t2,b2) = (t2, T(t2,t1)b1 + b2).

Associativity follows from the transition composition law. It allows independent
causal packet schedules without a shared timestep. If packets arrive out of
physical-time order, choose a lawful common reference and transport/assimilate
there; the formula does not license retroactive predictions. Parameter-dependent
transitions require source-version binding and the actual intervening transitions.
An old residual is not transported with newly changed poles as if they had always
applied.

Endpoint state AND derivatives agree under exact affine packet aggregation when
the identity holds as a function of the parameters. However, nonlinear emissions
inside a packet generally cannot be reconstructed from its endpoint. Example:
scalar additive states after two writes are(1,0) versus(0,0). Both endpoints0,
but sum of squared intermediate outputs1 versus0. Thus identical final state
does not imply identical next-layer input, credit or behavior. Preserve necessary
intermediate nonlinear events, augment sufficient packet statistics when possible,
or label aggregation as an architectural approximation and measure its errors.

Residual composition at a prediction interface can instead use

    logits(q)=logits_parent(q) + sum_e A_e phi_e(q),

with each available correction projected into the SAME logit space and each
phi queried/transported under its declared temporal law. A logit correction is not
a hidden-state increment; probabilities must be renormalized. Scalar probability
mixing is another distinct construction, with distinct gradients and semantics.
No level must wait for all other levels to produce a correction before answering.
Silence means no delivered increment under that interface, not necessarily no
information or no elapsed-time evolution.

## 4b. Residual design space: the algebra does not choose the architecture

There is no canonical asynchronous residual scheme. Choose the corrected object,
reference, arrival policy, update algebra and gradient law together:

| Scheme | What is added/corrected | Causal scheduling | Main benefit / principal failure |
|---|---|---|---|
| Event-paired payload residual | x_e + delta_e for the same source-event ID | Emit after branch completes, or issue base and a distinct later revision | Familiar serial refinement; waiting changes latency, revisions change the stream |
| Fast bypass plus delayed correction | Immediate base packet; tagged branch increment later | Receiver applies each available contribution once | Preserves fast access; needs version/ancestry ownership and stale-message semantics |
| Persistent-state increment | A write into a declared receiver state | Receiver-local update at arrival/assimilation | Natural sparse memory; nonlinear/noncommuting writes need explicit local order |
| Continuous-flow perturbation | Change to drift/decay/dynamics | Integrate the actual piecewise event-driven dynamics | Residual temporal computation; payload identity does not imply identity of future flow |
| Held predictive baseline plus innovation | Observation minus a specified causal predicted value | Emit selected innovations; query/propagate baseline between arrivals | Multirate compression; wrong baseline, stale correction or ignored silence changes meaning |
| Query/logit correction | Increment in a shared output coordinate system | Combine available corrections at each prediction query | Clean growth containment; can become a shallow ensemble rather than learned hierarchy |
| Optional temporal branch/race | Choose parent path versus a new computation | Hard supported choice, with actual delivery times | Native optional depth; winner selection is NOT additive merging and needs alternative credit |
| Causal packet residual | Transported sum/statistic over a local event set | Packet closes causally, independently of other levels | Exact for specified affine state; nonlinear downstream information can be lost |

These schemes can be composed, but each interface must name its type. A state
increment cannot be added to a logit; an absolute estimate cannot be treated as
a fresh additive delta every time it is delivered. If a producer sends revisions
r_v of the same reference, an additive consumer might apply r_v-r_previous,
transported consistently, rather than adding r_v again. That rule is valid only
for declared additive revision semantics; general Bayesian state assimilation
requires the correct dependence model, not automatic subtraction.

A residual packet contract should specify:

    (event_id, parent_ids, reference_id/version, coordinate_type,
     source_time, availability_time, destination, delta_or_absolute,
     transport_version, supersedes_id, validity, learning_provenance).

The schema is semantic, not a requirement to transmit every field in every byte
packet; an implementation may infer fields from owned queues. A branch evaluated
against state version v may deliver after the receiver has reached version v+k.
Choices are: apply a justified transported increment; recompute against current
state; replay a versioned internal world; or reject/expire it under a trained,
audited policy. These are different models. Naively adding a stale absolute
correction can double-count evidence even if all packets are causally delivered.

Two candidate schedules are behaviorally equivalent only if their observable
query outputs agree. Sufficient local conditions include identical causal input,
exact transition composition, commuting independent updates and no omitted
intermediate query/nonlinearity. Equal endpoint sums alone are insufficient.
When shared nonlinear updates do not commute, pick a receiver-local causal order;
no need to order unrelated events globally. If timing is informative or a query
falls between deliveries, even commuting endpoint updates can produce different
observable predictions at that query.

The first LOW-RISK growth experiment should use the protected optional branch
in USEFUL_DEPTH_REPAIR.md, with typed causal delivery and no parent-path waiting
at neutral growth. It isolates added useful computation without first solving
all forms of multirate communication. Output correction can supply a containment
contract, but must not replace the required integrated temporal hierarchy with
an unlabelled independent ensemble. Then compare fast-bypass/delayed-increment
and receiver-state innovation when data/error analysis motivates those choices.
This is an experiment order, not a proof that one residual scheme is universally
best. Do not force arbitrary independent streams into event-by-event pairing.

## 5. What makes a representation genuinely useful

There are three legitimate forms: compositional features, iterative refinement,
and temporal summaries at different predictive horizons. A useful architecture
can mix them. No theorem requires each successive layer to be more human-readable,
more invariant, lower dimensional or slower.

For discrete Y, unrestricted Bayes log-loss using representation Z is H(Y|Z).
Adding W to the SAME retained Z improves ideal risk by

    H(Y|Z)-H(Y|Z,W) = I(Y;W|Z).

If W is a deterministic function of Z, this gain is ZERO. Depth can nevertheless
make the same information accessible to a restricted decoder with fewer samples
or less work. A temporal upper state may contain older history absent from the
current lower message, so it need not be a function of that message alone.
Do not compare whole lower histories with single deeper messages and call their
information difference a depth advantage. Y's horizon, history budget and decoder
class must be explicit. Continuous-target statements require appropriate finite
entropies/KL integrability; use risk differences when entropy is undefined.

Let Q_l be the allowed decoder family. Its approximation/learning excess E_l>=0
satisfies R_l=H(Y|Z_l)+E_l under well-defined population log-loss. Therefore

    R_l-R_(l+1) = H(Y|Z_l)-H(Y|Z_(l+1)) + E_l-E_(l+1).

This separates information retention, computational accessibility and fitting.
Frozen linear/nonlinear probes, matched continued-parent fits and state/content
interventions test different terms; none substitutes for all of them. A feature
may be linearly decodable yet unused by the trained predictor. A destructive cut
may show dependence yet not useful incremental depth.

A predictive sufficient state at horizon h satisfies
P(Y_(t+h)|H_t)=P(Y_(t+h)|Z_t). Approximate sufficiency is measured by the expected
conditional KL between these laws, not by feature variance or reconstruction of
all raw input. An abstraction should retain task-relevant distinctions, including
uncertainty and timing, while allowing nuisance distinctions to be ignored.
Slowness by itself can erase a rare critical event; high variance can be nuisance.

## 6. Communication, silence and learned timescales

Let B be what a receiver already knows and Pi the COMPLETE observation of a
communication policy: payload, arrivals, timestamps, and observed no-arrival up to
a query/deadline. With unrestricted decoders its predictive value is I(Y;Pi|B).
It is generally NOT the emission probability times a conditional payload value:
the selection rule and silence themselves reveal information. Example: Y is a
fair bit, emit a constant symbol iffY=1. An observer of emission or silence knows
Y exactly; payload alone is constant. This is an information example, not a
permission to use an unrevealed future target to decide an emission.

Design a causal emit/hold race using current evidence and predicted future utility.
Teach both alternatives on eligible TRAIN histories, including suppressed events.
Otherwise the learner cannot discover valuable events it never sends. Observe
survival/no-event probability under the actual policy and sensor availability.
Silence, dropout and an unobserved stream require different conditioning.
Observation of silence itself has a causal deadline/query cost; it is not an
instantaneous globally available message. Arrival-time precision can carry bits;
charge timing precision and scheduling, not just payload bytes.

For a restricted one-upward-message architecture with source rate lambda_l,
causally averaged emission probability p_l and additional timer/top-down traffic
mu_l, lambda_(l+1)<=p_l lambda_l+mu_l. This relation is accounting, not proof of
quality. Multi-message emissions require their expected multiplicity. Full cost
includes key discovery, selected writes, transport, heads, timers, silent/losing
alternative learning, optimizer and retries. Low traffic with no predictive value
is extinction; high traffic with repeated messages is an activity cascade.

## 7. Why learning gets harder with depth, even with nonzero gradients

On a FIXED declared differentiable branch, lifted Jacobians act on content,
time and persistent-state coordinates. For genuinely contained residual payload
maps J_l=I+E_l with ||E_l||<=epsilon_l<1,

    sigma_min(product J_l)>=product(1-epsilon_l),
    ||product J_l||<=product(1+epsilon_l).

Summable epsilon budgets can bound transport with depth; normalization alone
cannot. These bounds do not certify the complete recurrent state/time Jacobian,
route switches, optimizer steps or long-time stability. For a nonlinear causal
DAG, cotangents sum over all downstream paths, not one depth-chain product.
Shared parameters require summing contributions before any norm/covariance claim.

For an issued fixed differentiable chain define cut-credit error e_l and local
consistency defect r_l. With P_k=J_1^T...J_k^T,

    e_0=P_D e_D + sum_l P_(l-1) r_l.

Anchor e_D to the actual terminal loss. All-zero predicted credits otherwise can
satisfy local consistency. Correlated transported defects reinforce; local MSE
alone misses that reinforcement. A contractive product can shrink true signal
as well as error. Use signal-relative/directional and actual-update utility.
The identity describes factual branch credit; supported counterfactual route,
optional-write and emit/hold contributions must be included separately.

If a fixed contribution G is observed at probability1/D with weightD, its raw
estimator variance is(D-1)||G||^2. A fixed teacher budget therefore dilutes site
coverage and increases noise. Local score-space allocation proxies need not
minimize full parameter or functional update error. An audit horizon shorter
than the time to a useful consequence estimates a different objective; unbiased
weighting cannot repair missing future outcomes.

For a version-bound raw-gradient control variate C_i with true contributions g_i,
issued before selecting K with full-support p_i, the estimator

    g_hat=sum_i C_i + (g_K-C_K)/p_K

has mean sum_i g_i. Its variance is
sum_i ||g_i-C_i||^2/p_i - ||sum_i(g_i-C_i)||^2. At equal per-audit cost, optimal
p_i is proportional to residual norm, with full support retained where required.
With random outcomes use their conditional residual second moments instead.
Variable costs require the constrained allocation derived in the linked notes.
Audit selection must have the recorded conditional propensities for the issued
world; do not select using an unrecorded future outcome. Potential-outcome replay
noise/coupling must match the declared objective. All baseline sums, discovery,
response transport and teacher fitting are paid.
This corrects the specified raw gradient, not arbitrary Adam updates or omitted
long-horizon contributions. Never train C on its own selected outcome before
using it as the issued prediction in that correction.

For L-smooth risk R, ordinary SGD displacement -eta*g_hat, conditional mean g+b,
and covariance Sigma, the descent lemma gives

    E[R(theta-eta*g_hat)]-R(theta)
      <= -eta <g,g+b> + L eta^2/2 (||g+b||^2+tr Sigma).

Even b=0 needs a noise-sensitive step. Clipping/Adam can introduce effective bias,
change direction and retain history; analyze their ACTUAL displacement d with
R(theta+d)-R(theta)<=g dot d+L||d||^2/2 when that smooth model applies. Across hard
route boundaries use replay/probability changes, not a fictitious global Hessian.
A longer bounded horizon reduces truncation only if the relevant consequence
falls inside it; variance and nonstationarity can increase simultaneously.

Stable individual modal decays do not certify a stable closed recurrent system.
In a linear witness z_next=A*z+B*u and u=K*z, the effective transition is A+BK;
A=.5,B=1,K=1 gives growth1.5 per event despite a stable private A. Native next-token
feedback and learned inter-level feedback therefore need lifted-state response
measurements, not pole checks alone. For irregular additive stable decay,
||s(t)||<=exp(-a(t-s0))||s0||+sum_i exp(-a(t-t_i))||b_i||. Arbitrarily many or
large writes can violate any uniform bound despite a>0. Record burst loads,
write magnitudes and feedback gain as well as average rates. On physical queues,
backlog changes what is available by a deadline; service/traffic and tail latency
are part of the architecture's observable behavior.

## 7b. Sparse causal paths have a second, multiplicative support problem

For a particular signal to traverse D emit/hold boundaries, let p_l be the
conditional probability of reaching boundary l+1 given that it reached l,
under the issued history and policy. By the chain rule, its path survival is
product_l p_l; this does NOT assume independent gates. If every conditional
probability is <=p<1, path survival is <=p^D. This is distinct from the linear
teacher-budget dilution above. An event-rate hierarchy is economical precisely
because it suppresses traffic; it can simultaneously suppress the examples that
teach its deepest paths. Measure conditional survival and useful signal delivery,
not only average firing rates. A skip can shorten this path while preserving
sparse private memory; it must deliver task-relevant content, not just activity.

Single-site counterfactuals need not solve a disconnected path. In the finite
Boolean example F=product_l a_l, all a_l=0 and D>=2, every single-bit intervention
leaves F=0. A supported coordinated multi-boundary intervention can reveal a
useful complete path, whereas a local-only infinitesimal or single-toggle signal
cannot. Real temporal paths have graded values and multiple routes, but E83's
zero deepest-only lower-layer effects instantiate the same practical warning.
Provide an initially connected causal route, a protected sparse skip, or a bounded
multi-hop counterfactual proposal with recorded support/cost. Merely increasing
single-site audit frequency cannot repair an identically zero effect.

Longer history is also not enough if the useful prediction is issued before a
message becomes available. On an actual path, each event must precede its causal
consumer; a loss query earlier than final delivery cannot depend on that delivery.
Training on a completed-answer objective cannot certify earlier response quality.
Neither a synchronized step nor fixed inter-level delay is required by this
argument: use actual causal ancestry and delivery times.

## 8. A realizable architecture recommendation

Retain temporal computation/races, separate keys/values, sparse addressed state,
content-memory mixing, alternative-route credit and silence-aware supervision.
Implement a protected parent pathway plus independently stateful temporal
correction branches, with causal queues and explicit query-time combination.
No dense conventional network is substituted for the substrate.

A neutral branch output A*phi has A=0 initially. It may read parent events, but
must not alter parent state, clocks, route-noise stream, old gains, feedback
context, observation availability or optimizer history while claiming containment.
Use a separate owned RNG stream. Only the branch's PRIVATE state can evolve
without affecting the parent. A protected parent answer can be issued without
waiting for the new branch; deadline comparisons then charge which corrections
were actually available. Shared CPU/memory contention can delay the parent even without an explicit join;
no-wait code alone does not certify unchanged physical latency. If the parent is
forced to wait for new work, completed
answer equality does not preserve latency. For a changed function reachable by
new branches, explicitly open versioned feedback edges after the neutral contract.

AtA=0, internal task gradients vanish; new output-map gradients may be live, but
can also vanish by symmetry/uncorrelated features. Use compatible joint learned
temporal operators and the contained cross-credit seeding construction in
THEORY_GUIDED_TEMPORAL_VARIANT_SELECTION.md where justified. Do not double-zero
both feature production and output maps or assume a tiny gradient means a tiny
Adam step. Feature seeding derivative policy and full cost must be declared.

Then add independently learned emit/hold decisions between levels. First preserve
all relevant causal opportunities; learn selective propagation with supported
suppressed-event audits. Higher states can summarize longer histories at lower
communication rates while hazard events propagate immediately. Choose horizon-
specific predictive tasks and useful output heads, with fixed declared weights;
retain the final/global objective so shallow local heads cannot win alone.
Auxiliary objectives change optimization and may conflict: measure their effect
on the shared final outcome. Different communication, memory, prediction and
learning cadences are allowed. No one-to-one event correspondence is required.

## 9. AWS decision procedure: evidence before the next long fit

Do not edit frozen active drivers or launch a new grid from this note. The current
priority remains admitted B10 controls and ongoing language acquisition/repairs.
For the next owned-model depth candidate:

1. Specify target, history, causal observations, event/parameter ownership,
   reference-time residual semantics, paths and complete resource boundary.
2. Implement the protected correction with unchanged existing per-stage gains;
   prove/test prefix predictions, old-state/old-gradient/clock/context/RNG and
   next-update containment. Keep completion equality and deadline equality distinct.
3. Measure live output teacher, feature/output cross-credit, actual post-Adam
   probability displacement, receiver coverage and per-level communication.
4. On a bounded TRAIN panel compare exact/declared factual and alternative credit
   with the candidate: transported error/covariance, relative error, actual-update
   utility and horizon reach. Charge all diagnostic work. Nonzero norms alone fail.
5. Fit one integrated candidate against equally funded parent continuation with
   same data/selection boundary; include inherited acquisition and branch/teacher
   preparation. Frozen parent alone cannot attribute a gain to added depth.
6. If selective propagation is integrated, compare always-on own-model correction
   and selective version with identical capacity/modalities/output service needs.
   Check event suppression, useful rare events, silence/dropout, query latency and
   total work. Exact endpoint pooling is not a substitute for this comparison.
7. Promote by completed quality/work/error evidence and repeat selected seeds;
   reject/repair a candidate with a written mechanism diagnosis. Do not interpret
   a tiny smoke or one member's loss as a verdict on the family.

Prediction: protection should remove initial destructive insertion; adequate
outcome-conditioned credit should reduce variance per useful parameter update;
multirate specialization should reduce redundant deep activity. Each prediction
has a distinct test above. None guarantees all three effects or a frontier win.

## 10. Reading map and established external foundations

Current AWS construction: ../USEFUL_DEPTH_REPAIR.md and its version-bound
optional-depth diagnostics. These are integration/contract evidence, not quality
claims. Read current HANDOFF.md for their latest status rather than freezing a
claim here that a rapidly changing gate is unimplemented.

Internal derivations: TEMPORAL_LAYER_FUNCTION_GEOMETRY_STATISTICS.md (actual native
transducer/geometry); ANCHORED_DEPTH_CREDIT_DEFECTS.md and
CROSS_AUDITED_CONNECTED_DEPTH_RISK.md (connected credit);
DEPTH_ALLOCATED_NATIVE_LANGUAGE_CREDIT.md (implemented allocator and its proxy);
LOCAL_UPDATE_GEOMETRY_AND_USEFUL_DEPTH.md (gain/optimizer/variance);
30_event_state_coalescing_and_reference_inclusion.md (exact affine packets and
nonlinear boundary);34_identity_depth_and_live_teachers.md (containment/latency);
126_depth_diagnosis_integration_decisions.md (historical gain/lineage pitfalls);
THEORY_GUIDED_TEMPORAL_VARIANT_SELECTION.md (neutral saddle, cross-credit seeding);
../architecture/MULTIRATE_INNOVATION_HIERARCHY.md (embodied proposal, not a fit).

External primary sources, attribution rather than a blanket guarantee:
- Residual optimization: He et al., https://arxiv.org/abs/1512.03385 .
- Iterative refinement alongside abstraction: Jastrzebski et al.,
  https://openreview.net/pdf?id=SJa9iHgAZ .
- Compositional feature reuse/sample complexity under a specified generative
  model: https://arxiv.org/abs/2307.02129 .
- Learned multiscale update boundaries: Chung et al.,
  https://arxiv.org/abs/1609.01704 .
- Predictive representation objectives: van den Oord et al.,
  https://arxiv.org/abs/1807.03748 .
- Avoiding constant-feature collapse: Bardes et al.,
  https://arxiv.org/abs/2105.04906 . Variance preservation alone does not certify
  predictive sufficiency or useful temporal abstraction.
- Representation identifiability needs assumptions: Locatello et al.,
  https://arxiv.org/abs/1811.12359 . Disentanglement is not guaranteed by depth.
- Normalization placement changes initialization gradients: Xiong et al.,
  https://arxiv.org/abs/2002.04745 . This supports conditioning design, not a
  theorem that normalization solves native sparse temporal learning.
- Recent LLM evidence supports mixing/compression/refinement rather than a
  universal monotone abstraction ladder: Queipo-de-Llano et al., ICLR2026,
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/1734b19d9afe7d2c7f1154954eaf0d5a-Abstract-Conference.html .

## Verification boundary

`tests/test_causal_asynchronous_depth_theory.py` checks irregular affine packet
state/parameter-gradient parity, nonlinear endpoint insufficiency, informative
silence, exact full-support audit mean/variance and noncommuting shared writes.
Floating-point addition is only approximately associative; mathematical packet
associativity does not promise bitwise schedule parity. These finite witnesses
check formulas and failure examples, not fitted-model quality or the complete
native asynchronous scheduler.
