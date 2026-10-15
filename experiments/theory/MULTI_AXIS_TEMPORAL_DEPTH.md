# Multiple depth axes in a causal temporal learner

15 October 2026. User direction: hierarchy is not layer count; distinguish sequential decisions, temporal composition and representation transformation. Design synthesis and source audit, not a new measured result or queue admission.

## 1. What the existing results decide

The failed P24 layer ladder establishes that repeated application of its present serial construction does not improve the measured language task. It does not test all forms of hierarchy. The keyed/episodic results establish useful retrieval of older evidence. The race-tree/copy result establishes a promising joint output construction; it does not isolate tree depth, establish hierarchical abstraction or teach downstream consequences of arbitrary unchosen state writes.

The missing design variable is the kind of composition performed, not just how many operators are stacked. Independent roles need independent controls and credit paths. They can share event/race primitives and parameters; a separate neural network per axis is not required. No theorem says every task requires every axis or that simultaneous maximization helps.

## 2. A taxonomy of depth, rate and meaning

These coordinates describe different properties, not fully independent physical quantities. Architectural controls can separate them while their learned consequences interact.

| Property | Measurement | Native interpretation and important distinction |
| --- | --- | --- |
| Historical span | Observation/event lag and wall-time lag of useful evidence | Durable addressed memory; long elapsed time alone is not many computations |
| Sequence/causal depth | Longest chain of required writes, decisions and reads | Stateful event lineage; direct retrieval can bypass intervening computation |
| Transformation depth | Nonlinear map applications at a fixed scale | Several local blocks, or a reusable local processor, inside one module update |
| Decision depth | Dependent branch/read choices | Tree/cascade, retrieve entity then relation; count candidate scoring separately |
| Relational composition | Consecutive bindings or joins needed for a query | A relation between relations can require several reads at the same event rate |
| Abstraction/semantic granularity | Which distinctions support transfer, prediction and control | Entity/process/situation representations; vector coordinates and norms do not order abstraction |
| Temporal scale/rate | Separate input, write, emission and read/query rates | Slow integration can condition frequent outputs; rates need not decrease monotonically along semantic links |
| Iterative refinement | Revisions at a fixed observation cutoff | Additional internal events or recurrent steps before halting |
| Receptive-field/compression depth | How many inputs or segments a representation combines | Pooling/segmentation; larger coverage is not automatically useful abstraction |
| Breadth/hypothesis multiplicity | Concurrent active evidence streams or alternatives | Parallel races and protected independent state; not serial depth |
| Learning/adaptation depth | Consecutive parameter/optimizer updates influencing later forecasts | Online learner trajectories; distinct from inference and factual gradient horizon |
| Physical dependency depth | Required serial operations, joins and communication hops | Actual critical path under a stated hardware model; not CPU time or parameter count |

Store capacity, message width, clock precision and candidate discovery are separate resource variables. Credit horizon W and delayed-consequence horizon C determine which dependencies learning teaches. Neither specifies numerical state lifetime.

### Four rates, plus the memory's own dynamics

For module m record r_in, r_write, r_emit, r_read, indexed by physical time and, where useful, by base observations. Also record state decay/rotation spectra and internally triggered event rate. A module can ingest 1000 events/s, commit an aggregated state 10 times/s, notify subscribers twice/s, and answer 1000 queries/s from that state. Those reads and their transport are paid work. A sparse emission rate does not prove cheap ingestion or cheap readout. A quiet module may evolve continuously by a cheap closed-form transport; silence is not necessarily a frozen state.

The hazard of a candidate computational timer is not automatically a semantic event rate. At fixed scores, multiplying all race rates changes waiting time while preserving winner probabilities. The system's actual rate additionally depends on arrivals, state, gating, resets and service rules. Do not conflate the race's summed hazard with the sensor's observation rate, learned boundary frequency or information rate.

A message's information depends on content and predictable timing, conditioned on what its receiver already knows. Event count times payload width is a traffic estimate, not a measure of novel information. Silence can carry information only under a known observation/delivery opportunity and timing model; a delayed packet cannot be silently treated as intentional non-emission.

### What abstraction means in vector states

There is no invariant notion that coordinate 70 is more abstract than coordinate 7. An invertible change of basis can preserve the same represented functions. Larger vector norms, low frequencies, a discrete concept index or greater distance from the input do not certify abstraction.

A useful higher representation pools variable instances while retaining distinctions needed for later predictions and actions. Test invariance to declared nuisance changes, sensitivity to consequential changes, compositional reuse, transfer to new instances and the accessibility of future-relevant relations. Its semantics may be distributed across slots and read operations, not a single vector. A fixed-size embedding can act as a query/program addressing richer persistent structure rather than compressing every fact into itself.

For a deterministic transformation of a complete state, the data-processing inequality forbids creating new Shannon information about a target. Transformations can nevertheless make a relation accessible to an economical decoder, combine separately accessible records, or preserve information against a later bottleneck. Compare practical predictors and their paid work, not an unsupported claim that each layer increases mutual information.

Abstraction and rate are related inductive biases, not synonyms. A machine's identity can be concrete and nearly static; a high-level conclusion such as a failure hypothesis can change on one sudden event. A hierarchy may therefore have fast abstract alarms, slow concrete calibration and several entities at one scale. The general ordering is a typed graph or partial order, not a single vertical stack.

### What tree depth actually buys

A balanced B-way path over V labels requires approximately ceil(log_B V) dependent rounds, but each round can require B scores and a winner reduction. For four 16-way rounds: 64 child scores, and 60 scalar comparisons if each winner is computed by a 15-comparison maximum. A parallel tournament can reduce each round's comparison span to four binary-comparison stages; query refinement still makes rounds dependent. Dot products, normalization, copy lookup, memory traffic and backward work remain additional.

A tree changes a conditional computation graph; it does not necessarily learn an abstraction hierarchy. A fixed vocabulary tree can be useful purely as a factorization. A decision chain can also select successive memory joins without changing abstraction scale. Parallel independent comparisons reduce dependency span by spending breadth; they do not remove total work or substitute for decisions whose inputs depend on preceding results.

A Transformer has position/history and layer coordinates, but a T-token context does not imply T sequential operations within a prediction. Autoregressive generation, attention paths, local transforms and recurrent refinement have different dependency structures. The same separation is required for our model.

## 3. Projection into the model family: a typed graph of persistent modules

The family already permits serial, parallel, recurrent and memory-linked modules. The design hypothesis is to expose the different roles through that interface, not invent one separate architecture per axis. A module owns protected facts, decaying/rotating state, addressed records, a timestamp, pending work and parameter versions. It can execute several local transforms, retrieve records, perform conditional choices, emit to peers or another scale, refine a query, or remain silent. These actions need distinct control and credit even if they share maps or race hardware.

### 3.1 Parallel races within an evidence stream

Use separate race groups when multiple complementary things must remain simultaneously represented: machine identity and operating regime; syntax and referent; recency and periodicity; local detail and long-range context. A winner within one group must not cancel an unrelated group's pending evidence. Per-group null/silence routes allow optional participation. Several independent groups may emit to a join or to separate consumers.

Use a single competitive race when alternatives really are mutually exclusive under the declared model, such as the next branch of one decision or the next event of an explicitly modeled competing-risk process. A global earliest winner is not a general fusion rule. If independent candidate timers in group g have total fixed rate Lambda_g, a global winner comes from g with probability Lambda_g/sum_h Lambda_h. A large group or large score offset can dominate without making its information more useful. Conditional within-group routing plus a separate admission/work controller can decouple representation competition from overall activation frequency. Group normalization or quotas change the model and must be learned/declared, not assumed exact repairs.

Parallelism can represent complementary factors or alternative hypotheses. Those roles differ: complementary evidence needs integration; alternative hypotheses may retain a normalized posterior mixture. Shared input events do not become independent evidence because several modules encoded them. Keep event provenance and learn interactions rather than multiplying duplicated likelihoods as if independent.

### 3.2 Mixtures, superpositions and feature fusion are different operators

- A probability mixture p(y|H)=sum_g pi_g(H)*p_g(y|H) chooses among complete predictive laws. It is normalized when its components and nonnegative weights are normalized. Exact mixture scoring evaluates every retained component, unless an exact sparse factorization is provided. Sampling one component produces a stochastic alternative, not the exact mixture log-likelihood.
- A competing-risk superposition has conditional marked intensity lambda_k(t|H)=sum_g lambda_(g,k)(t|H), with survival exp(-integral sum_g lambda_g). That is a different timing model. Product survival holds for conditionally independent clock construction; arbitrary interacting state does not inherit independent timer laws automatically.
- Feature fusion combines complementary messages in compatible typed spaces: a learned projection, concatenation, gated residual or an explicit relation/join. Averaging unrelated semantic vectors has no default interpretation. Gates selecting expert modules, gates selecting evidence, and probability mixture weights have different meanings.

Thus the answer to mixtures is conditional: mix alternative explanations or predictions; combine complementary evidence; race mutually exclusive actions. The same primitive can implement these with declared semantics, but one generic gate cannot be assumed to do all three correctly.

### 3.3 Lower-rate modules should condition higher-rate computation

Yes: a lower-rate representation can be read by fast consumers and can send sparse downward or lateral corrections. Let a slow module commit (t_s,z_s,v_s). At a later fast query t, its context can be c_s(t)=D_s(Phi_s(t,t_s;v_s)z_s, age, validity, query). A fast readout combines its own state with this context. The decoder may execute per query even while the slow state is unchanged; its cost is accounted separately from slow writes.

Choose push notification when an upper state changes, pull retrieval on a fast query, or a learned hybrid with cached projected context. A common upper state can condition many fast observations without replaying the upper inference each time. Learned transport can make that context evolve during silence. A periodic/timer-triggered correction can occur without a new raw event, using only past evidence. Do not overwrite a prediction already issued with later information.

For additive fixed linear transport, timestamped increments can be merged at a common reference: (t1,b1) merge (t2,b2)=(t2,Phi(t2,t1)b1+b2). Nonlinear gates, intermediate emissions and changing versions generally require more than a sum. Sufficient statistics must preserve what the next consumer needs. A mean of bare type embeddings and a histogram cannot distinguish AB from BA; ordered contextual encoders or additional sequence statistics may. This matters when coarsening evidence before a higher relation is learned.

Upsampling is conditional generation/readout, not recovery of discarded detail. Preserve an addressed fast path so slow context helps interpret detailed evidence instead of replacing it. Memory retrieval can cross scales in one hop. A useful abstract message might set priors, specify an address, modulate a route, predict a boundary or request additional evidence; it need not be another token-like payload.

### 3.4 Independent transformations, decisions and refinement inside each module

Use ragged controls K_m for local transformations, J_(m,e) for dependent decisions, and R_q for refinement. There is no need to execute an A*K*J*R tensor or populate every combination. A slow state can require many local steps when it changes; a fast prediction can need no refinement. Memory span H, gradient W, consequence C, fanout and candidate capacity remain explicit.

Separate object/process streams can overlap rather than nesting in a segment tree. Bottom-up composition and top-down context form a graph with cycles. Prevent zero-time self-excitation and duplicate evidence by event/query IDs, local conflict ordering, exactly-once committed writes, bounded refinement and declared stopping. Clock advancement is not automatically evidence that a new observation became available. Query cutoffs and service deadlines remain distinct.

### 3.5 Concrete first construction and architectural rationale

Failure addressed: current stacks make every stage process the token-rate stream, while plan C couples upper concept decisions with coarsening and reconstructs lane-local state. An upper module can therefore be inactive, redundant or insufficiently trained without telling us which role failed.

Start from the strongest acquired keyed temporal-memory member. Preserve its fast predictive path and useful retrieval. Add a small number of persistent, separately addressed evidence streams, initially two with learned rates rather than a fixed required semantic meaning. Each stream keeps an order-sensitive segment summary plus a protected fact component and time-evolving component. It has separate write/emission control, configurable local transformations and per-stream retrieval. Slow state enters fast predictions through a query-conditioned gate with recorded age/availability. Query decisions can fetch across streams without requiring another abstraction level. The output race tree remains a separately selectable head, not evidence of hidden abstraction.

Two streams are an initial engineering choice to isolate complementarity, not a theorem or new training admission. A return to a single stream remains valid if evidence shows it sufficient. Counterfactual credit must cover losing route, emit/silence and optional-state actions where integrated; maintain exact/read-time credit for continuous accesses. Straight-through alternatives remain labelled as such. Do not require groups to be statistically independent or artificially force equal rates.

Implement persistent per-stream carry and pending-summary state, then source-bound numerical contracts for chunked/unchunked causality, ownership, independent races, duplicate handling, gated-parent behavior and credit support inside R1's pipeline. Initialize new corrections without destructively changing acquired predictions, while checking actual internal gradient activation; a zero gate is not proof the new internals learn immediately. No new long fit is admitted before the capability/resource declaration and integrated contracts.

## 4. Learning and evidence for each role

Abstraction should be useful for future observations or actions, not merely predict its own learned code. Predicting the next lower-level concept and enforcing concept usage can produce balanced but irrelevant codes. A slower emission rate can produce a segmentation with no abstraction. Local next-token heads can encourage every scale to duplicate the fast state.

Use future targets that are fixed independently of the emission policy; otherwise a model can improve a score by declining to emit or selecting easy target positions. Account for silence and all required scored positions. Attach causal, horizon-specific predictive targets to issued upper states, using later revealed base-stream observations and exact target definitions. Train upper representations on future blocks/relations or residual predictive value beyond a lower-only predictor, while retaining global forecast loss. The residual target depends on the actual lower predictor and its version; it is not an instruction to discard all repeated information. Stopped references versus joint optimization need explicit gradient ownership. Event identity supervision that is unavailable to competitors remains forbidden.

Representation transformations learn task-relevant maps at their own scale. Decision credit learns consequences of selecting a read/branch. Emission credit must compare emit versus silence including subsequent persistent state and forecast effects. Refinement/halting credit compares incremental future quality with paid compute/delay. Intermediate predictions before the decision cannot serve as evidence that the subsequent decision helped. Read-time credit can reach an old stored address without a long recurrent backward path, but does not reconstruct absent writer credit.

A finite sequence of normalized branch probabilities gives an exact target-path likelihood; its sibling gradients supervise branch probabilities. It does not enumerate arbitrary unchosen downstream states, include their future utility or supply gradients through hard address membership. Those require supported alternative actions or an explicitly labelled surrogate. Straight-through routes in current plan C are surrogate credit, not integrated counterfactual consequence learning.

## 5. Current plan-C implementation audit

Source: `experiments/r1_async_hierarchy_lm.py`, inspected 15 October.

- `--base` and `--levels` already separate token-level transformation count and temporal coarsening. This is a start toward multiple axes.
- Each upper Level contains one Block. Local transformation depth at each upper scale is not independently configurable.
- One hard concept choice is made per level event. It is coupled to upward composition rather than an independent within-scale sequence of conditional decisions.
- No per-query recurrent refinement/halting interface is present.
- The `states(x)` path rebuilds finite lane-local event streams. It does not accept/return persistent per-level numerical carry, pending segments or clocks across training lanes. This blocks interpreting slower cadence as document-scale persistent hierarchy.
- Emissions are restricted to lower-event times. This excludes independent silent-time timers from this embodiment.
- Emission and concept choices use straight-through estimators; sparsemax reads score candidate keys before selecting nonzero weights. Nonzero-weight sparsity is not free key discovery.
- Auxiliary targets mainly predict lower emissions/concepts and repeated token losses. They do not independently demonstrate upper-scale future predictive abstraction.
- Compressed-event memory is promising, but segmentation matrices, per-token held context and deep readouts still incur work. Geometrically fewer emitted events do not establish geometrically bounded total work with arbitrary per-event cost, attention or mandatory top-down broadcasts.

Preserve source-bound queued/running jobs. These are limitations of this implementation, not limits of the model family and not reasons to abandon independent timing or persistent state. The next integrated construction should expose upper local blocks and per-level carry before scaling claimed temporal abstraction; sequence decisions and query refinement can then be implemented as separately measurable actions.

## 6. Decisive test design, not a Cartesian sweep

First add a path/cadence/credit trace to the selected construction and test each claimed causal contract inside its battle pipeline. On the existing token task, retain the strongest acquired member and common scoring. Keep proper GPT-2 tokens and CPU execution. Avoid a new synthetic benchmark or external-model fit.

Use diagnosis-led interventions: (a) reset only slow state at credit boundaries versus keep it, holding truncation unchanged; (b) replace a selected higher representation with matched mean/control context and evaluate horizon-specific deltas, with retrained/comparable lower references where needed for achievable-value claims; (c) fix decision budget and vary local representation work for one selected construction; (d) hold abstraction and representation constant while testing a learned extra decision/refinement. An isolated output tree gain is not an upper-representation gain. Selected manipulations are diagnostics, not the strongest-model competitive verdict.

Report pre-update prediction quality, history horizon, per-level event counts, local transform counts, conditional decision length, refinement rounds, trained credit horizons, complete arithmetic/special functions, discovered keys, state/message traffic, synchronization and causal dependency depth. Charge dense control-plane scans and all learning. Cost can be written W=sum_l N_l*C_l + W_queries + W_discovery + W_learning + W_coordination, where C_l includes actual K/J/R work; rates alone do not determine it.

Distinguish the value of persistent slow state, independent parallel streams, slow-to-fast context, local transformations and decision/refinement by successive diagnosis-led interventions rather than a full combinatorial sweep. Include fast-only and unchanged-parent saved references, but field the strongest integrated construction for competitive evaluation. Scale the axis whose intervention provides useful prediction at its paid cost. The objective is an integrated learner that can grow along independently useful roles; monotonically increasing one scalar layer count is not a sufficient hierarchy criterion.

## Sources and continuity

- [Clockwork RNN](https://arxiv.org/abs/1402.3511): modules with distinct prescribed update rates.
- [Recurrent Independent Mechanisms](https://arxiv.org/abs/1909.10893): selectively updated modules with sparse communication.
- [Sparsely gated mixture of experts](https://arxiv.org/abs/1701.06538): conditional capacity; gating alone is not semantic abstraction.
- [Perceiver IO](https://arxiv.org/abs/2107.14795): query-driven interfaces separate latent state and output queries.
- [Universal Transformers](https://arxiv.org/abs/1807.03819): iterative processing distinct from input positions.
- [Hierarchical Multiscale RNNs](https://arxiv.org/abs/1609.01704): learned boundaries/multiscale updates.
- [MEGABYTE](https://arxiv.org/abs/2305.07185): separate local/global processing; cited for factorization, not permission to replace required subword tokenization.
- Existing native notes: DEPTH_PLAN_C_ASYNC_HIERARCHY.md; CAUSAL_ASYNCHRONOUS_DEPTH_SYNTHESIS.md; CREDIT_HORIZON_AND_ADDRESSED_MEMORY.md; MULTIHORIZON_FORWARD_TAUGHT_CREDIT.md; report/model_family_composition.md.

The proposed factorization and next tests are engineering hypotheses. No trained multi-axis hierarchy win or new hardware result is claimed. This note admits no new job and leaves active source-bound runs intact.
