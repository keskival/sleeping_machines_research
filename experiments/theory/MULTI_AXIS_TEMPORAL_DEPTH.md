# Multiple depth axes in a causal temporal learner

15 October 2026. User direction: hierarchy is not layer count; distinguish sequential decisions, temporal composition and representation transformation. Design synthesis and source audit, not a new measured result or queue admission.

## 1. What the existing results decide

The failed P24 layer ladder establishes that repeated application of its present serial construction does not improve the measured language task. It does not test all forms of hierarchy. The keyed/episodic results establish useful retrieval of older evidence. The race-tree/copy result establishes a promising joint output construction; it does not isolate tree depth, establish hierarchical abstraction or teach downstream consequences of arbitrary unchosen state writes.

The missing design variable is the kind of composition performed, not just how many operators are stacked. Independent roles need independent controls and credit paths. They can share event/race primitives and parameters; a separate neural network per axis is not required. No theorem says every task requires every axis or that simultaneous maximization helps.

## 2. Coordinates and measured meanings

| Role | Coordinate | Actual computation | What is not sufficient evidence |
| --- | --- | --- | --- |
| Causal history | memory span H, lineage/path depth D_H | Influence of past state/writes on a current answer, potentially through retrieval | Long context alone; long elapsed silence with no transformations |
| Representation | K_l local transformations at scale l | Re-encode the available evidence at the same semantic/event scale | Slower clock, more allocated state or extra parameter count |
| Abstraction/composition | A levels, learned cadence rho_l | Build persistent entities/segments from lower-scale events; upper state can condition later lower events | Arbitrary decimation or fewer upper updates |
| Conditional decisions | J_e sequential branch/read choices for event/query e | Later choices depend on earlier outcomes, e.g. retrieve entity then its relation then an attribute | A vocabulary taxonomy by itself |
| Refinement | R_q revisits/iterations for query q | Revisit a provisional representation/decision using additional computation or evidence under an unchanged observation cutoff | More externally observed events or repeated service of the same payload |

Gradient horizon W and delayed consequence horizon C are separate learner controls. They determine which compositions are taught, not which facts exist in persistent state. Store capacity, width, number of scored keys, value fanout and numerical clock precision are also separate from the five roles.

Use a ragged specification K_l, J_(l,e), R_q, not a mandatory rectangular A*K*J*R tensor. One event may make a short decision chain without upward emission; a slow-level event may require substantial local transformation. Record actual causal paths and counts. Effective dependency depth is the longest required chain of dependent operations before an answer, while arithmetic is summed over all executed branches. Neither equals allocated layer count.

A Transformer has distinct position/history and layer coordinates, but a context of T tokens does not imply a T-step sequential computation for each prediction: attention can access earlier positions in one layer. Autoregressive generation adds inter-token dependencies. Similarly, one retrieval can shorten a temporal dependency path without deleting its history. This distinction should carry into native accounting.

## 3. Proposed native factorization

Each node owns persistent addressed state at a scale, with a last-state timestamp and parameter version. An event is admitted by observation provenance and modeled readiness. It transports local state through elapsed time, retrieves relevant past records, performs K_l local transformations, and can execute J conditional races whose later queries depend on prior winners. Each choice can expose a payload without forcing an irreversible state commit.

An independent emission/silence controller decides whether integrated evidence warrants a higher-scale event. This is different from choosing the next decision branch. An upper event can represent a repeated process/entity, a segment, or a latent relation; it need not always be a contiguous nested segment. Separate object-addressed streams can overlap in observation time. A strict nested segment tree is a useful initial embodiment, not the definition of asynchronous hierarchy.

Higher-scale state returns timestamped addressed context to lower nodes. A local query can execute another refinement step using already admitted information, or stop. Each query fixes its observation cutoff, required output schema, stopping/resource bound and version policy. Refinement cannot import the target, backdate a late message or overwrite an issued prediction. Concurrent upper context may legitimately change later predictions. For comparable answers at the same cutoff, specify whether a query waits for dependent work or returns the latest available state.

Timers and silence can create internal events even when no fresh lower event arrives. Their firing times must depend only on available past information. Restricting all upper events to lower-event timestamps is one implementation restriction and should be named. Object composition, within-event decisions and refinement must not accidentally advance physical observation time; modeled computational delay is explicitly separate.

A graph with feedback needs finite service bounds, event/query IDs, local conflict order and exactly-once committed writes. Retry/replay/refinement messages cannot count the same observation as fresh evidence repeatedly. Share the primitive; retain distinct state ownership and action types.

## 4. Why separate learning signals are needed

Abstraction should be useful for future observations or actions, not merely predict its own learned code. Predicting the next lower-level concept and enforcing concept usage can produce balanced but irrelevant codes. A slower emission rate can produce a segmentation with no abstraction. Local next-token heads can encourage every scale to duplicate the fast state.

Attach causal, horizon-specific predictive targets to issued upper states, using later revealed base-stream observations and exact target definitions. Train upper representations on future blocks/relations or residual predictive value beyond a lower-only predictor, while retaining global forecast loss. The residual target depends on the actual lower predictor and its version; it is not an instruction to discard all repeated information. Stopped references versus joint optimization need explicit gradient ownership. Event identity supervision that is unavailable to competitors remains forbidden.

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

Scale the axis whose intervention provides useful prediction at its paid cost. The objective is an integrated learner that can grow along independently useful roles; monotonically increasing one scalar layer count is not a sufficient hierarchy criterion.

## Sources and continuity

- [Universal Transformers](https://arxiv.org/abs/1807.03819): iterative processing distinct from input positions.
- [Hierarchical Multiscale RNNs](https://arxiv.org/abs/1609.01704): learned boundaries/multiscale updates.
- [MEGABYTE](https://arxiv.org/abs/2305.07185): separate local/global processing; cited for factorization, not permission to replace required subword tokenization.
- Existing native notes: DEPTH_PLAN_C_ASYNC_HIERARCHY.md; CAUSAL_ASYNCHRONOUS_DEPTH_SYNTHESIS.md; CREDIT_HORIZON_AND_ADDRESSED_MEMORY.md; MULTIHORIZON_FORWARD_TAUGHT_CREDIT.md; report/model_family_composition.md.

The proposed factorization and next tests are engineering hypotheses. No trained multi-axis hierarchy win or new hardware result is claimed. This note admits no new job and leaves active source-bound runs intact.
