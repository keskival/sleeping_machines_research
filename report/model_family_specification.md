# A compact specification of the model family

[Overview](model_family_overview.md) · [Design choices](model_family_design.md) ·
[Complete examples](model_family_members.md) · [Composition](model_family_composition.md)

**Definition.** Relative to a declared operator library, Sleeping Machines
comprises causal networks of stateful temporal programs composed by typed
events, state-access interfaces and explicit scheduling/learning contracts.
A member can mix selective and full-support regions, local and barrier-based
schedules, and statistical and learned state. Its actual operators and support
determine its capabilities. The integrated research target combines
computational time, hard selective updates, small learned messages, separate
selection/value roles and useful credit to unrealized consequences.

This document specifies membership and compatibility. The design guide supplies
options and reasons to choose them; the evidence map supplies the status of
claims. A specification can be meaningful before its implementation succeeds.

## 1. A member has twelve declared fields

Write M=(E,O,G,S,I,P,T,Q,L,U,B,X). These are semantic fields, not a required
software API or a claim that all existing modules use one configuration format.

| Field | Required meaning |
| --- | --- |
| E: evidence | Observation schema, available-information boundary, time/address adapters and allowed histories |
| O: operators | Actual local maps and numerical domains, including precision/rounding when numerical identity is claimed |
| G: composition | Units/modules, admissible connections, shared paths and any structural-action space |
| S: operational state | Persistent facts, timestamps, pending work, caches and random state; allocation, ownership and reset |
| I: interfaces | Payload/address/time schemas, state read/write permissions, projections and explicit conversions |
| P: participation | Discovery, scoring, selected deliveries/writes, reception/aggregation and alternative support |
| T: scheduling | Causal ordering, modeled-time conversions, delays, ties, deadlines, joins, cancellations and barriers |
| Q: queries | Observation cutoff, accessible snapshot, completion and readout rules |
| L: objectives | Task/self-supervision, observed outcomes, scoring order, and any resource penalty |
| U: learner | Credit estimator/horizon, parameter and optimizer ownership, versioning and update order; may be disabled |
| B: bounds | Admitted histories, finite execution or stopping contract, state/queue/work limits and overflow behavior |
| X: execution/evidence | Reference versus deployed realization, claimed equivalences, measured/projected costs and validation status |

Learned parameters theta and initial operational/optimizer state instantiate
the specification. Changing a dimension, connection or policy can define a
different member; a declared structural policy can make such changes inside
one member. Fixed policies are included cases of that policy space.

## 2. Operational meaning

Distinguish observed availability/order from modeled computation time and
physical runtime. An event has a typed source/destination, content, an
observation provenance/cutoff and a modeled readiness coordinate. Provenance
may be implicit in a serialized implementation; its causal meaning is required.
A timer or learning event need not contain a new external observation.

For a fixed member and current parameters, its admitted transition is

    Advance_M(state, admitted event or query, assigned randomness)
        → next state, finite boundary trace, accounting record

The trace contains the declared messages, state-visible effects and query
outputs. Internal programs can perform flow, proposals, discovery, selection,
integration, commits or readout in the order their specification declares.
The external observation boundary controls admissible information even when
modeled computations finish later. State/parameters and queues are read at
declared versions. An update event applies U using the permitted objective and
retained credit; it need not reproduce a dense full-sequence gradient.

**Membership requires well-defined causal transitions**, compatible interfaces,
declared state/update ownership and an execution/admission bound. These are
semantic obligations: listing all twelve field names does not prove them.
An implementation is conformant only within its validated input/numerical
domain. A timestamped wrapper around an existing program can be an endpoint;
the wrapper alone supplies no new efficiency or learning result.

## 3. Construction and closure

The allowed construction grammar is:

    program := a declared local state transition from O
    member  := program | serial(member, member) | parallel(members, fusion)
               | recurrent(member, feedback, bounds)
               | memory_link(member, store, read/write contract)
               | region(member, participation, schedule)
               | controlled(members/actions, policy, migration)

This is a specification grammar, not an executable parser. Constructors retain
the fields above and must satisfy their compatibility obligations. `region`
allows dense support or synchronous barriers as well as selective/event-driven
policies. `controlled` includes event-level choices and separately declared
structural choices; structural changes require a state/version mapping or a
deliberate reset. Each local program must state its real operators; a reference
attention block is not silently identified with a winner-only temporal unit.

For finitely many admissible components with compatible interfaces and a
terminating composed schedule, their product state and declared communication
define a member of the same envelope. This is **conditional closure**. Finite
components connected by unbounded feedback do not automatically terminate.
Composition does not automatically preserve local expressivity, optimization
quality or sparsity; the composition guide states the relevant conditions.

The library-relative definition makes breadth explicit. Adding operators
extends the envelope. An incumbent containment must identify the library and
construction that reproduce its transition/query semantics, then separately
establish learning and resource claims. Computational universality alone is
not the project's distinguishing result.

## 4. Compatibility rules for actual choices

| Combination / boundary | Required rule | Consequence when absent |
| --- | --- | --- |
| Different widths, state types or addresses | An explicit map/lookup and declared information loss; matching shape alone is insufficient | A legal tensor operation can erase a necessary distinction |
| Observation clocks and modeled timers | Units/origins/order adapters and nonnegative causal readiness; queries preserve observed cutoff | Age may be meaningless or later evidence may leak |
| Parallel values and a join | Completion/timeout and transport policy; specify which pending branches can be inspected | Latest-arrival joins may block or change earlier content |
| Multiple writers / shared state | Ownership, atomicity/order, or a valid commutative reduction | Execution order changes facts and predictions |
| Private facts and cross-source queries | An explicit read, delivery, fusion or retrieval path | Shared weights cannot make inaccessible facts visible |
| Dense island in a selective graph | Required full aggregation plus entry/exit scheduling and accounting | More candidates alone cannot emulate dense attention |
| Synchronous region in an event graph | Read-old/write-new snapshot, barrier and surrounding causal admission | Equal timestamps can still execute sequentially |
| Hard race and continuous credit | Interior clock/content derivatives plus a stated choice/outcome estimator | Fixed-history gradients omit unrealized consequences |
| Learned window / silence reception | Boundary/option utility, timer ties and pending-query semantics | Membership changes or absence are uncredited or noncausal |
| Sparse inference and sparse learning | Explicit key, proposal, replay, gradient and optimizer supports | One selected write can conceal dense training work |
| Online updates and stored credit/caches | Producing versions, refresh/replay/stale policy and scoring-before-label order | Old derivatives may be mistaken for current ones |
| Coalesced input and nonlinear/routed state | A sufficient descriptor and closure proof, or a declared approximation | Packet summaries can destroy order or interaction |
| Growth/pruning and persistent state | Migration/reset, queued-event identity and update ownership; preservation contract if claimed | Equal current outputs can hide different future behavior |
| Recurrence and asynchronous delays | Event/depth bound or another termination argument and overflow policy | Positive delays alone permit finite-time accumulation |

Two choices can be compatible for forward inference yet incompatible with a
claimed learning equivalence. Likewise they can be semantically compatible
but too costly for a proposed deployment. Record which contract is being tested.

## 5. Profiles and claims

Describe a member by a **mechanism profile**, not a single dense/sparse label:
temporal flow/races/reception; state and parameter capacity; discovery support;
deliveries/writes; key/value roles; factual/alternative credit; persistence;
scheduling; adaptation and structural control. Profiles are orthogonal and
are not a ranking of model quality. Dense and synchronous settings are permitted
endpoints on specified axes, with actual aggregation and barriers.

The target profile preserves the core temporal/selective mechanisms. A model
can partially instantiate that target; missing depth, writer utility, silence
or discovery coverage must remain named. Statistical endpoints and dense
controls help diagnose it. Successful controls do not substitute for an
integrated demonstration.

Keep four claim levels visible: **specified capability**, **implemented and
contract-checked**, **completed empirical evidence**, and **aspirational
integration**. A derivation supplies its stated conditions, not a fitted score.
An implemented path can still lack trained parity or economic evidence.
The [complete examples](model_family_members.md) use this same specification;
the [evidence map](architecture_evidence.md) locates actual project results.

## 6. An adaptive member still owes a prediction at each declared query

The evaluation protocol fixes the admitted observations and scored queries
independently of learned routes, clocks and structural actions. A member can
choose how much work to perform, but cannot silently choose which targets enter
an ordinary accuracy or log-loss denominator. If selective prediction is the
task, declare its coverage/risk objective and compare at the same coverage.
This is a family-level comparison contract, not a finding that the saved native
language evaluations omitted difficult targets.

| Adaptive choice | Prediction / learning obligation | Resource obligation |
| --- | --- | --- |
| Slow route or unfinished branch | State the accessible query snapshot and fallback, or declare the run invalid; never silently drop the target | Charge attempted work, pending state and any permitted wait |
| Learned stopping or optional depth | Score every declared target with the actual stopped readout | Charge discovery and executed continuations, including rejected proposals |
| Silence timeout or no emitted message | Distinguish an internal absence from an absent system prediction; specify how the query reads it | Charge timers, retained state, cancellation and readout |
| Online learning or changing structure | Score before consuming that outcome; preserve the declared reset, feedback and version rules | Charge credit, replay, migration and optimizer work |
| Stochastic routes | Declare the seed/repeat or expectation protocol; keep outcomes and costs from the same realization | Report resource distribution/tails when deadlines or budgets matter |

A modeled readiness coordinate is not elapsed hardware time. A deadline must
identify which clock it constrains; hardware latency requires measurement on
the stated realization. Reducing a modeled delay does not by itself establish
faster wall time. Work and quality should refer to the same admitted workload,
with whole-fit and per-target denominators shown together.

For cost-sensitive route learning, an alternative's utility includes the
declared query consequences, fallback/lateness policy and complete incremental
work. A local value teacher can improve the current delivered content without
estimating this system utility. Widening the counterfactual pool only helps
when its estimator teaches the relevant consequences. See the
[comparison and credit derivation](../experiments/theory/152_primitives_integration_and_capability_bounds.md#19-adaptive-execution-needs-a-fixed-query-and-utility-contract).
