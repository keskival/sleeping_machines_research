# Sleeping Machines: the family at a glance

Sleeping Machines studies **networks of stateful temporal programs**. A message
carries content, an arrival time and an address. A receiver consults local
evidence, transforms it, chooses what to retain or communicate, and may emit
later. Messages and persistent state compose these programs into layers,
graphs and learning systems. Time can perform an operation rather than merely
label an observation.

The research target combines learned small messages, computational delays and
races, sparse addressed state changes, separate selection/value roles, and
credit to alternatives that did not execute. Useful stored capacity should
grow beyond expensive selected work. A larger state bank still costs discovery,
storage and learning; we measure those separately.

The enduring aim is a **trainable computing substrate with useful capacity
beyond expensive activity**. The architecture is the organization of temporal
programs, evidence and credit; individual benchmark recipes instantiate it.
Progress means better useful prediction and learning per total resource, while
retaining the integrated mechanisms that motivate this family.

![The model-family landscape](figures/architecture_landscape.svg)

## What makes it a family?

The common **content/time/address/state interface** lets us vary the local
program, routing, reception, memory, composition and learning policy without
requiring one uniform layer recipe. A particular member specifies its actual
operators and contracts. The breadth is an architectural design space; the
integrated temporal/selective target is the direction we want to make efficient.

The envelope, an individual member and the integrated target are distinct.
The same interface permits different local programs and mixtures per region;
each member's actual support and operators determine its capabilities. The
[composition guide](model_family_composition.md) explains how those local
choices carry through to the whole system. Properties must survive interfaces;
collecting capable units alone does not establish a capable composition.

| Level | Computational power and work | Representation | Trainability |
| --- | --- | --- | --- |
| Atoms | Delay, first/latest arrival, flow, gates and statistics | Values, order, intervals, phase and repeated evidence | Continuous sensitivities and event-boundary utility |
| Units | Local transformation, state update and emission | What a local program remembers and distinguishes | Content, timing, state and parameter credit |
| Modules | Search, selection, retrieval and reception | Which evidence is accessible and can meet | Candidate support, alternative utility and exposure |
| Layers / interactions | Mixing, shared paths and temporal joins | Relations across messages, programs and sources | Coupled derivatives and discrete consequences |
| Stacks | Depth, recurrence and optional continuations | Hierarchy and persistent multi-timescale context | Conditioning, useful depth and writer credit horizon |
| Compositions | Neural state plus statistics, history or dense interaction | Abstraction and direct evidence together | Responsibilities, retrieval and future write utility |
| Systems | Scheduling, learning, batching and serving | Capability within actual information/resource limits | Objectives, update consistency and full cost |

**Synchronous schedules and dense participation are included endpoints.** A
synchronous region imposes snapshot/commit barriers; a dense region recruits
every required operator/value with the proper aggregation. Other regions can
remain selective and event-driven. No global periodic update tick is required
by the general semantics. Local timers, joins and causal order remain necessary.

The closest category is **hybrid dynamical event-processing with configurable
stateful conditional computation**. It intersects recurrence, SSMs, attention,
MoE, statistics, retrieval and event/spiking systems. Its useful contribution
must come from the integrated construction and its quality/resource behavior.

## Match rich functions where needed, remain economical elsewhere

A supported filter, causal estimator, gated recurrent cell or attention block
can be reproduced when the required local operators, information, precision,
state and scheduling exist. With sufficient compatible resources, finite
reference computations have an event-graph emulation path. A fixed winner-only
or narrow-state member has more restrictions than that broad envelope.

We can vary width, accessible memory, delivery fan-in, depth and local program
per layer. Expand at the stage that loses required information or useful
credit; downstream width cannot recover erased facts. Restoring a reference
function, learning it successfully and running it more cheaply are separate
contracts. This gives a constructive route toward a better attainable frontier.

## The same structure can learn and adapt

Online learning and TTT can update the deployed units' maps, clocks and routes
from observed causal outcomes or self-supervision. State/count adaptation and
gradient-based parameter adaptation are distinct. Independent streams can also
be batched. Alternative credit, optimizer work, historical versions and shared
parameter coordination belong in the learning budget.

An earlier causal neural pilot improved prediction through online adaptation
at substantially higher processing work; the evidence map preserves its scope.
Integrated native TTT, drift retention and fully asynchronous training remain
separate demonstrations.

The further ambition is **automatic design within the family**: data can teach
operator, reception, optional-depth and state-allocation choices. Learned
event routes already adapt execution inside a fixed pool/stack. Broader graph
growth, pruning and morphing require state migration and future task/resource
credit. NAS, network morphism and elastic networks provide relevant precedents.

Dense frames, language, irregular events and queries should ultimately update
one shared world state through explicit information paths. Input density,
computation density, arrival cadence and physical clocking are independent.
Current clocked emulators do not establish clockless hardware energy gains.

Adaptive computation still owes a prediction at every declared query. Routes,
delays and stopping can change the internal work, while the comparison fixes
the observation boundary, scored targets and treatment of unfinished work.
Quality and resource use must refer to that same workload; selective prediction
requires its own stated coverage/risk protocol.

## A short reading path

Read this overview → the formal core → the complete examples → the design
and composition guides. Use the architectural review and inventory when you
want implementation detail; use the evidence map when assessing a claim.

| Purpose | Document |
| --- | --- |
| Identify the defining fields and compatible constructions | [Compact formal core](model_family_specification.md) |
| See contrasting complete members under one specification | [Three complete examples](model_family_members.md) |
| Choose a member and understand the design rationale | [Design choices and family positioning](model_family_design.md) |
| Understand how local choices become system capabilities | [Composition and preservation guide](model_family_composition.md) |
| Explore how different reception policies compute | [Interactive atlas](architecture_atlas.html) and [worked example](model_family_example.md) |
| Inspect native units, layers and wider integrations | [Detailed architectural review](architecture_review.md) |
| Check precisely what a claim rests on | [Evidence and capability map](architecture_evidence.md) |
| Inspect conditions and bounds | [Theory 152](../experiments/theory/152_primitives_integration_and_capability_bounds.md) |
| Find implementation branches and source records | [Family inventory](model_family_inventory.md) and [source navigation](architecture_source_inventory.md) |

The progression is manifesto → constructions and tests → family definition →
integrated validation and scaling. The evidence map distinguishes derivations,
implemented references, completed scoped experiments and open integrations.
