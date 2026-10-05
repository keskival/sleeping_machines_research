# Sleeping Machines: the family at a glance

Sleeping Machines studies **networks of stateful temporal programs**. A message
carries content, an arrival time and an address. A receiver consults local
evidence, transforms it, chooses what to retain or communicate, and may emit
later. Messages and persistent state compose these programs into layers,
graphs and learning systems. Time can perform an operation rather than merely
label an observation.

The research target combines learned small messages, computational delays and
races, sparse addressed state changes, separate selection/value roles, and
credit to alternatives that did not execute. Useful stored capacity grows
beyond expensive selected work; discovery, storage and learning of the larger
bank are counted in the accounting.

The enduring aim is a **trainable computing substrate with useful capacity
beyond expensive activity**. The architecture is the organization of temporal
programs, evidence and credit; individual benchmark recipes instantiate it.
Progress means better useful prediction and learning per total resource, while
retaining the integrated mechanisms that motivate this family.

![The model-family landscape](figures/architecture_landscape.svg)

## What makes it a family?

The common **content/time/address/state interface** lets us vary the local
program, routing, reception, memory, composition and learning policy without
requiring one uniform layer recipe. Each member specifies its operators and
contracts; the integrated temporal/selective target is the direction we make
efficient. Different regions of one model can use different local programs.
The [composition guide](model_family_composition.md) explains how local
choices carry through interfaces to whole-system capabilities.

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

A filter, causal estimator, gated recurrent cell or attention block can be
reproduced from the family's local operators, state and scheduling; finite
reference computations have an event-graph emulation path. Members then
specialize: a winner-only or narrow-state member trades generality for work.

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

Online adaptation already improves prediction on a new stream (3.191 → 3.096
bpc, causal predict-before-update). Native test-time training, drift retention
and fully asynchronous training are the next demonstrations.

The further ambition is **automatic design within the family**: data can teach
operator, reception, optional-depth and state-allocation choices. Learned
event routes already adapt execution inside a fixed pool/stack; graph growth,
pruning and morphing extend this with state migration and task/resource credit,
building on NAS, network morphism and elastic networks.

Dense frames, language, irregular events and queries update one shared world
state through explicit information paths, so learning in one domain can transfer
to another. Input density, computation density, arrival cadence and physical
clocking are independent design choices.

Routes, delays and stopping change the internal work per query; comparisons
fix the observation boundary and scored targets so quality and resource use
refer to the same workload.

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
| Follow application opportunities and their first proof conditions | [Opportunity register](model_family_opportunities.md) |
| Inspect conditions and bounds | [Theory 152](../experiments/theory/152_primitives_integration_and_capability_bounds.md) |
| Find implementation branches and source records | [Family inventory](model_family_inventory.md) and [source navigation](architecture_source_inventory.md) |

The progression is manifesto → constructions and tests → family definition →
integrated validation and scaling. The evidence map distinguishes derivations,
implemented references, completed scoped experiments and open integrations.
