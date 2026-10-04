# Architecture review history and checks — 4 October 2026

Scope: the whole implemented/theoretical model family, not one benchmark.
Documentation changes only; no model runtime, new training or queue admission.

## Pass 1: coverage, interfaces and mathematical scope

| Requested axis / level | Coverage after review | Gap exposed or clarification made |
| --- | --- | --- |
| Computational capacity, representation, trainability | Common matrix at every level; same columns for15family branches | Available state, selected execution and learning work explicitly differ |
| Primitive and unit | Timing logic, statistical, modal, nonlinear, historical and reception units | Native receiver is one example, not the whole family; input-conditioned age flow differs from autonomous transport |
| Module and routing | Pools/races, indexing, memory, reception and query | Source address, candidate search, winner, temporal reception and continuation/stopping are separate decisions |
| Interaction and layer | Content/time/state cycle, key/value roles, head mixing and latest-arrival join | Shared parameters do not automatically share private facts; cross-source interaction needs a path |
| Stack and composition | Serial, parallel, recurrence, KV/outcome, count hybrid, timeout and decoder compositions | Parallel residual additions are not serial depth; execution packing is not new expressivity |
| Learning and system | Objectives, factual/counterfactual gradients, write utility, horizon, optimizer and serving | Silence exposure, causal query cutoff versus completion, cached versions and online-state versus weight learning added |
| Whole family |15semantic branches;110source/driver AST records; mechanism coverage matrix | Statistic-valued hybrids are implemented small-fit members, not merely future proposals; reception remains partial |
| Comparisons and bounds | Counts, supported SSMs, RNN/MoE, retrieval, exact/sampled attention, ideal universality | Expected output differs from nonlinear-loss equality; class count alone does not prove a Littlestone lower bound |
| Advantage / novelty | Explicit opportunities, costs, prior attribution and actual evidence | No automatic paradigm victory, cheap arbitrary exact retrieval or measured clockless energy claim |
| Per-layer capability / cost | Lean-to-rich design continuum and conditional operator-containment table | Expand at the actual bottleneck; restoring representation does not guarantee optimization; new race candidates need an explicit parent-preserving contract |
| One-model aspiration / clocks | Shared-world diagram; four independent density/cadence/clocking axes | No required global periodic tick; local causal joins/timers remain; current emulators/optimizer are clocked/coordinated |

The substantive research gaps remain visible: future-write utility, useful
credit horizons, candidate coverage, full reception/silence scheduling,
shared multimodal grounding, repeated benchmark evidence and quality-matched
serving/energy. A review does not close those gaps by naming the operators.

## Pass 2: coherence and readability

The reading path now starts with a generative family definition, design-space
diagram and reusable-rule/private-evidence thesis. The catalogue follows as
supporting evidence. It defines the three review axes and component vocabulary,
then shows unit types, a representative expanded native unit/stack, routing,
composition and the complete learning system. The plain-language review is
separate from the detailed bounds and source-level inventory. Class names are
navigation aids, not unexplained accuracy labels or proof of capability.

The first stage added seven standalone SVG diagrams and one local interactive HTML atlas sharing colors:
blue computation, green persistent state, orange learning/whole-system concerns.
Controls expose available memories, scored keys, selected writes, evaluated
proposals and state scalars. These are structural counts, not advertised FLOPs
or measured savings. Every diagram states its scope and validation boundary.

The comparisons distinguish trained weights, current state and explicit
statistics; dense carrier work versus selected receiver work; exact attention
versus stochastic approximation; theoretical containment versus a tested
implementation; and prefill/decoding/streaming versus physical hardware.
Earlier evidence and failed variants remain preserved. The owner-rendered PDF
is unchanged; report Markdown and new local visual documentation are updated.

Verification records SVG XML/canvas geometry, local links, self-contained HTML,
unique element IDs, JavaScript syntax/controls, source-inventory hashes and the
unchanged 53-source clock-admission binding. Algebraic scalar witnesses check
selected identities; they are not numerical model or benchmark tests. No
browser renderer is available locally, so browser font/pixel layout has not
been visually certified. SVGs include explicit dimensions and bounded boxes.

Reproduce the low-impact checks with
`python3 scripts/check_architecture_review.py` after rebuilding with
`python3 scripts/build_architecture_atlas.py`. The verifier uses Python's
standard library and Node built-ins, imports no model, and launches no queue.
The first-stage record at
`experiments/results/diagnostics/architecture_review_stdlib_20261004T094000Z.json`
contains that stage's results: seven SVGs, local Markdown links, 110 source hashes,
53 unchanged frozen sources, JavaScript control contracts and 14 scalar witness
cases. These witnesses check selected identities, not full learning theorems.
The earlier `20261004T092600Z` record is retained as an intermediate six-diagram
snapshot; the `20261004T094000Z` record validates the first-stage seven-diagram revision.

## Pass 3: family definition, position and inclusion endpoints

The family is now defined by the content/time/address/state interface and
compatible local programs, reception and causal query rules. Local operator
libraries, support/schedule restrictions and learning define subfamilies.
The category is hybrid dynamical event-processing with configurable stateful
conditional computation, intersecting recurrence, SSMs, attention, sparse
experts, statistical memory and temporal/event systems.

Synchronous execution is an allowed scheduled/barrier endpoint. Dense execution
is an allowed full-support endpoint of configurable sparsity. Equal timestamps
do not ensure read-old/write-new semantics; fixed winner-only support does not
include arbitrary dense aggregation. Function/state, execution and learning
inclusions are separate contracts. These statements avoid treating sparsity or
asynchrony as hard ceilings, while retaining the selective temporal target and
its still-required advantage evidence. Added flexibility is relative to fixed
reference templates, not a claim other families cannot build hybrids.

## Pass 4: atomic-to-system closure and design rationale

Rechecked the common axes across atoms, units, modules, interactions, layers,
stacks, memory compositions and systems. Computational capacity includes
transformations, discovery, scheduling and resource costs; representation
includes retained evidence, precision, access and readout; trainability includes
factual and alternative utility, support, exposure, horizon and actual updates.
The design rationale connects choices to specific failures rather than merely
listing components. Information loss must be repaired before it occurs;
uncredited alternatives need useful credit rather than just wider content.
Multi-message integration needs matching aggregation and causal reception.

Shared rules versus private facts, source-private versus shared-world state,
selection versus discovery, retained capacity versus horizon, and sparse
inference versus full learning costs remain distinct at system level. No
subfamily is credited with a different branch's unimplemented operators or
benchmark success. The reading order is definition → choices/containment →
native examples and wider compositions → bounds/evidence → source navigation.

## Pass 5: learning, automatic design and historical continuity

The same trainable units, clocks, routes and persistent state support online
parameter learning and TTT as family capabilities. The documentation separates
ordinary state updates, statistical increments, learned fast memory and global
parameter updates. It states causal scoring, stale/cached version contracts,
local ownership, counterfactual work and streamed/batch update differences.
Implemented batching is separated from fully asynchronous hardware learning;
no static benchmark result is re-labelled as a TTT gain.

Automatic model design is connected to NAS, morphism and elastic networks,
with primary references to DARTS, Net2Net, Once-for-All and ProxylessNAS. Learned
event routes are distinguished from broader operator/allocation adaptation.
State migration, parent preservation, alternative future utility and complete
resource cost are required. Data-driven structural adaptation remains an
ambition; it is not promoted to an implemented general optimizer.

The historical manifesto was re-read. The introduction now explicitly follows
motivation → tested constructions → coherent family definition → integrated
validation/scaling. Explicit keys/indexing remain in the modern construction;
no historical aspiration is silently converted into a measured property.
Broad finite-execution containment states sufficient primitives, information,
storage, precision and scheduling; it supports reach without promising equal
learning or cost. Every added aspiration is located in the same design space.

## Pass 6: a shorter entry and operational definition

Added an overview that leads with the family definition, common axes, included
synchronous/dense endpoints and design/learning ambitions. Its reading map
separates definition, worked semantics, implementations, evidence, bounds and
source navigation. The detailed definition now states graph/operator, state,
schedule, query, learner/version and resource contracts. A glossary resolves
key/value, active/discovered/available, several sparsity meanings, native,
capacity/retention/access/horizon and different containment claims.

Observation time, modeled event time and measured runtime are distinct semantic
roles. Composition must preserve information and ownership as well as shape.
Independent operations can commute under stated conditions, while queries,
shared updates and random draw order can break equivalence. Finite recurrent
execution needs a bound/termination contract; positive delays alone need not
prevent infinite event accumulation. Theory152§17 records the scope.

## Pass 7: concrete semantics, provenance and interaction

Added an eighth SVG and reception explorer: first arrival, fixed window,
silence timeout and all-at-query compute different first-group content/time.
The example declares query cutoff, arrival-before-timer ties, pending state and
no EOF flush. Sixteen independent expected cases check causality, deadline
ties, delayed C membership and increased reception that remains pending.
No gradients, fitted quality or integrated native scheduler are claimed.

The generated claim/evidence map labels 13 claims and binds four completed native
language results, preserving result and actual execution-source hashes. It
checks declared matched settings and consistent whole-fit/per-presentation
units. Saved “exact BPTT” metadata is qualified beside its original scope;
continuous factual derivatives are not complete discrete future-write credit.
Quarantined target-dependent scores remain excluded. Numerical descriptions
are derived from the saved result fields, rather than transcribed constants.

Tab/panel accessibility references and keyboard navigation now pass. Invalid
dimensions/times clear stale results. Source hashes, result bindings, links,
SVG bounds, controls and 16 scalar witnesses pass with no numerical model import.
Current record:
`experiments/results/diagnostics/architecture_review_stdlib_20261004T103000Z.json`.
It also binds the reviewed documentation artifacts by SHA256. Browser pixel
rendering remains unperformed; the earlier validation records are preserved.

The final mathematical pass tightened the finite-event bound: b is effective
continuation fan-out after grouping terminating zero-delay work, not raw graph
degree. External and initially pending seeds are included. An additional
bounded-tree scalar witness checks that stated contract. The earlier102000Z
grooming record remains an intermediate snapshot; the103000Z record includes
the refined condition and17scalar witnesses.

## Pass 8: keep the whole family visible through composition

The overview and definition now distinguish the family envelope, concrete
member specifications and the integrated temporal/selective research target.
The new composition guide treats a unit as a product of evidence, flow, write,
selection, reception, output and learning choices. This permits combinations
inside a unit and heterogeneous regions inside a system, rather than a fixed
list of species or a uniform benchmark recipe.

The guide tracks all three review axes through each boundary from atoms to
systems. It covers serial, parallel, persistent, statistical/retrieval,
dense/synchronous and adaptive compositions, with their sufficient conditions
and costs. A ninth SVG keeps the family scales and hierarchy visible together.
The positive whole-model design rationale remains explicit: persistent
selective evidence, rich interaction where needed, computational time and
statistical/neural evidence can cooperate in one causal model. Implemented
branches illustrate the envelope; their union is not a completed model.

## Pass 9: future continuation, learning and resource preservation

Theory152§18 derives state sufficiency and sufficient transition/trace
conditions for parent-preserving growth or replacement. Pending clocks and
deadlines belong to operational state. Equal current predictions can hide
different future emissions. Functional reparameterization can change an SGD
trajectory; a parent-preserving extra product branch can have zero initial
first-order credit. Forward, learning and resource preservation therefore have
separate contracts. Intentional learned deviations remain permitted; exact
parent preservation is required only when that equivalence is claimed.

The accounting pass distinguishes summed invoked work, ideal critical-path
latency, actual contention and learning overhead. Shared/cached executions are
charged once along with validity/refresh costs. No isolated count of active
messages is converted into a hardware benefit.

Final standard-library/Node record:
`experiments/results/diagnostics/architecture_review_stdlib_20261004T104200Z.json`.
Nine SVGs and keyboard tabs, 16 causal reception cases, 21 scalar witnesses,
four completed result bindings, 13 scoped claims, 110 source hashes and all
53 frozen clock sources pass. New witnesses cover count/order sufficiency,
isolated state growth and a lost deadline, reparameterized SGD, and dormant
versus live growth directions. These are finite algebra/semantic checks, not
trained-model proofs or new benchmarks. The record binds current documentation
including the overview, guide, report entry, theory index and local handoff.
Previous validation records and the existing rendered PDF remain intact.

## Pass 10: one canonical core and explicit compatibility

The formal core now supplies twelve fields for evidence, operators, composition,
state, interfaces, participation, schedule, query, objective, learner, bounds
and execution/evidence. Its library-relative definition and conditional
construction grammar avoid granting every restricted member all envelope
capabilities. A compatibility table states the requirements for different
widths/clocks, joins/writers, shared information, rich islands, hard reception,
online credit, sparse training, coalescing and structural migration.

The overview gives one reading path; design/composition guides explain choices
and their consequences; review/inventory illustrate implementations. Repeated
core descriptions point to the specification. The whole-family target remains
temporal computation, learned small messages, selective state, key/value roles
and useful unrealized credit, evaluated on all three axes through every level.

## Pass 11: complete contrasting members and evidence consistency

C, R and H each declare all twelve fields: a causal statistic endpoint, a
selective temporal learner and a mixed reception/dense-query/finite-option TTT
composition. Initial state/maps, event order, stopping, queries, learning and
bounds are explicit. They are illustrative specifications, not implemented
benchmarks or new queue proposals. H charges its retained learning prefix as
well as its two-record forward bank. R names its local estimator and missing
future-write utility. Field-completeness checks do not prove conformance.

The evidence review corrected overly broad “no TTT win” wording by preserving
the earlier causal neural online pilot: 3.190859→3.095738 BPC on 8,191 development
targets, for 10.736× total processing work. A fifth, separate result binding
retains its protocol/historical sources; it is not mixed into the four T256
native language rows. Native-backbone integration, drift retention, economic
advantage and asynchronous hardware remain open.

## Pass 12: report and investor view preserve the same family

Added the model-family landscape as the tenth SVG/atlas tab. The report's
11-page vector chapter leads with the family and its relationships, then
reviews all three axes at every level, design options, actual unit/layer
examples, mixed worlds, learning, formal core and preservation. The original
215 report pages retain identical extracted text and original page content.
Earlier PDFs are archived.

The deck now has 18 main slides plus 16 diligence slides. Two new main slides place
the family landscape and heterogeneous design before mechanisms/benchmarks.
Definitions/capabilities/results/aspirations are distinct; the financial and
benchmark ledgers remain unchanged. Speaker notes and the private pack include
the same core and evidence. No external distribution or outreach occurred.

Bounded ReportLab/PyMuPDF rendering uses one thread, nice19, 300,000 KiB RSS
watchdog, 1,000,000 KiB address-space cap, 120-second timeout and 8 GiB available
floor. The report avoids the NumPy/plotting rebuild. The normal report builder
now includes the chapter and Markdown entry on later authorized rebuilds.
Rendered report pages and new deck slides were visually reviewed; page bounds,
pagination, current source bindings and preserved evidence are checked.
No HTML browser rendering is claimed.

Current documentation record:
`experiments/results/diagnostics/architecture_review_stdlib_20261004T113300Z.json`.
Ten SVGs/tabs, 16 causal reception cases, three complete twelve-field members,
25 scalar witnesses, four native result bindings plus one causal online binding,
13 scoped claims, 110 source hashes and 53 unchanged frozen clock sources pass.
New algebra checks cover the causal count prefix and complete finite-timeout
query outcomes/option derivatives. These are scalar semantics, not model fits.
Publication records:
`report/publication_model_family_release_20261004T113000Z.json` and
`investment/publication_pitch_deck_family_release_20261004T113100Z.json`.
The112200Z and112500Z snapshots are intermediate reviews. Final resource review
copies retained contiguous page ranges and deduplicates PDF objects:226pages
occupy5,069,175bytes, down from15,114,402bytes in the intermediate merge.
All215retained pages have identical extracted text; six original cover/chart
pages also have pixel-identical renderings at the checked resolution. The
bounded independent pixel check peaked at60,904KiB. Source/publication bindings
and the new private pack are refreshed after that optimization.
