# Select a scalable member from the generative family

User direction,5October: implementations are points in a design space, not
the model-family definition. We make the implementation to realize the chosen
member. Read the generative definition in report/model_family_design.md,
formal specification, composition guide, current THEORY synthesis and original
motivation before treating an existing native kernel as canonical.

Sleeping Machines pursues a general-purpose substrate spanning language and
reasoning, multimodal world models, embodiment, event-native analytics,
continual learning, communication, self-design and hardware. The language
program supplies evidence and a scaling test for one front.

## Research principles and selectable coordinates

The integrated target computes through causal time and stateful events, uses
selective communication and persistent representations, separates selection
from delivered content, and learns from unrealized alternatives. A member
specification chooses compatible programs implementing those principles.
The following are choices, not required constants of the family:

| Design axis | Current token lab point | Degrees of freedom worth selecting deliberately |
|---|---|---|
| Event schedule/topology | One token event per layer, fixed feed-forward depth | Genuine interacting pending events, recurrent/branching topology, learned stopping, variable internal work and input cadence |
| Time computation | Exponential race and .001+.010*T/(1+T), token timestamps spaced1 | Threshold accumulators, delay-coded algebra, different rate/clock distributions and gauges, timing bandwidth and trainable computational cadence |
| Local state/representation | Small vectors with gated damped rotations | Protected or historical memory, statistic-valued state, richer flows/filters, multiple retention scales, structured/low-rank learned representations |
| Candidate discovery/routing | Global one-source receiver pools; score every key | Learned addresses, hierarchical/indexed discovery, cross-source/state access, multiple receivers, dynamic capacity and selective head interaction |
| Communication/reception | One value per head, aligned and densely mixed | Small learned typed messages, conditional reception, coincidence/hold/veto, sparse inter-head exchange, multi-stage time/content composition |
| Capacity and exposure | Tied or private maps, fixed pool size | Shared bases plus private adapters, learned recruitment/threshold prices, protected evidence banks, growth/reuse and useful capacity beyond activity |
| Learning/credit | Local value teacher plus one bounded suffix intervention | Actual joint outcomes, memory/clock/support terms, deeper horizons/replay, sampled counterfactuals, eligibility/statistical updates, learning-aware optionality |
| Input/readout | GPT-2 token lookup, frequency-initialized adaptive likelihood | Different supporting embeddings, normalized decoder organization/rank, statistical/learned output composition, temporal output decisions |
| Execution | Python/torch CPU, parameter banks, cached projections | Fused event operators, packed sparse updates, compiler/runtime choices and hardware mapping; these implement the member rather than define it |

Our small P16/D2/H2/U4 fits are diagnostic coordinates. They have no privileged
claim to be the benchmark-sized member. Their negative findings attach to
that configuration and recipe. Their positive contracts and measured gains
remain valid at that coordinate. A parity contract ensures that an engineering
transformation preserves a chosen member; it does not make its choices sacred.

## How experimental evidence constrains the design space

A completed result belongs to the declared member, learner, data and resource
protocol. Generalizing it requires an argument identifying which design choices
caused the result and which other members share those conditions. Failure of a
small-vector, single-delivery, short-credit construction does not establish a
failure of protected memory, richer reception or deeper counterfactual credit.
Likewise, a successful isolated primitive is a useful construction to compose,
not automatic evidence that every composition will learn or scale.

Choose the desired member before choosing a convenient implementation. Write
down the information it must preserve, the temporal interactions it must
perform, the alternatives its learner must credit, and its work budget. Then
implement those requirements. Existing kernel interfaces, vector sizes,
parameter sharing and detach boundaries are changeable coordinates. Numerical
contracts validate a stated construction; they do not require preserving an
old construction when the intended function changes.

Scaling laws are conditional on that choice of member and learning protocol.
Poor scaling of one choice calls for diagnosis and reselection, not an imposed
family-wide ceiling. The objective is a construction with useful predictive
capacity and economical activity, rather than maximal occupancy or a larger
count of allocated parameters.

## Selection order for the current objective

1. Keep the useful completed token-interface, persistence, normalization and
   exact-continuation contracts. Remove avoidable input, data-allocation and
   execution faults. Source storage/optimizer representation is free to change.
2. Combine sparse candidate-value execution with actual future-write credit
   in one integrated fit. Separate diagnostic kernels are not the final member.
3. Test whether useful time interactions, information retention and decoder
   capacity are present. The current timestamp/delay ratio leaves inter-token
   waiting inactive; this is a concrete temporal coordinate to revisit. The
   adaptive tail rank is also a selectable capacity choice, not a mandated cap.
4. Choose state/routing/exposure changes by diagnosed information or learning
   failure. Favor changes that recruit useful predictive functions at measured
   cost. More slots, uniform traffic or higher state rank alone are not the goal.
5. Select a coherent promising member and freeze its protocol for the rough
   scaling measurements in TOKEN_SCALING_PLAN.md. Those laws describe that
   selected member family, not all Sleeping Machines designs. If a new design
   changes the active/capacity relationship, measure its scaling separately.

Before each substantial substitution record the failure, derivation, retained
mechanisms and resource consequences, then verify small integrated fits. We
can change any current implementation choice; a slow or faulty backend is
engineering work, not a ceiling on the chosen architecture or ambition.
