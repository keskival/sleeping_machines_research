# Higher abstraction through addressable lower-level memories

15 October 2026. G-HIER architecture development; no trained quality claim.

## Failure being repaired

The parent-role construction retained every lower state but let each new parent
read only its immediately preceding level. A persistent collection of states is
not addressable memory if the computation that needs it cannot query it. A
lossy intermediate summary could therefore permanently hide supporting detail
from a higher concept. This contradicts the intended compositional architecture.

`sleeping_machines/addressable_parent_hierarchy.py` retains the input feature
surface and all level states, and gives each parent role direct typed access to
addresses in every lower level. It preserves the 8 abstraction x 4 refinement
construction, hard races, separate keys/values, relation terms, selected-content
transport, persistent state, delayed top-down conditioning and final-task
counterfactual credit. Existing source-bound architectures/jobs are unchanged.

## Formal information distinction

Let X be lower evidence, Z=C(X) an intermediate representation, Y the target.
If x_a and x_b have C(x_a)=C(x_b), any deterministic parent F(Z) gives the same
answer for them. If their labels differ and they are equally likely, any
classifier based only on that Z has error at least 1/2 on this pair; best
conditional log loss is log(2). A larger downstream network cannot invert this
collision without further information.

With retained memory M(X), refinement can instead compute
h_(n+1) = Update(h_n, Bind(Query(h_n), M(X)), Relations).
If an accessible address distinguishes the pair, the collision no longer
forces equal predictions. This enlarges available information; it does not
prove learnability, correct grouping or classification improvement. The
implemented collision contract holds intermediate representations exactly equal
and shows that a high parent can retrieve distinct raw details. Blocking direct
ancestor access makes its candidate banks identical again.

This separates abstraction from irreversible disposal. A concept is a useful
summary and an addressable hypothesis about supporting detail. Its vector need
not carry every child fact: later queries can request a different role or source.
A fixed stencil only provides some requests; genuinely content-addressed
candidate discovery and explicit child ownership remain future engineering.

## Implemented read/compose/refine schedule

A parent at level l has four role races per refinement. Candidates contain
(content, source-level coordinate, relative pose). The dense reference gathers
nine fixed addresses in each source memory, at that source's native resolution,
spanning the parent's approximate footprint. It does not average source maps
before retrieval. A null candidate adds no content.

At each abstraction level:
1. Build an address bank from retained input and newly computed lower states.
2. Query it using persistent parent state and previous-call upper-level context.
3. Select one child per role with independent exponential races.
4. Compose role-specific values and pair relations; interpolate the parent state.
5. Repeat four learned refinements, then expose this parent memory to higher levels.

The returned memory preserves all levels without in-place mutation or hidden
detach. Calling with no new observation permits further refinement against the
same retained input. Feedback is previous-call upper state; there is no implicit
cyclic solve. Repeated calls refine memory; they do not extend the observation
window or manufacture new evidence. No automatic forgetting/deduplication or
raw-event asynchronous dispatch is implied.

Direct reads also shorten selected information/credit paths: a high parent can
read a lower source without traversing every intervening compression. This is
not evidence that intermediate levels become semantic. Direct raw reads may
bypass them. `blocked_reads=((parent_level,source_level),...)` supports controlled
source interventions with identical candidate/tape shapes, so we can test which
learned levels contribute beyond raw access. Source 0 is input; source s+1 is
abstraction level s. No immediate-only vs all-level quality claim is made.

## Capacity, activity and charged work

Each cell queries up to K_l=9(l+1)+1 candidates per site, but selects R roles.
The reference still computes every candidate key and value. At level 7, K=73
and R=4: sparse selected deliveries are not sparse discovery or value execution.
The returned work descriptor charges every key score and selected delivery.
It is not a complete FLOP ledger; projections, pair relations, gather traffic,
state updates, replay and optimizer work must be charged separately.

For an exactly halving 2D grid before saturation, N_l=N/4^(l+1). With T refinements,
total key scores are RT sum_l N_l[9(l+1)+1]. The infinite geometric sum is
(13/3)RTN. Immediate-only 10-candidate scoring gives (10/3)RTN. Thus bounded
multi-scale stencils can add information access without an asymptotic quadratic
cost in image area. These are analytic score counts, not measured speed/energy.
Once a grid reaches 1x1, the geometric bound no longer applies; extra levels
incur growing candidate counts. Width, candidate materialization and memory
traffic can dominate. The finite actual geometry is reported by the descriptor.

Retained state uses input plus all parent grids: approximately (4/3)NW scalar
slots under ideal 2D halving and fixed width W. Selection cannot erase the cost
of storing or gathering them. This design currently exposes useful memory; it
does not demonstrate capacity beyond activity at execution level.

## Learning and evidence

The final objective combines factual selected-path differentiation and exact
conditional categorical derivatives from alternative full forward replays under
common random tapes. Cell-balanced site sampling preserves deep audit coverage;
its actual probability is used in the importance correction. Losing downstream
choices are recomputed. Replays reuse old memory without mutation. This covers
the current forward computation, not future optimizer/world changes.

29 combined numerical tests pass. New contracts cover collision recovery,
source identity and valid geometry, all32-cell force-factual replay, persistent
input reuse, highest-level role-key credit, all-source scoring accounting, the
event adapter and controlled source interventions. These establish execution
and gradient behavior, not learning or novel-concept generalization.

The source-bound real-TRAIN preflight uses one 128x128 first-second gesture,
32-width/32-cell construction, one exact route audit/backward and a retained
memory refinement. It admits no optimizer, test access or long fit. It measures
RSS, time, finite gradients and replay count through the existing six-slot
coordinator. It can wait behind the slot's loaded batch; queue publication does
not mean it is running.

Next decision: measure whether real TRAIN credit and replay costs permit a
bounded learning experiment, then test source contribution and unseen part
arrangements. Preserve useful data and acquired checkpoints. A gesture score
alone does not establish pixels-to-fur-to-cat semantics. Raw asynchronous
execution, dynamic parent allocation/ownership, content-indexed candidate
discovery, complete work accounting and learned compositional transfer remain
explicit implementation/evaluation tasks.
