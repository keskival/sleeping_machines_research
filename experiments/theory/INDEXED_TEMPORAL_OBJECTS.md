# One indexed temporal object: persistent memory and event generation

15 October 2026. Native hierarchy prototype; no trained semantic result.

## The architectural decision

The user's correction is essential: a memory is not its notification stream.
Replacing memory with an event log would discard efficient indexed access and
force reconstruction. Keeping a static memory beside an unrelated event model
would duplicate state and leave temporal semantics ambiguous. Scheduling an
emission after every read would turn queries into repeated observations and can
create feedback storms. None is the selected construction.

Use one indexed temporal object with persistent state, closed-form evolution,
native race updates and two interfaces:

read(address,t) -> representation of the published object at t
receive(address,t,message) -> pending new state and native completion time
arm(address,t) -> native race proposing a future emission
fire(timer) -> representation at firing time, if its version remains current

Indexed reads and event generation are views of the same persistent state.
There is no periodic recomputation of inactive objects and no required event
log. The model retains state at stable addresses. Update notifications, query
responses and timer outputs can all be inputs to higher objects, with distinct
types. They are not automatically independent new observations.

## Native mechanism selection, not benchmark inheritance

The implementation reuses `SplitEventHeads` with protected/dynamic state pairs,
shared rules and separate addressed state; `TemporalUnit`, `TemporalRoute` and
native alignment/transport are retained. Addressed state is allocated only on
use. Independent native heads race within candidate pools. A parent's children
enter one common receiver stream (source0); geometric source identity, relative
position, source abstraction and event kind are message metadata. Making each
child a private source would isolate memories and defeat cross-child composition.

This is the existing integrated native event transition, not a reproduction of
the entire winning EasyTPP model/objective and not evidence of inheriting its
scores. Heads are parallel computational channels, not demonstrated semantic
part roles. Relations are learned through ordered content/pose-bearing native
updates; explicit variable object ownership and learned binding remain open.

`TensorTimedEventCore` adapts the existing transition only to preserve tensor
input timestamps. The old `float(timestamp)` boundary loses upstream timing
derivatives. Scalar-time forward state/logit/arrival parity is bitwise in the
contract. Native delayed clocks, evolving memory, selected writes and the
existing counterfactual route teacher remain. No active/frozen source is edited.

## Persistent state, publication and causality

For a published record R=(version,ready,{v_h,t_h}), a read computes
V(t)=head(concat_h Phi_h(t-t_h) v_h), t>=ready.
Phi retains the native protected prefix and decaying/rotating suffix. Read does
not overwrite R, add an evidence item, increment the native event count or emit.
Thus repeated reads do not make the source stronger by observation repetition.
They do cost transport/projection work. A computational query reply may refine
the receiving parent; its type and identity distinguish it from a new sensor write.

A transition may be computed ahead in this reference executor, but is private
until its completion event publishes. Direct reads and timer proposals use only
published snapshots. Timer scoring must not inspect private future memories.
Publication follows local versions; pending contexts are immutable tensor
references rather than an ever-growing history. A per-(receiver,source-stream)
last sequence number provides bounded retry/stale-message handling. It grows
with communication edges, not with every event. Parameter changes during pending
graphs require an explicit online/checkpoint policy; this prototype does not
claim that integration yet.

## Why time now performs computation

For candidate scores s_i, raw exponential times are E_i exp(-s_i). A common
score shift changes arrival time without changing the winner. Our native race
uses its established bounded monotone delay map. A generated event reads Phi at
that delay. Therefore even with the same winner, its delivered value can change.
The clock-speed contract verifies exactly this, and a generated-value objective
gives a nonzero clock-bias derivative through elapsed-time evolution. Changing
which event arrives first is a separate discrete consequence.

The native route teacher is a local counterfactual surrogate; the chosen clock
has its interior time derivative. Neither is an exact derivative through all
possible queue reorderings or changed future event counts. The earlier exact
suffix audit remains useful engineering, but cannot be claimed to cover this
new event topology without extension. Timer alternatives currently use native
winner-time differentiation, not an exact loser timer-consequence audit.

## Lean hierarchy and bounded activity

`NativeObjectHierarchy` is a thin dispatcher over eight levels of the same
primitive, with four native race stages per level by default. A child publication
notifies its geometric parent; a parent can explicitly query any indexed lower
object. The query policy is not yet learned. Model rules are shared across
objects within each level; memory is separately addressed. No dense spatial
sweep or fixed number of all-parent refinement iterations is imposed.

The reference heap orders causal availability events on one CPU; it is not a
measured distributed asynchronous implementation. Each object's admission uses
its own native readiness. A timer fires once, is superseded by a newer arm, and
is invalidated by a newer published version. There is no automatic rearming.
This prevents an accidental self-exciting event loop. Independent ingest/write/
read/emission rates are possible, but current default write notifications are
one-for-one; learned coalescing and emission/halting policies are not implemented.

Virtual arrival times are model computation coordinates anchored to input times,
not measured hardware latency. Their scale relative to observed timestamps needs
calibration. Counters report candidate scores, selected updates, losing-value
work, timer scores, publications, reads and state-tensor storage. Input records,
queue metadata, published aliases, parameters and graphs are separate. No full
FLOP/traffic/energy claim is made. Known addressed subscriptions avoid an all-object
scan; learned content-indexed discovery and capacity beyond activity at useful
trained quality still need measurement.

## Completed contracts and next decision

Twelve new contracts pass: scalar native parity; incoming-time derivative;
published-state visibility; lazy non-mutating reads; bounded stream deduplication;
common-speed/same-winner/different-content; timer single-fire and invalidation;
clock learning through generated content; independence from unpublished future
work; eight abstraction/four-native-stage causal propagation and early credit;
indexed query plus timer refinement without new observations; future-input
exclusion; and event-time-ordered version assignment (some combined in one test).
These are mechanism tests, not proof of learned abstractions.

The existing cached spatial learning pipeline remains an explicitly labelled
composition diagnostic. Native raw-event resource admission is the next gate for
this construction. Then test whether the same learned lower representations
support different parent compositions, with controlled input, write notification,
indexed query and timer paths. Preserve observed evidence and actual state
lifetimes; do not silently replace the native learner with the earlier synchronous
pooling prototype. No external reference is trained by this development work.
