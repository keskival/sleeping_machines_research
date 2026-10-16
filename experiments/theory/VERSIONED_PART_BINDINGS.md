# Persistent wholes need stable access to their bound parts

16 October 2026. Native construction and numerical contracts; semantic quality
remains an experimental endpoint.

## Why this change follows from the evidence

Local geometry learning reduces calibrated root reconstruction error by34.6%,
but the acquired frozen actor still leads the trained continuations on whole
labels. Better preservation of a mixed vector is not enough. A parent needs
both a reusable representation and a way to recover the particular parts and
roles it composed. Reading whatever a lower memory contains *now* can silently
change the meaning of an older whole.

Separate three identities. An **instance address** locates a persistent memory;
a **version** identifies its immutable publication; a **concept identity** groups
semantically equivalent instances. The first two are now explicit and tested.
They do not,by themselves,learn the third. A timer/query notification sequence
is a fourth identifier:it names a computation,not a new memory version or a new
sensor observation. Confusing these identifiers is particularly damaging in an
asynchronous hierarchy.

## State and binding algebra

A published whole is (r,D),where r is the existing evolving native state and
D is a directory of role-bearing part references:

    D = { (lower level,address,version,issued time,relative pose,event kind) }.

Each accepted child view refreshes its source's current directory entry.
Native temporal state still integrates every accepted message. A publication
freezes the current directory;later child writes and parent computations do not
rewrite that publication. D describes currently bound child views,not an
inverse of every historical influence in r. It neither replaces persistent
state with an event log nor pretends the native update was a simple sum.

A bound read at q uses the saved native publication,its per-head timestamps and
the original native alignment/head function. It matches the ordinary native
read exactly while that version is current and keeps reading that same version
after the live lower memory changes. A higher memory can explicitly query a
bound older part and deliver that view through its existing native receipt
races. No sensor replay,new latent labels or symbolic replacement tree occurs.

**Distinguishability.** If r=A(a,b) is non-injective,there can be distinct child
bindings with identical r. A commutative sum,for example,cannot distinguish
(left=a,right=b) from (left=b,right=a). An ordered directory can distinguish
those cases without asking r to invert the sum. This removes a forced access
bottleneck;it does not guarantee that a learned query or decoder uses the
information. Existing native composition is already nonlinear/role-sensitive;
this argument explains the value of an additional recoverable path,not a claim
that its actual forward computation was commutative.

**Causality.** Every edge strictly descends an abstraction level. A child is
published before its view is issued,and issuance precedes parent publication.
Thus a directory graph is acyclic and a bound read before child availability
is rejected. Changes after a version's publication cannot change its tensors or
its directory. A learned policy may inspect more parts only after paying their
indexed read/transport costs.

## The same native race learner can consume the part pool

`BoundRecursiveRaceHeads` applies the existing competing defective-clock
readout to all memories in a declared bound directory,with the parent as
context. Each slot retains its actual lower publication clock;queries do not
reset it. Local silence/event losses use the partition-invariant exposure
ledger. Gradients reach the real lower native memories,not a detached symbolic
surrogate. Every candidate slot is charged;the current implementation does not
claim free key discovery or selected-only scoring. The pool declaration must
fit the entire directory;undersized declarations fail rather than dropping
children silently.

The integration contract uses an arbitrary legal future mark solely to test
credit connectivity. A semantic experiment must instead define stable causal
concept marks or observable relation targets,with TRAIN-only fitting and an
explicit relevance test. Predicting arbitrary marks is not evidence of
hierarchical learning.

## Retention and work

No event history is replayed. An explicit mark/sweep retains current input and
native publications,live directories,pending publication parts,in-flight
bindings and their transitive parts. Unreferenced versions are removable. For
R live roots,depth D and bounded directory fanout b,the straightforward upper
bound is R sum(i=0..D)b^i nodes;sharing reduces this bound. No constant-memory
claim follows when root counts or directory sizes grow. Autograd can retain
older computation even after directory GC;graph memory is separate.

The prototype adds no actor parameters or random draws. Contracts require
bitwise actor logits,completion schedules and every original parameter gradient
to match the acquired construction. Indexed bound reads preserve native state
and event counts. Extra tensor lifetimes,directory edges,metadata,GC visits,
read transport and auxiliary race training remain paid work. Existing native
counterfactual learning and timer/value races are unchanged.

Next decision:can stable learned concept identity plus these accessible
role-bearing parts improve nonadditive relations and held-out whole labels?
Require identity stability,sensitivity to role changes,cross-part predictive
value beyond additive/single-part controls,and improvement over the acquired
frozen actor. Preserving geometry and recovering parts are separate capabilities;
neither substitutes for that semantic test.
