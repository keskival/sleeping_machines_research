# State ownership makes lower representations reusable

15 October2026. Structural analysis and numerical contract; learning is separate.

## The representation is an object, not only its last message

A compositional state consists of persistent indexed lower objects, parent
hypotheses, typed relationships and available read/generation operations. The
last vector emitted by a child is a view, not the whole represented system.
Different parents can interpret a shared child without duplicating its state.

Let M_i be canonical child memory and H_a,H_b two parent hypotheses. Our current
indexed operation is a side-effect-free temporal view V_i(t), interpreted by
native parent updates F_a(H_a,V_i,pose_i) and F_b(H_b,V_i,pose_i). Changing H_a
must not silently rewrite M_i or H_b. A deliberately requested child update is
a separate operation with its own identity/credit, not an implicit consequence
of asking a question. This is inference-state noninterference; shared learned
parameters can still couple learning across tasks/parents.

The new contract perturbs the highest object's private hypothesis, then supplies
the same next observation under the same race tape. Every lower-level published
representation remains bitwise identical; the highest representation changes.
This prevents accidental top-output-to-bottom recurrence from making every
local update traverse the whole hierarchy.

## Added abstraction depth versus temporal recurrence

In a simplified publication chain with D independent persistent objects and N
ordered receipts, each object uses d native stages. If one receipt per level is
preserved in order, the dependency grid has recurrence
L(l,n)=d+max(L(l-1,n),L(l,n-1)).
Thus the longest processing path is d(N+D-1), although total work is dND.
If the top output is instead fed into the bottom of the next receipt, the serial
path can become dND. This is a structural argument for independent state
ownership, not a hardware timing claim or a bound for arbitrary feedback graphs.
Queues, joins, queries, candidate work and unequal cell costs still count.

Our hierarchy has independent addressed state at each object and upward
publication; the read round directly queries lower states without rewriting
them. It therefore removes that particular avoidable top-to-bottom dependence.
It does not eliminate long local temporal credit paths, guarantee well-conditioned
gradients, or prove that deeper concepts will be learned.

## Time-dependent views and finite information

For linearized latent flow z(t)=Phi(t)z_0 and compressed read y(t)=W Phi(t)z_0,
a set of reads has observation matrix [W Phi(t_1); ...; W Phi(t_k)]. Its rank can
exceed that of one compressed read. For example, a rotating pair (a,b) read by
W=(1,0) reveals a at time0 and -b after a quarter turn. A learned race can decide
when a useful latent direction becomes accessible. This is the standard
observability argument, not a novelty claim or new information creation.

Important scope: the current published-context head is square and may already
be full rank. Multiple timed reads need not reveal extra Shannon information
there. Time can still change which representation is accessible to an economical
nonlinear parent. The native private unit memories are richer than this published
context; our current indexed read exposes the published view, not arbitrary
private future state. Query-conditioned access to finer internal snapshots needs
its own causal/publication design; do not claim it already exists.

## Why a read round is not yet a learned abstraction

The current round explicitly gathers all active lower learned memories. This
ensures useful access opportunities but is not learned sparse discovery. Higher
state could ignore the content and use addresses, timing or prior upward events.
The planned frozen intervention zeroes only query response content, preserves
its source/pose/type and event count/order, and retains factual input processing.
All query deliveries target the root, so the same root race draws are assigned
to the same receipt positions. Clock values/winners may change as the causal
consequence of the altered content. This isolates the incremental value of
queried content beyond inherited state and query structure. It does not test all
hierarchical meaning, and it does not erase legal positional evidence.

Only after a useful learning result should we spend this attribution compute.
Then test held-out constituent reuse. No unchanged deeper ladder is admitted
merely because the mechanism contracts pass.
