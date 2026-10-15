# Carry composed representations through native memory

15 October 2026. Decision from the completed paired fit and TRAIN credit audit.

## Measured failure and hypothesis

The full-width role actor scores DEV NLL0.694245, versus prior0.693147 and
frame0.728851. Whole-label training has not earned transfer. On16TRAIN scenes,
interaction output median norms stay0.915–1.270 across parent levels, while
median gradient-times-value norm falls from0.004035 at level7 to5.072e-8 at
level1 (79556x). These are first-order sensitivities under the acquired joint
head and local native teacher, not causal effects, exact expected gradients or
comparable information measurements. Level populations and paths differ.
The geometry also gives some early parents one occupied child: bilinear
capacity there does no work. Layer count and interaction order remain distinct.

Each parent previously had to push its composed role value through four native
receipt transformations before publishing it. The next hypothesis is that
native transformed memory should refine a composition while keeping the
composition itself directly available, rather than requiring every native
stage to rediscover or faithfully relay it.

## Persistent carried view

Retain the latest computed ordered child interaction c and receipt time t in
the parent's private state. It becomes readable only with the corresponding
published native version. At read time s use

    emitted_view(s) = native_view(s) + tanh(a) T(s-t)c.

T is the parent's existing protected/rotating/decaying transport. Set initial
tanh(a)=0.5 at every parent: no grid, no zero-gated stack. Child views, role
binding and bilinear composition are unchanged. The carrier is computed from
already published child memories, not raw sensors or hidden identities. It is
persistent numerical state, not an event queue; a native timer can emit it
without a new query. Receipt races, selected native writes, keys/values, local
counterfactual credit, pure timer credit and all32native stages remain active.

For fixed published native memory and c treated as an independent argument,
D_c emitted_view = tanh(a) T(s-t). This removes four native transformation
Jacobians from this particular derivative path. It does not promise a global
identity gradient: role composition, transport, upper learning and timing still
matter. Native timer scores also see the changed view, so this is a changed
actor, not forward containment. Zero carrier strength exactly recovers the
old actor; nonzero strength is the proposed construction.

The contribution is integrating a conventional residual principle with native
persistent composition, lazy indexed reads and race-generated views. Carrier
state bytes, transport calls/additions and seven scalar optimizer states must
be charged. No hardware speed or energy follows from this path argument.

## Admission decision

First run one real paired optimizer update from the acquired relational raw
weights, comparing identical TRAIN scenes/common tape. Measure per-level
sensitivity, finite all-level gradients, resource envelope and new work. A
passed numerical gate is not permission for an unchanged64-update retry:
select a new learning comparison only if the carried path materially improves
the diagnosed sensitivity without a resource failure. DEV/TEST are excluded
from the gate. Validate exact zero-strength migration, persistent causal reads
and live carrier/native derivatives before admission. No external model fit.
