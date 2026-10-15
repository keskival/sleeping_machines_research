# Persistent parent composition: binding, relations and information transport

15 October 2026. G-HIER development, implemented and numerically checked;
untrained reference, not a benchmark or semantic hierarchy claim.

## Why this construction

Races choose alternatives. Pools combine evidence. Neither alone identifies
which children belong together or represents their relationships. The new
`parent_race_hierarchy.py` explicitly separates eight abstraction levels from
four refinement steps at each level. Its 32 learned cells do not represent
32 time steps or 32 abstraction levels. The event wrapper reuses the existing
learned temporal decay/rotation surfaces on a fixed observed window.

At level l, parent j retains vector h_lj. A local addressed child candidate
contains content v_i and relative pose p_ij. For each role r:

s_jri = <Q_r(h_lj + B old_h_(l+1)), K(v_i,p_ij)> / sqrt(d)

A hard exponential race chooses candidate a_jr, including a zero-valued null.
Separate value maps deliver selected child content. Roles cooperate; candidates
within a role compete. Distinct role queries permit complementary selection,
but collisions are allowed: this is not an injective object-part assignment.
Current roles address a fixed local 3x3 neighborhood; learned variable-sized
object grouping is absent.

For selected values u_r, relational features are
R_rs = tanh(A u_r) * tanh(B u_s), r<s.

They preserve role identity and encode conjunctions. Relative positions travel
with candidates; permuting candidate storage while preserving content and pose
leaves deterministic composition unchanged. Permuting actual geometric roles
need not. Parent templates are shared spatially, permitting reuse across sites;
no trained generalization to unseen part arrangements is claimed.

## Transport repair derived from a measured failure

The first update was h' = h + gate(h) tanh(W[roles,relations]). It relies on
repeated randomly initialized projections to carry child information through
all levels. A synthetic signal contract found float32 final-loss contrasts at
only 16/32 audited cells. Float64 resolved all cells, with extremely small early
contrasts. Correct counterfactual credit cannot recover a numerically erased
forward effect.

The repaired update uses an initially identity/rectangular-identity content map
T, learned corrections V, and relational proposal f:

u_r = T v_selected + 0.1 V(v_selected,pose)
z = mean_r(u_r) + 0.1 tanh(f(roles,relations))
h' = h + active * gate(h) * (z-h).

The write gate is initialized at a neutral bias rather than strongly closed.
Null has exactly zero value; all-null leaves h unchanged and emits zero.
The direct path keeps selected child content available while relations learn.
It supplements hard selection and relational computation rather than replacing
them. Interpolation limits repeated evidence accumulation during refinement;
with fixed z and scalar gate g, n refinements give
h_n = (1-g)^n h_0 + [1-(1-g)^n] z,
which is bounded for 0<=g<=1. In the actual model gates and proposals vary, so
this fixed-proposal identity is intuition, not a global stability guarantee.

For one step, the state Jacobian contains diag(1-g), and the child Jacobian
contains diag(g) times a direct selected-content path. This removes mandatory
multiplication through a learned nonlinear proposal on every child-to-parent
edge. It does not give a lower singular-value bound: role averaging, routing,
compression and learned cancellations can still lose information.

## Persistence, feedback and credit

Explicit state is returned and accepted across calls, never detached or reset
implicitly. Current lower states flow upward. Previous-call upper states query
lower binding, avoiding an implicit cyclic solver. This is delayed top-down
conditioning, not an active query/reply scheduler. Numerical null silence is
supported, but raw event absence is not automatically interpreted as null;
spatial event surfaces are currently dense evidence.

The final loss supplies factual selected-path gradients plus exact conditional
categorical credit via full forward replays under common random tapes. Forced
choices recompute downstream winners. A cell-balanced audit sampler first
chooses a cell uniformly and then a site/role uniformly within that cell:
q(cell,site,role)=1/(number_of_cells * sites_in_cell * roles).
The surrogate divides by this actual q. Thus high-resolution lower levels do
not starve upper levels of audit opportunities. This changes sampling variance,
not the expected sum of gate derivatives. Audit RNG must be independent of the
race tape. All valid candidates, including null, are enumerated; padding is not.

## Completed evidence and limits

`experiments/results/diagnostics/parent_binding_signal_contract_20261015_v5.json`
records one initialization, fixed 16x16 synthetic feature surface, eight levels,
four refinements, width 8, three roles. Each cell audits role 0 at spatial site
0 and enumerates valid alternatives. Both constructions use the same initialized
parameters and input per precision; the transport flag changes the direct path,
update rule and gate bias together. This is a construction comparison, not an
ablation attributing the gain to a single component.

| Construction | Float32 cells with nonzero contrast | First-cell contrast |
|---|---:|---:|
| Proposal-only update | 16/32 | 0 |
| Selected-content transport + interpolating update | 32/32 | 0.00000143051 |

Float64 first-cell contrast increases from 6.79e-14 to 9.86e-7. Float32 and
float64 can differ in downstream race outcomes; neither contrast is a quality
score. Earlier v1/v2 diagnostics used the pre-transport source; v1 did not match
input draws across precisions. They remain numerical evidence with their source
hashes; v5 is the current matched-input construction check.

22 tests across the three hierarchy modules pass: role complementarity,
permutation with pose, null-state preservation, persistent state replay without
mutation, feedback-conditioned binding, final-loss early-key gradient, correct
cell-balanced probabilities, float32 early credit through all eight levels,
and integration of the existing event surface with temporal-parameter gradients.

All spatial candidates, states and proposals are evaluated by dense tensors.
No sparse-dispatch, execution-latency, energy or complete-FLOP claim is made.
One update costs a factual forward/backward and K valid-candidate full replays.
Persistent state is dense, capacity/activity are not yet decoupled, and
alternative-value learning is sampled through factual choices. Raw asynchronous
event scheduling, dynamic parent allocation, cross-parent ownership,
query/reply access to supporting children and calibrated uncertainty are missing.
This development construction cannot silently replace the integrated flagship.

## Next decisive work

Before a long fit, measure real TRAIN contrast distributions, audit variance,
RSS and replay work. Preserve acquired spatial inputs and source-bound v1 jobs.
If signal/resource admission passes, compare learned child reuse and geometric
sensitivity at fixed observation horizon, with held-out arrangements when data
supports that protocol. Gesture labels alone test recognition, not a cat/part
hierarchy. No new training was launched for these synthetic numerical checks.
