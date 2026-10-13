# Train the critic's writes by their future consequences

B10, 13 October 2026. Formal design; no completed memory-write intervention or
quality result. This extends the issued-action teacher, rather than changing
the fixed confirmation currently running.

## The missing derivative

At an issued cut x, the native critic returns an actor correction a_psi(x) and
its next persistent state c_psi(x). Define U_x(a,c;psi_bar) as future supervised
loss after actual actor updates, optimizer evolution and subsequent critic
decisions. Future critic parameters psi_bar are frozen at the issued version.
The current-decision derivative is

    d_psi U = (D_psi a)^T partial_a U + (D_psi c)^T partial_c U.

The existing paired action teacher changes a while holding issued c fixed.
It teaches the first term, including current writes' effect on the current
action. It does not teach their distinct effect through future stored state.
This is a structural omission even if the action teacher is perfectly accurate.
It is not evidence that the omission caused the measured parent-dependent
quality outcomes. Differentiating shared future policy parameters adds another
term; the expression above is a local, frozen-future-policy objective.

In this implementation c includes value banks, arrivals, contexts, private
random state and detached numerical history. These are not interchangeable
differentiable coordinates. Start with critic-owned value-bank writes only;
retain factual arrivals, topology, history and random state. The frozen parent
bank is not an intervention target. Selected memory values are produced by
native temporal units, not by an added dense recurrent network.

## Economical projected write credit

Choose a small, causal write chart y with fixed rank b at issue. Define c(y)
by substituting bounded value coordinates at selected native addresses; all
other state remains factual. Address/selection decisions are held fixed during
this local intervention. Let z be the four active actor-action chart coordinates
and F_x(z,y)=U_x(a(z),c(y);psi_bar). Paired sphere directions in dimension4+b
give the known uniform-ball smoothing estimator

    g = (4+b) * [F_x(v+r u)-F_x(v-r u)] * u / (2r),  v=(z,y).

Retaining the current computation graph permits a detached-label surrogate
g_z dot z_psi + g_y dot y_psi. This trains the current action and the current
writer from the same matured supervised consequences, without differentiating
through the future actor trajectory or imitating BPTT labels. It estimates a
smoothed conditional projected derivative, not the full state gradient or a
guaranteed descent step after critic Adam. Small b limits exploration variance
and receipt size, but loses directions outside the selected chart. Two paired
worlds suffice per receipt; larger perturbation dimension can still require
more receipts to obtain useful accuracy.

Sampling separate write blocks with inverse-probability weights deserves care:
finite-radius smoothing each block generally gives components of different
smoothed functions. Do not claim a single joint smoothed-objective gradient
without a compatible joint perturbation distribution or a zero-radius limit.
Predictable selection and weighting do not remove that mathematical issue.

## Activation and causal receipt contract

Keep the issued writer activations and their parameter/version bindings until
feedback matures. Counterfactual worlds receive detached state copies; detaching
the factual replay must not discard the retained learning graph. The current
NativeCarry.detach intentionally cuts earlier graphs, and its histories are
NumPy observations. This proposal teaches the issued writes; it does not turn
that history into a differentiable unlimited sequence tape.

Receipt identity binds observed-prefix cut, actor/optimizer version, critic
version, selected addresses, write chart, horizon and issued random tape.
Apply matured credit only under an explicit stale-version policy. Online
memory adaptation and online critic-parameter learning are different: current
DEV comparisons evolve critic memory but freeze its weights. A later causal
test-time critic-learning evaluation must train only after labels arrive and
retain a paired frozen-critic control.

For the current fixed-count future replay, paired global/private RNG endpoints
must match and the factual tape must be restored. If an asynchronous extension
changes event counts or execution order, equality of RNG consumption is no
longer an adequate coupling contract. Random keys tied to event/address/draw
purpose can preserve each world's marginal law despite different schedules;
that needs its own contract before enabling variable-work races.

## Required numerical witnesses

1. Zero intervention preserves factual actor forecasts, critic carry and tapes.
2. With actor action held fixed, a selected critic write changes a later critic
   decision or forecast; the first actor update remains exactly unchanged.
3. On a small fixed-route cut, finite write differences agree with a functional
   derivative within a declared step-size/roundoff range. Hard-route smoothing
   is a separate estimator claim.
4. Retained writer activations receive nonzero credit where action-only teaching
   gives none, and detached replay copies do not mutate factual banks.

Only after these witnesses should a same-data B10 fit compare action-only with
joint action/write teaching. Charge intervention selection, retained activation
bytes and lifetime, extra world state copies, messages, optimizer work and all
teacher forecasts. Memory is useful capacity, not free capacity. The immediate
priority remains the admitted cohort and query numerical contract; this note
does not queue an additional fit or assert a new benchmark win.
