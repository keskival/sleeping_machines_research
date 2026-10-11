# A boundary critic for the learning influence that survives sparse events

R1/B1,11 October2026. Companion to CAUSAL_CREDIT_HORIZON_ECONOMICS.
The native tail measurement motivates an explicit replacement for discarding
all consequences after eight targets. This is a proposed integration, not an
implemented critic or a new claim that temporal-difference learning itself is
novel.

## The state and the objective

Let S_t include native addressed/temporal state, fast parameters, optimizer
moments and any policy/credit numerical memory. H_t contains the observed
history and issued parameter versions. Hold a continuation policy version
fixed for the objective being considered. Its conditional downstream cost is

    V_t(S_t,H_t) = E[sum_(j>=t) gamma^(j-t) L_j | S_t,H_t].

V depends on time/history when observations and useful biases change causally.
A stationary V(S) is not required. Define the task return after one issued
intervention a over an H-step prefix and a later continuation:

    C(a) = sum_(j=0..H-1) gamma^j L_j^a + gamma^H R_H^a.

The expected route contrast is exactly the expected prefix contrast plus

    gamma^H E[V_(t+H)(S_H^a,H_H^a)-V_(t+H)(S_H^b,H_H^b)].

Future observations may be common in supervised streams, but branch learners
and future route policies can differ. Forcing a shared future action sequence
is an intervention diagnostic, not automatically a policy-value comparison.
Include changing optimizer and controller state in S rather than treating the
old numerical activations as a sufficient Markov state by assertion.

## The mutually teaching loop

The forward execution exposes candidate perturbation geometry, observed loss,
updated parameters/moments and subsequent responses. A connected conditional
critic V_phi uses these to learn downstream value, with dense per-event targets

    L_t + gamma stop_gradient(V_phi_target(S_(t+1),H_(t+1))).

Finite observed returns can replace bootstrap targets on selected trajectories.
Forward losses train critic parameters toward predictive consequences, not
imitation of a backpropagated gradient. Route updates use critic contrasts to
improve forward computation. Route discovery/exploration may additionally be
chosen to improve critic information subject to a separately stated prediction
and work objective. Such an acquisition term changes the optimized objective;
it must not be silently added to a task-gradient claim.

Credit-side activations must survive to feedback: issued phi version, critic
input, route/reference identities, score/selection probability, elapsed time,
state representation/feature version, prediction and declared reset boundary.
Subsequent forward feedback supplies a training target. A network shared across
routes and events can represent connected evolving inductive biases; a scalar
value is its output, not a limit on the critic's representational capacity.
Capacity/depth are measured engineering choices, not reasons to assume tiny
independent local modules suffice.

## What keeps the actor update correct

A reference trajectory can approximate alternative boundary states economically
using shared native feature geometry and transported learning differences. That
state approximation and V_phi give any predictor h(a). For issued pi and a
common predictor fixed independently of sampled A, the full observed return
correction gives

    g = sum_a pi(a) h(a) grad log pi(a)
        + [C(A)-h(A)] grad log pi(A).

Its conditional mean is the exact issued expected-cost gradient, regardless
of predictor accuracy. If only the first eight losses are observed in C, the
update targets that prefix objective; calling the bootstrap a correction does
not make it unbiased for the full future return. Delayed full returns, valid
probability-weighted residual audits or an explicit approximation claim are
required. Shared boundary-state construction must count candidate discovery,
optimizer influence transport, normalization and losing-route work.

When future corrections arrive after actor parameters change, the stored score
belongs to its issued parameter version. Applying it to current parameters is
a delayed online update, not a freshly evaluated gradient of current-policy
risk. State the bounded-delay/update semantics and retain versions. A narrow
first implementation can pin actor parameters throughout each return horizon;
this costs response latency but has a clear objective. General asynchronous
learning needs measured staleness and transport bounds.

## Next decisive implementation

First measure a finite48-target route objective with a pinned issued controller,
retained native/learner/moment state and exact factual/reference continuation.
Then fit a connected boundary predictor from forward-generated TRAIN returns;
compare reference-only, reference-plus-boundary and capacity-matched predictors
on fresh route contrasts and future task performance. Count all acquisition,
retained activations, fitting, audits and serving work. Existing measured tail
influence justifies this test; it does not prove that its critic generalizes,
scales better than BPTT or is cheaper on asynchronous hardware.
