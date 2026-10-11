# R1 continuous proper-token route learning: development admission

## Decision and failure addressed

The completed real TRAIN calibration improves route-credit error34.4% beyond
static response, but it does not train a policy. The continuous execution
comparison removes repeated context acquisition and reduces complete measured
pipeline time15.977s to6.901s with identical state. The next decision is whether
forward-informed credit improves a learned hard-write policy on fresh development
episodes. Neither calibration nor ordinary lexical adaptation answers that.

## Retained mechanisms and issued information

Use the existing pretrained two-layer GPT-2-token native temporal carrier,
separate native key/query content, persistent addressed state and optional hard
key writes. Adopt contracted vectorized transport. Keep the available source
checkpoint fixed in this first isolating stage; only the64-dimensional lexical
learner and route controller adapt. This is a restricted family member, not the
final full-carrier credit architecture.

Issue the candidate pool and actor features before observing the next eight
supervised targets. Use observed document addresses, native produced write
content, query, current carrier output and incoming lexical state. Capture these
at issuance; the packet's final state is future information and cannot be an
actor input. Directions are normalized native per-head writes with fixed signed
amplitude. Preserve no-op and positive probability for every admitted route.

Pin actor parameters and common reference choice during the window. The reference
learner starts from the same incoming state and optimizer snapshot as factual,
but receives the no-op intervention. Only factual state is committed. Source
parameter version stays fixed; this avoids retroactive lazy-decay changes.

## Gradient and controls

For cost minimization use

    g = sum_a pi(a) h(a) grad log pi(a)
        + (C(A)-h(A)) grad log pi(A).

Detach h and observed C from actor differentiation. Sample A from the issued
policy; retain the full factual correction, with no unrecorded clipping or
selection. Charge all candidate discovery, normalization, reference adaptation,
projections, copies and optimizer updates.

Compare zero, current-snapshot static and online reference credit with identical
initial actor, source, data and optimizer settings. Static freezes the incoming
lexical vector, resetting on observed EOS; it is not a zero-state baseline.
The source-preserving static helper passes independent dense CE contracts.
Carry numerical native/lexical/optimizer state between windows; reset on EOS.
Specify controller recurrent-state truncation separately from numerical carry.
Eight-target consequences are finite-horizon credit, not complete BPTT.

## Gates before fitting

1. Independent finite-route expected-gradient equality against exhaustive costs.
2. Suffix invariance of issued features, candidate pool and probabilities.
3. Continuous factual forecast/state/moment parity and input snapshot independence.
4. Current-snapshot static parity, reset and zero-learning equivalence.
5. Source/data-bound bounded TRAIN smoke through a unique run_safe queue.

Before longer fitting freeze exact TRAIN and private DEV offsets, seeds, updates,
actor shape, learning rate, route amplitude, checkpoint selection and work
boundary in the driver and admission record. Public reserved scoring remains
untouched. Development improvements determine engineering changes; no first-pass
variant is published as a family loss. Evaluate stochastic and greedy endpoints
separately, with primary endpoint matching the optimized stochastic policy.

## Next analytical test

Forward-derived feature geometry suggests teaching an optimizer metric through
later losses. Keep its parameter/version fixed within each issued credit window
and retain prior gradient/metric activations. Existing finite identities do not
constitute an implemented metric learner or a scaling result.

## Completed bounded integration

The v2 smoke completes four eight-target windows per zero/static/reference arm
on32 proper TRAIN targets after16 context tokens, source checkpoint unchanged.
Connected router29,057 parameters, all gradients finite; factual key/fast
vector/Adam carry;6.46s total and360MiB peak. Every window retains65 routes
from up to8 observed document addresses. Full-prefix and carried-event route
features/probabilities agree at depths2/4/8. Duplicate issuance and all reference
work are charged. This checks the learning cycle only. Register longer TRAIN
and fresh private DEV offsets before quality development; no DEV selection or
utility result has been measured. Recurrent controller state, full-carrier
learning and economical vocabulary normalization require further implementation.
