# Bounded token suffix credit for alternative writes

Concrete failure: the local value teacher cannot distinguish routes with equal
immediate delivered values. Those routes can write different receivers and
therefore have different future losses. The token intervention contract
constructs exactly this case. This is a missing credit term, not a conclusion
about the full substrate's attainable quality.

Retain hard races, first-time clock credit, deep persistent memories, separate
keys/values, sparse selected writes, temporal transport and the local teacher
at other decisions. At one selected decision per chunk, replace the immediate
local value teacher with a paired actual suffix-loss teacher. The first site's
local teacher is masked to avoid double-counting its value term. The factual
pathwise clock and selected-map gradients are retained.

Conditional on scores s, exponential-race identity has probabilities pi =
softmax(s), independently of the first arrival T. For this bounded intervention
hold the sampled first arrival at the selected site fixed, force receiver j,
perform its actual write/delivery, and let downstream clocks/routes respond
using common future noise. Let L_j be the chunk's per-lane mean target NLL under
that intervention, and w the factual winner. Choose a nonwinner j with proposal
q_j, a mixture of .9 conditional pi and .1 uniform nonwinner probability.

Add the zero-forward term

    (pi_j - stop_gradient(pi_j)) * stop_gradient(L_j - L_w) / stop_gradient(q_j).

Its expected score gradient, summing over j != w, is

    sum_j grad(pi_j) (L_j - L_w) = grad_pi sum_j pi_j stop_gradient(L_j),

because the probabilities sum to one and the omitted winner's difference is
zero. The enumerated numerical contract matches this conditional expected-loss
gradient. It is a selected-decision, bounded-horizon outcome teacher, not an
unbiased whole-network gradient theorem. Losses include the immediate target
and future targets until the chunk ends; an intervening EOS naturally erases
the write's remaining effects. Targets enter losses only. Alternative choice,
factual prediction and state updates never inspect targets.

The factual race generator is preserved. A separate generator samples the
alternative. Replay starts from the same chunk boundary state and same factual
noise state; it does not advance the factual generator or modify factual state.
Depth/head sites rotate over updates. Numeric state persists between chunks,
with credit detached at those boundaries.

Inference is unchanged. This first diagnostic implementation evaluates all
proposals on factual and one no-gradient replay path; both paths, normalized
vocabulary likelihood, backward and optimizer are charged in measured fitting
wall time. Combining the teacher with the winner-plus-sampled-value kernel is
future engineering, not an already measured work reduction. Alternative value
maps do not yet get direct replay gradients; sharing maps and factual exposure
remain the mechanisms that train their values.

Required comparisons: numerical no-intervention all-gradient parity; equal
immediate delivery/different future outcome; enumerated conditional gradient;
small real-token smoke; same-data/budget local-teacher vs suffix-teacher fit;
then larger small-data/state-carry and credit-horizon comparisons. No long fit
is admitted before those contracts and its integrated fit.
