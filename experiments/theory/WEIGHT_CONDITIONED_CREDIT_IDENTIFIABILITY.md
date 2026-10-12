# B1: why a connected credit packet must identify its current operator

The new event optimizers lose against matched acquired-core/head-only learning in seven of eight development cases. More events, shared parameters and a globally connected temporal critic are useful construction features; they do not by themselves guarantee sufficient information for deep credit. The full-cohort/depth run tests data and capacity scaling before another architecture is admitted. The following is a conditional representation diagnosis, not a proof of the cause of the measured failure.

## An explicit ambiguity

Consider a two-layer model h=W1*x, y=v^T*h with squared future loss. A packet contains observed inputs x_i, hidden states h_i, outputs and terminal errors, but not v. If the observed hidden vectors span a proper subspace, choose nonzero n with n^T*h_i=0 for every packet event. Actors v and v+n produce identical packets and identical losses on those inputs. Their future parameter derivatives nevertheless differ:

    grad_W1 L_future(v+n) - grad_W1 L_future(v) = error_future * n * x_future^T,

including a future input drawn from the same observed inputs. Equality of the loss at the current weights does not imply equality of its sensitivity to a new hidden-state direction. A deterministic field conditioned on the common packet cannot produce both correct local directions. Under a distribution of such actors, the best squared-error predictor has irreducible risk E[Var(g_future | packet)]. Increasing neural capacity cannot remove information excluded from its input.

This uses derivatives to exhibit ambiguity, not to declare gradient imitation the training objective. Under a small norm-constrained intervention, smooth-loss first-order optimal updates point opposite these different gradients. Therefore the task-valued update policy also needs to distinguish the actors. Finite steps, curvature, optimizer state and future-distribution uncertainty add further information requirements.

## A sufficient direction for construction

Condition credit on current native operator information and learner state, not only forward activations. Build connected messages through depth, rather than only a global pooled context supplied separately to each layer. A local message can take the form

    z_l = A_psi(P_l, operator_l, learner_l, elapsed_time) z_(l+1) + b_psi(...).

The operators and offsets are trained on actual future intervention utility, with receipts retaining their issued version. Local known weight contractions or cached weight sketches may provide geometry; learned terms can change future demand, allocation and optimizer behavior. This is a learned causal operator, not an assertion that exact past backprop is the future optimum. A recurrence supplies a genuine cross-layer information path; any amplification and approximation error must be measured with depth.

The present event field shares typed decoders and a global temporal context, but does not explicitly supply current weight operators or pass learned messages recursively through depth. Its all-layer mean can discard layer relations. Holding psi for16blocks while actor weights move can worsen this mismatch: incoming packets evolve while the hidden operator also changes. This is a concrete hypothesis for the failed held-version variant, not a demonstrated causal attribution.

## Learned inductive bias as transformation law

A permutation P of hidden units changes h to P*h and a linear native weight to P_out*W*P_in^T. Consistent parameter updates must transform by the same law. A responsibility/input outer product q*x^T respects that law when q transforms as P_out*q and x as P_in*x. A fixed learned coordinate transport T must also transform as P_in*T*P_in^T; leaving T unchanged generally breaks the law. Sharing tensor dimensions alone does not enforce this inductive bias.

A weight-conditioned construction can derive transport from actor tensors and permutation-invariant scalar summaries, then learn its coefficients. This retains dependence on the current actor basis and can reduce acquisition needed after actor drift or transfer. General invertible basis changes require transforming primitives and optimizer metrics as well; LayerNorm/GELU do not grant arbitrary-basis equivariance. The permutation claim is the explicit limited symmetry used here.

## Economics and next admissible comparison

Operator information is paid: reading all weights afresh, candidate discovery, state summaries and stale-cache repair cannot be hidden. Maintain sketches when the selected operator changes; sample producer/event queries with full support; update only selected producer matrices when testing sparse learning activity. Exact future teachers still cost their measured native adjoints. A cheap query key is not a cheap full value update, and a reverse-depth message path still has dependencies even on clockless hardware.

If the complete B1 cohort/depth stage fails its quality/work gate, the next architecture comparison should distinguish weight-blind versus weight-conditioned recursive credit under the same actor/data/teacher/work protocol. Validate basis-permutation behavior, operator freshness and native finite-update gradients in that integrated pipeline; then fit on real task targets. Do not queue another standalone variance grid or infer that every learned-credit formalism fails from the current weight-blind member.
