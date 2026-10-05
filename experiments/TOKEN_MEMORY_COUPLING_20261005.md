# Test predictive memory coupling without replacing temporal state

5 October 2026 · Diagnosis-driven integrated comparison.

## Failure and retained mechanisms

The completed8Kpaired fits all beat initialization. Credit16selected NLL is
8.297491/8.286427(seeds6/7);credit64is8.300131/8.290236. Constant TRAIN-mean
features cost0.017759/0.027515NLLfor credit16, but addressed-state erasure costs
only0.000381/0.000980. Credit64addressed erasure effects are−0.000268/+0.000449.
The recurrent message is useful:erasing it costs0.017843–0.021955NLL. Stored
state is numerically alive and affects keys/values; predictive use is weak.
Wider normalized adaptive tails lose at seed6and are effectively tied at seed7:
8.302969/8.286345versus8.297491/8.286427. Retain the narrower decoder.

These are frozen interventions on2040development targets, not retrained
ablations or bounds on model-family capacity. Both horizons have initial-inclusive
selection and equal16368presentations/32updates. Longer credit and output-tail
width alone have not improved this selected two-seed recipe.

## Derived intervention

The selected unit already emits

    v = x + g * LN(W_out m_new + x) * sigmoid(G GELU(LN(...)) + b).

Its stored-value path therefore has Jacobian

    dv/dm_new = g * J_gate_and_normalization * W_out,

conditional on the factual winner. Input residual x has a direct path independent
of g. The current g=.5/sqrt(depth)=.353553at depth2is a fixed construction choice,
not a learned scalar. It is reasonable to test whether this balance leaves
stored-value contribution weak during a small fit. Nonlinear gates, downstream
norms, changed routes and optimization mean doubling g need not double whole-model
sensitivity or improve NLL. We will measure the integrated effect.

Compare exactly one alternative:gain scale2(g=.707107), against the saved
scale1member at8K,batch64credit16. All numerical state equations, temporal
rotation/decay, race clocks, hard selection, sparse writes, key/value separation,
depth and actual future-write teacher remain. Memory time constants are unchanged.
The change alters factual message content and downstream routes, rather than
adding a dense carrier/readout bypass. No added parameters or selected activity.
Its scalar multiplication has the same arithmetic shape; charge the complete
executed fit because selected routes and adaptive-target work still require audit.

## Contracts and promotion

- Scale1 must reproduce the existing engine's token features, state, route RNG,
  all parameter gradients, and actual tiny-fit trajectory.
- Scale2 must retain causal chunk partitioning, finite gradients to addressed state,
  EOS lane reset, exact normalized probabilities, sparse forced-write continuation
  and actual-driver interruption/resume. Gain scale belongs in settings/source pins,
  so resuming with a different scale is rejected.
- First run seed6at the existing8Kbudget and selection cadence. If it improves
  selected dev by at least0.002NLL, run seed7for the paired confirmation. Otherwise
  record the loss and retain scale1; do not launch a gain grid.
- On a successful pair repeat context/state utility. Promote only when quality
  gain and complete work support the stated conclusion. Greater memory dependence
  alone is not better prediction. Reserve64Kdata before a scaling-law selection.

This is a scalar construction comparison within the existing family. It proposes
no substitution of dense computation, removal of hard alternatives or redesign of
memory transport. Next repairs follow the completed result, including losses.

## Completed decision

Eight numerical/actual-fit contracts passed. The seed6scale2fit selects8.310055
versus the preserved scale1control8.297491:loss by0.012563NLL. The predeclared
0.002improvement gate fails, so no seed7gain2fit or gain grid is admitted.
Retain the original gain and all evidence. The next64Ktest measures greater
training exposure with the retained recipe, after complete8Kwork parity.
