# B10 regularized future utility for native optimizer controls

12October2026. One diagnosis-led intervention after curie_b10_native_optimizer_20261012_v1_fit. Unregularized online guidance trails online BPTT0.015535nats/event; multiplier quarter means1.192/1.609/2.698/3.498, while frozen guidance stays about1.03–1.07. Proposal/actual optimizer parity passes; this is an objective/control problem to test, not a mathematical derivative failure. Preserve the failed variant as development evidence, not a family LOSS.

Retain every model, actor signal, interface, source recipe, date/target count, native history and six controls of NATIVE_OPTIMIZER_FUTURE_PROTOCOL.md. No capacity or data-size change. Change only the native critic's future teacher objective:

    L_teacher = next_window_NLL + 0.05 ||theta_proposed-theta_exact||²
                                      / max(||theta_exact-theta_current||²,1e-20).

The coefficient is predeclared as a modest0.1quadratic restoring strength relative to the ordinary update. The reference uses complete exact gradients, the same current Adam moments/clipping and base step, with no learned field. It requires a charged additional producer VJP and proposal; no extra future-label access. Its parameters and normalizer are stopped in this local teacher; the learned proposal retains derivatives through native field and multiplier. Penalty is in the implemented parameter coordinates, not an invariant information metric or a universal stability guarantee.

This tests whether the myopic teacher's excessive step amplification is responsible for the observed degradation. It favours a justified deviation from a strong exact-gradient reference rather than automatic amplification. Future outcomes still teach the native event model jointly; the teacher does not imitate individual gradients. Zero regularizer applies to fully-frozen/BPTT/local-only arms; frozen learned guidance is not retrained. Source acquisition is repeated under the changed teacher, with identical common acquired bundle across every arm.

Numerical16TRAIN+16DEV smoke before4096target fit verifies six arms, actual Adam parity and finite regularizer gradients. Preserve each multiplier, relative step deviation, every control and full cost. Quality gate online better than every control; no direct claim of efficiency. Source/drivers freeze after results. Longer horizon only after the source/development decision. OneCPU/run_safe/RSS2.2GB/9GiBhostfloor/2GiBcgroupreserve/1800s. No sealedTEST, trader identity, profitability or sustained regime advantage claim.
