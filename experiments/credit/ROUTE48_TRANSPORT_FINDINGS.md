# R1 carried 48-target route-policy development: completed diagnosis

11 October 2026. Result `curie_route48_transport_20261011_v1_fit.json` completes
259.259 seconds, peak RSS 524,664 KB. One fixed source and initialization,
32 controller updates /1,536 TRAIN targets; eight fresh private scoring prefixes.
No confirmed utility gain: transport-over-static -2.722e-7 nats/target,
paired-prefix95% [-1.009e-6,+2.722e-7]. Transport-over-zero -2.474e-6,
interval [-7.155e-6,+7.246e-7]; over initial -1.512e-6, interval crosses zero.
This is a development result about this variant, not a family verdict.

Static and transport take identical sampled actions in all32 training windows.
Mean observed gradient norms are .0004447 static/.0003488 transport/.0011701
zero. These are realized norms, not unbiased noise-variance estimates. The
policies' expected costs differ, but training never produces a distinct actual
trajectory between static and transport. Only32 learning opportunities, fresh
scoring state versus carried training state, and small counterfactual headroom
are concrete diagnostic issues. None proves a cause of the unclear endpoint.

Fresh scoring uniform-to-best headroom ranges .000575–.004898 nats/target.
Fit costs: zero56.667s/51.280GF, static63.799s/51.499GF,
transport86.973s/80.936GF. Shared exhaustive scoring51.627s/330.503GF,
520worlds/24,960updates. This integrated driver duplicates reference work;
the separate2.09x acquisition measurement does not apply to its fit.

Next development decision: measure policy probability movement and alignment
with exact route costs, then evaluate the issued controller on carried versus
fresh learner state using causal disjoint prefixes. Do not increase a learning
rate or launch a large fit just because the endpoint is unclear. Incoming Adam
moments influence transport but are not explicitly supplied in the controller's
current descriptor; test whether learner-state descriptors and connected
long-range supervision address the diagnosed mismatch. The native carrier
remains frozen in this test; complete deep native learning is the larger target.

Preserve existing credit accuracy and execution evidence independently of this
utility endpoint. Joint depth composition/anchors/future-loss objectives are
specified in LEARNED_DEPTH_CREDIT_REPRESENTATIONS.md; they need integrated
implementation and complete work-to-quality comparisons.
