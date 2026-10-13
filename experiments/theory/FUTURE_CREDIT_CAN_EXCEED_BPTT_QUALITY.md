# Future credit can exceed standard BPTT quality

12 October 2026, user question. Nats/event is a loss; lower is better. Exact BPTT differentiates the chosen past-data objective exactly. It is not an oracle for the best future-data update, finite-time optimizer, architecture or compute allocation. Perfectly imitating its gradient with the same optimizer/state/data reproduces its update and supplies no quality improvement by itself. Predicting a conditional mean of noisy gradient examples can also denoise; this is distinct from exact per-example imitation.

Let u_BP be the baseline parameter step, mu the conditional future-risk gradient, H a positive semidefinite local curvature model, and D the admitted correction directions. Consider u=u_BP+D*a with regularized local future objective

    J(a)=mu^T u + (1/2) u^T H u + (lambda/2)||a||², lambda>0.

Then a*=-(D^T H D+lambda I)^-1 r, r=D^T(mu+H*u_BP), and

    J(0)-J(a*)=(1/2)r^T(D^T H D+lambda I)^-1 r >= 0.

A zero correction remains feasible. When r is nonzero, this oracle correction strictly improves the local future objective; its nonnegative regularization cost means it also improves the unpenalized quadratic future-risk model. This is a local conditional existence result, not a guarantee for nonconvex long-horizon training or a fitted neural critic. After replacing exact history derivatives, u_BP is not automatically available cheaply; do not transfer this inclusion guarantee to a graph-cut implementation without a baseline/fallback comparison.

The critic can exceed the selected standard-BPTT training baseline by learning future utility, denoising and selecting useful temporal/shared-leg updates. Exact local derivatives can teach/calibrate it without imposing BPTT as its sole objective. Standard BPTT can also train sophisticated meta-learners; no claim that differentiation itself has an intrinsic quality ceiling follows.

More relevant experience can improve a learned conditional update function across shared contexts/branches and amortize acquisition. If per-target work is cheaper, fixed acquisition A becomes a smaller fraction of A+N*C. Neither prediction improvement nor cheaper per-target work is automatic. Changing regimes require causal context and adaptation/forgetting; raw old data can become misleading. Measure relevant experience, update counts, novelty of branches, complete acquisition and quality/work as data/depth grow.

Current evidence: bounded market critic improves frozen guidance0.000026nats/event over4096DEV forecasts, trails ordinary online BP0.000047; exact joint sweep saves execution with equal scores. Integrated local eligibility preserves128forecast scores and own-memory/readout gradients while exposing omitted remote gradients. These support the development direction, not a sustained learned-credit or scaling win.
