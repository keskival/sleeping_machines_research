# Credit quality determines the audit work frontier

B1/R1, 11 October 2026. Extends [connected paired actor–critic](SPARSE_PAIRED_ACTOR_CRITIC.md). This is conditional estimator algebra, not a measured convergence or depth-scaling result. Known control-variate and optimal sampling principles apply to our native, versioned future-loss contrasts.

## The object to economize is the policy gradient

Fix a causal history H, its current route policy pi, and a reference route b. Let A be drawn from pi, s(A)=grad_theta log pi(A), and D the learner-aware future loss contrast L(A)-L(b). The continuation and branch learner belong to the target law. Let q(A)=E[D|H,A] and let h(A) be the critic prediction issued before feedback. Conditional score gradients and predictions are fixed during consumption. A positive audit probability p(H,A) may depend on the issued packet and sampled action, but not the unrevealed outcome. With I conditionally Bernoulli(p), the raw estimator is

    G = g_h + I/p * (D-h(A)) s(A),
    g_h = sum_a pi(a) h(a) s(a).

The audit coin must be conditionally independent of the future loss given H,A. Then E[G|H]=g=sum pi(a)q(a)s(a). Neither clipping nor an Adam parameter update inherits this unbiasedness identity. Carrier/continuation-law derivatives require their own terms.

Define r=(D-h(A))s(A) and v(a)=E[||r||^2|H,A=a]. The exact trace of the conditional covariance is

    V(h,p) = sum_a pi(a) v(a)/p(a) - ||g-g_h||^2.

Proof: g_h is deterministic conditional on H, and the corrected random term has first moment g-g_h and second moment sum pi v/p. Its subtraction term couples actions; route-wise prediction MSE alone is not the full variance objective. Since

    v(a) = ||s(a)||^2 * [Var(D|H,a) + (q(a)-h(a))^2],

credit should resolve errors where the current policy has large score sensitivity. Uniform prediction error treats cheap/insensitive and expensive/sensitive routes equally. The conditional future noise term survives a perfect conditional-mean critic. Paired outcomes remove shared nuisance fluctuations but do not remove unpredictable futures.

For uniform p, perfect h=q attains V=sum pi ||s||^2 Var(D|H,a)/p. For general h, optimizing prediction MSE is sufficient to reduce an upper bound on V, not a guarantee of the globally smallest gradient variance. A route-independent shift also changes this variance while leaving g_h unchanged because sum pi s=0. The paired critic anchors h(b)=0; that constraint can preclude the best score-weighted baseline. Account for an additional causal baseline separately if one is implemented.

## Predictable audit allocation

Let c(a) be the incremental cost of an audit. The factual selected world is already executed. Only the additional reference world and audit handling enter c; equal/no-op routes have exact zero contrast and require no extra world. For fixed h and a bound on sum pi v/p, minimizing expected extra work sum pi c p yields the Lagrangian solution

    p(a) = clip(sqrt(lambda*v(a)/c(a)), p_min, 1),

with lambda chosen for the variance budget and p_min>0 where the residual is unknown. This follows by differentiating c p + lambda v/p; zero-cost or exactly known residuals need separate handling. Estimates of v and c must be learned from past issued packets only. The formula identifies a design target; it is not implemented by the present fixed-p pilot. A poorly calibrated v estimate affects economy, while recording the actual positive propensity preserves the estimator mean. Any sampling-biased critic objective needs the declared distribution/importance weights.

An asynchronous scheduler can request expensive credit where expected sensitivity-weighted residual is large, retain cheap model credit elsewhere, and consume delayed packets with their issued pi,h,p and version. Recomputing these coordinates after feedback changes the estimator. A packet can remain valid for its old target while being stale for the current target. Changing the carrier, learner law or causal horizon needs explicit target treatment, not a stationarity assumption about inductive biases.

## Investment and a concrete falsification test

Let C0 be factual forward learning, discovery, controller updates and critic query cost per episode. Let CA be average extra audit cost. A frozen acquired critic with audit rate p costs C0+p*CA per episode. If acquiring it costs K beyond a reference and permits a reduction p_old-p_new at matched learning quality, its cost can pay back after

    N >= K / [(p_old-p_new)*CA],

provided the denominator is positive and reuse introduces no other extra cost. With ongoing critic training cost CT per episode, replace the denominator by (p_old-p_new)*CA-CT. This is cost algebra with explicit assumptions, not a proof that our critic reaches the required quality or audit rate. If C0 dominates, reducing audits has a small ceiling regardless of critic accuracy. Full variance and quality under coupled learning still need measurement; a variance reduction does not directly set matched-quality runtime.

The current 128-to-512 episode ladder measures acquired-frozen versus reset credit at identical actor initialization, then compares continuing acquisition. All acquisition costs follow the clones. Next use completed histories to identify whether sensitivity-weighted residual, continuing critic work or factual/discovery work is the bottleneck before admitting a lower-audit or larger-model stage. Report wall time with all optimizer work; supported-operator FLOPs omit unsupported pointwise/scalar/special operations and cannot alone certify complete arithmetic efficiency.
