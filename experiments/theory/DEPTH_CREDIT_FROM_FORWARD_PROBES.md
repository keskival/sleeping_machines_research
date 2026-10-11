# Connected depth credit taught by forward probes

R1/B1, 11 October 2026. This extends the compatible-depth objective in
LEARNED_DEPTH_CREDIT_REPRESENTATIONS and reuses AWS's exact joint content/time
packet as an execution reference. No active AWS packet or model is replaced.

At a fixed issued native state/parameter version, let J be a cut-to-cut response
operator, c a downstream loss query and b=J^T c its true upstream cotangent.
A connected credit function produces b_hat=C_psi(c,X), using observed native
packets X and retained credit activations. For an issued direction v,

    y(v)=<c,Jv> = <b,v>.

The forward response therefore supplies a scalar teaching target for backward
credit. Random directions need not reconstruct a full Jacobian. With zero-mean
v of covariance Sigma, fixed independently of the response/critic error,

    E_v[(<b_hat,v>-y(v))^2] = (b_hat-b)^T Sigma (b_hat-b).

For identity covariance this is exact Euclidean cotangent risk. For a declared
learned/proposed covariance, it measures that metric, including its nullspace.
Rank-deficient probes cannot identify unobserved components. Direction sampling
and forward response acquisition are part of the cost, not free supervision.

An economical direction can additionally correct a biased predicted cotangent:

    b_corrected = b_hat + Sigma^{-1}v [y(v)-<b_hat,v>].

Its conditional mean is b when Sigma is positive definite and the issued
predictor does not depend on the sampled v. If a critic is trained on that
same direction before issuing the correction, this independence no longer
holds. Retain issued activations/versions or use independent correction probes.

Whitened Rademacher probes v_i=+/-1 give variance

    E||b_corrected-b||^2 = (n-1)||b_hat-b||^2

for one identity-covariance probe in n dimensions. Thus a cheap forward query
can have prohibitive correction noise in a large cut. For k independent probes
averaged, this variance is divided by k. Learning a good joint credit function
reduces the residual norm, but random projection alone is not an efficiency
proof. Charge temporal/content dimensions and explicit gain/precision.

Coordinate sampling is naturally sparse: with positive q_i, audit
b_i through J e_i and use b_hat+(b_i-b_hat_i)e_i/q_i. Its conditional correction
variance is sum_i (1/q_i-1)(b_i-b_hat_i)^2. For known residual magnitudes and
unit coordinate costs, minimizing the uncentered second moment gives q_i
proportional to abs(b_i-b_hat_i). Residuals are unknown before auditing; learned
uncertainty can propose q, with positive support and calibration measured.
Nonunit costs require an explicit expected-work constrained allocation.

Across depth, global cut-to-cut probes teach composed credit and expose errors
that isolated local fitting misses. Pair them with local consistency and useful
nonzero anchors. Allocate probes where the upstream amplification-weighted
residual contribution is large. Shared elapsed-time cotangents sum across
layers exactly once; treating each layer as an independent global-clock error
would double-count the producer. Causal future-consequence credit and current
first-order cotangents are separate targets; use both with their own risk/audit
rules rather than call one the other.

Next integrated comparison: connected predicted content/time credit versus
exact packet reverse transport, on owned native depth2/4/8, full actor updates
and future quality measured. Exact cache may be cheaper than extra forward
queries in the current implementation; require complete measured work and
variance, then work-to-quality. This derivation defines a learning/correction
path, not a proof that learned credit already beats backprop or BPTT.
