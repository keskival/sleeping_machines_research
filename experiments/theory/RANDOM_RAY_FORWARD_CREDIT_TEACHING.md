# Two forward worlds can teach all admitted credit coordinates

13 October2026, B10. Integrates AWS's causal teacher-admission principle with Curie's forward-taught future-state credit. Known stochastic directional probing supplies the sampling identity; the native state interface and realized future-loss teacher specify its application here.

At an issued boundary let E map d parameter directions to complete state effects, D contain fixed positive TRAIN scales, b=E^T lambda, y=D^-1 b and p be the normalized native critic output. Draw a vector a of independent Rademacher signs before observing the future block, independently of that block conditional on the boundary. Query two forward worlds:

    q = [L(s + epsilon E D^-1 a) - L(s - epsilon E D^-1 a)]/(2epsilon).

In the exact directional limit q=a^T y. Train with (a^T p-q)^2/d. Since E[aa^T]=I, its conditional expectation is ||p-y||²/d and its expected output gradient is the complete-coordinate squared-error gradient. No BPTT label or additional reverse teacher is needed. Conditional future randomness is permitted; treating feedback changed by our own actions as the same exogenous future would require a different intervention model.

For fixed residual e=p-y, the added Euclidean output-gradient variance from a single ray is4(d-1)||e||²/d². Averaging k independent rays divides this sampling variance by k. This is additional ray variance, not removal of unpredictable future variance. Actual Adam acts nonlinearly on the gradient/moment sequence and is not path-equivalent to full-coordinate fitting; quality must be measured.

For d=4 the teacher buys2worlds rather than8, while keeping all four learned output coordinates. Native key/value inference, retained decision activations, eligibility construction and critic optimizer costs remain. Finite differences add directional curvature/cancellation error. Set epsilon=1e-4/||D^-1 a|| so the coefficient-space perturbation norm is1e-4; with Rademacher signs and fixed D, epsilon is sign-independent. A paid mixed-ray adjoint contract audits the initial state interface. The comparison uses the same128TRAIN frames,32DEV frames and native31,804parameter capacity as the coordinate teacher; DEV labels remain full-coordinate evaluation only.

This directly attacks paid credit acquisition. It does not remove the measured own-trace or refresh overhead in deployment. The successful current actual-update path still costs more declared arithmetic than exact eight-event BP on the measured prefix. Reducing teacher worlds is a component saving; only a completed actor-quality/full-work integration can turn it into a practical learning advantage.

Completed same-data/seed acquisition: one-ray DEV MSE0.001343988 and two-ray0.001246334, versus zero0.001197747 and retained full-coordinate0.001133650. Their75%/50%teacherworld reductions do not retain useful acquisition. Mixed-ray initial adjoint error0.000389 at epsilon1.224e-7; normalized finite-difference cancellation and sampling variance remain explicit. Do not adopt either ray variant. User quality/data priority permits more fitting work; fixed-label reuse is the current better direction.
