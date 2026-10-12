# Future utility teaches credit through a producer JVP

The B10 direct critic computes a one-update future objective with an immutable actor producer graph. The corrected teacher retains all factual paths inside its next feedback window; detaching that window's first produced state omits a real term and fails finite differences. Source-bound v2 fit is therefore a teacher-protocol error, not a quality verdict. v3 passes both residual branches at1e-4 perturbation: absolute directional errors1.45e-7 and2.86e-7. At1e-6, recording-cell arithmetic is noisy; tolerances and perturbation sizes are explicit in the result.

Let z=F_theta(x;s) be the current produced state with carried numerical state s held fixed, J=d z/d theta, and h_psi the connected critic output. With independent audit A of probability p, kappa=1-A/p and g=g_local+J^T[kappa h_psi+A g_exact/p]. Treat g_local, g_exact, theta and prior moments omega as constants for critic teaching. Let theta_plus=Phi(theta,g,omega) be the actual clipped optimizer update. On the next revealed window, take the full factual gradient u=grad_theta_plus loss_future with incoming numerical state held fixed. Then

    grad_psi loss_future = kappa (d h_psi/d psi)^T J (d Phi/d g)^T u.

This is a chain rule, not a novelty claim or a descent theorem. It does not require an actor Hessian: first pull u through the optimizer, then push the resulting parameter tangent through the old producer J. A stored event packet/current-operator JVP can replace the retained differentiable producer VJP. The critic's own activations remain necessary for its final backward pass. Charge the optimizer pullback, old-version producer replay/JVP, future teacher and activations. No economy claim follows without measuring them against the existing implementation.

Global norm clipping contributes a scaled identity plus a rank-one term. For active clipping gbar=tau*g/(r+epsilon), r=||g||,

    D_clip = tau/(r+epsilon) I - tau/[r*(r+epsilon)^2] g g^T.

Adam contributes an elementwise diagonal derivative. Write mhat=m_const+a*gbar, vhat=v_const+c*gbar², b=sqrt(vhat), with bias-corrected a,c and old moments fixed. The parameter step eta*mhat/(b+epsilon) has derivative

    d_i=eta*[a/(b_i+epsilon)-mhat_i*c*gbar_i/(b_i*(b_i+epsilon)^2)].

For theta_plus=theta-step, the gradient tangent is D_clip^T(-d*u). The numerical-zero denominator convention must match the actual proposal contract. These diagonals need not be positive: a Euclidean credit MSE or PSD quadratic-risk proxy can disagree with actual future utility.

Incoming numerical state, old optimizer moments and historical actor parameters are stop-gradient objects in this one-step objective. This is not exact all-history BPTT/meta-learning or a proof that causal biases are stationary. Persistent critic state can condition the direction on observed changes. Hard discrete route changes need separately priced counterfactual/replay or score-function terms; the factual JVP alone supplies no unrealized-route derivative.

Decision order: first prove task utility against zero credit and native BPTT using the corrected teacher; if it helps, implement this lower-retention teaching path and compare complete work. Optimizing an ineffective critic first would not advance the target.
