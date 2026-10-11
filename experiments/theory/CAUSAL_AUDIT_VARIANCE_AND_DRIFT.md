# Causal calibration separates useful decisions from economical learning

B1/R1,11 October2026. This note develops the [task-valued credit gate](TASK_VALUED_CREDIT_GATES.md) after the observed common-mode audit failure. It does not replace the connected route critic, temporal forward computation or persistent learner state.

## Two objectives from the same forward evidence

A route decision depends on cost contrasts. A supported residual audit also depends on cost levels. Let d=p0−p, L=sum|d| and r=|d|/L. Given an issued policy/prediction packet, the audit prediction qhat+c gives gain estimate

    Ghat + d(A)/r(A) [Y_A−qhat(A)−c],
    Ghat=sum d qhat.

Because sum d=0, c changes neither contrasts nor the prediction sum. With conditionally unbiased observations of the declared learner-aware future, the estimate's mean equals the real policy difference for any issued c. Its conditional variance is minimized at m=E_r[Y−qhat]; the excess variance is exactly L^2(c−m)^2. Thus the same future observations can teach two distinct outputs: action contrasts for task value and common-mode calibration for credit economy. A low task loss does not guarantee low audit variance. This is known control-variate/sample-mean algebra applied to the versioned native route/learner construction, not a claim that scalar centering alone is a novel general learning rule.

The branch learner, optimizer state, causal horizon, discovery and sampled support are part of the target. Predictions are issued before future feedback. Later calibration changes cannot rewrite an issued packet. Wrong-target and duplicate feedback is rejected; out-of-order feedback can update current calibration while its old issued estimator remains fixed. Delayed evidence can be less useful for the next forecast even though it is valid evidence for the old target. All packet storage/feedback work remains paid.

## Tracking nonstationary residual bias

Index t by feedback arrivals. Let the raw sampled residual be R_t, conditional mean m_t, noise xi_t=R_t−m_t with E[xi_t|H_t]=0, and conditional second moment bounded by sigma^2. The current c_t is measurable before R_t. The update is

    c_(t+1)=(1−eta)c_t+eta R_t,       0<eta<=1.

With e_t=c_t−m_t and drift Delta_t=m_(t+1)−m_t,

    e_(t+1)=(1−eta)e_t+eta xi_t−Delta_t.

No stationary inductive-bias premise is needed. If ||Delta_t||_L2<=delta and a_t=||e_t||_L2, then orthogonality of e_t and xi_t plus the L2 triangle inequality gives

    a_(t+1) <= sqrt((1−eta)^2 a_t^2+eta^2 sigma^2)+delta.

This allows drift correlated with the observed noise, as in a coupled learner. Define

    B=delta/eta + sigma sqrt(eta/(2−eta)).

Then a_t<=B+(1−eta)^t max(a_0−B,0). Proof: the scalar map on the right is increasing and(1−eta)-Lipschitz. Write B=D+S with D=delta/eta and S=sigma sqrt(eta/(2−eta)). Direct expansion yields

    (B−delta)^2−(1−eta)^2 B^2−eta^2 sigma^2
        = 2eta(1−eta)DS >=0,

so the map sends B to at most B. Induction/Lipschitz control bounds any initial excess. At eta1, the bound becomes delta+sigma and the initial excess vanishes after one update. If delta0 and noise variance is constant, the exact stationary mean-square term is eta sigma^2/(2−eta), under the martingale-noise premises. Increasing eta tracks faster drift but pays more sampling noise. Unknown sigma/delta are not measured constants in the current native pilot.

Since calibration-induced excess audit variance is L^2 e_t^2, bounded L<=2 gives an expectation bound4 a_t^2. Future innovation leaves an irreducible variance floor. For independent repetitions of an issued packet with variance V and variance budget v, n>=V/v is sufficient; adaptive histories and correlated outcomes require their covariance terms. The measured finite-continuation/prefix variance is not a population confidence bound. Choose economy by charging the predictor, calibration, candidate discovery, delayed storage and actual replay, not by counting only selected messages.

## Execution and first evidence

The main task proposal and shared-prefix response remain fixed in this stage; four readout Adam updates occur within each synthetic continuation. Causal calibration then learns across episode prefixes from sampled earlier feedback. First fresh stream472 gives31.5% lower full route-plus-continuation audit variance than zero prediction (0.000491 versus0.000717), while fixed TRAIN calibration does not improve. This is one frozen-actor history. Two fresh independent histories473/474 are admitted with unchanged eta1/8 and TRAIN initialization, with a predeclared10%-reduction gate. No real nonstationary-market, deep actor co-adaptation or whole-fit scaling claim.

Addressed native replay scores only specified worlds and agrees with selected72-world outputs/Adam updates within7.3e−16. The72-versus3 kernel work ratio is24, with different output coverage; it is not a24x full learning speed claim. Operational sampled comparisons need two worlds (chosen action and common no-op); this research stage pays three to measure variance. Native eight-receiver policy discovery remains charged.
