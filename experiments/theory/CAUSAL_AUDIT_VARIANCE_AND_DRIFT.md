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


## Confirmation and packet clock correction

Confirmation gate fails. Causal/zero variance ratios are0.685/0.888/1.066 for histories472/473/474; the new474 history worsens6.6%. Fixed TRAIN ratios1.021/0.776/0.845. Retain zero prediction for single-route auditing. Scalar tracking algebra remains valid, but the existing eta1/8 update acts eight times per issued prefix. For identical residuals its effective prefix rate is1−(1−eta)^8≈0.656 rather than0.125. This is a learning-clock choice, not extra information. Packet-normalized asynchronous updates carry mass eta/n and issued coordinate c_issue, giving additive increments eta/n*(R_j−c_issue). Once a complete packet arrives, the sum equals one SGD step eta*(mean R−c_issue), independently of replica count and arrival order. Interleaved/stale gradients and missing feedback require separate stability/inclusion handling. The component and finite replication/order/version contract are implemented; no variance gain is inferred from that contract.

## A stronger observation design: paired alternatives need no no-op world

For two deterministic route alternatives, let Y_old and Y_new be their future-loss effects relative to the same no-op future. Then

    D=Y_old−Y_new=Loss_old−Loss_new.

The no-op branch cancels, including its own readout learning trajectory, because each branch owns independent optimizer state. Two addressed alternative worlds therefore suffice. They use exactly the same branch count as one sampled alternative plus no-op. Preserve the same continuation random draw and independent branch-local Adam; task proposal/baseline/native carrier remain unchanged. This changes the credit observation design, not the actor architecture or the learning target.

Write mu_old,mu_new for effect means, S=Var(Y_old)+Var(Y_new), C=Cov(Y_old,Y_new). The paired estimator has variance S−2C. The uniform single-route residual estimator, with prediction sum h=qhat_old+qhat_new, returns2Y_old−h or h−2Y_new and has variance

    V_single=2S+(mu_old+mu_new−h)^2.

Cauchy–Schwarz gives C>=−S/2, hence V_pair<=2S<=V_single for every h, under this common joint future law. This includes future noise and arbitrary prediction calibration. Equal two-world count is not automatically equal wall time; verify full replay work and label parity. Paired finite-data variance is a descriptive measurement; this inequality is an algebraic conditional guarantee. No stationarity assumption is used. Fixed actor/proposal here; selecting the pair after seeing these same futures would invalidate the comparison's causal interpretation.

For general policy differences d, m=sum d_positive=L/2, sample a positive and negative route under a coupling with marginals d_positive/m and d_negative/m. Then m*(Y_positive−Y_negative) is unbiased for sum d q. Coupling chooses which counterfactuals share a future; its creation/route discovery cost remains paid. For identical policies m0, no comparison audit is required. Losing/unchosen routes remain represented by supported interventions; this is not a dense scan made free.

A learned contrast critic still matters when paired audits are sparse. With prediction ghat and an independent Bernoulli audit I of probability p>0, return ghat+I/p*(D−ghat). Conditional variance is Var(D)/p+(1/p−1)*(E[D]−ghat)^2. At p1 the critic cancels from this exact comparison; at p<1 better contrast prediction can reduce expensive audit frequency. Thus paired forward evidence should train the connected contrast critic, which can then price/allocate its next audits. This connects the useful decision function to economical residual correction without training irrelevant cost common mode. Sparse-audit depth/whole-model integration is the next stage, not demonstrated by pair replay alone.

Admission: contract two-world score/Adam/cost-difference parity against three-world no-op reference and finite variance dominance; fresh stream475,32 prefixes×8 continuations with unchanged models/proposals. Generate three worlds for the diagnostic comparison only; operational paired comparison uses the two alternatives. Report all three-world diagnostic work and actual two-world kernel measurement, no projected total-fit win. Earlier histories are development analysis; new475 is a prospectively declared execution/variance measurement, not an official benchmark.


Completed new475 measurement:paired variance0.000243887 versus zero single-route0.000621066, ratio0.392691 (60.7% lower). Same32 prefixes×8 futures;13 prefixes change. Even choosing the best of all four single-route arms separately for each prefix gives paired ratio0.618162; that best-arm choice is diagnostic, not deployable selection. Native two/three-world score, optimizer and cancellation errors<=8.9e−16. Ten measured two-world replay iterations cost2.641920MFLOPs in both designs, instrumented wall2.643/2.717s. Whole research phase30.60s,317.1MiB RSS; policy9.034MFLOPs, diagnostic teacher41.214MFLOPs/312 worlds, TRAIN preparation2.040MFLOPs. Operational pair counts208 worlds; actual fresh measurement pays312 worlds for reference coverage and104 legacy calibration updates. All earlier fits remain additional. This is a single-actor estimator-economy result, not a model-quality/full-gradient/depth win. Source-bound results `credit/curie_paired_audit_20261011T0740Z_contract/smoke/measure` JSON/PT; drift and packet-normalization contracts also completed. Next integrated target is learned contrast prediction with positive-probability paired residual audits, evaluated by causal forward utility and complete fitting cost.

Resource-boundary condition: equal two-world work above is the measured full-replay boundary. If an existing factual/no-op trajectory can be reused, compare the actual marginal work of both estimators, including version alignment and replay. Variance dominance survives the declared joint-law assumptions; equal marginal compute is not automatic under another reuse protocol.
