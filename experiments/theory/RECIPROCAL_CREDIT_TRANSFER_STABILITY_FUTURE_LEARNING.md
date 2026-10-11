# Reciprocal credit that transfers, remains stable and improves future learning

Current constructive theory, 11 October 2026. The target is a general-purpose substrate for language, reasoning, world models, embodiment and continual learning: temporal computation, sparse addressed state, hard-route alternatives and capacity beyond active work. This note derives mechanisms for achieving transfer, stable credit and economically useful exploration. Its equations specify what to build and what its numerical contracts must preserve.

The construction combines standard directional-derivative, conditional-expectation, implicit-function and information-filter identities. The research contribution is their integration into forward-taught, state-conditioned credit and future-value intervention policies, with complete acquisition and correction accounting.

## 1. Put credit in coordinates that survive a change of state

Let a continuous factual boundary be

$$h=F_\theta(x,\tau;s),\qquad c=\nabla_h\ell,$$

where $s$ is persistent numerical state and $\tau$ is computational elapsed time. The true boundary credit is $g=J^Tc$, with $J=D_{(x,\tau)}F_\theta$. Hard-route changes have a separate intervention target; their value is not obtained by differentiating only the factual route.

A matrix fitted at one point learns $J(z_0)^T$. A transferable construction instead separates current-state geometry from reusable influence:

$$g(z,c)=g_0(z,c)+T(z)K_*(e(z))R(z)c.$$

Here $z=(x,\tau,s,\theta,\text{learner state})$; $g_0$ is a cheap known contribution; $R,T$ use the current forward packet; and $K_*$ is the expensive residual influence in canonical coordinates. Train $K_\psi$, rather than making it rediscover clocks, gates and normalization at every event.

**Transfer bound.** If the residual has approximation error $\epsilon_{\rm rep}$ and $\|K_\psi(e)-K_*(e)\|\leq\epsilon_K$ on the represented domain, then

$$\|\widehat g-g\|\leq \|T(z)\|\epsilon_K\|R(z)c\|+\epsilon_{\rm rep}.$$

This follows by subtracting the two factorizations and applying the operator norm. A change of state is handled by recomputing $T,R$ from the new primal, while shared learned influence remains reusable. Balanced coordinates reduce the amplification factors; state/version descriptors handle the remaining dependence of $K_*$.

**Native realization.** The forward packet already exposes temporal decay/rotation, write gates, writes, normalization centers/inverse scales and GELU slopes. These supply the state-dependent local factors. For a temporal state

$$z_t=D_tz_{t-1}+b_t,\qquad D_t=\exp[(-\rho+i\omega)\tau_t],$$

its exact local adjoint is

$$\lambda_t=a_t+D_{t+1}^T\lambda_{t+1}.$$

Learn the costly future boundary influence; transport it through the current local factors. A boundary error obeys $e_t=D_{t+1}^T\cdots D_T^Te_T$, so the temporal transport component is bounded by

$$\|e_t\|\leq\exp\!\left[-\rho_{\min}\sum_{j=t+1}^{T}\tau_j\right]\|e_T\|.$$

Cross-layer couplings are included in their own transport factors. The construction keeps exact cheap geometry and spends learned capacity on the expensive future influence.

## 2. Train representations to preserve interventions, not merely predict the past

Write the complete learning state as $S$: forward memory, credit memory, parameters, optimizer moments and version information. An intervention $a$ changes it by a causal map $T_a$. A latent representation $e(S)$ should preserve both ordinary transitions and the interventions that the learner might take:

$$e(T_a(S,u))=\overline T_a(e(S),u),\qquad \ell(S,u)=\overline\ell(e(S),u).$$

**Intervention-transfer theorem.** Under these equalities, every finite-horizon future loss and every paired intervention advantage is identical in the full and latent systems. Proof: induction preserves the represented state at every step; equality of each loss preserves its sum. The same reasoning covers a learning intervention when parameters and optimizer state are included in $S$.

For approximate representations, suppose each latent transition has error at most $\epsilon_T$, loss matching error at most $\epsilon_\ell$, latent dynamics are $L_T$-Lipschitz and latent loss is $L_\ell$-Lipschitz. Starting from an exactly represented initial state, for discounted future loss with $\gamma L_T<1$,

$$|V(S)-\overline V(e(S))|
\leq {\epsilon_\ell\over1-\gamma}
 + {\gamma L_\ell\epsilon_T\over(1-\gamma)(1-\gamma L_T)}.$$

Proof: state discrepancy at step $t$ is at most $\epsilon_T\sum_{j=0}^{t-1}L_T^j$; sum its loss effect geometrically. A paired advantage has at most twice this bound.

**Training consequence.** Use forward transition consistency, observed loss matching and counterfactual contrast consistency together. Condition on the actual change in addressed state, clocks, parameters and moments; an action identifier alone need not preserve its effects. Counterfactual packets are feedstock for learning an intervention-preserving representation. Allocate additional observations where estimated representation error matters to future decisions.

## 3. Forward observations train credit and its reliability in one finite loop

Let a shared producer have parameter $p$ and outputs $(x,\tau)$, with tangents $(A_x\xi,A_\tau\xi)$. Given issued predictions $(\widehat g_x,\widehat g_\tau)$, the forward residual observation is

$$\delta=c^T(J_xv+J_\tau w)-\widehat g_x^Tv-\widehat g_\tau^Tw.$$

For independent isotropic cut probes, $E\delta^2$ is the sum of the two cut residual energies. For a common producer probe $v=A_x\xi,w=A_\tau\xi,E\xi\xi^T=I$,

$$E\delta^2=\|A_x^Tr_x+A_\tau^Tr_\tau\|^2.$$

This includes the cross term between content and clock credit. Train the credit field with $\tfrac12\delta^2$, keeping the measured response and output cotangent fixed for the issued observation. Train its conditional reliability from the same squared observation. The credit estimator's local parameter gradient is obtained by ordinary finite differentiation; no new learned teacher is needed to train that gradient.

For a positive semidefinite producer metric $Q$, use $E\xi\xi^T=Q$. Then

$$E\delta^2=r_p^TQr_p,\qquad r_p=A_x^Tr_x+A_\tau^Tr_\tau.$$

Thus a future-learning metric can change what credit error is valuable to reduce, without changing the basic observation mechanism. Fresh probes independent of the issued estimator give conditional expectation identities. Reused probes train an empirical objective; fresh held-out observations measure its transfer.

For shared elapsed time across layers, $h_l=F_l(h_{l-1},\tau)$, define $A_l=D_hF_l,B_l=D_\tau F_l$. The exact composition is

$$v_l=A_lv_{l-1}+B_lw,\quad c_{l-1}=A_l^Tc_l,\quad
g_\tau=\sum_l B_l^Tc_l.$$

Every shared-clock contribution enters once. Discounting belongs to a declared future objective, not to this factual chain rule.

## 4. Preserve memory; stabilize the error that tracks its credit

Let $e_t$ be credit-state error, $A_t$ its transported dynamics and $d_t$ drift from a changed model or target. An exact audit can reset the tracking error, giving

$$e_{t+1}=(1-I_t)(A_te_t+d_t),\qquad P(I_t=1\mid\mathcal H_t)=\pi_t.$$

Without drift, if $\|A_t\|\leq M_t$,

$$E[\|e_{t+1}\|^2\mid\mathcal H_t]\leq(1-\pi_t)M_t^2\|e_t\|^2.$$

**Constructive stability rule.** Choose a desired contraction $q<1$ and impose

$$\pi_t\geq\max\!\left(0,1-{q^2\over M_t^2}\right),$$

with additional positive support on the audited domain. Credit tracking then contracts in mean square. A neutral memory mode with $M_t=1$ needs correction opportunities, not forced forgetting of the forward memory. When useful computation amplifies sensitivity, audit demand increases accordingly.

For drift bounded by $\|d_t\|\leq D$, Young's inequality gives, for any $\kappa>0$,

$$E\|e_{t+1}\|^2\leq
(1+\kappa)(1-\pi_t)M_t^2\|e_t\|^2
 +(1+1/\kappa)(1-\pi_t)D^2.$$

If the first coefficient is at most $q^2<1$ uniformly, the error is bounded by its decaying initial term plus the drift term divided by $1-q^2$. This supplies a version-aware correction rule. Bounds on $M_t$ can use current packet factors; learned estimates can propose corrections and measured anchors certify them.

**A stable last layer.** A canonical credit last layer with an $m$-strongly convex, $L$-smooth observation objective tracks its optimum with contraction $q$ under a suitable step size. If $\psi_*(\theta)$ is $C$-Lipschitz and actor movement per step is at most $d$, then

$$\|\psi_{t+1}-\psi_*(\theta_{t+1})\|
\leq q\|\psi_t-\psi_*(\theta_t)\|+Cd.$$

Its tracking floor is $Cd/(1-q)$. A regularized linear influence layer is a concrete implementation. More informative forward observations increase its precision; slower shared-representation drift and audit resets reduce the tracking floor.

These bounds concern credit error, not erasure of persistent information. Long-lived forward state and stable learned-credit tracking are compatible.

## 5. Train the allocation policy on its actual resource objective

For true producer gradient $g_i$, prediction $\widehat g_i$ and residual $r_i$, use

$$\widetilde g_i=\widehat g_i+{I_i\over\pi_i}r_i.$$

With an issued, version-bound prediction and positive inclusion probability,

$$E\widetilde g_i=g_i,\qquad
\operatorname{tr}\operatorname{Cov}(\widetilde g_i)
 =R_i(1/\pi_i-1),\quad R_i=\|r_i\|^2.$$

For independent audits, summed-gradient covariance traces add; averaging divides all policies by the same squared batch size. In metric $Q$, replace $R_i$ by $r_i^TQr_i$.

With audit costs $k_i$, budget $B$ and support/stability floors $f_i$, the allocation problem is

$$\min_\pi\sum_i R_i(1/\pi_i-1),\quad
\sum_i k_i\pi_i=B,\quad f_i\leq\pi_i\leq1.$$

The KKT solution is $\pi_i=\operatorname{clip}(\sqrt{R_i/(\lambda k_i)},f_i,1)$, with $\lambda$ enforcing the budget. A trainable policy can optimize this fitness directly using forward labels $y_i$ with $Ey_i=R_i$, rather than fitting an intermediate risk by squared error.

For equal costs, set

$$\pi_i=f_i+(1-f_i)\sigma(s_i-\tau),\qquad \sum_i\pi_i=B.$$

With $q_i=(1-f_i)\sigma(s_i-\tau)[1-\sigma(s_i-\tau)]$, the exact implicit adjoint is

$$D_s\pi^Tu=q\odot\left(u-{\sum_iq_iu_i\over\sum_iq_i}\mathbf1\right).$$

The scalar threshold and its credit close analytically. For unequal costs, replace the subtracted vector by $k(\sum_iq_iu_i)/(\sum_i k_iq_i)$. Feasibility requires $\sum_i k_if_i<B<\sum_i k_i$. If stability floors exceed the available budget, change the transport geometry, step size or resource allocation; do not silently violate the stability rule.

## 6. Turn variance reduction into a future-learning objective

Let a future loss be quadratic with Hessian $H\succeq0$. Let the next update be $\theta'=\theta-\eta M\widetilde g$, where $M$ is fixed before the current audit draw. For two policies with the same mean gradient,

$$E\mathcal L(\theta'_{\rm old})-E\mathcal L(\theta'_{\rm new})
={\eta^2\over2}\operatorname{tr}\left[HM
(\Sigma_{\rm old}-\Sigma_{\rm new})M^T\right].$$

**Future-learning consequence.** Optimize audit residuals in metric $Q=M^THM$. Lower weighted audit variance then gives strictly better expected next-step future loss whenever the weighted trace gap is positive. Raw Euclidean variance is its $H=M=I$ instance. This states the mechanism that connects an allocation gain to learning, and specifies the metric to learn when future loss geometry is different.

For a Hessian-Lipschitz smooth loss with constant $L_H$, the third-order remainder of an update is at most

$${L_H\eta^3\over6}E\|M\widetilde g\|^3.$$

A sufficient local learning-gain rule is

$$\eta< {3\Delta_Q\over L_H(E\|M\widetilde g_{\rm old}\|^3+
E\|M\widetilde g_{\rm new}\|^3)},$$

where $\Delta_Q>0$ is the weighted covariance-trace improvement. This gives a trust region in which the second-order improvement dominates both remainder bounds. Positive audit support controls extreme correction magnitudes; step size, metric and correction probabilities are coupled design variables.

Learn the future metric from observed future losses, paired learner interventions and directional curvature observations. A positive semidefinite low-rank-plus-diagonal parameterization preserves the metric's meaning. Current-step Adam moment updates are nonlinear; a predictable preconditioner supplies the update identity above. Actual optimizer transport and future returns supply their own measured targets.

## 7. Explore for future information value, not uncertainty alone

Let canonical influence parameters have posterior covariance $\Sigma$. A query observes $y=a^T\beta+\epsilon$, with noise variance $\sigma_a^2$. Under a linear Gaussian observation model,

$$\Sigma'=\Sigma-{\Sigma aa^T\Sigma\over\sigma_a^2+a^T\Sigma a}.$$

If future learning loss attributable to influence uncertainty is $\operatorname{tr}(Q\Sigma)$, its exact expected reduction is

$$\Delta(a)={a^T\Sigma Q\Sigma a\over\sigma_a^2+a^T\Sigma a}.$$

**Exploration rule.** Query when $\Delta(a)$ exceeds its acquisition cost in the declared future-loss/resource units. A highly uncertain direction can have little future value when $Q$ assigns it little demand. A moderately uncertain direction used repeatedly can justify investment. For $N$ comparable future uses, value scales as $N\Delta_{\rm per\ use}$; the acquisition crosses its payoff threshold at $N\Delta_{\rm per\ use}>\text{cost}$. Drift shortens effective reuse and appears as process noise in the information filter.

This supplies a fitness for a trainable exploration-value estimator: predict the improvement in future learning, including precision gained and the demand for that precision. The information-filter update and local calculus provide credit without another learned trainer. Learn $Q$, query noise, drift and representations from realized future use and paired intervention returns. Maintain declared support for potential interventions so an initially wrong demand forecast cannot permanently exclude useful routes.

For finite learner-state interventions, the observed target is

$$U_H(a)=\mathcal L_H(S^{\rm skip})-\mathcal L_H(S^{a})-\lambda\,\operatorname{cost}(a).$$

Use the same future stream and matched random choices where appropriate. A structured influence predictor uses the causal change $\Delta S_a$:

$$\widehat U_H(a)=-\mu_\psi(S)^T\Delta S_a
-\tfrac12\Delta S_a^TH_\psi(S)\Delta S_a-\lambda\,\operatorname{cost}(a).$$

It is zero for a zero intervention before cost, shares influence across actions and makes learner/optimizer geometry explicit. Observed paired future losses anchor its residual. This is the future-return target; audit variance is an economical surrogate when the learning metric just derived applies.

## 8. A finite reciprocal cycle with a constructive implementation path

1. The forward model predicts before feedback and records causal state/clock/write geometry.
2. Feedback supplies local observed credit. Current packet factors canonicalize it; a learned influence layer supplies the expensive remaining credit.
3. Measured forward responses train the influence and reliability layers. Reuse observations, with version and horizon bindings.
4. The audit policy optimizes weighted correction variance under cost and stability floors. Audits correct raw gradients, teach the influence layer and anchor credit state.
5. Learner parameters and moments advance. Information and drift state record what credit can be reused.
6. Exploration acquires the intervention whose expected future learning improvement pays for its work. Paired future losses train the value and demand estimators.

All training targets ultimately terminate in observed outcomes, measured directional responses or declared exact audits. Learned estimators improve reuse and allocation; analytic local derivatives and information-filter updates supply a finite base. Shared parameters accumulate the connected contributions of the declared losses. Detaching an issued teacher defines a semi-gradient objective; retaining a finite causal graph carries gradients across flips. Arbitrary loss weights do not establish a stability contraction.

For a stationary linearized two-way credit system,

$$\lambda_F=b_F+\gamma A^T\lambda_C,\qquad
\lambda_C=b_C+\gamma B^T\lambda_F,$$

the fixed point is

$$\lambda_F=(I-\gamma^2A^TB^T)^{-1}(b_F+\gamma A^Tb_C)$$

when $\rho(\gamma^2A^TB^T)<1$, so the corresponding Neumann series converges. This is a local solver identity. A relaxation can change iteration behavior while preserving the fixed point; changing the transport discount changes its target unless that discount belongs to the declared objective. The stable-error and future-value mechanisms above address transfer and learning beyond this local algebra.

### Current implementation anchors and next decisive contracts

- Shared content/time native transport and all-parameter Adam parity at depths 2/4/8; fused joint teaching acquisition through depth 32.
- Neural forward-teaching loss gradients match exact cut and shared-producer objectives. Local learned fields transfer across terminal directions; new-state tests identify the need for current geometry.
- State-conditioned audit-risk acquisition at64/256/1024contexts; all raw records preserved. Direct variance-fitness policies reuse them, with fresh128-context evaluation, fixed budget and measured inherited work.
- Next mathematical contracts: intervention-representation error bound; credit-reset mean-square contraction without forward-memory decay; posterior query value versus future quadratic loss; weighted audit variance versus expected next-step learning loss.
- Next native construction: current-packet transport around a reusable low-rank influence layer, uncertainty/drift state and metric-aware direct allocation. Preserve hard-route counterfactuals and persistent state; compare actual future prediction quality per complete learning work.

The numerical evidence and protocols are linked in [forward-observed credit risk](FORWARD_OBSERVED_CREDIT_RISK.md), [native temporal packet](../credit/NATIVE_TEMPORAL_TEACHING_PACKET.md), and the [two explanatory diagrams](../../report/LEARNED_CREDIT_DIAGRAMS.md).
