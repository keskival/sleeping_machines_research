# Build the forward process to make future learning predictable

Curie, 10 October 2026. B1/R1 formal development. Algebraic propositions below are proved under stated assumptions; numerical witnesses validate finite instances, not general neural convergence. This extends [reciprocal closure](CLOSED_LOOP_EXPLORATION_AND_CREDIT.md), [event credit economics](ECONOMICAL_EVENT_CREDIT.md), [supported route credit](UNSEEN_ROUTE_CREDIT.md) and [depth scaling gates](../DEEP_LEARNING_SCALING.md). It does not replace their architectures or completed evidence.

## 1. The joint design objective

An event-native learner has forward state S, persistent credit state C, forward parameters theta, credit parameters psi, optimizer state O and versioned retained packets D. Issued predictions precede their feedback. Hard routes, timed races and sparse addressed messages remain explicit transition choices. The augmented transition includes learning, so the objective is expected later prediction loss plus charged work, rather than reconstruction of the current backward pass alone.

Use two separately declared targets: (i) a finite or discounted future intervention value, whose continuation includes subsequent learner updates; (ii) a parameter contribution Y to a specified smooth objective or a specified stochastic score estimator. A cotangent lambda becomes parameter credit Y=A lambda through its actual producer map A. Fixed-path optional key-write effects in the completed receiver experiment are a tractable instance of (i), not complete changing-weight intervention values. The results motivate shared receiver-conditioned representations; they do not certify the full objective.

A causal packet Z=f_phi(H) and persistent credit memory predict h_psi(Z). H includes the causal history, current feedback when legally revealed, relevant state and parameter versions. All information must exist before the audit draw. The packet can carry race margins, alternative receiver summaries, decay, activation projections, elapsed time and compressed producer information. Its construction, reading and training cost count. A connected credit model may be large and share across receivers; economizing means sparse evaluation and sufficient packets, not imposing tiny disconnected predictors.

## 2. Credit information theorem: separate innovation from compression

Let Y be square-integrable parameter credit and let Z be measurable with respect to H. Define m(H)=E[Y|H], h*(Z)=E[Y|Z]. All quantities refer to one declared conditional data/learner law. Then

    R*(Z) = E||Y-h*(Z)||^2
          = E||Y-m(H)||^2 + E||m(H)-E[m(H)|Z]||^2.                 (1)

Proof: write Y-h*=(Y-m)+(m-h*). The cross term has expectation zero because E[Y-m|H]=0 and m-h* is H-measurable. The tower property gives h*=E[m|Z]. For any fitted h(Z),

    E||Y-h(Z)||^2 = R*(Z) + E||h*(Z)-h(Z)||^2.                  (2)

Thus there are three errors: future innovation, information discarded by the packet, and credit function approximation/fitting. Additional critic capacity addresses the third; better forward packets and persistent memory address the second; neither predicts genuinely unobserved innovations. The forward pass should teach both packet sufficiency and the credit function, rather than supplying only current activations to a fixed estimator.

These are projection identities, not new conditional-expectation mathematics. Their architectural implication is a distinct training objective: preserve information about conditional future *learning consequences*, not just present predictions. If A is H-measurable but absent from Z, predict Y=A lambda directly or expose adequate producer information. Minimizing unweighted cotangent MSE can preserve producer-null directions while losing important parameter directions. If a causally fixed preconditioner P matters, use Y=P A lambda and charge its application; this does not cover nonlinear Adam with an update-dependent denominator.

**Prediction sufficiency is not credit sufficiency.** Let x be equally likely +/-1, y=1, f_theta(x)=theta*x, and loss=(f-y)^2/2. At theta=0 every forecast is zero, yet its derivative is -x. A packet retaining only the forecast discards all sign information needed for credit. Retaining x makes the derivative exact. There is no contradiction: a sufficient statistic for one prediction at one parameter version need not be sufficient for learning that prediction function.

For nested packet information Z1 contained in Z2,

    R*(Z1)-R*(Z2) = E||E[Y|Z2]-E[Y|Z1]||^2 >= 0.              (3)

This quantifies the ideal benefit of persistent credit state, counterfactual summaries or receiver descriptions. Net benefit subtracts their actual work and approximation cost. Finite training can still get worse after adding information.

## 3. Learn a credit-relevant geometry, not a fixed symmetry list

Two causal histories are credit-equivalent for this target when m(H1)=m(H2). This is a target-, version-, horizon- and regime-dependent equivalence relation. A shared function can extend credit across these equivalence classes even when the forward observations look different. An admissible relabeling T obeys m(T H)=T_* m(H); scalar intervention values have the corresponding invariant law. The completed native receiver relabeling contract tests a known coordinate action. Learning useful classes beyond those coordinates is the next generalization problem.

Equality of means is sufficient for squared-error-optimal credit, not for tail-sensitive routing or acquisition. If those decisions depend on uncertainty, the conditional response distribution, risk and delay must also be represented. Nonstationary regimes change these equivalences causally: C carries a belief about the current regime, and m_t(H) changes accordingly. Stationarity of the raw data or of a fixed symmetry group is not required by (1); tracking error and insufficient regime information enter the measured risk.

A simple capacity witness makes geometry concrete. Center m and let its covariance have eigenvalues lambda1>=...>=lambdap. If the packet can encode exactly a rank-k orthogonal projection of m and decode that projection, the best linear reconstruction risk is sum_{j>k} lambdaj. By (1), its total risk is innovation plus this spectral tail. Proof: captured variance is tr(P Cov(m)); maximizing it over rank-k orthogonal projections selects the top eigenvectors. This is the known principal-component optimum, an *oracle witness*: m is not available free during inference. Audited outcomes must teach a causal packet approximating those useful directions. It suggests measuring credit spectra and their change with depth rather than arbitrarily making every backward module small.

## 4. Predictability translates into exact-audit work

At a versioned boundary let Y be the exact declared contribution, h a pre-audit prediction, and B independently Bernoulli(a), 0<a<=1. Set

    Yhat = h + B/a * (Y-h).

Conditional on H,Y,h, E_B[Yhat]=Y and

    E||Yhat-Y||^2 = (1/a-1) R,  R=E||Y-h||^2.                 (4)

Also tr Cov(Yhat)=tr Cov(Y)+(1/a-1)R, by conditional variance. An exact conditional mean h=m(H) preserves expected future credit while avoiding some expensive factual realization work. Its innovation still creates audit error when the target is the realized Y. Choosing a future expected-gradient objective does not make a learned m exact; supported residual correction anchors its mean.

For allowed *additional* audit MSE tau>0, constant auditing needs

    a >= R/(R+tau).

Let w_h include packet discovery, prediction, amortized credit fitting, retained state and extra optimizer work; let c be incremental exact-audit work. This mechanism costs w_h+a*c versus c for the same direct exact contribution. It saves expected work precisely when

    w_h < c*tau/(R+tau),                                    (5)

at the minimal constant a. Count local factual work common to both separately, and include earlier teacher/pretraining work in whole-fit comparisons. At zero allowed audit error and R>0, a=1. If R=0, prediction is exact almost surely and the limiting audit cost can vanish; finite evidence never proves R=0. An exploration floor adds its cost.

This is an expected-work comparison for an estimator, not matched-quality training superiority. To link it to optimization, assume a fixed L-smooth objective F, an H-measurable unbiased stochastic gradient G, and a corrected estimate Ghat with added variance bounded tau. For theta'=theta-eta*Ghat,

    E[F(theta')|H] <= F(theta)-eta||grad F(theta)||^2
                    + L*eta^2/2 * (E[||G||^2|H]+tau).       (6)

This follows from the descent lemma and unbiasedness. Nonstationary next-step F, stale targets, clipping and update-dependent nonlinear optimizers require their own analysis. Equation (6) gives a declared local tolerance, not a claim that existing real-stream pilots reach matched quality.

## 5. A depth criterion that can fail

Suppose disjoint producer contributions sum to G=sum_{i=1}^D Y_i. Each Y_i already includes the correct map to the shared parameter space; overlapping cuts must not double-count. Conditional on the full current targets and predictions, audit draws are independent, so cross audit-error terms vanish. Write r_i=E||Y_i-h_i||^2 and exact cost c_i. The added MSE is

    V = sum_i (1/a_i-1) r_i,
    W = W_h + sum_i a_i*c_i.                                (7)

Targets can share future noise; the zero cross term here concerns independently centered *audit corrections*, not independence of the targets. Correlated race audits need joint-propensity covariance terms. If predictions are densely summed, that cost belongs to W_h. Sparse baseline aggregation or the separately proved supported route sampler is required to avoid that work; its route-sampling variance is additional.

Minimizing audit work with V<=tau gives, for positive r_i and c_i and no active caps,

    a_i = sqrt(r_i/c_i) * S/(tau+R),
    S=sum_i sqrt(r_i*c_i), R=sum_i r_i,
    W_audit = S^2/(tau+R).                                  (8)

Proof: minimize sum c_i*a_i with constraint sum r_i/a_i<=tau+R. Stationarity gives c_i=lambda*r_i/a_i^2, and equality fixes sqrt(lambda)=S/(tau+R). Convexity in the probabilities establishes optimality. Clamp probabilities at one and solve for the remaining multiplier when caps activate; supported floors similarly modify the active set. Actual implementation substitutes causally estimated r_i, retaining recorded propensities, so oracle optimality is a benchmark rather than an assumption.

For equal per-cut r_D and c_D, a=D*r_D/(tau+D*r_D), and the work ratio to exact auditing is

    W/(D*c_D) = W_h/(D*c_D) + D*r_D/(tau+D*r_D).              (9)

Under this fixed unnormalized-gradient audit tolerance, a nonvanishing residual floor per cut makes the second term approach one with depth. Tiny isolated predictors with persistent error do not automatically become economical on deeper models. A shared critic/forward packet must improve aggregate residual risk, or exact-audit cost must grow relative to its overhead, or the sampling/aggregation design must improve. For constant c_D and negligible overhead, r_D=o(1/D) yields a vanishing audit ratio. That is a sufficient asymptotic witness, not a proposed universal requirement on all architectures.

Changing gradient normalization, tolerance, correlated audits or aggregated cuts changes this conclusion. In particular, averaging D contributions scales each contribution and its squared risk; report that denominator explicitly. No universal lower bound against every alternative credit method follows from (9). The decisive experiment records r_i, producer maps, W_h, c_i and tau at each depth, and then tests actual work to matched quality. The theoretical prize is *jointly better predictability and cheaper execution*, not depth alone.

## 6. Asynchronous versions change economics before they change mean

Let h_fresh and h_stale predict the same current contribution Y. If their L2 difference is at most d, Minkowski gives

    sqrt(R_stale) <= sqrt(R_fresh)+d.                       (10)

A proved Lipschitz condition can supply d<=K times state/parameter drift plus a separate distribution-tracking term. Elapsed wall time alone supplies no such bound. Plugging (10) into (4) bounds the extra audit variance of stale predictions.

If an audit recomputes Y at the current intended version, using a stale h in the residual formula still preserves the current conditional mean. If the audit instead returns Y_old, the estimator targets Y_old: residual correction does not repair stale labels. This separates *prediction staleness*, which costs variance, from *target staleness*, which changes the objective. Online events need addressed producer/learner versions and an explicit rule for replaying or transporting targets.

A clockless implementation can prioritize packet refresh by estimated increase in producer-risk divided by refresh cost. Factual messages and credit messages may travel on different schedules. Pending storage, discovery, refresh, exact replay and race covariance remain charged. Refresh priority is a hypothesis to test; the theorem does not make a universal hardware scheduler optimal.

## 7. What to build and measure next

The complete construction combines deep timed state, hard sparse routes, separate keys/values and forward-produced counterfactual packets with connected persistent receiver-conditioned credit. The packet learner trains to preserve conditional future producer credit and response uncertainty; residual audits ground that target; credit updates teach later forward behavior. Optimizer policy and acquisition remain part of the augmented state/return in reciprocal closure.

Three practical gates serve B1/R1: (1) measure information/approximation/innovation risk using independent continuations at the same causal prefix, keeping finite-rollout uncertainty explicit; (2) compare packet and persistent-state variants under matched information and complete work, including held-out receiver addresses and chronological regime shifts; (3) measure depth-dependent producer risk and audit costs before an end-to-end matched-quality fit. Confirm benefits from forward-to-credit teaching separately from forward access to credit memory.

The receiver experiment already supplies evidence of learned generalization across omitted training addresses. It does not supply m(H), an innovation floor, current-version online transitions, sparse discovery, or the inequalities required for a scaling win. Equations (1), (5), (9) and (10) make those next measurements concrete.
