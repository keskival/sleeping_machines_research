# Forward-observed credit risk closes audit learning

B1/R1 AWS supporting primitive; Curie retains coupled controller/credit ownership. This adds an observation channel without changing the actor architecture or active sources.

Let h=F(x), J=∂F/∂x, and c be an issued downstream loss cotangent. A credit predictor supplies ĝ=Bφ(c,context); freeze the issued predictor and actor version before measuring its residual r=Jᵀc−ĝ. A forward teaching direction v supplies y=Jv. Its scalar reciprocal error is

    δ(v)=cᵀy−ĝᵀv=vᵀr.

For directions with E[vvᵀ|issued coordinates]=I,

    E[δ(v)²|issued coordinates]=||r||².

Thus the same forward response that teaches reciprocal transport can supply a noisy target for the conditional residual-risk estimator. There is no reverse-teaching pass in target construction. Exact reverse is an optional audit/validation instrument, not another learned trainer. An observed downstream cotangent still costs its actual local derivative work; a predicted cotangent defines a different target and must be labelled accordingly.

## Producer-sensitive risk without a dense Jacobian

For producer x=Gθ(z), write A=∂G/∂θ. Choose a parameter-space probe ξ with conditional covariance I and form v=Aξ by a producer forward tangent. Then

    E[δ(Aξ)²|issued coordinates]=rᵀAAᵀr=||Aᵀr||².

This measures the correction in producer parameter space, rather than treating every cut-coordinate error as equally costly. Native embedding/gap producers construct Aξ with an addressed embedding lookup and a small local affine operation. Directions, producer features, full native output work and responses are charged; no dense producer Jacobian is stored. A fixed optimizer metric can enter via directions A P ξ to measure ||PᵀAᵀr||². Changing Adam states, clipping and nonlinear optimizer transitions need their own targets; this identity alone does not preserve the mean of an Adam update.

## Risk-learner closure and sampling

For an issued risk estimate q_w and feature f, the local squared-error gradient is

    (q_w−δ²) ∂q_w/∂w.

Its conditional mean equals the gradient using exact ||r||² because the target enters linearly. The sampled squared loss itself includes an extra target-variance term; fixed issued probes/actor/credit make that term independent of w. If probes depend on w or φ, stop their target path or account for those derivatives. Probes used to fit credit cannot simultaneously be treated as independent residual validation for that same fit: issue risk observations before the update or use a fresh held-out direction.

Selected observations require the recorded positive propensity and inverse-propensity weighting already implemented in predictable_audit_allocation_v2. Risk for a future audit decision uses causally earlier packets. A fresh probe reveals an observation only after its own acquisition decision; it is not a free pre-decision risk oracle. Random projections add conditional noise; probe count, fitting, drift, feature construction and audit expenditure decide their economy.

The finite closure is: actual forward responses/observed loss → reciprocal-error observations → credit and risk fitting → future credit/audit decisions → new actual observations. No recursive trainer stack is required. This closes a residual-risk measurement problem, not optimal exploration or its future-learning objective.

## Registered native contract

One four-event Taxi TRAIN prefix, native temporal layer and actual observed-loss cotangent. Exhaustive Hadamard probe ensembles have exact covariance I in cut and producer coordinates, validating both risk identities and the issued softplus risk-training gradient. Numerical verification uses the exact packet transpose and local producer derivative, charged separately. The declared candidate ĝ=0.5c is a diagnostic, not a trained credit model. No DEV/TEST, learning-quality, sparse-risk or efficiency claim. `forward_residual_risk.py` supplies scalar residual observations; `check_native_forward_residual_risk.py` runs only through the AWS bounded scheduler.

## Completed numerical evidence

aws_native_forward_residual_risk_20261011_v1 completes:32 cut probes/128 producer probes for112 producer parameters. Exact cut risk0.0146475782, producer risk0.0780145134; errors5.20e-18/0 and conditional risk-training gradient error5.55e-17. Label acquisition0.956ms, separate reverse verification0.658ms, whole gate0.049s/305.9MiB. Reverse validation is cheaper here; exhaustive forward labels are not an efficiency win. They provide the observation needed to teach risk from existing teaching packets. Sparse probe variance and learned-risk allocation require the next owned integration experiment.

## Joint content/time and shared-producer correction

For deep output h=F(x,tau), issued terminal cotangent c and predicted fields (g_x,g_tau), a joint forward observation is delta=c^T(J_x v+J_tau w)-g_x^T v-g_tau^T w. Independent isotropic cut directions estimate ||J_x^T c-g_x||^2+||J_tau^T c-g_tau||^2. A shared producer requires v=A_x xi,w=A_tau xi using the **same** parameter-space direction xi. Its squared observation estimates ||A_x^T r_x+A_tau^T r_tau||^2, including2< A_x^T r_x,A_tau^T r_tau >. Adding separately estimated output risks omits this term and can misallocate audits.

B1 registered numerical gate: native depths2/4/8, actual Taxi TRAIN loss cotangent, four events; diagnostic positive internal tempo shares content parameters via0.1mean(content). This exposes cross terms without changing active fits, physical addressed-memory or race clocks. Exhaustive orthogonal directions validate both cut and producer identities and the issued positive risk learner's expected gradient. Forward responses generate labels; separate reverse validation and all preparation are paid. This is a calibration-target contract, not a fitted-risk/audit-utility claim.

Completed aws_joint_forward_risk_20261011_v1: depths2/4/8 combined producer risk error<=1.12e-16 and conditional risk-training gradient error<=5.56e-17. Cross terms1.59e-5/1.37e-4/8.47e-4 are nonzero; separate-output risks omit them. This validates shared directional labels, not learned calibration or sparse-audit utility. Full forward-label and separate reverse-validation timings are retained in the completed record; no almost-free or reverse-cost saving is inferred from the identity.

## Differentiable credit teaching, not just risk labels

The residual-risk helper deliberately detaches its predictions because it produces risk observations. A credit estimator itself instead minimizes L_psi=0.5 E[(g_x(psi)^T v+g_tau(psi)^T w-c^T(J_x v+J_tau w))^2]. Probe directions, actor responses and terminal cotangent are held fixed for this issued observation; gradients flow into the estimator predictions. For isotropic independent cut probes this equals joint credit regression in expectation. For common shared-producer probes it equals0.5||A_x^T r_x+A_tau^T r_tau||^2, preserving cross terms. It identifies only the probe-supported subspace.

Registered B1 gate: small context-conditioned neural estimator at native depths2/4/8, all estimator parameter gradients against exact cut and producer-projected objectives, then four finite local SGD steps. Reverse maps validate targets only; teaching labels use forward responses. One four-event TRAIN context per depth is an optimizer/gradient contract, not acquired held-out structure, downstream utility or efficient complete learning. This finite local loss implements the closed loop depicted in report/learned_credit_explained.pdf without another learned teacher.

## Registered local acquisition certificate

After the joint neural-gradient contract passes, fit a1152parameter linear map from terminal cotangents to joint input/time credit at depths2/4/8. Actor and one four-event Taxi TRAIN context remain frozen. Fixed recipe:512 Gaussian normalized receivers,4 independent Rademacher probes each,400 full-batch Adam steps,lr0.01,zero initialization. Teaching targets use forward responses only. Freeze fitting before64fresh receivers/4fresh probes each. Measure complete joint field error using a separately charged exact VJP after fit, with paired receiver intervals, probe MSE, all acquisition/fit/validation time and RSS. Repeat unchanged recipe at seeds0/1/2 after the bounded first fit is finite.

These are3unique task targets, not512examples of new task data. Success is acquisition of a local transpose map on fresh receiver directions; no state/version transfer, task improvement, hard-route utility, whole-fit savings or deep learned-credit supremacy follows. The exact cached reverse remains a measured alternative. This certificate informs the compact-credit owner's integration rather than replacing its future-learning objective.

## Registered state/version transfer diagnostic

The unchanged local acquisition recipe is repeated only to retain portable field/actor checkpoints and add a distinct transfer endpoint. Evaluate the frozen learned map on the actual observed-loss cotangent in its training context, two new four-event TRAIN contexts at the same actor weights, then the original context after one conventional Adam actor update(lr0.001). No credit refit or selection on these evaluations. Measure joint-field and shared-producer residual energies versus zero credit, separate diagnostic wall time, and preserve the trained version/context in the bundle. An expired map is an approximation requiring revalidation/audits, never an exact teacher.

Fixed-context fresh-receiver success alone cannot certify state-conditioned credit or changing-weight transfer. This test addresses that gap explicitly and informs which conditioning/transport the compact integration needs. No future task-quality or whole-learning advantage claim is admitted from it; repeated acquisition investment is paid again. Seeds1/2 repeat only after seed0 is finite.
