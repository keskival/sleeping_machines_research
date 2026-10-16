# Why readable parts need not yield a learnable whole: task credit geometry

16 October2026. Formal analysis and a frozen actual-actor audit proceed while the already admitted data/training run continues unchanged. Its final endpoint measures budget adequacy;it cannot by itself identify a cause. No new ablation or fit is required for the analysis below.

## Measured distinction

At both128and512additional updates,the fixed actor's actual parent-used level-one versions support512/512correct shape classifications on64reused DEV scenes (eight known templates). That is direct evidence for lower-level readability under this actor/readout/common clock tape. At512updates,the actual jointly trained whole-object head has approximately0.6956TRAIN/0.6947DEV loss and50%accuracy;the separately fitted legacy head has0.6651TRAIN/0.8040DEV loss,and the broader TRAIN head0.6435TRAIN/0.6945DEV. These are different heads and must remain separate. Perfect lower-view classification is not a measurement of upper use or a new public benchmark.

The original64update family verdict was premature. Equally,promising that the2048update endpoint will reveal the mechanism would be premature. It can show recovery,continued inability to fit,or failure to transfer. The following constraints already narrow the engineering question.

## An additive representation cannot solve this balanced task with a linear head

Write L,R for the two three-bit child arrangements and q for root orientation. The task audit establishes E[Y|L,q]=E[Y|R,q]=1/2 on the complete balanced TRAIN set. This also holds conditional on each label-independent nuisance tape. Therefore,for arbitrary fixed vector functions f,g,the representation

    h = f(L,q) + g(R,q)

has E[(1/2-Y)h]=0. The logistic risk with d=w^T h+b is convex in(w,b). At w=0,b=0,both gradients vanish,so the constant predictor is a global optimum for that fixed additive feature family. Concatenating the two separate feature blocks gives the same conclusion. This is an optimization-independent statement about this restricted representation/readout,not a proof that the implemented nonlinear native actor is additive or cannot learn.

Thus stronger independent child representations alone cannot make the present affine output solve the task. Somewhere before the output,the learner must create a feature involving both child subtrees and orientation jointly. The current actor already has nonlinear gates and bilinear relation operators;their presence establishes possible interactions,not acquisition of the required relation. The empirical question is whether the learned interface and credit train those operators effectively. More data can help discover this feature;the lemma does not prescribe a new regularizer or hidden relation labels.

## The relevant paired signal is a contrast,not a nonzero gradient norm

Let d_a,d_b be log odds for an observed root-swapped TRAIN pair with labels y,1-y. Let J_a,J_b be the implemented log-odds VJPs into a parameter block. Define alpha=sigmoid(d_a)-y and beta=sigmoid(d_b)-(1-y). The averaged pair credit is

    G = (alpha J_a + beta J_b)/2.

With J_plus=(J_a+J_b)/2 and J_minus=(J_a-J_b)/2,

    G = [(alpha+beta)J_plus + (alpha-beta)J_minus]/2.

At d_a=d_b=0,this is(1/2-y)J_minus. Identical sensitivities cancel the task signal even when each individual norm is large. We measure the odd-sensitivity norm,the cancellation ratio ||G||/[(|alpha|||J_a||+|beta|||J_b||)/2],and root feature/logit contrasts per block. A low ratio identifies local paired cancellation;it does not prove that cancellation explains every update or the population outcome.

For d=w^T h_theta+b,upper feature credit contains w:J_theta^T w. If the head contrast is small and paired feature differences are small,both the head's discriminative update and the upstream task signal can be weak. Local identity teaching does not automatically cure this readout/representation coupling. The frozen audit records the actual head norm and parameter-block signals to test this possibility. No threshold or failure attribution is invented before measurement.

## Hard routing requires the correct derivative interpretation

The native `TemporalRoute.backward` is a local counterfactual surrogate for nonlinear sampled losses. Its score credit is not the exact derivative of an expected route intervention. It is linear in the incoming cotangent,so the scalar CE chain rule and paired decomposition above still hold for the implemented teacher. A finite-difference mismatch of its score credit would not by itself be a bug. Readout derivatives are ordinary exact derivatives. The audit explicitly checks scalar linearity before interpreting block norms and does not call surrogate VJPs exact expected-loss Jacobians.

`composition_credit_audit.py` uses the frozen128/512update native checkpoints,eight original TRAIN pairs and a common native race tape. It runs factual forward computation and actual credit traversals only;no content cuts,ablated models,parameter updates,DEV or TEST. Each parameter group separates native state/routing from relation operators and the output head. Runtime,all counted native/counterfactual work and source/checkpoint hashes are retained. Two numerical algebra contracts pass. Earlier initialization/carrier checks established live paths;this audit asks about task-facing signal after local identity has actually been learned.

## Decisions the evidence can change

If opposite-label scenes already have substantial root feature contrasts but the raw head does not exploit them,investigate output optimization/calibration before redesigning all upper layers. If task-facing contrast is lost and most paired credit cancels at particular interfaces,repair that interface or its teacher rather than adding depth. If signals remain substantial and training loss falls with the admitted budget,the scaling trajectory remains the strongest route. None of these observations alone proves a unique cause;a concrete repair must subsequently improve the complete native learner. The fixed2048update run and final fresh DEV endpoint remain unchanged.

## Completed frozen audit: responsiveness is acquired, alignment is missing

The completed source-bound audit uses eight original TRAIN pairs at each frozen checkpoint, not DEV selection. Root feature distance grows0.030306→1.564505(51.62×);absolute log-odds contrast0.004863→0.134859(27.73×). The top relation block's retained paired-credit ratio grows0.0683→0.6479,and its credit norm0.01030→0.04436. Head contrast norm is0.8341→0.6392,not zero. Scalar CE chain-rule agreement is within2.39e-7 for the implemented teacher.

Correct pair rankings are5/8then4/8;mean label-aligned margin0.001229→−0.013159. These observations reject complete upper insensitivity and wholly cancelled relation credit on the audited samples. They do not isolate a unique cause. In particular,large contrast can encode the wrong relation. Source:`investment/composition_credit_evidence.json`. The actual joint head also fails to fit TRAIN,so describing this as a demonstrated memorization-before-grokking phase is incorrect.

## Next native capability: local acquisition of content-to-role binding

Failure addressed: local identity teaching rewards readable children but gives no direct local objective for a parent to retain which child content occupied which observed role. The current upper operator can respond strongly without learning that association usefully. A candidate training-only **binding race teacher** compares the actual issued tuple against counterfactual assignments of the same contents to the same observed poses. It uses no whole-label target,latent constituent ID,held-out arrangement or future observation.

For child values v_i and centered observed poses p_i,form a candidate key

    k_sigma = sum_i A normalize(v_sigma(i)) * B normalize_pose(p_i) / sqrt(n).
    s_sigma = <Q normalize(h_parent), k_sigma> / sqrt(rank).
    L_bind = log sum_sigma exp(s_sigma) - s_identity.

The candidates are the factual binding and each distinguishable single pair swap. Identical content or identical pose swaps are omitted. Softmax(s) is the exact first-arrival probability for exponential clocks with rates exp(s). These are local *teaching* clocks;the native actor's predictive receipt/timer races,temporal transport,persistent memory,counterfactual route credit and sparse state updates remain the actor. Standard local linear maps support the teaching clocks;they do not replace the substrate.

Read the parent at its immutable publication time and each child at its actual issued version/time. Stop lower-view teaching gradients;whole-task gradients continue normally. All latest published parents at levels≥2 and all their pair-swap alternatives are enumerated and charged. This is not sparse candidate discovery. Inference receives no additional teacher computation. The candidate adds teacher parameters and backward work during learning;no efficiency improvement is claimed.

The gradient is sum_sigma pi_sigma grad(s_sigma)−grad(s_identity):it directly rewards parent information that distinguishes factual assignments. This is a density-discrimination objective,not a certificate of complete joint sufficiency. The compressed content×pose key can alias higher-order configurations;its optimum does not prove the seven-bit whole rule is linearly accessible. Part readability,single-swap binding discrimination and whole-task transfer must therefore be reported separately.

`sleeping_machines/binding_race_teacher.py` implements this candidate beside the actor,without changing any active trainer. Five numerical contracts check exact clock normalization,parent credit with stopped lower targets,enumeration/translation invariance,content-to-pose sensitivity and indistinguishable-swap exclusion. An integrated TRAIN-only resource/credit gate is required before any acquisition. The fixed2048update run finishes unchanged. Its final result decides whether an additional local-binding acquisition is warranted;no new long fit is automatically admitted and no ablation is proposed.
