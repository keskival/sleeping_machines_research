# Analytical audit: which credit paths support composition?

16 October 2026, R1. This audits actual implementations before the user-requested empirical advantage comparison. It builds on COMPOSITION_LEARNING_CONTRACT, COUPLED_NATIVE_QUERY_CREDIT, native receiver eligibility and the existing joint_race_credit / exact hierarchical suffix replay work. New numerical gates are queued; algebraic results and integration status are distinct.

## 1. Exact prediction credit is already present

For p(y)=sum_c pi_c q_c(y), pi=softmax(s), posterior rho_c=pi_c q_c(y)/p(y), the normalized NLL has

    dL/ds_c = pi_c-rho_c.

`native_indexed_cause_head.indexed_cause_nll` and CERL implement this through the difference of two logsumexp terms. The depth mixture supplies the same direct responsibility-based signal to each included depth. This is exact for the selected predictive candidate law. It does not differentiate hard candidate-set inclusion or teach historical detached value producers automatically.

A soft memory read v=sum_i pi_i v_i has dL/ds_i=pi_i*g^T(v_i-v). This is the exact gradient of the actual soft-read computation, including nonlinear suffixes via g. It is not the gradient of expected loss under sampled hard reading. RMR's soft reads therefore differ mathematically from the sampled native-core route teacher. Top-k write/read support remains a discrete selection boundary; gradients through selected entries alone do not establish credit to excluded addresses.

## 2. Legal joint query credit fixes constraints, not suffix curvature

The native query chooses two distinct versions. Its law is

    P(i,j)=softmax(s0)[i] * softmax(s1 with i excluded)[j], i!=j.

For any factual tuple w and an alternative a sampled from q(a|w)>0 over every other legal tuple,

    E_a[ grad P(a)/q(a|w) * (L_a-L_w) ] = sum_j grad P(j)*L_j.

Proof: sum of all probability gradients is zero. This identity holds for every w, including nonuniform legal proposals with their actual probability divided out. Direct factual/pathwise derivatives are a separate term. Second-race completion depends on the excluded first candidate: reusing E1=T1*Lambda1(w_first), T1_alt=E1/Lambda1(a_first) preserves the appropriate conditional clock coupling. The existing coupled query repair respects that law.

The current vector teacher instead supplies grad P(a)/q times g_w dot(A_a-A_w), where A is the assembled representation and g_w is the downstream loss cotangent at the factual representation. It is exact for that conditional tangent objective, not the true nonlinear task objective.

Two counterexamples matter for composition:
- L(z)=softplus(-z1*z2), factual z=(0,0), alternative=(1,1): the tangent signal is zero, but actual alternative loss is lower. A relational suffix can require a joint move invisible to the factual tangent.
- L(z)=softplus((z-0.25)^2), factual z=0, alternative=1: the tangent predicts improvement, while the actual loss increases. Nonzero credit is not enough; its direction can be wrong.

The immediate joint composer is evaluated for each alternative by the coupled teacher, so it already captures interactions inside that composer. The remainder concerns nonlinear computation after that assembly, including deeper compositions and changes of later discrete decisions.

If the conditional suffix is smooth with Hessian norm <=beta along every segment, the norm of this conditional score-credit bias is bounded by

    (beta/2) sum_a ||grad P(a)|| * ||A_a-A_w||^2.

Hard downstream route changes can violate the smoothness premise. Neither small bias nor a correct sign follows merely from efficient individual stages.

`composition_task_credit.actual_joint_loss_credit` implements the exact local score surrogate using supplied actual conditional suffix losses. It does not obtain those losses or install suffix replay in the native learner. Existing joint winner/time and hierarchical suffix replay machinery provides a reference for that integration. A predicted critic without an actual residual correction is another biased approximation, not an exact substitute. Clock-pathwise or joint winner/time terms must remain accounted for.

## 3. Own-write traces are conditional, and long memory can retain omitted producer credit

For the complex receiver recurrence m_t=A_t m_prev+b_t, fixed-input own-parameter eligibility obeys E_t=A_t E_prev+partial_theta(A_t m_prev+b_t). Existing factored traces implement this without retaining an event autograd graph.

For an input producer parameter psi, the missing recurrence is instead

    E_t^psi = A_t E_prev^psi + (partial m_t/partial x_t)(partial x_t/partial psi).

Replaying only the latest write input omits earlier producer contributions. Receiver transport satisfies |A_t|=exp(-r*f(x)*Delta), but there is no uniform contraction away from1: persistent memories deliberately permit tiny decay and simultaneous arrivals permit Delta=0. A short credit horizon is therefore not automatically harmless for long-lived memories. Preserve memory independently of credit truncation; measure or repair the omitted paths rather than calling the whole derivative exact.

Historical child replay repairs a distinct abstraction path: the lower memories actually used to compose a stored write. Default two-hop/six-edge replay reaches only two child edges from a root. It is useful conditional credit, not coverage of arbitrary abstraction depth.

## 4. Full recorded abstraction depth with linear replay work

At a producer with k actual child roles, let v_r be the contribution of role r to its downstream content VJP. Sample R with positive probability p_R and propagate v_R/p_R. Then

    E[v_R/p_R]=sum_r v_r,
    E||v_R/p_R-sum_r v_r||^2 = sum_r ||v_r||^2/p_r - ||sum_r v_r||^2.

Apply this recursively, conditioning at each node. The tower property and linearity of the local VJP recover the complete available conditional producer-DAG content derivative in expectation. Shared children appearing along multiple paths receive the sum of their path contributions in expectation; enumerating a tree expansion does not require unique-node ownership. The child graph strictly descends in abstraction level, so one sampled path needs at most depth-1 additional reads rather than expanding every branch.

`sleeping_machines/sampled_compositional_credit.py` implements this in the native core. Saved factual role views receive zero-forward inverse-inclusion child surrogates. Uniform selection uses a separate checkpointed credit RNG; factual and joint-alternative RNGs stay unchanged. Every available recorded abstraction level can receive credit. The eight-layer case caps extra child reads at7 per root.

This theorem concerns the recorded frozen-parameter conditional content graph. It does not recover unrecorded lineage, historical routing/clock decisions, prior temporal writes' input provenance or base-input/inbox provenance. Changed-parameter local replay is approximate. Inverse inclusion can amplify variance multiplicatively with depth (binary branches permit weight up to2^(depth-1)). Global clipping and Adam are nonlinear: an unbiased raw conditional VJP is not an unbiased clipped/Adam update. The paired run records preclip norms/clipping, work and actual prediction. More variance reduction must earn its cost empirically; no low-variance theorem is asserted.

## 5. Contracts and empirical decision

Queued analytical contracts verify exact finite-sum score gradients under actual nonlinear losses, the zero/wrong-sign tangent witnesses, and existing joint winner/time estimators. New sampled-path contracts enumerate every path of an actual native four-level saved producer DAG and compare the expected mix/composer gradients with an independently rebuilt complete conditional DAG. They also require factual prediction/RNG parity, a depth-1 read cap and exact recovery of the dedicated credit RNG. An acquired eight-layer real-token update/recovery gate precedes training.

After both gates pass, the admitted pair starts from one frozen acquired checkpoint: two complete native learners, both with legal joint queries and current local producer credit. The candidate replaces the two-hop cutoff with full-recorded-depth sampled content credit. Both receive the same65536 additional TRAINtargets and then8192 chronological onlineDEVtargets, score-before-update with state carried continuously. Frozen flagship8192/1024 evaluation is separate; novel-token loss and per-window differences are reported. No TEST or external architecture is fitted. This isolates whether better content-credit coverage improves actual learning; both arms still retain the tangent historical choice teacher.

A positive online or frozen result is reported under its own protocol as single-parent/seed development evidence. Extra learning work is disclosed; no iso-FLOP, full-history-exact, hardware-energy or scaling-supremacy claim follows. A failed result is evidence about this construction and triggers signal/variance/cost diagnosis, not a family verdict. The comparison is not allowed to start on derivative tests alone if the acquired learner gate fails.
