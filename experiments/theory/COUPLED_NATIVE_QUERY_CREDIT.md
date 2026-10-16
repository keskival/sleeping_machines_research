# Legal joint credit for without-replacement memory composition

16October2026,R1. Formal review discovers a constraint mismatch in the local query teacher. This repair changes credit,not the factual forward query policy.

## Concrete counterexample

Two lower memories have scalar values0and2. Two without-replacement query roles always produce either(0,2)or(2,0). If the local composition is addition,both legal assemblies equal2 and the true expected-output gradient with respect to ordering logits is zero. The old per-role first-query teacher can replace its0with the memory2already selected for the second role. It evaluates the invalid duplicate(2,2),creating a nonzero ordering gradient. This is a constraint-coupling error independent of whether the downstream loss is nonlinear. It can teach selection using an assembly the actual model forbids.

## Correct joint distribution and teacher

Keep the factual sequential native races exactly. For two query score rows s0,s1 and distinct indices i,j,

    P(i,j) = softmax(s0)[i] * exp(s1[j])/sum(k!=i)exp(s1[k]).

The second eligible set depends on the first choice. Enumerate these scalar probabilities over the declared root pool;for N=16 there are240legalpairs and256matrixentries,not256memory reads. Choose one nonfactual legal pair a from q(a)=P(a)/(1-P(w)). Decode only missing selected alternative memories and use cached actual reads when shared. Run the same joint role composer on that complete pair. Attach the zero-forward score surrogate

    (P(a)-stopgrad(P(a))) / stopgrad(q(a))
        * stopgrad(A(a)-A(w)).

For any fixed factual pair w,the mean over a of its score gradient is sum(a)dP(a) A(a),because sum(a)dP(a)=0. This is exact for the declared conditional linearized joint-assembly objective. It is not the full nonlinear suffix/sequence gradient;actual prefix state and exogenous clocks remain conditioned. Existing local receiver/producer credit continues separately.

Timing also couples to the excluded candidate. The first clock's total rate is unchanged by its winner. The second total depends on the first winner. Reuse the observed independent E1=T1*Lambda1(w_first)and set T1_alt=E1/Lambda1(alt_first). Do not reuse the old second time after changing its eligible set. Transport both alternate replies to their legal common completion time before composition. Actual forward times,direct selected-value gradients and factual clock derivatives remain intact.

## Scope,work and validation

Three contracts pass:the two-memory spurious-gradient witness;exact finite-sum joint-score gradient for every factual pair in a three-memory pool;and bitwise actual native predictions/bindings/main-race RNG equality with live query gradients. Alternative RNG consumption changes because one joint audit replaces two separately sampled role audits. These draws do not alter factual predictions before learning. No parameters are added.

A private acquired86048checkpoint passes64TRAINtargets plus32discardedresumeaudit with full serialized recovery. Source:`experiments/results/diagnostics/aws_r1_coupled_query_credit_gate_20261016_v1.json`. No DEV/TEST,ablationorqualityclaim. All256joint-scoreentries per upper query,240legalcandidates,conditionalnormalization/clock arithmetic,actual alternative decodes,transports/composer products and backward work are charged. The past claim that the old surrogate was only a nonlinear approximation needs this additional constraint correction.

Explicitly checkpoint-migrate the document-aware producer-credit run with all parameters,Adam states,numerical memories,binding/producer records,bank,inboxes,RNGs and corpuscursor retained. Two new integer counters record joint-scoreentries and coupled audits. Keep the4Mcumulative data/selection/flagship protocol. Existing source/log/checkpoints remain evidence;record any discarded unsavedtail. This is a derived repair now,not a retrospective explanation of a disappointing final score.
