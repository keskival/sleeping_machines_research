# Learning investment, scale and delayed payoff

B1/R1 design amendment, 11 October 2026. The user identifies a possible small-data disadvantage: learning routing, exploration and credit may require an investment whose return appears only with accumulated experience. This hypothesis changes the experiment horizon, not numerical correctness or evidence standards.

## Two different curves

Let N be unique causal TRAIN targets consumed, P total target presentations, U optimizer updates, and W complete cumulative learning work. Let R(N,P,W) be expected held-out predictive risk, including learner randomness. Repeated epochs increase P but not N. Parameter capacity, depth and stream duration are separate axes. Neither a repeated-epoch improvement nor a larger model alone establishes unique-data scaling.

Compare the complete adaptive system A to reference B by Δ(N,W)=R_B(N,W)−R_A(N,W), with update schedules and validation opportunities recorded. A temporary negative Δ is compatible with later benefit. A favorable endpoint alone does not establish a crossover trend; record all scheduled checkpoints, uncertainty and complete cost from initialization.

A stylized falsifiable model is

    R_A(N)=R* + a_A N^(−α_A) + b exp(−N/τ),
    R_B(N)=R* + a_B N^(−α_B).

The exponential term represents transient learner acquisition, not a proven law. Early loss does not identify α_A or τ. Do not fit a power law to a few points and extrapolate a win. Nonmonotone adaptation, saturation and drift can invalidate this representation.

## Economic closure and amortization

Let I be initial credit/discovery/teacher investment and w_A,w_B subsequent learning work per target at a stated quality. If w_A<w_B, amortized cost advantage requires

    N > I/(w_B−w_A).

This condition is necessary only for this stationary cost model; it says nothing about quality. For changing rates, break-even requires ∫_0^N(w_B(n)−w_A(n))dn>I, alongside a registered quality criterion. Charge unsuccessful exploration and auditing, all optimizer updates and prerequisite fits. A small scalar-fit cost does not erase the cost of the predictors producing its inputs.

For a sequential future-loss objective, an exploration intervention e at time t has value

    V_t(e)=E[Σ_{k=1}^H γ^(k−1)(ℓ^0_{t+k}−ℓ^e_{t+k}) | issued history] − λ C_t(e).

The paired worlds share exogenous future observations and differ in the intervention and induced learner/optimizer state. Both worlds may continue learning. A frozen-model prediction contrast is a different estimand. Increasing H can reveal delayed benefit; replaying H futures for every intervention is not affordable by definition. Learn the conditional value from occasional bounded paired continuations and observed returns, reuse causal forward packets, and report the teacher cost. This remains the finite reciprocal actor/critic loop in CLOSED_LOOP_EXPLORATION_AND_CREDIT.md; measured returns anchor it without a hierarchy of learned trainers.

## Staged experiment contract

1. Keep the existing compact-credit owner and native architecture. Before any long admission, retain causal/versioned numerical contracts, finite updates, positive-support audits and measured memory/work bounds. A small immediate task-utility win is not mandatory. Require evidence that the credit learner acquires useful held-out structure, or run a bounded acquisition stage explicitly intended to locate its learning timescale.
2. Register increasing unique-TRAIN prefixes and cumulative presentation checkpoints. Start each matched arm from identical actor initialization and causal data order. A single continuation ladder measures online accumulation; independent-prefix fits measure sample-size dependence and must be labelled separately. Match scheduled DEV opportunities. DEV selects; TEST stays sealed.
3. Within each native model compare acquired credit to exact affordable credit, frozen credit and reset-credit state. These isolate reuse; they are not classical architectures. The learned system may lose early. Stop for numerical/resource failures, not for an early quality deficit. Stop a bounded stage at its declared budget and report unreached targets rather than promising later success.
4. Report quality against N, P, U and W, with initial investment included. Select fixed quality targets from existing references before candidate scoring. Describe post-hoc target crossings separately. Separate loss from calibration, audit variance and resource measurements.
5. Test reuse on a later causal stream or regime: preserve learned credit weights while matching actor initialization/state rules, compare to freshly initialized credit, and charge source-stage work. Lower marginal cost and total amortized cost are distinct claims. Confirm a selected trend with three independent seeds before a scaling claim.
6. Use published classical quality references only under matching data/tokenization/scoring protocols; no new dense training. Existing native controls can isolate credit but cannot substitute for a classical scaling curve. Missing classical work records prevent matched-work claims, not a scoped quality comparison.

## Current decision

Existing online3/online5 curves vary repeated presentations, update frequency and credit transport, not unique-data scale or acquired compact-credit amortization. The completed B4 confidence diagnostic reuses fixed predictors; it does not resolve the scale hypothesis. Audit these saved curves before choosing the next ladder. Curie owns live compact-credit integration; AWS owns economical exact references and the horizon/cost protocol. Do not duplicate that implementation or redirect properly tokenized language to easier characters.

## Completed saved-curve audit

`aws_learning_horizon_audit_20261011_v1` completes through AWS slot 1/run_safe, binding completed input files by SHA256 and excluding pending results. No data/model/TEST evaluation or fitting occurs. The reference-final quality targets below are post-hoc descriptive thresholds, not registered adoption gates.

| Native dataset/seed | Trace minus BPTT DEV LL, pass 50 | Difference, pass 100 | First trace pass reaching BPTT final quality | Recorded trace/BPTT epoch wall |
|---|---:|---:|---:|---:|
| Amazon 0 | +0.005303 | +0.011377 | 67 | 5881.5 / 1511.8 s |
| Amazon 1 | −0.002768 | +0.003100 | 81 | 5994.2 / 1485.0 s |

The initial audit excluded pending Amazon seed 2. The completed follow-up `aws_learning_horizon_audit_complete_20261011_v1` binds all three seeds: seed2 final gain +0.008102, first reaching native BPTT final quality at pass65, trace/BPTT recorded epoch wall5959.1/1484.4s. Mean final three-seed gain +0.007526 nats/target. Both rows use the same fixed TRAIN corpus repeatedly. Trace takes 1,770,900 updates versus 40,400 for BPTT. Seed 1 therefore demonstrates a later repeated-presentation quality reversal, not unique-data scaling, matched-update superiority, learned-critic amortization or a classical-reference win. The wall boundary is recorded epoch time; setup and prior work are excluded, so it is not complete-system cost.

Taxi two-layer seeds 0/1/2 all finish ahead of native BPTT after 20 passes (+0.027221/+0.019469/+0.058912 DEV LL), but original dense trace fits take 4874–5269 recorded epoch seconds versus 77–83 for BPTT. Trace uses 63,440 updates versus 1,760. Existing economical block execution reduces this cost and retains the separately registered quality gate; these saved curves still do not isolate update frequency or the return on learning credit machinery.

Next admission depends on the owned compact-credit integration and measured workload. Preserve the completed Amazon seed 2 evidence. Reuse these exact references; do not queue another control grid. For the new learner, collect persistent-credit versus reset/frozen-credit curves at registered geometric experience checkpoints, with an explicit bounded acquisition phase and complete accumulated cost. A negative early difference is recorded rather than used as an automatic rejection.
