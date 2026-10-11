# Connected policy and credit learn from paired native futures

B1/R1,11 October2026. Follows [paired observation economics](CAUSAL_AUDIT_VARIANCE_AND_DRIFT.md) and [reciprocal actor–critic](RECIPROCAL_ACTOR_CRITIC.md). Prioritized native integration:depth4 persistent temporal/keyed carrier,hard optional addressed four-dimensional key write at event3,then four supervised predictions and independent readout Adam updates. Carrier is fixed in this stage; route-policy and credit networks co-adapt across episodes. This is route learning, not all-carrier BPTT/depth or continuous online market learning.

## Failure and retained mechanisms

Frozen-actor variance improvements do not establish useful learned credit. Isolated learning-effect correction and scalar calibration fail utility/replication gates. Retain the valid connected shared-prefix structure and learn its whole contrast function directly from learner-aware forward futures. Both actor route controller and critic are copies of the10584-parameter shared context/receiver recurrent structured-response network, initialized from the same saved credit checkpoint. Each has one connected context GRU and receiver-conditioned representation; these are not disconnected per-layer linear predictors. No substitution of temporal computation/hard routes with dense control architectures. Candidate discovery/scoring still enumerates six TRAIN receivers/nine messages; one hard write is executed. Evaluation includes eight receivers, including never-TRAIN-labelled6/7.

## Declared objective and unbiased route-policy credit

Given causal prefix H and fixed reference b(H), choose one write a from

    pi_theta(a|H)=(1−epsilon)softmax(−Q_theta(H,a)/tau)+epsilon/K.

Epsilon0.1 preserves finite support. Tau is TRAIN-only RMS of saved learner-aware response (floor0.001). The actual episode learner supplies L(a):mean of four forecasts, each scored before its branch-local readout Adam update. Q(a) is E[L(a)|H] under that fixed continuation/learner law. Reference b is the saved baseline critic's greedy choice, independent of current theta. The paired contrast D=L(a)−L(b) includes later readout learning.

For critic contrast h_psi(a)=Qhat_psi(H,a)−Qhat_psi(H,b), Bernoulli audit I of probability p>0 issued before future feedback, policy credit is

    grad_theta sum_a pi_theta(a) h_psi(a)
      + I/p * [D−h_psi(A)] grad_theta log pi_theta(A).

Conditional expectation equals grad_theta E_A[Q(A)] for the route-policy parameters, with critic detached and fixed during consumption. Proof:conditional audit expectation repairs Q−Qhat, then the score identity combines it with the model term; the reference constant cancels. It does not differentiate the carrier, the continuation learner or future policy changes. Gradient clipping and Adam are nonlinear optimizers; the theorem applies to the raw gradient estimator, not an unbiased parameter update. We minimize cost; no gold-backprop gradient labels train the critic.

Factual supervised feedback occurs in every episode:one selected world is always replayed and its readout learns causally. An audited nonidentity reference adds a second world. Equal route/reference has known zero contrast and needs no additional branch. Compare all workload components, not just audit counts. Main production baseline stays unchanged; this is an integrated development controller.

## Acquisition and reuse protocol

Three native arms share carrier, initialization,prefix/future order,uniform action draws,positive-support mixture,route-controller updates and validation opportunities:

- full fixed critic:p1,critic frozen; paired full-feedback reference;
- sparse fixed critic:p0.25,critic frozen;
- sparse learned critic:p0.25,critic trains on paid paired future contrasts.

Critic fitting uses at most128 FIFO evidence packets, four minibatch updates per new nonidentity audit, batch<=16. Raw prefix packet, selected/reference actions, contrast,target version and learner horizon are retained. Carrier/continuation learner are fixed, so these labels keep their declared target across policy/critic changes; no target relabeling from future data. Newly issued predictions bind current model versions, and graphs are consumed before updates. Critic optimizer lr0.001,policy lr0.0003,weight decay0.001,clip1. Scalar loss scale comes from saved TRAIN residual RMS times sqrt2, floor0.0001. No DEV tuning.

Contract finite exhaustive raw-policy-gradient equality with mixed exploration,conditional Bernoulli correction and stale context rejection; native paired replay parity already passes. Smoke four episodes/two DEV prefixes/one future. Bounded acquisition128 unique episodes,DEV32 prefixes/eight paired continuations, followed by a registered reuse stage to512 total episodes if finite/resource contracts pass. Reuse freezes acquired critic; a reset-credit arm starts from the same acquired route controller and resets only critic to the saved initial checkpoint. Reference/frozen-critic policies continue from their own128-step states. No immediate small-scale utility win is required to admit this bounded experience ladder. Each persistent state/optimizer checkpoint is retained. Equal updates/data are recorded separately from resource cost; prior source model pretraining stays charged.

Primary development effects:learned versus fixed sparse credit future route utility; acquired-frozen versus reset credit during reuse at equal audit probability; full-versus-sparse complete fitting work and quality. Track critic pair error on issued pre-audit predictions,actual factual loss,paid extra worlds,unique scored targets,policy/critic updates,capacity/selected activity,whole fitting supportedFLOPs,per-target work,CPU wall and RSS. Shared DEV teacher cost is reported separately and all-arms accounted; sparse inference/discovery and optimizer work never silently free. A failed early point diagnoses the variant; scaling beyond512 needs the completed learning curve and written diagnosis.

The intended outcome is connected forward policy generating teaching evidence for connected credit,which teaches later hard route choices. Actor acquisition/information objectives,persistent across-event credit state,all-carrier gradients and real-domain changing regimes require subsequent integrated tests; the present raw-gradient identity does not stand in for them.
