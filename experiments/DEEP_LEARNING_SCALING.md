# Efficient learning across depth and capacity

User direction, 10 October 2026: the valuable learning result is improved scaling with depth and model size. This is the next shared learning-enabler objective inside B1 and R1. Completed benchmark confirmations remain protected. Curie owns the compact-credit implementation and resource measurements; AWS retains its admitted online3 two-layer grid. No AWS job is moved or duplicated by this plan.

## Decision and current evidence

Decide whether a compact causal credit learner reaches the same development quality with less complete fitting work, and whether that advantage improves as depth or capacity grows. The one-layer Taxi results establish quality per data pass, with different update counts; they do not supply this scaling result. Note 160 section 12 and online_deep.py give a two-layer exact-gradient witness, not a scalable implementation: cross-layer forward sensitivities require state dimension times upstream parameter count per stream. Exact traces remain the small-scale correctness reference. The existing online3 AWS grid is reused when its completed files arrive.

## Model and credit design

Prioritized integrated model: R1 keyed temporal persistent memory with predecessor messages, followed by B1 event likelihood. Preserve temporal computation, separate keys/values, persistent state, hard selection, counterfactual credit, and silence-aware likelihood where applicable. A new learner changes credit transport, not the forward architecture or prediction objective.

Concrete failure: local mode detaches continuous key/query producer paths, while full depth traces transport all upstream sensitivities at growing storage and arithmetic cost. Start from note 163's predicted cotangent plus positive-support residual VJP audits for these actual producer boundaries. Note 162 supplies optional-route expected-loss credit when optional writes are present. Deep factual temporal credit and residual route credit must both be represented and measured; a cheap next-event predictor alone does not certify delayed memory credit. Feedback-weight alignment by itself also does not certify affordable cross-layer temporal credit.

At depth, place each producer cut explicitly and avoid double-counting overlapping cuts. Freeze credit predictions within each audited update. Sample residual audits with recorded positive inclusion probabilities; preserve their conditional gradient mean for the declared causal replay horizon. An unbiased sampled audit must be compared with the exact objective for that horizon; truncation versus full persistent credit is a separate measured bias. Charge predictor fitting, discovery, losing routes, replay, state transport and all optimizer updates. Derive variance/work allocation before admitting long fits. If full support makes audits expensive or variance grows with depth, report that failure and revise the learner.

## Reciprocal inference and persistent credit state

User direction, 10 October: inference trains the credit-assignment functions, which train the forward weights; investigate carrying credit activations into subsequent forward events. Represent forward state s, persistent credit state c, forward weights theta and credit weights psi separately. A causal event cycle is:

    (prediction_t, s_next, packet_t) = F_theta(s_t, input_t, c_t)
    (credit_t, c_next) = C_psi(c_t, packet_t, observed_outcome_t)
    theta_next = update(theta_t, credit_t)
    psi_next = train_credit(psi_t, forward_evidence_t, sparse_audits_t)

The observed outcome enters after its prediction. Credit state may affect subsequent forward events; it cannot enter the prediction it evaluates. Predictions and audits bind the forward/credit parameter versions, state versions and causal horizon. Simultaneous unversioned updates would change the target while it is being evaluated.

Forward packets should expose sampled causes, local eligibility information, addressed writes, elapsed time and learned compact messages useful for credit. Known causes in generated samples supervise attribution, but selected factual causes alone do not certify responsibility for observed data or counterfactual improvement. Use the existing closed-credit real-data correction and sparse audited derivatives/effects to train those distinct outputs. Note 160 sections 3–6 already develop this reciprocal mechanism; reuse the matched AWS pipelines rather than duplicating them. Continuous cotangents, discrete attribution and finite update utility have distinct targets and must not be conflated.

Credit activations become a persistent addressed numerical state, updated on informative events and transported through time. This is a proposed extension, not a mathematical requirement to retain every backward activation. Compact state may summarize useful learning history; exact full-history gradients require sufficient sensitivities or replay, whose cost remains charged. Merely caching the previous error or aligning backward weights does not establish that the state is sufficient for deep credit.

Measure the reciprocal loop against frozen-credit, credit-state-reset, no-forward-access-to-credit-state, and no-forward-supervision controls using the same sparse execution and complete accounting. Freeze/reset arms distinguish a learned credit function from useful persistent credit memory and from inference improvements caused by that memory. Include credit-state updates and slow credit-model training in learning cost, and any use of credit state in serving cost. If forward parameters or credit parameters change, record whether stored state is reused, corrected or invalidated; quantify drift rather than assuming exactness. The intended outcome is sparse forward activity teaching sparse persistent credit activity, which improves future forward learning.

## Bounded progression

1. Complete the already-admitted native continuous-producer contract. Generalize the numerical witness to depths 2 and 4, then 8, with small dimensions: every parameter, factual paths, delayed writes, silence, overlapping cuts and causal suffix checks. Fixed-weight exactness and changing-weight drift are separate checks.
2. Measure one bounded smoke per new driver before fitting: depth 2/4/8 at fixed width and pool; width 16/32/64 at fixed depth; pool 2/8/32 at fixed selected activity. These are separate scaling axes. Derive memory bounds first; exact traces run only at affordable points. Measure length 32/128/512 separately to distinguish history savings from depth costs. Initial grid points are measurements, not automatic fit admissions.
3. Fit the same native forward model under full backpropagation, exact forward traces where affordable, compact predicted-plus-audited credit, and a local-credit ablation. Reuse completed compatible controls. No new Transformer/LSTM training is admitted. Match initialization, training events/order, scored targets, precision, update grouping and validation opportunities. Include an equal-update comparison so per-event optimizer frequency cannot masquerade as better credit.
4. On R1, require an actual depth dependence: ablate intermediate-layer state/message paths and test delayed binding at 8/16/32 pairs. On Taxi DEV, report time and mark likelihood separately. If the task is solved equally well by the shallow model, use it only for resource scaling and select a harder R1 development regime before claiming deep learning advantage.
5. Select the compact learner using DEV only, then confirm the selected depth/size trend on three seeds. Keep benchmark TEST sealed. Resource failures stop admission; they are recorded, not retried with silently changed caps.

## Small-scale learning gate before scaling

A learned credit system must first show acquired structure, even when its overhead does not yet pay. Use the [credit-learning certificates](theory/CREDIT_LEARNING_CERTIFICATES.md): fixed-packet fitting, disjoint held-out producer-weighted calibration and audit-variance reduction, then later-forward-loss utility under fixed updating. Match persistent/reset state availability between fitting and assessment; prefix-assisted adaptation and zero-shot transfer are separate tasks. Confirm the selected small-scale signal across three seeds before scaling it. Include all pretraining and teacher work. Larger scale is an amortization hypothesis, not a substitute for a learning signal.

## Required evidence and pass criterion

Primary endpoint: cumulative complete fitting work needed to reach a fixed DEV quality target chosen from a completed full-credit reference before evaluating the candidate. Include unsuccessful steps and credit-model pretraining; a run that never reaches the target is marked unreached at its actual budget. Report final quality at equal total work as the paired endpoint. Use whole-fit and per-scored-target operations in the same units for every arm, measured CPU wall time, peak process-group RSS, trace/credit storage, events/passes, optimizer counts and inference work. Separate arithmetic from special functions and estimated bytes from measured traffic; make no energy claim from FLOPs.

A deep efficiency win requires better or equal quality at lower complete fitting work on at least two depths of 4 or greater, confirmed on three seeds. A better-scaling claim additionally requires the candidate/reference work ratio to improve across the registered depth or size axis with seed uncertainty reported. A memory win or per-pass quality gain is reported on that axis alone. Increased available pool capacity counts only when quality improves with the same selected activity and full learning/discovery costs charged. The target is useful trained capacity beyond active work, including deep persistent learning.

No scaling win is currently claimed. Next concrete gate: the already-admitted native state-key cotangent contract, followed by small-scale learned state-credit fitting and held-out calibration. Note-163 continuous-producer and depth-2/4/8 reciprocal execution contracts are completed; execution correctness does not establish acquired credit structure.

## Exact execution enabler: CPU temporal scan

The native B10 long-window fit motivates measuring recurrence dispatch/backward cost. `tpp/temporal_scan_cpu.py` and `.cpp` implement the unchanged affine complex-state recurrence and its first-order adjoint in one CPU call per layer. Projection/gating, elapsed-time decay/rotation, normalization, dense local operations, mark memory and race heads are retained. Explicit initial state supports numerical persistence and differentiable chunk carry; no implicit detach is introduced. This is an execution change, not a substitute learning architecture or a sparse-credit win.

Two R1/B10 jobs in `queue/enabler/curie_cpu_scan_v1_20261010T1510Z/manifest.json` are admitted after conditional-noise diagnostics and before the existing report build: recurrence/carry/full-native-parameter-gradient contracts (depths 2/4/8), then one bounded kernel forward/backward timing observation at lengths 128/1024. C++ compilation uses one worker and a 2.5 GB process-group RSS cap with the existing 9 GiB host/2 GiB cgroup reserves. Private build artifacts live under `.git/temporal-scan-build`. Parent and individual compiler-child RSS are separate metrics; the runner guards the process group. Compilation and contracts are pending.

The backend deliberately supports first-order autograd only. Higher-order meta-Hessians require an independently validated differentiable backward before substitution into those contracts. No current fit uses the new backend. Passing parity and kernel timing admits an integrated whole-fit measurement under the original forward/data/update protocol; kernel speed alone is not complete learning efficiency. Arithmetic and state sizes of the kernel are explicitly charged, including stored trajectory and carry; all remaining model/credit/optimizer work remains in the final comparison.
