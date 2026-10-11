# B4 one-core mark readout development

Decision: retain the confirmed conditional calibration benefit while eliminating the second native core at serving. The three-seed teacher improves DEV marks1.979±0.537 sample SD nats/sequence; both teacher fits remain charged. This is Github split1 development, not the sealed B4 TEST record.

## Design and contracts

Concrete failure: a transferred relative mark branch does not preserve its distribution in another core; serving a calibrated mixture of two complete predictors doubles core evaluation. Retain the repaired core's temporal layers, addressed mark state, physical state clock, normalized race likelihood and time density. Add only a supporting conditional mark readout. Features are existing causal hp, addressed slots, preceding-mark embedding, baseline conditional log probabilities, aged physical state and candidate-gap features. They contain no scored target-mark identity. Candidate time is the argument of p(k|tau,history), not an illicit future label. The joint density remains f_core(tau|history)*p_readout(k|tau,history), normalized over marks with the original time marginal.

The confirmed teacher supplies full conditional-mark vectors on TRAIN. Reuse those forward values as soft targets. No new learned trainer is needed: ridge linear algebra or the ordinary soft-label loss anchors the readout. Teacher features/outputs, core fits and all cached-target presentations are paid. The student carries no teacher at serving; checkpoint includes one core, feature normalization and readout.

Fixed gates: mark gain>=1nat/sequence and retention>=80% of the corresponding confirmed teacher gain, exact unchanged core/time and first-target suffix-mark invariance. No TEST loading, DEV hyperparameter grid or early stopping.

## Completed development

| Seed0 readout | DEV mark loss | Gain vs repaired core | Teacher gain retained | Gate |
|---|---:|---:|---:|---|
| Fixed ridge logits | 66.014 | 1.222 | 60.10% | FAIL |
| Probability-weighted linear KL | 66.011 | 1.226 | 60.29% | FAIL |
| Linear KL + nonlinear residual | 65.502 | 1.735 | 85.30% | PASS |

Ridge fits144 standardized features and9 centered logit residuals with fixed penalty1;8.74ms fit after2.553s teacher preparation. KL uses the same features/readout, bounded full-batch LBFGS100 iterations and penalty1,1.173s; final max gradient0.000113, not a certified exact optimum. The negligible DEV change from this objective refinement motivates testing nonlinear feature interactions, not continuing a linear tuning grid.

The residual is a64-wide Tanh supporting map, zero-initialized output, fixed AdamW0.003/weight_decay1e-4,50 TRAIN passes/batch512 with no DEV selection. It uses the same144 frozen native features. Residual fitting1.386s on665,350 cached target presentations;1300 optimizer updates. This does not increase unique data. Seed0 student32,938 parameters vs50,826 teacher pair including calibration coefficient; matched full-distribution DEV serving0.188s vs0.419s (2.23x),2094 targets. Timing/core unchanged and suffix-mark error0. Fits7040.78s are prerequisites; preparation and both linear/residual fits are additional. Teacher quality is higher, so this is a scoped quality/work tradeoff, not matched-quality supremacy. Leading joint-model DEV total is unchanged.

Evidence: aws_single_mark_readout_20261011_s0_v2, aws_single_mark_readout_kl_20261011_s0, aws_nonlinear_mark_readout_20261011_s0. Initial v1 timer-shadowing bootstrap failure is preserved; v2 fixes wiring only.

## Independent-seed confirmation

Before seeds1/2 scoring, lock the same144-feature/64-wide/50-pass recipe and require every seed to meet the unchanged pilot gate. Each seed uses its own completed predictor pair, TRAIN-only coefficient/features and frozen core. One Github DEV split is reused; these are independent training seeds, not independent datasets. All three fixed-recipe seeds pass: gains1.735/1.239/2.391, retention85.30/87.44/96.14%. Mean gain1.788, sample SD0.578; mean per-seed retention89.62%. Serving speed versus teacher pair2.23/2.23/2.28x. Decision aws_single_readout_decision_20261011 is complete.

Teacher-free restoration contract aws_single_readout_serving_20261011_v1 passes all three: exactly identical DEV scores/time, normalization error<=4.45e-16 and first-target suffix error0. Each290,035-byte bundle includes one native core and readout metadata; no teacher checkpoints or TRAIN loaded to restore. Sequence-level residual diagnosis is the next decision before broader-domain development; no TEST or external deployment.

Completed sequence diagnosis aws_single_readout_sequences_20261011_v1: readout helps 28/23/26 of34 DEV sequences across seeds0/1/2. Mean scores reproduce exactly. Losses concentrate in particular sequences; no DEV-driven fit or candidate selection follows this diagnostic. Multi-position prefix-equivalence contract aws_single_readout_prefixes_20261011_v1 passes129 checks per seed,387 total; maximum time/all-mark prefix discrepancy5.33e-15.

## Observation-anchored readout test

Concrete failure: the compact readout loses teacher quality and harms6–11 of34 DEV sequences. Test a fixed TRAIN target q=.9*p_teacher+.1*one_hot(observed_mark). Observed marks enter only the TRAIN loss, never causal features or serving selection. Keep the same core,144 features,64-wide residual,50 passes and training optimizer. Linear KL initialization also uses the fixed target; no DEV-selected mixture coefficient or early stop. This anchors a learned teacher on actual forward observations without a third learned teacher. It measures mark prediction quality, not future intervention value or sparse credit.

Register all three seeds before any new DEV scores. Adoption requires mean DEV mark-loss reduction>=0.1 against the saved distilled student, each seed no worse by more than0.25, and existing baseline gain/80% teacher-retention gates all pass. Original teacher fits and both readout-development rounds are charged. A failed arm is diagnosed, not retuned against DEV. Frozen TEST and leading joint-model DEV total remain unchanged.

Completed aws_observation_readout_decision_20261011 passes: gains over prior student0.109/0.614/0.378, mean0.367, sample SD0.253, all seeds positive. Mean repaired-core gain2.155; mean teacher-gain ratio110.9%. Seed1/2 exceed their teacher DEV marks. Original timing/core fixed. Readout fit2.60–2.62s, teacher preparation2.58–2.61s per seed, inherited core/prior-readout fits additional. Teacher-free portable serving contract aws_observation_readout_serving_20261011 passes all seeds with exact score/time parity, normalized marks and first-target suffix invariance. Adopt the fixed observation-anchored objective for the current compact readout. Per-sequence gains are diagnosed before further development. No future-learning/independent-dataset/TEST claim.

Anchored sequence comparison aws_observation_readout_sequences_20261011_v1 improves19/27/20 of34 DEV sequences versus the prior compact readout; gains reproduce the registered means. Anchored prefix contract aws_observation_readout_prefixes_20261011_v1 passes387 multi-position checks, maximum error3.56e-15.

## Registered readout work-to-quality curve

Compare observed-label weights0 and0.1 at unchanged initialization/data/order/AdamW/50-pass budget across seeds0/1/2. Fix each seed's target to its completed teacher-only compact student's DEV mark loss before these runs. Score only passes0/5/10/20/35/50; never stop or choose weights from these snapshots. Report first scheduled attainment and unreached budgets, equal-pass final quality, unique targets versus cached presentations/updates, teacher preparation and inherited original core fitting work. DEV evaluation is separately charged; supported fitting FLOPs are unmeasured. This isolates readout learning economy with the core frozen, not whole-substrate/depth or sparse-credit efficiency. No TEST.

Completed aws_readout_curve_decision_20261011: teacher versus anchor first scheduled target passes50/35,50/0,5/5. Incremental preparation+fit wall ratios0.909/0.711/0.984; inherited-core/prior-readout supported training ratios0.999934/0.999752/0.999989. This is a readout-learning improvement, not whole-pipeline efficiency. DEV evaluation is recorded separately; inherited core evaluation/earlier failed investigations are additional. Final scores reproduce saved50-pass students within1e-9. Driver legacy human-readable rule string incorrectly describes both arms as90/10; authoritative observed_train_weight and executed source are0/0.1, with final-score reproduction verifying the intended algorithms. No TEST or depth claim.

## Prospective new-partition gate

Next B4 partition is Github split2, fresh seed0, original own-model configuration and recording-cell rules. Reuse the established log-domain clock repair and full-gradient checker against the saved guarded split2 model; run five fresh TRAIN passes only after that contract passes. No new long fit is admitted automatically: require zero skips, finite TRAIN/DEV values, measured RSS and wall budget. Compare the pilot to the same saved guarded split2 prefix where available. This changes the data partition while holding the numerical repair fixed; it is the same Github cohort, not an independent dataset or extra unique-data evidence. Existing TEST scores and checkpoints are untouched. The eventual mark/readout target would use newly fitted split2 teachers; split1 checkpoint outputs must not be transplanted as validation evidence. Current admission is only backward contract plus five-pass pilot, no new mark objective or tuning grid.

Split2 backward check v1 stopped by RSS watchdog at2,363,516KB against2,000,000KB, before any pilot fit. Failure logs preserved; host~29GiB available. Fresh v2 check/pilot caps4,000,000KB, same model/data/settings and8GiB host reserve. This explicit resource amendment uses measured demand; no numeric/protocol gate relaxed.
