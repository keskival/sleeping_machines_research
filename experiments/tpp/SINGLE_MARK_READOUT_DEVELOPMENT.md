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

Completed sequence diagnosis aws_single_readout_sequences_20261011_v1: readout helps 28/23/26 of34 DEV sequences across seeds0/1/2. Mean scores reproduce exactly. Losses concentrate in particular sequences; no DEV-driven fit or candidate selection follows this diagnostic. Multi-position prefix-equivalence contract queued next before portable serving promotion.
