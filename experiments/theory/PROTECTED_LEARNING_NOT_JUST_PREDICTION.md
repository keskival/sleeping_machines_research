# Protect learning policy, not only the initial prediction

15 October2026. Governs the next native depth repair; no new long fit admitted.

The controlled result is unambiguous: continued parent6.065562542 versus grown6.095076744. Frozen grown-actor ablations give zero-correction6.072677415 and parent-feature substitution6.200268526. Therefore the measured excess loss decomposes exactly as

    grown - continuedparent
      = (grown - grown_without_correction)
        + (grown_without_correction - continuedparent)
      = 0.022399329 + 0.007114873 = 0.029514201.

The learned correction harms this DEV endpoint. The substitution result is evidence that its learned projection is specialised to new features; it is not evidence that those features improve attainable prediction. Initial neutrality and nonzero gradients did not settle the quality question.

## A concrete learning-policy confound

The earlier growth helper required identical old-bank caps; the later integrated pilot calibrated complete parent/grown schemas independently after warmup. That changed acquired learning policy. For example transport-frequency cap0.040775249 became0.637506503, and transport-rate0.007847940 became0.005190203. The expanded transport array also mixed old and new depths inside one bank norm/Adam clock. These changes are not prediction containment. They prevent interpreting the quality gap as added depth alone, even though the complete tested construction clearly trails.

ProtectedGrowthAdamW now retains the original transport-prefix bank and adds separate new-depth views. Old caps/moments/versions and bank shapes are preserved; new caps must be supplied separately. A fixed aggregate cap budget must not silently renormalise old thresholds when capacity grows. If the old and new cap squared budgets are B_old^2,B_new^2, their combined radial bound is sqrt(B_old^2+B_new^2); report that paid budget instead of pretending it stayed B_old.

## Block-triangular learning

Let theta be acquired parameters, phi new parameters and s the complete acquired learner state (numeric memory, clocks, RNG, caps, Adam moments and versions). With growth, retain the parent's learning operator

    (theta_next,s_next) = U_parent(theta,s,observed_data).

Train phi from the augmented predictive law while holding the old-parameter update fixed to U_parent. Then induction gives exact equality of the acquired trajectory with the continued-parent control, assuming identical issued observations/noise and arithmetic. This protects continued learning, not just initialization. New-branch credit must not change old cap normalisation or send additional gradients into theta. Shared tensor storage is allowed; shared update ownership across the old/new boundary is not.

This can first be checked with a paired implementation that pays for a reference parent's raw credit, then implemented with reused parent features/teacher packets rather than duplicate inference. Charge the prototype's extra work. Preserve full supported new-route credit and new private state; an inactive output must not be mistaken for zero key-scoring or teaching cost.

## Earn the right to emit

An always-on learned feature correction is not an optional computation policy. Train emit/hold decisions from causal subsequent loss contrasts on TRAIN, with positive proposal support and owned factual/counterfactual clocks. Preserve parent service at its actual deadline; later branch computation cannot be backdated. Compare the learned conditional emission with always-on and parent-only outputs on the same stream. The next fit requires acquired next-update parity with the real calibrated learner and demonstrated new-path utility under the actual update rule. No unchanged deeper or larger acquisition follows from the completed deficit.
