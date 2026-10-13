# Forward responses teach future state credit

13 October 2026, B10. Integration of state-interface rank, Curie shared-path eligibility and native temporal credit memory.

Let s be the complete producer boundary, E its sensitivity to an admitted parameter-direction vector a, and L_future(s) the next observed block's loss at fixed actor parameters. The useful coordinate credit is

    b = E^T lambda, lambda = gradient_s L_future.
    L_future(s + epsilon E_j) - L_future(s - epsilon E_j)
      = 2 epsilon b_j + O(epsilon^3).

This supplies a forward-only teacher for the credit predictor. Fitting squared error on issued causal context h learns the conditional mean E[b | h], subject to function capacity and acquisition. The unpredictable remainder has conditional variance; a realized derivative is not guaranteed predictable before its observations arrive. Predicting a realized cotangent and choosing an update with useful expected future risk are different endpoints. State probes must stay within a numerically valid local region; finite-difference cancellation and curvature need auditing.

The implemented E is the existing rank4 analytic clock-parameter sensitivity through all two temporal layers and addressed slots,88state coordinates/352eligibility floats. E includes cross-layer content writes. It is not a rank4 representation of every parameter effect or of the optimizer-inclusive learner cut. The exact contract compares E^T lambda with full-history parameter derivatives minus future-only direct derivatives, not with independent own-leg traces. Error2.22e-16. Three forward-probe scales yield maximum absolute discrepancies below4.56e-7 at the selected recording-cell likelihood precision.

The native critic's readout predicts four b coordinates, retaining hard temporal races, persistent private banks and selected-value inference. Delayed forward labels train stored decision activations. No GRU or separate dense recurrent estimator is introduced. Training-time counterfactual native values remain paid.

Route/leg similarity becomes demand and operator conditioned. For admitted E, E^T E exposes state-effect overlap; E^T s and E^T delta_s expose response relative to the factual context. These summaries can support transfer but are not sufficient statistics for E^T lambda in general: equal Gram matrices can have different orientations relative to future loss. Operator-conditioned prototypes should state exactly which information they expose. Already revealed previous credit may enter the next causal receipt, never the current unrevealed target.

The first128TRAIN-pair/32DEV-pair acquisition with a fixed restored actor gives DEV MSE0.001211182 versus zero0.001197747:1.12%higher. This scalar-admission-to-vector readout has not earned actor-update deployment. Per-direction scale imbalance and missing operator/credit-receipt inputs motivate a same-size native variant exposing16Gram entries,8state/change projections and4previous revealed coordinates, with TRAIN-only per-direction normalization. No architectural family verdict follows from this first diagnostic. Both fits pay8teacher worlds per pair; changing the deployed critic requires future-loss utility and complete work evidence under actual clipped Adam.

Completed same-seed ablation: per-direction scaling alone gives DEV MSE0.001133650,5.35%below zero; geometry/receipt variant0.001144868,4.41%below zero and0.99%above scaling-only. TRAIN-mean baseline0.001239154 and causal previous-credit baseline0.002059063. All128paired DEV coordinate labels identical across variants. Retain the simpler scaled variant. The comparison identifies useful credit prediction and scale conditioning, not a benefit from the tested geometry summaries. Original native31,804parameters unchanged.128TRAIN frames pay1024teacher worlds/8192targets;32DEV frames pay256worlds/2048targets solely for scoring. Actor quality and complete work remain separate integration endpoints.
