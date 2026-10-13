# Optimizer-context identifiability for native credit

13 October2026, B10. A correct full-state teacher need not be learnable from an insufficient critic input.

The current19coordinate operator descriptor contains four projected own gradients, four parameter means, four first-moment sums, four second-moment means and norm/step/clipping statistics. Native before/after/change state and causal private memory accompany it. A fixed-cut witness holds actor parameters, numerical state, issued gradients and observations fixed while jointly permuting each clock group's first/second moments. Such permutation preserves pooled statistics and moment validity.

Three real actor versions give descriptor differences4.34e-19/1.08e-19/6.51e-19 while functional refreshed-future action demands differ0.072669373/0.000188479/0.002096309(maxcoordinate); demand signs change. Future loss also changes. The optimizer cuts are diagnostic states, not established identical-history reachable pairs. This does not measure alias frequency in the actual acquisition or exclude reconstruction from richer causal memory.

For two cuts with identical critic information z but demands q1,q2, any deterministic prediction has equally weighted squared error at least ||q1−q2||²/4. This follows by completing the square around their mean. General conditional variance is the irreducible error for that information set. Larger parameter capacity or more samples cannot remove exact information loss; richer receipts or enough informative history can. These are standard identifiability facts applied to our measured learner cut, not a new impossibility result for learned credit.

## Preserve mode relationships without a dense extension

Use typed route/node response receipts rather than averaging away the association between a mode and its optimizer state. For actual global clip/Adam, write the preclip future demand as

    w_i = D_i lambda_theta_i + (1−beta1)lambda_m_i
          + 2(1−beta2) z_i lambda_v_i.

Here D_i is the known local Adam parameter sensitivity, z_i the clipped issued gradient and lambda the future learner-cut demands. A sparse action direction B_j then receives, in the clipped region,

    q_j = k[B_j^T w − (B_j^T g) Gamma/(r(r+epsilon_c))],
    Gamma = g^T w.

Unclipped q_j=B_j^T w. Current four clock supports contain64mode coordinates. Mode-specific signed optimizer responses and route identity can be supplied as native event tokens; the connected temporal critic predicts future demands or finite response values conditioned on these tokens and its causal memory. Existing local optimizer arithmetic provides response coefficients. This avoids treating mode identities as interchangeable merely because their means match, and avoids prescribing a large new dense context projection.

The global Gamma contraction remains part of actual globally clipped Adam; sparse support does not eliminate its effect on every parameter bank. Known signed response coefficients alone are not sufficient for future value: future state/moment demands still require learning. Future tests should compare old summaries with mode-specific receipts under identical teacher/data/deployment-state protocols and include their event/key/selected-write cost. Existing parent predictor is retained until that integrated comparison improves actual forecasts.
