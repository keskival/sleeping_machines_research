# Future learning value of route memory under depth and drift

13 October 2026. R1/B1 theory extension to LEARNED_ROUTE_SIMILARITY_MEMORY.md and DEEP_TEMPORAL_CREDIT_CLOSURE.md. Derives a solvable conditional model for deciding which route receipts to acquire and retain. It is a design instrument, not an implemented native architecture, measured advantage or universal optimal exploration policy. The running D8/D16 and8M/32M acquisition program remains the priority.

## 1. Similarity and exploration need a future-demand distribution

An activation-history summary describes where a route operated. A future-consequence receipt describes what its intervention did. Neither alone determines which experience will help future learning. The missing third object is the distribution of future demands on that learned relation.

Fix a causal learner cut and a versioned learned feature map. Let phi(r,omega) in R^k describe a route r and a demand omega, using only information available when the query is issued. Suppose the reusable conditional effect is

    u(r,omega) = phi(r,omega)^T beta.

Here u is a declared finite-horizon actual-learning utility or producer-response target, not necessarily immediate prediction loss. Prior notes define how actual forward continuations supply such labels. A linear final effect head is a local model of the learned geometry; the encoder and native temporal state need not be linear.

Condition the current belief on legitimately acquired receipts:

    beta | memory ~ Normal(m, Sigma).

This assumption produces exact conditional formulas below. It does not assert that the full nonlinear actor/critic has a Gaussian posterior. Empirical covariance or learned uncertainty can approximate it, but oracle optimality then no longer follows automatically.

Define a future deployment measure nu and nonnegative weights w. Its demand matrix is

    M = E_(r,omega~nu) [w(r,omega) phi(r,omega) phi(r,omega)^T].

The posterior expected squared effect-prediction error of the posterior mean is

    Risk = tr(M Sigma).

Proof: at each query the error is phi^T(beta-m), whose conditional second moment is phi^T Sigma phi; take the weighted expectation and use the trace identity. This is a future-weighted prediction risk. To interpret it as actual learning harm, choose response coordinates and weights from the declared update/future-loss geometry; otherwise it remains an auxiliary prediction objective.

## 2. Closed-form value of one future learning receipt

A candidate audited receipt has observation

    y = x^T beta + epsilon,   epsilon ~ Normal(0,sigma^2),

with sigma^2 > 0, noise independent of beta conditional on the issued context. For a paired route contrast, x = phi(r,omega)-phi(s,omega), and sigma^2 is the paired observation-noise variance, including common-future covariance. Sampling and continuation work are paid.

The Gaussian conditioning equations are

    K = Sigma x / (sigma^2 + x^T Sigma x),
    m^+ = m + K (y-x^T m),
    Sigma^+ = Sigma - Sigma x x^T Sigma / (sigma^2+x^T Sigma x).

Therefore the expected reduction in future prediction risk is exactly

    Delta(x) = x^T Sigma M Sigma x / (sigma^2+x^T Sigma x).

It is nonnegative because M is positive semidefinite. It depends on whether this observation resolves uncertainty in directions future decisions will use. The innovation y-x^T m trains the value; the known update trains its reliability at the same cut. There is no additional learned supervisor above this conditioning identity.

Acquisition priority can be Delta(x)/c(x), or Delta(x)-lambda c(x), where c includes discovery, descriptor reads, both worlds, forward/learning replay, receipt processing and storage. This is the one-receipt optimum for the stated surrogate. It is not the optimal multi-step adaptive exploration policy: acquisitions can overlap or change subsequent actions, states and demand distributions.

Information gain or activation novelty alone can choose the wrong route. For example, if Sigma=I and M=diag(1,0), an observation in direction(0,1) has positive information gain but Delta=0; a direction(1,0) observation reduces the risk future deployment actually uses. The zero represents an assumed irrelevant direction at this cut, not a reason to remove exploration coverage indefinitely.

## 3. The value of stored counterfactuals is not additive

For a batch of observations y = X beta + epsilon with conditional covariance R positive definite,

    Sigma^+ = Sigma - Sigma X^T (R+X Sigma X^T)^(-1) X Sigma,
    Delta_batch = tr[M Sigma X^T (R+X Sigma X^T)^(-1) X Sigma].

Counterfactuals sharing an exogenous future can have correlated observation noise. R must retain that covariance or a justified approximation. Counting their individual values as independent can overstate acquisition evidence. Identical effect directions are also redundant even with independent noise; conditioning on one receipt changes the marginal value of the next.

This defines the learning value of the counterfactual pool as feedstock for future interventions. Store the reference action, shared-future identity, noise/covariance provenance, horizon, learning policy, state/operator/optimizer version and observation propensity. A direct factual/alternative loss difference without these types cannot identify which future-learning target it teaches.

## 4. Drift changes durability, not just recency

Let the reusable relation evolve at known conditional dynamics

    beta_(t+1) = A_t beta_t + eta_t,   Cov(eta_t)=Q_t.

Before a new receipt, the belief predicts

    m^-_(t+1) = A_t m_t,
    Sigma^-_(t+1) = A_t Sigma_t A_t^T + Q_t.

A_t and Q_t may be learned from causal successive receipt innovations. The exact covariance interpretation requires the stated linear-Gaussian model and calibrated dynamics; a learned plug-in is an estimator, not a theorem about the deployed distribution.

For a fixed future query plan with demand matrices M_(t+h), define P_h=A_(t+h-1)...A_t and nonnegative horizon weights gamma_h. A present covariance reduction D changes future summed risk by

    tr(Gamma_t D),
    Gamma_t = sum_(h=0)^H gamma_h P_h^T M_(t+h) P_h,
    P_0 = I.

Proof: under the same future dynamics and query plan, independent process-noise contributions are identical with and without the receipt; the covariance difference at horizon h is P_h D P_h^T. Sum and move factors around the trace. Thus the future-learning value of receipt x is

    Delta_H(x) = x^T Sigma Gamma_t Sigma x / (sigma^2+x^T Sigma x).

This distinguishes three phenomena: uncertainty that a receipt can reduce now, the durability of that relation under drift, and how often future computation will need it. A recently observed relation may be low value if it is irrelevant; an old but stable relation can remain high value when its demand recurs. Q increases forecast uncertainty but cancels in this paired covariance difference only when its dynamics are unaffected by the intervention. If the receipt changes later learning/actions or observation schedules, use an actual causal continuation teacher for total value instead of this fixed-plan formula.

The future demand and drift operators must be forecasts from available history at decision time. Future observations can train those forecasts when revealed; they cannot be oracle inputs to the original query.

## 5. Reuse similarity and uncertainty are distinct

For delta_phi(omega)=phi(r,omega)-phi(s,omega), the posterior expected squared effect distance is

    E_beta E_omega [(delta_phi^T beta)^2]
      = E_omega [(delta_phi^T m)^2 + delta_phi^T Sigma delta_phi].

The first term measures predicted behavioral difference. The second measures uncertainty about that difference. A large second term is an opportunity to acquire knowledge, not proof that routes are behaviorally dissimilar. Do not collapse both into a single retrieval rejection score.

Retrieve by learned mean-effect geometry, then qualify transfer by uncertainty, present operator/optimizer compatibility and forecast demand. A jointly permuted route/address descriptor should transform responses consistently; address coincidence alone is not transferable knowledge. Activation distributions, signed response sketches and temporal/version metadata supply conditioning, while consequence labels learn which aspects matter.

Depth contributes more producer cuts and reuse opportunities. It also increases receipt horizon and cost. Shared effect coordinates are valuable when many deep producers reuse the same learned conditional relation while current native transport handles their local geometry. They are harmful when the compression aliases producer-specific directions. Increasing data cannot repair exact aliasing; enrich the typed packet before increasing head capacity.

## 6. Train all estimators from the same finite stream

Each issued causal record contains: native route/cut and activation-history sketch; current operator and optimizer-response information; phi and its encoder version; predicted effect, noise and future-demand descriptors; proposal/audit propensity; shared continuation identity; actual acquisition/reuse work. When feedback arrives, append the typed consequence and innovation.

That single receipt can support:

* Effect regression from y-x^T m.
* Calibration of predicted innovation variance against squared revealed innovations, with version and selection corrections.
* Drift/demand forecasting from successive revealed receipts and later query usage.
* Acquisition-value training from paired future performance with and without that receipt, using actual later learning when the fixed-plan approximation is insufficient.

Using all observed receipts estimates the collected distribution. Importance weighting to another supported target distribution must use actual propensities; reused labels do not become independent examples. Encoder changes require retained raw descriptors to be re-encoded or an explicit coordinate transport; old numerical feature vectors are not automatically valid in a new basis. Rank-k covariance costs O(k^2) storage/update before any sparsity or sketch approximation; this is not free work inside a forward pass. The same response and innovation can be reused, but storage, covariances and additional heads remain charged.

Closure occurs at forward-observed consequences and known conditional/differentiation identities. Auxiliary estimators can share a native parameterized function and one existing backward call. That does not make their chosen weights or surrogate objectives optimal: actual future task utility remains the adoption criterion.

## 7. Engineering consequence for the current program

Do not displace the actual D8/D16 and8M/32M fits with a new small critic grid. Use the saved per-layer gradients/occupancy, causal activation histories and error buckets to identify which deep cuts merit richer descriptors. Curie's allocated audit program and AWS uniform teacher program supply complementary receipt streams, not identical acquisition policies.

The next compact-credit implementation should record mean effect, uncertainty and future-demand weighting separately, then compare prospective quality and complete work of retained-memory versus no-memory learning under changing dependencies. A fixed learned geometry can be tested on disjoint routes or regime recurrence, with causal numerical state updates explicitly declared. Public language validation remains sealed during design. This note supplies a constructive choice rule and memory schema for that future integration, not a current language win.
