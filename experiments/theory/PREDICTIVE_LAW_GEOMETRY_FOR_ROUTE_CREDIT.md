# Predictive-law geometry for route credit

12 October 2026. B10/native credit direction. This analytical note connects the user's route-similarity and interchangeable-strategy intuition to the forecast-response interface. Fisher geometry, conditional projection and smooth-risk descent are established mathematics; the contribution here is their concrete role in the reciprocal temporal model, not a claim that these tools are new.

## Similarity concerns consequences

For observed history H, let Q_theta(E_1:h|H) be the model's normalized causal future-event law over horizon h, with numerical state advanced by the observed/generated events. A route or optimizer control induces an actual parameter-response direction d. Its predictive score is

    s_d(E_1:h,H) = d^T grad_theta log Q_theta(E_1:h|H).

Define K_H(d,e)=E_Q[s_d*s_e]. This is positive semidefinite and measures common predictive effects, rather than shared node IDs, parameter coordinates or activation proximity. Directions with zero score are locally indistinguishable under this history and horizon. Identical policies or permutations of equivalent mixture clocks need not be separately identified when they induce the same predictive law. The finite predictive distribution, and this local differential metric, are distinct notions.

The horizon matters: two writes can have equal immediate forecast yet change different persistent memories and later laws. A one-event kernel does not justify long-term route equivalence. Version the history, operator and horizon; online mixture changes can change the useful similarity. No stationary-bias assumption is required.

For B10 use the normalized recording-cell likelihood, including marked zero-gap cells and survival, rather than a continuous-density score substituted after the fact. At finite parameters and integrable scores, differentiating normalized probabilities gives E_Q[s_d]=0. Information within a cell cannot be identified beyond the recorded resolution. The kernel can be estimated on sampled causal forward outcomes and selected directions; no dense parameter-square Fisher matrix is required. Forward sampling, direction responses and delayed checks are real work to charge. Learned probes may approximate this geometry but raw clock-coordinate distances are not this kernel.

## What a useful critic must learn

Let f_H be conditional future risk, and D(H) a fixed matrix of admitted actual-update directions. At a reference parameter vector define T=D^T grad f_H. The best square-error prediction from causal features Z is m(Z)=E[T|Z]. With an update delta=-eta*D*m and conditional L-smooth future risk,

    E[f_H(theta+delta)-f_H(theta)]
       <= -eta E||m||² + (L*eta²/2) E||D*m||².

The first term follows from conditional expectation; the second is the smoothness remainder. Under finite moments and a sufficiently small nonzero step, a nonzero predictable teacher gives a descent opportunity in the admitted span. If D is random it must be fixed/measurable in Z when applying this conditional identity. The bound is local, assumes the conditional future objective used by the teacher, and is not a guarantee for adaptive Adam, regime extrapolation or arbitrary learned directions.

Adding causal response R increases E||E[T|Z,R]||² by

    E||E[T|Z,R]-E[T|Z]||².

Thus better information can increase the useful predictable component; curvature and extra computation can still erase the advantage. A learned critic succeeds by predicting future utility in inexpensive admitted directions, not by reproducing every coordinate of past-data backprop. Mean-changing prospective credit and unbiased factual residual correction remain separate targets.

## Engineering implication

The current162-field packet provides typed factual clock parameters and their local-Adam response through a native16-dimensional adapter. It is a first explicit predictive-effect input, not a Fisher sketch or a sufficient statistic for all future event laws. Compare the common-acquisition masked-input arm to see whether this information is used. Then charge direction discovery, forecast probes, native state, audits and actual future teaching against the exact propagation replaced. Increasing critic capacity without informative effects or an economical replacement budget does not meet the objective.
