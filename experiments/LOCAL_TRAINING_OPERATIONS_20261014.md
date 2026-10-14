# R1 local training operations — 14 October2026

User authorizes replacing global gradient clipping and other global operations with local alternatives. User clarifies that the preferred bound preserves vector direction within each meaningful parameter bank; coordinate-wise saturation is not the default. Existing global-clipping acquisitions are preserved as controls. No new external architectures.

## First declared rule and comparison

`local_clip_receiver_token_language.py` changes the integrated allocated D8 trainer to per-bank L2 clipping, cap1 per receiver/head/shared parameter bank, before its local AdamW update. It removes the model-wide norm/scale entirely; each packet performs its own schema, version, finite and norm checks. The unused cross-layer balance reduction is skipped when balance0. Existing per-bank clipping is a known primitive, not claimed as an invention.

This differs from global clipping: unrelated large gradients no longer rescale a receiver's update, while the total model norm can exceed1. With B banks each bounded by c, the combined clipped gradient norm is at most c sqrt(B), versus c for global clipping. Same-data quality/work is the comparison; thresholds are not claimed to match total update energy. Shared embedding/readout banks remain large. Raw gradient bounds are not Adam parameter-step bounds.

Three-update manual per-bank clip/AdamW/weight-decay/resume contract passes exactly; changing another bank's gradient from scale1to1000 and reversing delivery order leaves the tested bank unchanged. NativeD8 smoke then reserved8M follow in guardedslot1. Controllers retain at least8GiBavailable/RSS3GB/VMS12GB/oneCPU; fit timeout from measured smoke throughput, at most7days. No small-quality veto. Compare with the same allocated/global-clipped D8 using paired8192DEV/1024history and full training cost; public validation sealed.

## Operations audit

| Operation | Actual boundary | Action |
|---|---|---|
| Model-wide gradient norm/scale | All parameter banks | Removed in local candidate |
| Global balance penalty with coefficient0 | Cross-layer summaries | Skip unused work |
| Adam moments/step/weight decay | Versioned local bank | Already colocated; all declared zero/nonzero updates retained |
| Norm clipping inside one bank | Receiver/head/shared tensor | First local alternative; shared banks still large |
| Factual backward aggregation | Event graph/credit window/shared parameters | Retained and charged; learned causal credit is next replacement project |
| LayerNorm | One payload vector | Local normalization, not a whole-model reduction |
| Race/proposal normalization | One head's candidate pool | Retain probability semantics; discovery still charged |
| Adaptive vocabulary normalization | Selected prediction head/tail | Required scoring work; explore hierarchical/local implementations under exact protocol |
| Batch loss mean | Known target count and shared gradient accumulation | Scaling denominator known; credit accumulation remains, not removed by deleting a scalar mean |
| Diagnostics and checkpoint selection | Read-only observation/DEV ledger | Not an actor-update dependency; charged in relevant diagnostic boundary |

## Smooth radial bound

For a bank gradient g, use S_c(g)=c tanh(||g||/c) g/||g||, S_c(0)=0.
This preserves direction and strictly orders finite mathematical bank norms under one common c, bounds each norm by c, and is close to identity for small norms. It compresses ratios and does not preserve whole-model direction when different banks receive different scale factors. Finite-precision saturation can tie very large norms. `learning/smooth_bank_bound.py` provides this primitive; no fitted smooth-bound result is claimed. The currently admitted comparison uses hard radial bank clipping, not coordinate-wise clipping.

Choosing a common factor from the whole-model norm preserves whole-model direction but retains a global reduction. Local bank factors deliberately change inter-bank weighting. Smoothness removes a hard kink; it does not remove stochastic estimator bias or restore unbiased actor updates after nonlinear clipping. With Adam, moments/preconditioning further change step size and direction, so actual future utility matters.

## Prior art and interpretation

- Pascanu, Mikolov & Bengio2013: norm clipping for exploding recurrent gradients. https://proceedings.mlr.press/v28/pascanu13.pdf
- Zhang, He, Sra & Jadbabaie2020: clipping as adaptivity under gradient-dependent smoothness. https://arxiv.org/abs/1905.11881
- Koloskova, Hendrikx & Stich2023: stochastic clipping bias and tight convergence guarantees. https://proceedings.mlr.press/v202/koloskova23a.html
- Brock et al.2021: unit-wise Adaptive Gradient Clipping based on gradient/weight norm ratio. https://proceedings.mlr.press/v139/brock21a/brock21a.pdf
- Barrett & Dherin2021: implicit gradient regularization from finite-step gradient descent, not proof that clipping equals a fixed penalty. https://arxiv.org/abs/2009.11162
- Cattaneo et al.2024: Adam implicit bias depends on optimizer parameters and training stage. https://proceedings.mlr.press/v235/cattaneo24a.html
- Smoothed Gradient Clipping and Error Feedback for Decentralized Optimization under Symmetric Heavy-Tailed Noise2023: assumptions are smooth/strongly convex local functions and symmetric heavy-tailed noise, not a theorem for our native learner. https://arxiv.org/abs/2310.16920

Deterministic global radial clipping preserves gradient direction. In continuous-time gradient flow, a positive scalar rescaling changes time parametrization, not the path. Discrete steps, stochastic noise, local rescaling and Adam break that simple argument. Generally clipped gradients are not the gradient of the original loss plus one fixed regularizer. A gradient-field integrability condition is needed before such an interpretation. Our empirical comparison will separate stability, attenuation, useful later quality and full work; no novelty or generalization claim is assigned from smoothness alone.

Additional direct smooth-saturation prior art: [SoftAdaClip, Soleymani, Dadsetan & Rudzicz2025](https://arxiv.org/html/2510.01447v1), Algorithm2, applies g'=tanh(C/(||g||+epsilon))g to per-example gradients for private training. It is radial and direction-preserving, with a bounded strictly increasing norm response; its inverse-argument tanh differs from the c*tanh(r/c) radial output bound above. Its utility/fairness evidence is DP-specific, not evidence of our deep language utility. The2023 decentralized smoothed-clipping paper provides another theoretical precedent under explicit heavy-tail/symmetry/convexity assumptions. No invention claim is made for gradient saturation itself.

The decentralized SClip-EF paper's specific smooth operator is component-wise y*phi/sqrt(y²+epsilon), applied to gradient-estimation errors with feedback. It is related saturation theory, not a direction-preserving radial precedent. SoftAdaClip is the directly relevant radial tanh precedent. Our native preference remains radial bounding within a declared receiver/head/shared bank.

NativeD8 local-cap smoke completed:32,768targets/64updates/47,488deliveredcreditpackets,106.255trainingseconds,843604KBpeakRSS, finite losses and gradients in everylayer. `aws_local_clip_receiver_d8_8m_20261014_s6_v1` is running under measuredtimeout/RSSguards. This smoke does not demonstrate quality or efficiency versus global clipping; both full8M variants must complete.
