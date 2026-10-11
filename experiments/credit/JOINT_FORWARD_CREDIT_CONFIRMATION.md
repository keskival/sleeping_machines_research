# B1/R1 forward-derived route-credit confirmation

Registered11 October2026 after completing original three-arm512 comparison, fixed-policy parameter-gradient diagnosis and full-policy DEV error analysis. This confirms one-seed synthetic development mechanisms, not a public benchmark or depth-scaling claim.

## What motivated this test

The original greedy primary interval crosseszero: reference over prefix+0.001101 nats/target. Post-run full72-route error analysis reveals a stochastic-objective improvement+0.000997, paired64-prefix bootstrap95%[0.000431,0.001559]. The training objective is precisely0.9softmax+0.1uniform expected route loss; greedy action selection is a different criterion. Keep the original gate false. The extra diagnostic is exploratory evidence, not retroactive confirmation.

At a fixed10584-parameter controller, reference credit reduces action-gradient covariance trace99.92% versus prefix, total estimated covariance94.14%, while full corrected expectations match exact policy autograd within5.6e-17. Future-gradient variation is98.79% of its remaining noise.

## Fixed protocol

Keep the same valid128 full_fixed controller/Adam checkpoint, frozen native depth4 carrier/readout start, three arms (prefix, exact frozen-forward, reference-learning),384 continued episodes, per-episode four causal readout Adam updates, policy rate0.0003/weight decay0.001/clip1,τ0.007422685957715026 and p1 correction. No architecture or hyperparameter tuning from the diagnostic.

Train two additional optimization streams r=1,2. Add1,000,000*r to original TRAIN observation seeds144701000+i, action seeds73000000+i and future seeds75000000+i. Both arms use the same stream within each replicate. All replicate controllers/optimizers are independently cloned. The original r=0 saved controllers are reused without retraining, preserving their paid acquisition/continuation costs. All three paths share the original128 acquisition and actor seed170; these are independent continuation streams, not three independently initialized native models.

Fresh shared DEV cohort:64 prefixes from native.batch(148124000+i), eight futures each from RNG89000000+i. No overlap with prior fit's147913000 cohort or information diagnostic146823000. Evaluate every72 route's independent actual learner cost, before any scorer/model selection. Scored policy is0.9softmax+0.1uniform over those72 routes. Both stochastic expected utility and deterministic greedy utility are retained. Reference choices, projected alternatives, optimizer work, copy/setup, all verification worlds and inherited fitting are paid.

Primary confirmation endpoint: mean reference-learning minus prefix stochastic utility across the three continuation streams, with a positive paired95% hierarchical bootstrap interval and positive point gain in each stream. Also require reference over its acquired128 starting policy to have a positive mean gain. Bootstrap resamples three streams and64 shared prefix indices, retaining arm pairing. With three streams the uncertainty is development evidence, not a distribution-wide guarantee. Static-forward versus prefix and reference versus static are secondary comparisons, not required pass gates. Greedy results are reported separately and do not replace the stochastic endpoint. No predicted/ongoing score fills a completed cell.

## Execution and next decision

One new-driver four-episode integrated smoke, then exactly one uniquely tagged replicate1 fit and one replicate2 fit through run_safe, serialized and guarded. Original stream is evaluated with the same fresh cohort in a separate one-job analysis queue. Commit source-bound completed JSON/PT; use no TEST or dense controls. After completion analyze paired quality and cumulative wall/supported arithmetic. A positive development confirmation admits a task-sized persistent-state integration; a failed interval triggers error analysis of remaining future-noise/optimizer effects rather than an unregistered seed extension.
