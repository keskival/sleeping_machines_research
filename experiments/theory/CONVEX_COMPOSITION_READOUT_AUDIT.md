# Separate native representation learning from readout optimization

15 October 2026. Conditional frozen audit following the bounded composition fit.

Prior-centred gain only searches the ray spanned by an acquired classifier.
A nonpositive association on that ray does not imply that all emitted features
lack useful label information. Conversely, fitting a flexible head on training
examples alone does not prove reusable representations. Score unseen combinations.

Reuse completed initial/final native cores and cache their32-dimensional root
features on64 TRAIN calibration scenes and64 DEV scenes. Fit a binary affine
native head with logistic loss and a fixed .01 ridge penalty on TRAIN-standardized
features. This is standard convex logistic regression solved by damped Newton
steps. Penalize weights, not intercept; both labels are supported. Frozen TRAIN
centres/scales convert the result back into the original native linear head.
No native parameter, event schedule or state trajectory changes. No DEV-selected
penalty, feature selection, new backbone fitting or TEST scoring is performed.
All native rescoring and head fitting remain charged work.

Compare initial and trained cores under the same head fitting procedure. If a
trained core only improves TRAIN, repair representation/generalization. If it
improves DEV while the jointly trained head did not, repair readout optimization
before more backbone acquisition. If the acquired initial core already matches
the final core on DEV, do not attribute the gain to new representation learning.
Measure an emitted-content intervention with the fitted final head frozen when
DEV beats its balanced prior. Use a combination-cluster bootstrap for paired
loss differences; correlated nuisance/root variants are not independent samples.
A single native seed and fixed noise tape remain exploratory evidence.

Alternating convex head fitting and native core learning is a candidate repair,
not automatically admitted. At an exact inner optimum, the envelope derivative
of minimized TRAIN risk eliminates the derivative through the optimal head,
subject to the usual regularity and correct treatment of feature normalization
and penalties. This does not guarantee DEV improvement. The actual native-core
teacher still has its local counterfactual and event-order boundary limitations.
