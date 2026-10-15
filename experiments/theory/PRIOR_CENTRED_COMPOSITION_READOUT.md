# Useful queried representations can be hidden by a poor readout

15 October2026. Derived from completed native learning and content intervention.

## What actually learned

The64-update native32-stage FIT probe improves loss2.547975→2.482746. Its
label-association measure moves-0.013494→+0.037650, positive on both fixed
noise tapes. Accuracy reaches2/12, equal to the majority predictor, and raw loss
still trails the empirical constant-label predictor2.369382. This is input-use
learning on a tiny FIT set, not a benchmark or semantic transfer result.

Zeroing only query-response content preserves source/pose/type, receipt counts,
ordering and random-draw consumption in the contract. It changes association to
-0.014054 while improving raw loss to2.452741. Thus the contents carry useful
FIT label association, but also a larger label-independent prediction penalty.
Do not infer from the lower ablated raw loss that memory contents carry no signal.

## Exact loss decomposition

Let p be the empirical label marginal and q_i the prediction for FIT input i.
Define A=E_i E_(Y~p)[-log q_i(Y)]-E_i[-log q_i(y_i)]. Then

L = H(p) + E_i KL(p || q_i) - A.

For full query content, the label-independent penalty is about0.151014 and
A=0.037650. For zero query content, it is about0.069305 and A=-0.014054.
Content adds about0.051704 association but about0.081709 penalty: net raw loss
worsens about0.030005. The penalty includes both class bias and unhelpful
prediction variation; calling all of it calibration would overstate the diagnosis.

For logits z_i, A=E_i[z_i(y_i)-p dot z_i]. Every shared class-bias vector cancels
exactly. This makes association a useful complement to raw cross entropy here.
It is a FIT statistic, not proof of held-out information or abstract concepts.

## A lean readout repair without refitting native computation

Keep the native representations, timing, state and races fixed. Let mu=E_i z_i
and epsilon_i=z_i-mu. Use
q_i(g)=softmax(log p + g epsilon_i), g>=0.
At g=0 this is the empirical constant predictor. Its FIT loss is convex in g,
and L'(0)=-A. Therefore positive A guarantees some positive gain beats H(p) on
this same FIT set. Negative A makes g=0 the optimum on this nonnegative ray.
This is a mathematical learnability consequence, not a generalization guarantee.

`prior_centered_readout.py` solves the one-dimensional convex problem by derivative
bracketing/bisection with declared gain cap256. It learns no new feature extractor.
For an external linear head W,b the equivalent parameters are
W'=gW, b'=log p + g(b-mu).
All native object/core parameters and internal trajectories are unchanged.
Centres/prior/gain are fitted on the same FIT logits being scored, so the result
is optimistic calibration evidence, not a sealed quality score.

## Frozen experiment and attribution scope

Use acquired initial/final native weights and the same12 prefixes/two shared
race tapes. Save all per-example logits for future analysis. Compare initial,
final, zero-query final, initial-core/final-head and final-core/initial-head under
the same calibration. Recompute raw endpoints as a consistency contract.
Only the external class head changes in the saved calibrated final artifact.

Swaps measure fixed-decoder effects and coadaptation. A representation can rotate
its basis during training, so a swapped-head loss alone is not an invariant
ranking of representation quality. Likewise this tiny FIT protocol cannot
establish learned object-part reuse. The experiment decides whether readout
conditioning deserves repair before more native fitting; held-out composition
and time/source contribution tests follow successful learnability evidence.

All native rescoring and one-dimensional calibration work is admitted through
the existing coordinator. No backbone is refitted and no external model is trained.

## Completed frozen result

The final core/head gives 1.925817 FIT NLL and 37.50% accuracy after gain
25.644832; the constant-prior baseline is 2.369382 and 16.67%. Initial and
zero-query variants choose gain zero. Trained core/original head gives 1.926302;
original core/trained head gives the prior. This locates useful signal in the
trained native core under these fixed decoders. All native parameters remain
unchanged by calibration. The five-way frozen experiment takes 164.565 CPU
seconds, 297,376 KiB peak RSS. Source: results/hierarchy/
aws_g_hier_native_frozen_readout_20261015_v1.json. Same-FIT scope above applies.
