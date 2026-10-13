# Preserve selected temporal outputs at the language interface

13 October2026; R1 owned quality repair toward the pinned external FineWeb target.

## Failure and hypothesis

Frozen1024-history development error analysis at8M finds frequent ranks0–2000 are70.7% of targets and52.5% of total NLL. The rarest rank30000+ group is3.6% of targets and7.2% of total NLL. Widening only its six-dimensional tail cannot address most current error. These are one frozen seed on8192existing DEV targets, not an external benchmark score.

The current two-layer native core exposes only its final48-dimensional feature to the prediction head. The top2000 logits are affine in that feature, so their centered logit matrix has rank≤48 on any collection of factual contexts. Earlier selected event outputs are already computed but discarded at the readout. This rank bound is an implementation fact, not a proof that rank causes the observed loss. The repair hypothesis is that both layers hold complementary predictive information that the final interface fails to retain.

## Candidate

Each depth d emits a selected head-concatenated output z_d at its own completed arrival tau_d. For final arrival tau_D, expose

    Phi = concat_d T_d(tau_D−tau_d) z_d,

where T_d uses the existing per-head decay/rotation. The final layer's term is reused exactly. No observed token or future label bypasses the races. Each earlier output is transported forward to the same read time, so elapsed time continues to perform computation. A96-dimensional exactly normalized adaptive head predicts the same50257GPT2token targets.

Concatenation and multi-depth readouts are known primitives. The candidate applies them to selected timestamped native outputs, rather than replacing the substrate with a dense carrier. It retains hard temporal races, all-key discovery, sparse state writes, separate keys/values, persistent numerical memory, factual credit windows and actual alternative-write teaching. Same-parameter core states, probabilities, selected writes and factual route RNG are unchanged before fitting. Training gradients change intentionally because the head can use earlier outputs directly; the final layer still remains available.

For P24D2H2U4, selected writes remain4/target and scored keys16/target. The feature width48→96 increases the readout-boundary feature payload192→384float32bytes/target before placement/fanout; existing interlayer value deliveries are accounted separately, plus earlier-output temporal transport and a larger readout/optimizer. All work is charged. State capacity and token input lookup width are unchanged. The widened head also changes adaptive tail widths24/12/6→48/24/12; a successful fit does not isolate increased top-logit rank from improved tail expressiveness or optimization.

## Owned development and promotion

The external milestone remains the pinned3.3406 published Transformer run and ultimately the3.28 benchmark target, under the exact public scorer. Our repeatedly used2040-target DEV score is an engineering metric, not a comparable public score. The public validation prefix remains sealed during this repair.

`check_multidepth_readout.py` checks native float32/64 state, last-leg features, probabilities, selected writes, TRAIN-prior normalization and gradients restricted to the retained last leg. All-depth transport is an extra inference/learning path; the contract does not claim that its work is free.

Run the same retained probability0.25 recipe at524272targets, seed6. Reserve2097088targets regardless of a modest small-point deficit, unless a numerical/resource failure occurs: this tests acquisition behavior rather than ruling on the family from a short fit. One fixed member and one change, no width/capacity grid. Host resource stop: measured fit above2.5ms/target or RSS above1.2GB triggers execution diagnosis; RSS watchdog1.5GB/at least8GiBavailable remain. Record quality, data, capacity/selected activity, full training time and inference/readout costs. Compare to the saved same-data fixed-admission fit and frozen1024-history DEV protocol; final model selection uses development only.

The active8Mseed7 fit completes. Unlaunched8Mseed8 replication is redirected to this quality repair; its stopped continuation and all completed evidence remain preserved. Learned residual credit remains a bounded supporting pilot; it does not own the main quality-development queue. A first-pass readout result is labeled in development, never a family verdict. Error analysis after the reserved2M point determines whether to keep the interface, improve credit/depth conditioning or choose the next information path.

## Acquisition-first extension (user direction, 13 October)

Reserve the identical member at8M targets after the2M numerical/resource checks, even when smaller points lag. Selection examines paired quality and empirical NLL reduction per data doubling across512k/2M/8M, with total fit time reported alongside. For budgets n_i<n_j define s_ij=(L(n_i)−L(n_j))/log2(n_j/n_i). A larger s is a measured local acquisition slope, not an asymptotic exponent or evidence that a crossover must occur. All fixed-member fits reuse the saved same-data recipe; this is no tuning grid. Quality versus targets and versus measured resources are separate views; CPU time does not represent asynchronous ASIC cost. The resulting evidence decides whether this information path deserves further acquisition or requires a different repair.

## Completed vocabulary diagnosis

On the identical8192-target1024-history DEV slice,512k scores6.814035 versus6.813951 and2M scores6.455753 versus6.440696. At2M top2000 improve0.017824 within-bucket NLL, contributing0.012606 total improvement; other buckets contribute0.027664 worsening. The weighted decomposition sums to the0.015057 total loss. This supports a vocabulary trade-off, not a causal identification of width or optimization. Preserve the reserved8M test.

If the larger point also loses, prioritize an extension that contains the retained probability model exactly at initialization rather than replacing all adaptive-head dimensions at once. One possible normalized construction is p_alpha(y|h)=(1−alpha)p_base(y|h)+alpha p_extra(y|h), with alpha in[0,1] conditioned causally on observed native features. At alpha=0 the baseline is recovered, but extra-head parameter gradients vanish; training must explicitly address that boundary rather than claim free learning. Such containment protects initial predictions, not future optimized quality. Charge both head evaluations, stored parameters and learning work. This is a candidate rationale for the next decision after8M, not an admitted fit or a proposed dense replacement of the temporal core.

## Depth-first continuation (user direction, 13 October)

Useful temporal depth is the scaling priority; exposing more layer outputs at fixed depth does not test depth scaling. Retain the deepest selected-event path and counterfactual teaching. A prepared decoder extension keeps existing tail ranks, normalization and learned intercepts while initializing each decoder input matrix as [0 ... 0 W_base] on timestamp-aligned depth outputs. Thus baseline probabilities are contained algebraically, and earlier-leg weights can receive gradients from an already contextual decoder. Extra input weights increase top-logit capacity without rebuilding tail output factors. This is a known linear extension, not a new credit rule. It does not establish a depth advantage; arbitrary direct-output shortcuts can also encourage shallow solutions.

`depth_preserving_readout.py` supplies the independent decoder transformation only. Native numerical/gradient/RNG contracts and integrated evaluation are required before fitting. It does not transport Adam moments or extend the native core; restarting an optimizer must be explicit, and a paired warm-start control must use the same restart if needed. The next depth program must compare quality and full learning work across actual temporal stages, rather than mistake an interface-width gain for deep computation. Preserve current8M acquisition and Curie's compact-credit/depth ownership before admitting another model.
