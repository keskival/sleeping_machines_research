# Local update geometry and useful temporal depth

14October2026. Prospective theory/variant-selection note for the admitted calibrated/global continuation pair and the next native depth repair. Claims below are conditional derivations, not measured language scaling laws. Known Adam, residual conditioning, importance sampling and control-variate primitives are attributed as such; the engineering target is their faithful integration with temporal races and addressed state.

## 1. Intuition that changes the next variant

Global clipping couples every bank to the largest simultaneous gradient demands. A readout/teacher spike can attenuate an otherwise modest receiver signal. Local clipping removes that dependency, but arbitrary cap1 perbank abandons the global budget and changes optimizer history. Our same-cut audit measures this: cap1 gives RMSgradient3.530 versusglobal0.995 and calculatedAdamdisplacement1.288 versus0.916. Fixed calibratedcaps give1.116 and0.928, respectively. This supports testing calibratedlocal first; it does not prove preserved displacement direction or prediction quality.

The sensible first replacement is a close local approximation to the acquired useful update rule. Preserve weights, optimizer moments, memories, routing noise, data order and update grouping. Change one law. Do not add horizon, decoder, resets and depth to that comparison. The current slots4/5pair follows this design.

## 2. Independent caps can enforce an aggregate budget

Partition g into disjoint parameter banks g_b. Radial hard clipping is

    h_b = s_b g_b,   s_b = min(1,c_b/(||g_b||+epsilon)).

For nonnegative epsilon and positive c_b, ||h_b||<=c_b. Therefore

    ||h||^2 = sum_b ||h_b||^2 <= sum_b c_b^2.

Choose fixed caps with sum c_b^2=C_local^2. This certifies a model-wide gradient bound without an online global reduction. The allocation is computed offline and shared once; each bank needs only its own norm and cap. It does not preserve concatenated direction, does not match global clipping exactly, and does not bound the Adam step. Aliased/tied parameters require a canonical nonoverlapping bank partition; duplicating a shared parameter invalidates the sum-of-squares accounting.

Nonuniform caps are necessary when bank sizes/roles differ. Our quantile proposal uses TRAIN norm samples attenuated by the reference global factor, then normalizes the cap budget. Budget normalization changes activation rates; the requested quantile is not a guaranteed quantile afterward. Cold banks need a positive fallback and subsequent exposure checks. A1.25budget is an explicit moderately permissive candidate, not an optimum or predicted-quality interpolation.

## 3. Why clipping magnitude is not Adam magnitude

Fix the prior scalar Adam moments m,v, current raw gradient g and next local step t. Absorb bias corrections into

    A=beta1*m/(1-beta1^t), B=(1-beta1)*g/(1-beta1^t),
    V=beta2*v/(1-beta2^t), W=(1-beta2)*g^2/(1-beta2^t).

With gradient scale s>0, the adaptive direction before learning rate/decoupled decay is

    f(s)=(A+B*s)/(sqrt(V+W*s^2)+epsilon).

For positive V+W*s^2, differentiation gives

    f'(s)=[B*V-A*W*s+B*epsilon*sqrt(V+W*s^2)] /
          [sqrt(V+W*s^2)*(sqrt(V+W*s^2)+epsilon)^2].

The sign need not be positive. If inherited momentum opposes the incoming gradient, the numerator A+B*s can cross zero at s=-A/B. A smaller gradient can then produce a larger step magnitude or a direction reversal. With zero past moments and negligible epsilon, f(s)=B/sqrt(W), independent of positive s: first-step Adam nearly cancels gradient scale. Consistently scaling an entire history also cancels in idealized Adam with epsilon0; changing scales over time breaks that invariance. AdamW decay contributes the same -eta*weight_decay*theta term at an identical issued cut, but later weights differ.

Consequences: equal rawgradient norms do not certify equal update geometry; equal Adam norms do not certify equal direction. Per-bank radial clipping preserves rawdirection within each bank, while coordinate-dependent moments can still rotate the actual adaptive update. Global clipping preserves concatenated rawdirection but also changes Adam's response against inherited moments.

## 4. The better offline cap objective

For issued TRAIN cut k and bank b, let d_b,k(c) be the exact optimizer displacement under localcap c, retaining that cut's moments/version. Let d*_b,k be the global reference displacement. A candidate offline objective is

    minimize sum_b E_TRAIN ||d_b(c_b)-d*_b||^2
    subject to sum_b c_b^2<=C_local^2 and c_b>0.

This anchors a local rule to a useful acquired reference in update space instead of norm space. It is a proposal, not a convex problem: Adam's response need not be monotone. For fixed multiplier lambda>=0, the penalized objective separates into one-dimensional bank problems

    E ||d_b(c)-d*_b||^2 + lambda*c^2.

The separation is exact; a global optimum requires solving those subproblems globally and appropriate constraint/duality conditions. A local search is not a proof of optimality. Full update vectors, moments, replay, search and heldout checks are paid. This is offline calibration; it does not add an online cross-bank norm dependency.

Do not silently change the admitted quantilecap candidate. Measure its useful quality first. If it fails despite similar update norm, direction/momentum distortion is a specific next diagnosis; if it wins, an additional offline optimizer-space fit needs an answer-changing resource case. Avoid an automaticcapgrid.

## 5. Depth has two separate response penalties

On a frozen differentiable route branch, a residual stage h'=h+alpha*f_theta has parameter derivative alpha*partial f/partial theta. In a local squared-error model with fixed Jacobian and ordinary gradient descent, the parameter Hessian scales alpha^2 and the correction's response timescale scales1/alpha^2. The constructor alpha=0.5/sqrt(D) therefore gives an illustrative per-stage timescale proportionalD. This model assumes a fixed task/output path, no Adam, no route switches and stable curvature; it is not a native scaling measurement.

For Adam's zero-moment/epsilon-negligible first update, gradient scaling cancels in parameter space and the immediate residual-output displacement instead scalesalpha. That difference is why gain-only theory must specify the optimizer and acquired moments. Preserve oldstagegains and moments in a growth experiment; multiplying every oldstagegain by a newdepth factor confounds growth with changing its learned function/response.

Teacher dilution is separate. If one amongDsites is selected uniformly and a fixed site contribution G is estimated as D*I*G, where I has probability1/D,

    E[D*I*G]=G,
    E||D*I*G||^2=D*||G||^2,
    trace Cov=(D-1)*||G||^2.

The mean is correct while the variance grows. Its nonzero observations occur only aboutN_teacher/D times. Nonlinear clipping/Adam can change how those rare weighted events actuate learning. Do not explain deeper failure solely with insufficient total tokens; inspect useful site coverage, teacher weighting, update geometry and consequence reach.

For a fixed family of issued vector contributions with second moments v_i, sampling probability p_i gives secondmoment term sum_i v_i/p_i. Under sum p_i=1, its minimum uses p_i proportionalsqrt(v_i); under an additional expected-cost constraint, stationary interior probabilities take p_i proportionalsqrt(v_i/(lambda+mu*c_i)). These identities assume fixed contribution laws and complete support. Native probabilities/labels evolve; clipped-update utility is not automatically represented by rawgradient v_i. Use the existing full-support allocator/control-variate construction with these boundaries explicit.

## 6. Temporal growth is not an ordinary identity layer

_unit returns incoming+gain*correction, but the full stage also mixes channels, chooses a race, writes memory, advances arrival and transports values to head completion. gain0 alone leaves those paths live. A new finalstage can also replace the context fed to the next token. Consequently adding zero-gain stages does not establish identity of predictions, clocks or future state.

The strongest next growth candidate is an acquired-parent correction with preserved oldstagegains/optimizer/state and explicit transport/context semantics. Required initial contracts: finalstage predictions and oldparameter cotangents, oldarrival/clock paths, causal prefixes, meaningful carried context, unchanged parent RNG policy or disclosed difference, live newoutput correction gradients and complete charged execution. Newinternal parameters may initially receive no prediction credit until zero-output corrections move; distinguish this from a disabled permanent path. A learned correction must become useful; identity initialization alone is not a depth result.

## 7. Predeclared interpretation of the coming experiments

- Calibratedlocal retains/improves pairedquality with favorable fullwork: advance this local rule to deeper/delayed-credit integration; physical async execution remains a separate measurement.
- Similar displacementnorm but worse quality: inspect direction, perbank moment response, teacher-tail actuation and cap drift. Do not assume largerdata fixes it.
- Better prediction but more work: retain as a qualityfrontier point; optimize measured overhead rather than mislabel efficiency.
- D16 improves against D8 but still trails richerD2: useful local depthtrend, not overall competitive language progress. Test predictionpreserving growth/credit before more freshdepth construction.
- Preservedgrowth improves beyond equallytrainedparent: evidence that added temporal capacity is useful at this acquisition. Then choose a larger point and replicate.
- Largerdata improves only ordinaryparent equally: continuedfitting explains the gain; do not attribute it to newdepth/locality.

The selected continuation pair's qualityprotocol is identical8192DEVtargets/1024history. Intermediate loss curves diagnose dynamics; finalpairedscores decide the registered comparison. Report inherited/calibration/smoke/fit/evaluation work separately and together. No interpolation theorem, fitted scaling exponent, novelty assertion or frontier-language win follows from these derivations.
