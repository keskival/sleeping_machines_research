# Stable causal standing-memory decay

16 October 2026, AWS. Feeds the existing ordered R1 RMR2 comparison; no architecture or recipe change.

## Failure identified before acquisition

`r1_rmr2.RMR2.standing_coeff` factorizes exp(-r(t-tau)) around chunk origin c0:

    exp(-r(t-c0)) exp(-r(c0-tau)).

The second factor grows for future writes tau>c0. The implementation evaluates it over the entire lane, multiplies by write weights, performs the matrix product, and only then masks future writes. For float32, exp overflows when its argument exceeds about 88.72. At the default r=softplus(-3)=0.048587, the first chunk of a 2,048-token selection lane reaches about99.46. The 256-token acquisition lane does not expose this. An impossible future write can produce infinity or zero-times-infinity before masking, and masking after the matrix product does not recover a finite answer. Short40-token contracts miss this failure.

This is a derived implementation defect, not a new measured learning result. Reproduction and repair checks are queued through the guarded battle pipeline; all six training slots are currently occupied.

## Equivalent bounded factorization

`r1_rmr2_stable.StableRMR2` removes writes later than the last query row before exponentiation. For own-history writes it also removes writes at or after the own-history cutoff c0. The remaining positive exponent is at most r(chunk_length-1); capped rate<=0.5 and enforced chunk<=128 bound it by63.5, below float32 overflow. Past-write factors and query factors decay rather than grow. The per-row causal mask still removes later rows within the same chunk.

For any legal (t,tau), the product remains exactly exp(-r(t-tau)) in real arithmetic. Every discarded term has zero causal contribution. Parameterization, forward law, gradients in real arithmetic, gates, standing capacity and learning objective are unchanged. This repairs arithmetic before further expensive acquisition.

The replacement fit driver includes standing count, selected standing count, activity cost and hard-fire setting in checkpoint identity. Its checkpoint also binds source hashes; nonfinite training objective fails immediately. The old wrapper stripped these settings before serializing its acquisition identity.

## Admission and scope

Unstarted RMR2v1 fits and their gates are withdrawn with queue text preserved. New RMR2v2 fits run only after their own gate passes. The gate compares the actual repaired operator with a direct causal decay oracle over8,192 possible writes, exercises early/middle/late chunks and own-history cutoffs, checks finite derivatives, checks a decay finite difference, and requires the old overflow witness to reproduce. Existing normalization/causality contracts are retained. Gates wait behind current K2/K8 fits; no active imported source is changed and no extra concurrent job is launched.

These models retain the temporal-memory carrier, BPTT, lane-bounded state and expected soft firing during training. That gradient is the gradient of the deterministic relaxation; it is not automatically the gradient of expected loss under sampled hard firing. Hard-fire quality and actual sparse dispatch require separate measurements. All standing keys are currently scored; dormant storage does not imply zero discovery or learning work.

## Why this advances the winning path

Persistent standing memory can pool evidence over many lower events and let higher levels refine reusable content. It is the most promising concrete compositional extension to the demonstrated causal-race language law. Protecting that construction from a late evaluation overflow is higher-value work than adding another unvalidated architecture. The next decision remains whether the admitted memory hierarchy improves comparable development prediction, especially for novel tokens and relations.
