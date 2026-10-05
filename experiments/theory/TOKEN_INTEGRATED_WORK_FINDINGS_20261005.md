# Integrated token work and next credit optimization

Actual driver resume parity completed18:12:14 UTC: uninterrupted6updates and
3+3 interrupted/restored updates match model, AdamW, numeric state, factual,
local-value-alternative, future-alternative and global RNGs, cursor, writes,
quality curves and48cumulative targets exactly. Peak RSS337428KiB. Evidence:
`results/diagnostics/curie_integrated_token_resume_contract_20261005_v1.json`.

## Executed operator ledger

P16/D2/H2/U4, GPT-2 vocabulary50257, frequency-initialized adaptive decoder,
untied maps, first AdamW update. Synthetic targets exercise each decoder tail
equally; this is not the empirical FineWeb frequency mixture. Both columns
below use identical accounting and denominator128targets. Random sampling work
is unquantified; assignments/indexing are memory work rather than arithmetic.
Arithmetic and special-function evaluations are separate, not hardware energy.

| Stage | Whole-step arithmetic MFLOPs | Arithmetic MFLOPs/target |
|---|---:|---:|
| Factual event computation |4.136320|0.032315|
| Factual likelihood |47.286432|0.369425|
| Alternative sampling |0.000248|0.000002|
| Winner-only replay event computation |3.006080|0.023485|
| Replay likelihood |47.286432|0.369425|
| Paired utility |0.000425|0.000003|
| Backward |118.366892|0.924741|
| Gradient clipping |6.346425|0.049581|
| First AdamW update |27.501063|0.214852|
| **Total** |**253.930317**|**1.983831**|

All executed floating operators have formulas in v3. Separate special-function
ledger totals7,712,456 evaluations. v1 is preserved with incomplete coverage
for reverse subtraction and indexed overwrite; v2 corrects those formulas on
the8-target fixture. v3 measures the pilot batch shape128targets. Evidence:
`results/diagnostics/curie_integrated_token_work_audit_20261005_v{1,2,3}.json`.
These are one-step audits, not extrapolated whole fitting costs or comparisons
with dense references. Counts include all-key scoring and current-weight cache
rebuild, sampled proposals, real suffix replay, likelihood, backward and AdamW.

## Next design choice: sample utility scoring positions

Concrete failure: scoring the entire replay suffix uses47.286432MFLOPs for
likelihood versus3.006080MFLOPs for its event computation in this fixture.
Counterfactual training should not pay an unnecessary full decoder pass.

For lane loss differences delta_t and n scoring positions, the existing
utility is mean_t(delta_t). If K positions are sampled uniformly without
replacement independently of outcomes, their sample mean has expectation
mean_t(delta_t). Substitution into the detached paired-route estimator preserves
its conditional expected route-utility gradient by iterated expectation.
This changes estimator variance, not the expected suffix objective. It supplies
no gradients through shadow maps and extends no credit horizon.

Retain the factual full likelihood, actual alternative write, causal replay,
race noise and all current state/routing mechanisms. Score only the sampled
shadow features and their matching factual losses. Charge position selection,
replay until the last selected position, selected likelihood and the optimizer.
Keep all positions as K=n numerical/learning parity control. A distinct saved
position RNG, exact resume/partition contracts, enumeration of utility sampling
and matched small-quality comparison are required before promotion. K=1,4,n
are possible coordinates; select by measured quality/work rather than assuming
the smallest K is best. Existing AWS source-pinned queues remain unchanged.

## Implemented sampled utility —18:19 UTC

`sampled_utility_token_language_lab.py` implements independent persistent
position sampling, matching factual/scored alternative targets, and causal
replay only through the last sampled position. The original driver and pinned
AWS packet are unchanged. Three numerical contracts passed. Actual sampled
K2 interruption/resume matches every learning state and RNG exactly; the K=n
control reproduces the original6-update model/AdamW/quality trajectory exactly.
Completed result files use `curie_sampled_*_20261005_v1` names.

On the same synthetic B8/T16 fixture, K4 sampled positions[1,3,6,9] use
**218.255413MFLOPs/step,1.705120MFLOPs/target**, compared with full scoring
253.930317 and1.983831. Total arithmetic is14.05%lower; replay likelihood is
12.713512MFLOPs versus47.286432. All floating operators have formulas. This is
one realization of estimator work, not a quality or hardware speed win.

Additional immutable AWS packet `addenda/aws_sampled_tokens_20261005T182000Z.json`
queues an8K K4 comparison after the existing full-score future-credit arm.
It reuses the same data, dimensions, target budget, seed and decoder. Future
quality and estimator variance remain for the completed comparison to measure.
