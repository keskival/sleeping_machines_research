# Credit across all event positions

Failure: the current paired suffix teacher always intervenes at token0 and
cycles depth/head. Later writes receive immediate value credit but no actual
future-write utility. This is a learning coordinate, not a family restriction.

For n tokens,D depths,H heads, uniformly sample one of M=nDH route sites
before reading outcomes. Conditional paired choice credit at that site has
expectation equal to its expected-utility score derivative. Multiplying by M
makes expectation over sites equal the sum of those route-choice contributions
to the chunk-mean objective. This is a discrete choice score estimator; factual
clock/content derivatives remain separate. It is not a claim of an exact
whole-network gradient including every clock and state distribution term.

Positions before the intervention cannot be affected by it. Uniformly sample
scoring positions from its remaining suffix. Multiply their mean loss
difference by (n-t)/n to recover the full chunk-mean utility. Combined with
1/site_probability=M and the alternative proposal importance weight, iterated
expectation recovers the sum of conditional route contributions. Position,
site and alternative sampling must be independent of targets; RNGs persist.

To avoid adding complete future utility on top of an immediate-only route
surrogate, the uniform-site member disables sampled local value credit across
the factual chunk. It retains winner computation, factual continuous learning,
race clock learning, sparse writes, persistent state, separate keys/values,
small messages and actual alternative-write replay. Its counterfactual learner
is the sampled actual suffix estimator. The old first-token/local-teacher
member remains the control, unchanged. The new member evaluates one factual
winner and one replay winner per race; key scoring and optimizer work remain
dense and charged. Importance weighting can increase gradient variance: inspect
gradient clipping, route utility spread and learned quality before promotion.

Contracts required: every site has known support, enumeration of site and
position estimators reproduces conditional utility-gradient sums, arbitrary
later forced writes affect their causal suffix, factual features do not change
when local surrogates are disabled, and actual resume includes site RNG. Start
with small fits. Full credit extends to the current chunk, not beyond its
detach boundary. Broader credit horizons and protected memory remain choices.

## Completed checks and small integrated fit

Guarded numerical queue completed18:29:03 UTC,2tests passed in1.47s:
site+causal-position enumeration matches summed conditional route-utility
gradients, site RNG resumes, and a forced token2write leaves the preceding
features unchanged while affecting its suffix. Actual6-update versus3+3
interruption/resume completed18:29:52, exact state/optimizer/all RNGs/quality
parity including the new event-site generator.

Seed6,2048training tokens,8160presentations,1024development targets,
P16/D2/H2/U4/B8/T16; same initialization/data budget as the completed first-site
K4 control. Uniform-site member best trained dev is**9.063122**, versus
**9.130868** for first-site K4: **0.067745NLL trained-quality improvement**.
Measured training throughput386.82 versus351.84targets/s. The intervention
combines site coverage, utility weighting and removal of immediate-only credit;
these are not three isolated ablations. This is a single-seed development
comparison, not an iso-FLOP or public benchmark claim.

Both trained members lose to the initial frequency model8.906910. New selected
entry point preserves that initialization and correctly selects it. The
uniform member's step64dev10.354817 is worse than first-site9.844873. Preserve
this late-overfit evidence beside the early gain. All receivers are used;
uniform member final route entropy1.2595 and cumulative max write share0.3227
do not indicate winner concentration. First-site entropy1.2048 and largest
share0.5810. These observations do not identify the cause of the later loss;
inspect decoder/generalization and estimator variance before changing routing.

The prioritized next integrated comparison is8K uniform-site K4, queued in
`addenda/aws_event_credit_tokens_20261005T183300Z.json`, after the first-site K4
control. Its selection includes initialization. Its publication wrapper bundles
raw results, selection, initial weights, final optimizer/state and best trained
weights so saved development selection has an actual transferable artifact.
Existing immutable packets/sources are preserved. AWS delivery remains pending;
GitHub shell/CLI lack authentication and the connected GitHub get-repo call
returns404 for `keskival/sleeping_machines`. No remote admission is claimed.
