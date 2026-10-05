# Sparse execution with actual future-write credit

## Failure being addressed

The local sampled value teacher observes immediate delivered-value utility.
Two receivers can deliver the same current value yet write different persistent
states, producing different later predictions. The completed full-proposal
future-write contracts demonstrate this distinction. Separately, evaluating
every receiver's value proposal makes proposal work grow with available pool
capacity. Combining the repairs is required before selecting a scaling member.

## Construction and retained mechanisms

`integrated_token_language_lab.py` combines packed parameters,
`sparse_counterfactual_episodes.py` and the bounded paired suffix teacher from
`TOKEN_FUTURE_WRITE_CREDIT_20261005.md`. The factual path evaluates the winner
and one sampled value alternative. A separately sampled forced write replays
the chunk with the same factual race noise; its later likelihood supplies
conditional receiver-choice utility. The shadow path evaluates winners only.
At the intervened site, the immediate local teacher is suppressed to avoid
teaching the same route twice through the two utility estimators.

The construction retains persistent state, computational race clocks and
transport, hard addressed updates, separate selection/value roles, small
messages, depth and alternative credit. Packing changes parameter storage and
optimizer dispatch. All keys are still scored; no sublinear discovery claim is
made. One site per credited chunk and chunk-bounded consequences remain explicit
learning choices. This construction does not supply direct derivatives through
the alternative map or consequences beyond that chunk.

## Resource consequences and required comparisons

Factual proposal count is independent of pool size; the shadow adds one
winner proposal per replayed race. Key scoring, rebuilding current-weight key
caches, shared-map expansion, readout, replay and optimizer work remain charged.
This is an execution/credit construction, not a completed speed or quality win.

Completed guarded queue:
`queue/curie_integrated_token_contracts_20261005_v1.txt`: **3 passed in1.49s**, solver
completed with exit0 at18:04:59 UTC. Tests cover parent
factual/local-credit parity after a weight change and EOS reset, packed leaf
gradients, actual future write effects, conditional choice-credit identity and
partition/RNG continuation. Admission used one CPU thread,270000KiB RSS cap,
5000000KiB address-space cap,8192MiB available-memory floor and60s timeout.
Actual integrated-driver interruption/resume parity is prepared in
`queue/curie_integrated_token_resume_contract_20261005_v1.txt`, unrun: prior
equivalent driver peak339MiB exceeds the current282MiB headroom. Next run that
contract when safe, then compare the
completed 8K token development budget with and without the future teacher.
Record full fitting and inference work before extrapolating to larger fits.

Selection remains open across the broader family: clock cadence, richer
reception, protected memory and decoder rank are revisable coordinates. Passing
these contracts validates this construction, not a requirement to retain it.

## Estimator contract —18:07 UTC

The driver now uses `paired_route_credit.py` to enforce the score-only estimator:
replay utility and sampling probabilities are detached inside the function.
Queue `curie_integrated_token_contracts_20261005_v2.txt` completed exit0 at18:07:47:
**4 tests passed in1.58s**. The added contract enumerates the actual mixture
sampling law and proves equality to the conditional expected-utility gradient
on nonuniform three-receiver fixtures, with zero forward contribution and no
replay-loss derivative. Numerical evidence is saved in
`results/diagnostics/curie_integrated_token_contracts_20261005_v2.json`.
The preceding v1 source is recoverable at main commit2c80abbd; its evidence is
preserved. Host available memory then fell to8321MiB; no larger job admitted.
