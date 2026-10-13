# Useful depth: prediction, learning and economic value

13 October 2026. R1 decision theory for the admitted D8/D16, 8M/32M acquisitions. No architectural substitution or additional fit is admitted by this note.

Sleeping Machines aims at a general-purpose temporal, addressed substrate. The next language evidence must connect its compositional depth and persistent memory to useful prediction, then connect the learning mechanism to an economical acquisition curve. More stages, nonzero gradients and more stored capacity answer different questions.

## What useful information would mean

Let Y be the next token, Z the legitimately available shallow representation and U additional deep evidence, all under one specified data distribution and observation policy. For unrestricted normalized predictors with finite conditional entropies, the optimal expected log losses satisfy

    inf_q E[-log q(Y|Z)] - inf_r E[-log r(Y|Z,U)]
      = H(Y|Z) - H(Y|Z,U) = I(Y;U|Z).

For actual fitted predictors q and r, their difference is instead

    L(q) - L(r) = I(Y;U|Z)
      + E KL(P(Y|Z) || q(.|Z))
      - E KL(P(Y|Z,U) || r(.|Z,U)).

This follows by adding and subtracting the conditional true log probability in each cross entropy. Better information and better learning are separate sources of gain. A failed larger decoder can reflect acquisition or optimization error; a good decoder does not prove that deeper state carries additional information. The identity is a population statement, not an estimator of mutual information from current checkpoints.

The existing D8/D16 final-stage heads are not a nested (Z,U) comparison: their internal representations and training trajectories differ. A frozen incoming-content cut is also not the best predictor trained without U. It changes the receiving computation and may move it outside its training distribution. Report its loss delta as **predictive dependence under the specified intervention**, never as measured conditional information or a lower bound on achievable shallow loss.

## Diagnose the failure in the right order

1. Complete the reserved acquisition budgets. Compare D8 at 8M and 32M, and D8/D16 at 8M, on identical saved DEV targets and history. Report acquisition, parameter counts, selected writes, scored keys, teaching, optimizer and wall time together. Curie allocated audits and AWS uniform teaching are distinct members; comparisons between them do not isolate allocation alone.
2. Use already-admitted per-stage content cuts while retaining timing and banks. Check exact restoration before the factual replay. Combine their deltas with stage occupancy, gradient flow and long-history quality. A positive cut delta shows the trained prediction uses that input; zero delta can reflect redundancy, weak learning or an insensitive intervention.
3. If gradients reach all stages but content cuts are small and quality plateaus, examine whether useful signals are acquired, retained and exposed at the readout. If content cuts matter but quality is poor, prioritize prediction calibration and task-relevant learning credit. If memory is unused across long history, examine causal writes and retention before increasing dormant capacity.
4. Integrate a shared cut-query credit learner only after the measurements identify the missing information or learning path. Retain typed target identity, issued actor/state, horizon, optimizer response and version compatibility. Train by paid causal consequences; numerical BPTT comparisons can audit it without becoming its operational teacher.

These are conditional engineering decisions, not claims that current measurements identify a unique cause. The existing audit allocation, persistent numerical state, temporal races, private addresses, key/value separation and counterfactual alternatives stay intact during the acquisition ladder.

## Translate quality into a resource decision

For deployment workload N, define total charged cost C(N)=C_acquisition+N*C_prediction, in one common unit and boundary. If a candidate has higher acquisition cost by delta_A but lower per-prediction cost by delta_I>0, its cost crossover is N>=delta_A/delta_I. State separately the quality of both members under the same scoring protocol. Lower total cost at worse quality is an efficiency point; cost crossover alone is not a quality win. CPU time, arithmetic, traffic and physical energy require separate measurements.

Resource interruptions belong in acquisition cost. With an unknown interruption tail, observed process intervals give a lower bound on total wall work. Do not turn that bound into an exact efficiency ratio. Successful-trajectory training time describes the retained optimization path, not all resources consumed.

## Evidence that changes investor confidence

The near-term deliverable is a reproducible table of completed deep acquisitions, source-bound checkpoints, matched DEV scoring and full work scope. Next comes development-selected language quality against the pinned public reference under its exact tokenizer, data and history protocol. Public validation stays sealed until development predicts a competitive result. Independent replication follows a selected member, rather than multiplying an unselected configuration.

Existing event-domain wins support a focused product proof while language tests the larger substrate. A design partner should supply a declared stream, target quality, latency and adaptation budget; its measured outcome would connect benchmark mechanisms to customer value. This is a future commercial test, not current customer traction. No outreach or fundraising publication is authorized by this note.

Related derivations: [acquisition response](DEPTH_ACQUISITION_RESPONSE_SPECTRUM.md), [deep temporal credit](DEEP_TEMPORAL_CREDIT_CLOSURE.md), [connected credit risk](CROSS_AUDITED_CONNECTED_DEPTH_RISK.md), and [credit representation](LEARNED_DEPTH_CREDIT_REPRESENTATIONS.md).
