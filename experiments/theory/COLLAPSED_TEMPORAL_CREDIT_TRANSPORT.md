# Delayed credit can travel by elapsed time

13 October 2026; B1/B10 integration of the AWS state-interface result with Curie temporal eligibility.

The concrete problem is the cost of carrying delayed state credit through many event boundaries. Preserve native persistent clocks, nonlinear writes/readouts and local eligibility. Replace only an explicit dense direct-state transition or its repeated replay with the existing clock algebra; do not substitute a dense recurrent critic or discard alternative-route learning.

For fixed clock version, each complex mode propagates by

    a(dt) = exp((-rate + i frequency) dt).
    a(dt_k) ... a(dt_1) = a(sum dt).

The real two-coordinate adjoint therefore sends (c_r,c_i) to

    (cr*c_r + ci*c_i, -ci*c_r + cr*c_i).

No neuronal history tape is required for this direct transport. An endpoint cotangent, the clock version and elapsed time suffice. Local write eligibility pulls the transported cotangent into producer parameters or inputs. Accumulated losses require a sum of endpoint demands, not replacing many demands with one terminal demand. If parameters change, split at version boundaries; if incoming writes depend on the transported state, their Jacobians add paths and invalidate a clock-only collapse. In stacked layers, downstream writes depend on upstream outputs: the complete cut must retain those cross-layer effects. Hard alternatives and surviving parameter/Adam effects are separate paths too.

For n complex modes the contraction is 4n multiplications and 2n additions. Clock construction additionally pays rate exponentiation, elapsed products, decay exponentiation, sine/cosine and coefficient products. A dense 2n-square multiply convention costs 8n² FLOPs. This comparison concerns contraction alone, not total fitting work, traffic or hardware energy. Existing diagonal eligibility code already exploits local rotations; this reusable transport is not an extra saving against that implementation without an integrated ledger.

`sleeping_machines/learning/temporal_credit_transport.py` provides the reusable adjoint. The native TemporalMemoryLayer contract uses width32, 16 complex modes and irregular histories of 2/8/32/128 events. Exact complete terminal-loss input derivatives match a paid local endpoint VJP, collapsed temporal transport and paid first-write VJP. Sequential and collapsed transport also match. Result: `experiments/results/credit/curie_b10_temporal_transport_20261013_v1_contract.json`. This extends the two-event AWS rank witness to a reusable long-history operator; it does not train a prospective cotangent estimator.

The next integration is a native critic-issued state cotangent plus this transport and Curie local eligibility, with versions/read sets enforced by local banks. Compare against saved exact future-loss proposals, retaining their actual Adam/global clipping semantics initially. Charge discovery, critic acquisition, endpoint derivatives, alternate worlds and packet dependencies in the complete budget. Learned cotangent error is multiplied by the local eligibility norm and decays along a stable clock mode; write/depth/learner paths can amplify it and must be included in the matched-quality test.
