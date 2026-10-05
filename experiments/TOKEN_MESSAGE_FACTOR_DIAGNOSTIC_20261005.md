# Factor the message-erasure diagnostic before redesigning transport

P24selected64Kseed6message erasure costs0.022623NLL;seed7erasure improves
0.032675NLL. Context and addressed-memory effects are positive in both seeds.
Investigate the bundled intervention, preserving both original measurements.

In sparse_counterfactual_episodes.token_features, read_time=max(input arrival,
previous message arrivals) when has_ctx is true; it becomes input arrival when
false. Likewise has_ctx selects layer_norm(input + gated transported context)
versus the unnormalized lexical input. The existing erasure zeros ctx_vals,
ctx_arr and has_ctx before every event, changing information, normalization
branch and read-clock policy together. Its scores are valid for that full
intervention, not an isolated payload-information measure.

The new token_message_factor_audit.py tests both selected P24seeds with intact,
payload-only and full-message interventions. Payload-only zeros ctx_vals but
retains arrival metadata/presence, preserving the normalization branch and
incoming metadata at the intervened event. Future choices/timestamps still
change; subtracting deltas does not give a unique causal decomposition.

Require exact selected-checkpoint intact NLL, token denominator, matching route
RNG and model source pins; retain all results. The diagnostic does not retrain
a carrier-only model, remove any core mechanism, alter source-pinned fits, or
supply a quality gain. It determines which proposed learning/transport repair
a later integrated comparison should test. No repair selected in advance.

Queue curie_token_message_factors_64k_20261005_v1.txt uses run_safe, one thread,
300stimeout,1.2GBRSS/5GBVMS/8GiBfloor. Bounded selected-checkpoint evaluations
need no65KTRAINfeature accumulation;300s allows all six2040-target evaluations
and two model loads. Waiting coordinator follows both the live work replay
and reserved256Kfit, preserving their priority. Numerical parity is pending.
