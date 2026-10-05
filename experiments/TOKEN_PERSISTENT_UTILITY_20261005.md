# Recurrent message carries useful history; addressed memory needs diagnosis

Frozen selected fixed-batch2K models; seeds6/7, credit16/64, GPT-2 FineWeb,
1,016 development targets. No fitting or public validation. Before each token,
erase addressed memory plus arrival/seen markers, the recurrent event message,
or both, keeping global position and identical route-noise streams. Intact
one-token partition reproduces the original chunked selected NLL within2e-6;
all interventions consume exactly the same RNG stream.

| Seed | Credit | Intact NLL | Memory erasure minus intact | Message erasure minus intact | Both erasure minus intact |
|---|---:|---:|---:|---:|---:|
|6|16|8.812526|−0.000606|0.067014|0.065513|
|6|64|8.819470|0.001388|0.046181|0.043539|
|7|16|8.813838|−0.000072|0.062753|0.061785|
|7|64|8.812853|−0.001080|0.067343|0.066497|

The recurrent message has useful history dependence in all four checkpoints.
It can itself encode earlier events; this is not a previous-token-only finding.
Addressed memory erasure has a small mixed-sign effect in these selected fits.
These are frozen out-of-distribution interventions, not retrained ablations;
component effects interact and cannot be added as independent attribution.
The earlier constant-TRAIN-mean intervention asks a different question: its
constant features can preserve useful marginal readout calibration.

## Next diagnostic and design criterion

Keep both bounded AWS8K comparisons already queued. On their selected models,
repeat the interventions before promotion. Diagnose trained memory-to-key and
memory-to-value sensitivity, transport age/decay and selected-write retention.
Separate useful recurrent-message history from addressed capacity; occupancy
and route entropy cannot establish predictive memory utilization.

Before changing transport or memory access, require a numerical read/write
sensitivity contract and a small integrated fit. Retain clocks, sparse writes,
separate keys/values, deep recurrent messages and actual counterfactual credit.
The failure being addressed is negligible selected-model development benefit
from the current addressed memory bank, not a limitation of the model family.
Larger capacity is justified by improved contextual prediction and resource
accounting, not by a count of populated slots.

Evidence: `results/diagnostics/curie_token_persistent_utility_audit_20261005_v1.json`.
Producer and uniquely named guarded queue are committed beside the evidence;
RSS cap700000KiB, available-memory floor8192MiB, timeout180s, one CPU thread.
