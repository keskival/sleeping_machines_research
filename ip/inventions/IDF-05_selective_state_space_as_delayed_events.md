# IDF-05 — Executing selective state-space models as content-delayed events over decaying memory

CONFIDENTIAL · Invention disclosure for counsel · Status: **construction unpublished** (7 Oct 04:05 UTC `cfb11282`:
`experiments/theory/MAMBA_CONTAINMENT_20261007.md`, `mamba_containment_check.py`). **Motivation public:** theory note 08
(28 Sep 2026) states that event identity gives selectivity "Mamba-style models have to learn". Inventive-step risk in
EP against that note; US open regardless (note 08 within the grace year until 28 Sep 2027).

## Content

Selective state-space models (Mamba-1/2) update h_t = exp(Δ_t A) h_{t−1} + Δ_t B_t x_t, y_t = C_t h_t, with an
input-dependent step Δ_t. The construction maps each token to an event delivered after a learned, content-dependent
delay Δ_t; memories decay over the elapsed delay (exp(Δ_t A)); writes are content-addressed (B_t) and reads
content-addressed (C_t). It reproduces Mamba-1 and Mamba-2 selective scans to 4e−13 (numerical check in the repository),
and complex rotations extend it to oscillatory memories.

A mathematical equivalence is not patentable as such. Candidate claim subject matter is the **implementation**:

1. executing a selective state-space layer on event-driven or clockless hardware in which the input-dependent step is
   realized as a physical or scheduled delay of the event carrying the token, and state decay is realized by the elapsed
   time (leakage) of a memory element, so that no clocked per-step update of idle state is needed;
2. a scheduler that groups tokens by realized delay to update only the memories they address (sparse addressed updates),
   with counted reductions in memory traffic;
3. conversion of a trained selective SSM's parameters into delay, decay and address parameters of such a device.

Technical effect to be measured before filing claims 2–3: memory traffic and energy on a reference event-driven
implementation (currently modelled, not measured). Without a measured implementation effect, EP prospects are weak;
consider US-only provisional coverage and file EP when hardware evidence exists (the 12-month priority window allows
adding measured embodiments in a later application only if they are new matter filed then; counsel to advise).
