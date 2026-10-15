# R1 scaling bottleneck: the language member does not use history beyond ~64 tokens

15 October 2026, Curie. Battle R1. Decision document and admission declaration for the addressed episodic memory pair.

## Diagnosis from completed evidence

| Evidence | Result | Implication |
|---|---|---|
| Frozen-weight state resets (aws_acquired_*history_transfer_20261013, 3 seeds) | Resetting all stored state every 64 tokens costs 0.004–0.008 nats at 8M; every 16 tokens 0.039–0.045; every 4 tokens 0.16–0.20 | Predictions use about 16–64 tokens of history. The 512k→8M acquisition gain (0.66–0.69 nats) is short-context statistics; long-history value did not grow with 16x data |
| Causal in-window copy/induction cache on the frozen 5.978388 learner (`results/diagnostics/curie_copy_memory_opportunity_20261015_v1.json`, `analysis/copy_memory_opportunity.py`) | Held-out (cross-fitted on alternating windows) **5.978388 → 5.684537, gain 0.294 nats**; ranks 2k–10k +0.59, 10k–30k +0.84, 30k+ +1.19; 51% of targets already occurred in their window | The largest available gain is long-range addressed recall, 3.4x the 0.087 nats bought by the last 4M training targets |
| Depth (AWS/Curie D4/D8/D16) | D8/8M 6.275 vs D2 6.160 at 3.4x work; grown D4 −0.030; erasing grown-stage memory changes NLL by 0.0002 | Added stages add about 24k weights each and their memory is unused; depth is not where capacity or memory lives in this member |
| Width/pool at 2M | pool 8 7.071 vs pool 4 7.052; payload 32 7.264 vs 24 7.252 | More receivers or payload without a path that credits long-range writes does not help |
| Readout width (8M) | richer readout 6.099 vs 6.160 | Output capacity binds on rare tokens; complementary to memory |
| Memory boundary policy (2M) | EOS-reset training 5.978 beats persistent memory 5.996 | Raw persistence carries stale state because nothing learned to address it |

Cause: the state-gradient window is 16 tokens. A write is credited only through backpropagation within that window, so the learner cannot learn to store information whose value appears hundreds of tokens later, and its 16 receivers per layer are overwritten within tens of tokens. Persistent numerical state exists; credit for long-range use does not.

## Construction: addressed episodic memory with read-time credit

`sleeping_machines/episodic_pointer_memory.py`. Every token event appends (key source = learner feature after the token, value = the following token) to a persistent per-lane store (capacity 1024 entries; capacity beyond activity). A query from the current feature scores keys through the CURRENT key map, adds a learned age clock (−softplus(ρ)·log(1+age)) and a learned document-boundary bias, keeps the top 32 candidates and races them against a null route. A learned gate mixes the pointer distribution into the native readout; the result is a normalized distribution.

Learning rule: an entry is credited when it is read. Gradients from a read reach the query and key maps, age clock, boundary bias, null route and gate for entries of any age, with no backpropagation through intervening state evolution. Old entries are stored detached; entries inside the current chunk keep their path to the writer features, so the core learns to write retrievable features within its credit window. This separates persistent memory from gradient truncation, as AGENTS requires, and turns the persistent-memory question into a learned addressing problem: the boundary bias is the learned EOS gate proposed after the memory-policy result.

Mechanisms retained: native temporal core with races, sparse addressed writes, key/value separation, counterfactual route credit (unchanged, on native readout terms), persistent numeric state (core corpus-EOS policy from the winning parent), calibrated local clipping. Added: sparse top-k addressed episodic read, age clock, boundary bias, read-time credit. Removed: none.

Contracts (`experiments/episodic_memory_contracts.py`, `results/diagnostics/curie_episodic_memory_contracts_20261015_v1.json`), all pass: normalization (max error 2.4e-7), target consistency, causality (prefix change exactly 0), chunk-split invariance (4.8e-7), read-time credit to detached old entries (key/query/clock gradients nonzero, zero gradient into old features), boundary bias isolation, capacity bound, and **bitwise equivalence to the published continuation step with the gate closed** (loss, weights and state, 3 updates).

Work per target (lanes 8, chunk 64, store 1024, key dim 32): about 3.3M key-map MACs per lane-chunk plus 2.2M score MACs, about 87k MACs per target before key caching; 32 selected reads per target. Recorded exactly in `memory_work`.

## Admission declaration (R1 pair, Curie, after D16 quality)

- Task: properly tokenized FineWeb GPT-2 language; non-stationary stream. Chronological fresh-TRAIN continuation with TRAIN losses scored before each update (prequential); persistent state carried across chunks.
- Strongest fighter: the 5.978388 learner (shared lanes, corpus-EOS training boundary, calibrated local clipping, 8M+6M acquisition), the strongest completed native language model.
- Arms: `episodic` (store added) and `none` (plain data-scaling continuation, the exact published `raw_step`). Same restored learner, optimizer moments, caps, numeric state, private RNG, teacher, and the identical next 2,097,152-target stream (start 14,942,224). Each: 64-update smoke, then resume to 4096 updates.
- State/history: core state persists across chunks with the parent's corpus-EOS reset; the episodic store persists across chunks and documents (learned boundary bias), FIFO capacity 1024. Scoring: recorded 8192 DEV targets, disjoint 1024-token windows, store and state reset per window (store never sees tokens outside the window). Native-readout score of the episodic arm is also recorded.
- Restrictions: capacity 1024 and top-32 for CPU cost on the shared host; single seed. No public validation.
- Resources: one Curie training slot, one thread, RSS about 0.7 GB measured in contracts, timeout from measured smoke rate (2x + 1 h, ≤1 day).
- Decision: adopt the store for R1 continuations if it beats the identical-stream control by ≥0.03 nats; then scale store capacity and acquisition with it. Otherwise diagnose read/gate/credit (pointer hit rate, gate, clock) before any larger fit. The control arm also measures the data-scaling gain per 2M at this point of the curve.
