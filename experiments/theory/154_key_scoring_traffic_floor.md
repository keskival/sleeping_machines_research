# 154 — Per-character traffic of the trained native models; the remaining floor (4 October 2026)

**Revision note (same day):** the first version of this note claimed a key-scoring traffic floor (76% of weights read
per character). That was an accounting error in hardware cost model v1: it charged every slot's P×P `key_read` on
every step. The existing winner-only evaluator (`sleeping_machines/sparse_inference.py`, §414; contract
`tests/test_sparse_inference.py`) caches `key + key_read · m` per slot. This is exact because scores read the stored
memory, which changes only when that slot wins. The corrected model's p96 count, 642K MACs ≈ 1.28 MFLOPs per character,
agrees with the recorded 1.3 MF/position winner-only trace. The erroneous proposal (low-rank/cached keys) is withdrawn:
cached keys already exist.

## Corrected result (results/diagnostics/hardware_cost_model_20261004.json, int8, batch 1)

| Model | Params | Weights read / char | Fraction | Bytes / char |
|---|---|---|---|---|
| native p64/d4/H2/U2 | 0.42M | 289K | 68% | 291K |
| native p64/d4/H2/U8 | 1.22M | 289K | 24% | 295K |
| native p64/d4/H2/U32 | 4.43M | 289K | 7% | 307K |
| native p96/d4/H2/U2 (1.888 bpc) | 0.94M | 642K | 68% | 646K |
| LSTM-512 (1.799) | 1.20M | 1.20M | 100% | 1.20M |
| Transformer-256×4 (1.908) | 3.24M | 3.24M | 100% | 3.76M (incl. 524K KV) |

Capacity beyond activity holds at the traffic level: 10.5× more parameters (U2 → U32) for 5% more bytes per character.
Quality at U32 is not established (pool 4/8 fits exist at 10M/90M; U32 does not).

## Remaining floor and a possible direction (proposal; not run)

Per-character traffic is now dominated by the shared dense maps: `channel_mix` (D·(HP)²), `queries` (D·H·P·HP) and
`source_gate` ((HP)²). At p64/H2 they are about 2/3 of the 289K. These are dense mixing between heads. Reducing them
would use the same addressed principle one level up: route each event to a subset of heads, or to sparse/low-rank
channel mixing. This changes information paths, so per AGENTS.md it needs a stated failure (none yet: this is a
resource floor, not a quality failure), contracts and an integrated 10M comparison before any long run. Priority stays
below P0/P1.
