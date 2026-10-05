# FineWeb MoE reference discovery — 5 October 2026

Question: a published MoE with lower complete training or inference FLOPs and
equivalent/better quality than our selected2025-01-26dense record?

**No verified qualifying comparison found in this search.** A relevant same-
distribution candidate supplies reported better quality, not an audited FLOP
win against the selected record. No baseline training launched.

## Closest candidate

Repository https://github.com/cat-state/modded-nanogpt-moe, pinned revision
b7d230616d020e031b2ebf30b6cb8018bd94cb1a. Sources downloaded read-only into
cat_state_moe_20261005 with SHA256manifest; no model execution.
README reports dense NLL3.276 after2.4Btokens and four-expert top1 MoE NLL3.218,
with threshold3.28 reached at4000steps versus4500dense. Same active parameter
count, larger total capacity. This is an11%update reduction against its paired
older dense recipe, not11%less complete work than our selected reference.

The source downloader uses kjj0/fineweb10B-gpt2 and the first validation shard.
Current defaults use512sequences/update and1024tokens/sequence:4000updates
would present2,097,152,000targets,3.013times our selected695,992,320. This is a
conditional calculation from present defaults, not a source-bound reconstruction
of the README result. A lower update count is not a lower FLOP count.

Current source has16hash-routed experts, whereas the reported README result
uses4top1experts. Do not bind that score to current source settings. Current
validation uses1024-token causal sequences over10,485,760targets, whereas our
selected reference uses40reset sequences of262144targets with document-aware
sliding attention. Matching total validation token counts does not equate
sequence boundaries/context. Current model vocab is50304(padded); our selected
record declares50257. Historic run provenance needs its own source/log.

Inference uses one selected expert of dense-MLP size, shared attention and
output projection, plus routing/aggregation; it does not establish lower
inference FLOPs than our different dense architecture. No complete training/
optimizer/inference work record for the reported score was found.

Primary explanatory companion: https://www.cerebras.ai/blog/moe-guide-debug
(August19,2025), describes the same GPT-2/FineWeb MoE construction, routing
issues and pedagogical implementation overhead.

## Decision

Retain3.218 as a reported exploratory quality target with its protocol, not an
eligible matched-work win/reference under our current endpoint. Preserve the
selected dense record. A qualifying MoE requires a source-bound completed run,
exact tokenizer/data/scored-target/history record and a consistent FLOP ledger.
Do not equate same active parameters with equal complete training work, or
FineWeb-Edu results/different tokenizers with FineWeb GPT-2 NLL.
