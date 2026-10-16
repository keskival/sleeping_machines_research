# R1 scaling supremacy: what proves it, what we have, what it takes

16 October 2026 (Curie lead; user: "do whatever it takes to achieve scaling supremacy and prove it").

## Claim to prove

On the same corpus, tokenizer and validation protocol, the native family's **loss-versus-training-compute frontier lies
below the Transformer frontier across a range of budgets, and the gap widens with budget** (a better compute exponent).
Single points against one reference checkpoint are efficiency points (WIN_CRITERIA.md), not scaling supremacy.

Protocol: FineWeb GPT-2 tokens; validation = the pinned modded-NanoGPT record's targets 1..10,485,760 of
`fineweb_val_000000` in disjoint 1024-token windows (`analysis/public_protocol_score_keyed.py`; contract: batched windows
equal the driver's flagship score to 4.5e-8, `curie_public_protocol_scorer_contract_L8_4M_20261016_v1.json`). Development
slices start at 10,485,760 or later, so the public range stays sealed until a selected model is scored once. Compute =
forward+backward matmul FLOPs per training token × presentations (`curie_plan_b_training_flops_20261016_v1.json`).

## Measured position

| | Training FLOPs/token | Budget reached | Loss |
| --- | --- | --- | --- |
| Plan-B L8 d64 (ours) | 25.8M | 4M tokens one pass: 1.1e14 | flagship 5.841 (development slice) |
| Keyed L2 d64 (ours) | ~21M | 4M tokens × 3 epochs: ~2.6e14 | flagship 5.601 (development slice) |
| GPT-2 124M record (published) | 800M (6N + 6·L·T·d) | step 125, 65.5M tokens: 5.2e16 | public 5.9598 |
| same | | step 250, 131M tokens: 1.05e17 | public 5.0586 |

**Published frontier at matched tokens** (`results/public_reference/published_record_curves_20261016_v1.json`): the current track-1 record (#89, July 2026) logs public validation 4.489 at 32.8M training tokens, 4.145 at 70.8M, 3.737 at 136M and 3.278 at 366M. Our strongest one-pass member scores 5.841 at 4M tokens (development slice) and the 16M L8 run is near 5.6 prequential at 6M: **at matched tokens we trail the modern record by more than one nat**. The gap is capacity (about 124M parameters and ~1e9 FLOPs/token against 3.8M and 2.6e7) plus two years of recipe engineering (Muon, multi-token prediction, hashed bigram embeddings, staged batch and context schedules). Closing it at matched tokens needs 10–30x wider native members; at our CPU throughput that is about ten days per 33M tokens, so the FineWeb record frontier is a hardware-bound target (user: no new hardware yet).

Depth is cheap in arithmetic: L2 → L8 adds 23% FLOPs per token (21.1M → 25.8M; the 50,257-way output dominates) for
−0.023 nats per fresh token at 4M, a gap that widens with data (theory note 5, P5c/P5d).

## Why existing evidence cannot yet prove the claim

1. **No overlap in budget with the published curve.** The record's first logged point is 5.2e16 FLOPs; our runs span
   1e14–2e15. Reaching the record's own budget takes ~2B tokens at L8 d64, or ~430M tokens at d256 — weeks of CPU.
2. **One reference size.** The record is a single 124M model tuned for its final score with 524k-token updates; its early
   points are update-limited warm-up checkpoints. A small Transformer would be far stronger at our budgets. Beating the
   record's early checkpoints is a scoped efficiency point against a pinned published run, not a family verdict.

## CPU-feasible published leaderboards (primary now; user: train our own models against published benchmarks)

Word-level Penn Treebank (929k training tokens, 10k vocabulary) and WikiText-2 (2.09M training tokens, 33k vocabulary)
carry long published tables of models trained only on their training split, with leaders at about 24–33M parameters and
a recognised online-adaptation category (dynamic evaluation: weights adapt on already-scored test text). Our keyed
token-slot read is a learned cache pointer, our temporal memory persists across the article, and causal online weight
adaptation is a primary capability (AGENTS.md); these are the mechanisms behind the best published results there
(continuous cache pointer, dynamic evaluation), but integrated and trained end to end. Budgets fit CPU: one PTB epoch at
~1k tokens/s is ~15 minutes, so hundreds of epochs fit AWS slots in parallel. Published references are verified from
the papers before the battle opens; only training-split models enter the main table, pretrained models (GPT-2
zero-shot) are a separate column.

Verified targets (Melis et al., ICLR 2020, Table 1; local copy `results/public_reference/melis2020_mogrifier_iclr.pdf`),
test perplexity, static / with dynamic evaluation:

| Dataset | Model | Params | Static test | Dyneval test |
| --- | --- | --- | --- | --- |
| PTB | Mogrifier (d2, MC) | 24M | 50.1 | 44.8 |
| PTB | AWD-LSTM (d3, MoS15) | 22M | 54.4 | 47.7 |
| PTB | Transformer-XL | 24M | 54.5 | — |
| WikiText-2 | Mogrifier (d2, MoS2, MC) | 35M | 55.1 | 38.6 |
| WikiText-2 | FRAGE (d3, MoS15) | 35M | 58.0 | 39.1 |
| WikiText-2 | AWD-LSTM (d3, MoS15) | 35M | 61.2 | 40.7 |

Our online-adaptation arm (weights adapt only on already-scored text, AGENTS.md) is compared with the dynamic-evaluation
column; the frozen arm with the static column.

## Program (in order)

1. **Our scaling laws, now (Curie + AWS, authorized).** One-pass plan-B ladder over data (4M/16M/64M), depth (2/4/8/16)
   and width (64/128): fit L(C) = E + A·C^−α per depth and width; supremacy-relevant result is α rising with depth at fixed
   width (compositional depth scaling) and the compute-optimal depth growing with budget. Curie: 16M L8 (running), 16M L2,
   L4 resume. AWS (ordered): 16M L4/L16, d128 L8, 64M L8.
2. **First exact public-protocol point (one sealed scoring).** Score the development-selected strongest member on the
   public protocol: an efficiency point against the record (compute to reach the record's 65.5M-token loss), reported
   with the scope above.
3. **Matched-protocol Transformer reference ladder (authorized 16 October as a bounded secondary cross-check; published
   benchmarks come first).** Small decoder-only Transformers at d 64–256, 2–8 layers, on the same shards, tokenizer, one-pass schedule
   and public protocol, at the same budgets 1e14–3e15 FLOPs. This is the only route to a family-level frontier comparison
   at budgets we can reach; without it every claim stays an efficiency point against a pinned record.
4. **Scale to overlap the published curve (needs provisioning).** A GPU or large-CPU worker to take the strongest native
   member to 1e16–1e17 FLOPs (0.4–4B tokens), where the record's logged points live; then the comparison is against the
   published 124M curve directly at matched compute. Prepared 16 October: exact GPU driver (`experiments/r1_gpu_ladder.py`;
   loss and every gradient equal DeepKeyedLM to 3e-6), ladder spec of flagship and memory-only members over 1e16–1e18 FLOPs
   with IsoFLOP arms, depth probe and per-layer scaling diagnostics, and a Hetzner GEX131 runbook
   ([runbooks/GPU_SCALING_LADDER.md](runbooks/GPU_SCALING_LADDER.md)); about 40–80 GPU-hours, awaiting a purchase order.

Decision rule: supremacy is claimed only when steps 1 and 3 (or 1 and 4) give two frontiers on one protocol with ours
below across at least one decade of compute and the gap not shrinking; otherwise report the measured gap and its trend.
