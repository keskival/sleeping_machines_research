# 159 — Event graphs: state freshness, containment of the heuristics, and what should win (curie, 9 Oct 2026)

Sections §§457–461. Context: battle B5 (Temporal Graph Benchmark), [tgb/B5_TGB.md](../tgb/B5_TGB.md). Written before the
race_link v4 and the tgbn-trade sealed results; the predictions in §461 are stated in advance and graded below them.

## §457 Setting

A temporal interaction graph is a marked point process whose marks are (source, destination) pairs. Link prediction
asks, at each event time t, for the distribution of the destination given the source and the history H_{<t}. Node
affinity asks for a node's share of next-period interactions over destinations. Both are questions about the race of
destination clocks for a given source: P(c | s, t) = h_{s,c}(t) / Σ_c' h_{s,c'}(t), the first-finisher law of the race
(note 158, §449) with hazards h_{s,c} = exp(score(s, c, H_{<t})).

## §458 State freshness (per-event exactness)

**Definition.** A predictor has *per-event causal state* if its prediction at time t is a function of every event with
time < t. A *batch-stale* predictor with batch size B uses only events before the start of the query's batch.

**Proposition 1 (the loss from staleness is the information in the hidden prefix).** For a scoring rule that is proper
for the conditional law of the destination given the history, the best batch-stale predictor's expected log-score is
lower than the best per-event predictor's by the conditional mutual information I(D_i ; E_{batch, <i} | H_{<batch}, S_i)
between the query's destination and the earlier events of its batch. *Proof sketch:* the per-event predictor conditions
on a finer σ-algebra; the gap of the log-score between the two Bayes predictors is exactly that conditional mutual
information (chain rule of mutual information). MRR is not a proper score, but it is monotone in the rank of the true
destination, which the hidden prefix changes when it contains the same pair or the source's newest destination.

**Measured size on tgbl-wiki (batch 200, validation / test):** 5.4% / 5.4% of queries repeat a pair visible only within
their batch; the next destination equals the source's newest destination in 80.3% / 77.9% of queries per event against
70.0% / 66.4% batch-stale. The information hidden by staleness is large because the process is bursty: a source's
events cluster within minutes, i.e. within one batch.

**Consequence for the family.** A race network's state update is O(state touched) per event (one addressed pair slot, one
destination slot, one source slot, one row and column of co-visitation), so per-event exactness costs no batching
penalty in inference. Memory-based temporal GNNs that update node memories once per batch (TGN lineage) carry exactly
this loss by construction ("memory staleness"); neighbour-sampling models (DyGFormer, TPNet) avoid it by re-reading raw
history at query time, paying a per-query cost that grows with the sampled neighbourhood.

## §459 Containment of the affinity heuristics

Let P_k(n) be node n's normalised affinity vector of the k-th previous period, and E_τ(n) the row-normalised
exponentially decayed affinity with time constant τ periods.

**Proposition 2.** The race_affinity feature map contains persistent forecast (rank by P_1), the simple moving average
of window w ≤ 4 (rank by mean_{k≤w} P_k, a linear function of the features) and the exponentially decayed affinity at
τ ∈ {1, 2, 4, 8}; a readout that is monotone in one of these reproduces the corresponding heuristic's ranking exactly.
NAVIS (arXiv 2510.06940) outputs a data-dependent *convex* combination of the previous affinity vector and an
EMA-like state (gates in [0, 1]); every such output lies in the convex hull of past affinity vectors.

**Proposition 3 (what convex combinations cannot do).** A predictor whose output lies in the convex hull of past
affinity vectors assigns each destination c a share of at most max_k P_k(c). If c's share is rising, max_k P_k(c) =
P_1(c) < current share, so c is under-predicted every period; a ranking error follows whenever a rising c should
overtake a destination c' with falling share whose past values exceed c's. A readout of (P_1, log P_1/P_2, global
growth) can extrapolate past P_1 and avoid it. *Proof:* each coordinate of a convex combination of vectors is bounded
by the maximum of that coordinate over the vectors.

## §460 Where the advantage should come from

- tgbn-trade: bilateral trade has persistent levels plus trends (rising partners) and reciprocity. Our readout has
  trend and reverse-flow features; Proposition 3 says convex predictors cannot use trends.
- tgbl-wiki: the signal is the source's own recent history (85% repeat pairs). Per-event exactness (§458) and ordinal
  recency carry it; collaborative identity and transitions serve the 10–15% new pairs.

## §461 Predictions (stated before the results)

1. **Freshness:** race_link v4 (per-event state) raises tgbl-wiki validation MRR over v2 (0.776, batch-stale) by at
   least 0.03, mostly through the queries whose pair or newest destination lay in the hidden batch prefix.
2. **Containment:** on tgbn-trade, the seed-0 development fit beats persistent forecast on every validation year (already
   observed: 4 of 4) and the sealed seeds beat NAVIS's test 0.863 on average.
3. **Trend use (ablation, after the sealed run; `race_affinity.py --drop trend` / `--drop reverse`):** removing every
   history-shape feature (lags 2–4, decayed affinities, pair/global/volume growth and their logs; level P_1, reverse
   flow, global share and context kept) loses most of the validation gain over persistent forecast; removing the reverse
   flow loses less.

## Grades

(Filled in as results arrive; a refuted prediction is stated beside the original claim with the reason.)

- **Prediction 1 — confirmed** (9 Oct 13:45, race_link v4 first epoch, validation): 0.8515 vs v2's 0.776 (+0.075, ≥ 0.03
  predicted). Repeat-pair share rises from 85.4% to 90.8% (the hidden in-batch repeats become visible) and repeat-pair MRR
  from 0.896 to 0.931 (fresher newest destination). v4 also adds identities (d = 16); the d = 0 run separates the two.
- **Observation (not pre-registered), 9 Oct 15:42 — inductive state vs transductive identity.** With identical state
  features, learned per-node identities (d = 16; 171,627 parameters) peak at validation 0.8515 after one epoch and then
  decline to 0.833 by epoch 4, while the identity-free readout (7,995 parameters) stays at 0.851–0.852 for six epochs.
  The state features are functions of elapsed time and observed history, so they transfer to a later period unchanged;
  identity tables fit the training period's node behaviour and drift. On interaction streams the family's inductive
  bias (relevance carried by time-decayed state, §458) is the generalising part; identity capacity is not needed for the
  validation lead over TPNet (0.842).

