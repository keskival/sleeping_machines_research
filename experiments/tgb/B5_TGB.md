# B5 — Temporal Graph Benchmark: dynamic link prediction (owner: curie)

Admitted 9 October 2026 under the founder's direction to prepare wins on new event domains (asynchronous event data,
mixed dense and asynchronous data). First dataset: **tgbl-wiki-v2**; next: tgbl-review-v2.

## Frozen protocol

- Data, splits, negatives and scorer: the official `py-tgb` package (2.3.0, `tgb.linkproppred`): `LinkPropPredDataset
  ('tgbl-wiki')` with v2 edge list and v2 validation/test negative sets; `Evaluator('tgbl-wiki')`, metric **MRR** with
  TGB's tie convention (mean of optimistic and pessimistic rank). Our drivers compute the identical reciprocal rank and
  assert agreement with the official Evaluator on the first 50 queries of every evaluated split.
- Streaming convention (TGB standard, as EdgeBank and memory models): a batch of 200 queries is scored with state built
  from events strictly before the batch; the batch's positive edges then enter the state. No information from a query's
  own edge or later edges is used to score it.
- Data facts (checked 9 Oct): 157,474 events, 9,227 nodes (8,227 users → 1,000 pages), 30.9 days; 110,232 / 23,621 /
  23,621 train/val/test; 18,257 distinct user–page pairs; 71% of test edges repeat a pair observed before the test window,
  16% of test edges come from users not seen in training.
- Development uses validation only. **Test is scored once per model** (`--score-test`), on the configuration selected on
  validation, with three seeds.
- **Win rule:** mean test MRR over seeds 0–2 above the leaderboard leader (TPNet 0.827 ± 0.001) and every seed above
  0.827. Below the leader but above DyGFormer (0.798): reported as "second/third on the leaderboard", plainly.

## Reference table (leaderboard, tgb.complexdatalab.com, verified 9 Oct 2026; tgbl-wiki-v2 test MRR)

| Method | Test MRR |
|---|---|
| TPNet | 0.827 ± 0.001 |
| Heuristic (LocalGlobal) | 0.821 |
| HyperEvent | 0.810 |
| DyGFormer | 0.798 |
| NAT | 0.749 |

## Design rationale (why the family should win here)

- **What the leaders do.** TPNet keeps time-decayed temporal walk matrices (how strongly, and how recently, the source
  reaches the candidate in k steps) compressed by random projections, and reads them with an MLP. DyGFormer encodes
  neighbour co-occurrence between source and candidate. A training-free recency/popularity heuristic is second. The signal
  is *how recently and how often* the pair, its neighbourhood and the candidate were active: elapsed-time state.
- **Our mechanisms map one to one.** Addressed sparse state per (source, candidate) pair holding a bank of clocks
  (exponential decays from a minute to a month, plus daily and weekly rotations: complex-diagonal temporal memory with
  decay and rotation by elapsed time); exact sufficient statistics (count, last time); decayed destination popularity;
  page–page co-visitation kept incrementally (three-step evidence user→page←user→page, the decayed walk count TPNet
  approximates by projection, here exact for the bipartite graph). Keys are pair addresses; values are clock sums.
- **The objective is a race.** P(next destination = c) is the softmax over *all* 1,000 candidate destinations, i.e. the
  first-finisher distribution of destination clocks with hazards exp(score). Every unrealised destination receives
  credit at every step (counterfactual credit to the losers), and the training target is exactly the ranking question
  MRR asks. Leaders train with one or a few sampled negatives per positive.
- **Where we expect to gain over the leader:** (i) multi-timescale *and periodic* pair clocks (a user returning to a
  page on a daily rhythm is visible to rotation clocks, not to monotone decays); (ii) full-race training instead of
  sampled negatives; (iii) exact co-visitation instead of random projections; (iv) the 16% new-user queries rely on
  popularity momentum and co-visitation, both explicit.
- **Inference work.** Per query: one pass over the source's touched pairs (typically a handful), a 1,000-vector of
  popularity terms and a 1,000×1,000 co-visitation product (or its sparse equivalent), then a 36→64→64→1 MLP per
  candidate. Measured CPU latency and operation counts will be reported beside quality.

## Development plan (rounds; each followed by error analysis on validation)

1. **Protocol check** (`heuristics.py`): EdgeBank-∞ and a LocalGlobal decayed recency/popularity grid through the official
   loader, negatives and Evaluator. Expect the heuristic near the leaderboard's 0.821 (validation).
2. **race_link v1** (`race_link.py`): pair clock bank + popularity + co-visitation + source context, MLP readout,
   full-softmax race. Error analysis split by query type: repeat pair / new pair of a known user / new user.
3. **Round 2 by diagnosis:** candidates are learned clock rates (the bank's span), destination identity embeddings
   (transductive), the source's recent-sequence context (the last k destinations, keyed read), and learned interaction
   between the pair clocks and the source's own activity clock.
4. Seeds 0–2 of the selected configuration; test scored once; measured inference latency.

## Development log

- 9 Oct: data and official loader verified; protocol-check and race_link v1 smoke/dev jobs queued on curie
  (curie_b5_*_20261009T09*.txt) after the running B3 Stage 4 seed and the remaining curie chain.
