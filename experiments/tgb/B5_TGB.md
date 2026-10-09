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

## B5-N — node affinity prediction (tgbn-trade first; genre/reddit/token by size)

- **Leaderboard (tgb.complexdatalab.com, verified 9 Oct 2026), test NDCG@10:** tgbn-trade NAVIS **0.863** (val 0.860),
  Persistent Forecast 0.855 (val 0.860), Moving Average 0.823, TGNv2 0.735, DyGFormer 0.388. tgbn-genre NAVIS 0.528, Moving
  Average 0.509. tgbn-reddit NAVIS 0.569, Moving Average 0.559. tgbn-token NAVIS 0.513, Moving Average 0.508. Training-free
  heuristics beat every temporal GNN; NAVIS (ICLR 2026, arXiv 2510.06940) is a gated linear state-space model over past
  affinity vectors (1–5K parameters; outputs are convex combinations of past vectors and the input) trained with a
  LambdaLoss plus margin term; its own ablation gives 0.859 with cross-entropy and 0.857 without the global state.
- **Protocol (frozen):** official `NodePropPredDataset` and `Evaluator` (sklearn `ndcg_score`, k = 10, mean over the
  nodes of a label time), TGB's example loop (batches of 200 edges per split, one label pointer, a label time fires when
  a batch's last timestamp exceeds it; split score = mean over its fired label times). Replayed on tgbn-trade: validation
  fires label years 2009–2012, test 2013–2015. Label(ts) equals the normalised flows of the period starting at ts
  (checked: label 1992 = 1992 flows), so predictions use only edges with t < ts.
- **Why we should win:** heuristics win because affinity is persistent and decays with time; a learned mixture of
  multi-timescale decayed affinities contains persistent forecast and moving averages as special cases. NAVIS is limited
  to convex combinations (cannot extrapolate a rising partner) and to a last-vector global state. Our readout is
  nonlinear over the pair's lags, decays, growth, the reverse flow (bilateral trade), the destination's global share and
  growth and node context; the race over all destinations is trained against the realised affinity distribution.
- **Win rule:** mean test NDCG@10 over seeds 0–2 above NAVIS (0.863) with every seed above it; test scored once.
- Driver `experiments/tgb/race_affinity.py` (v1); class mapping verified for tgbn-trade (label index = node id).

## Development log

- 9 Oct: data and official loader verified; protocol-check and race_link v1 smoke/dev jobs queued on curie
  (curie_b5_*_20261009T09*.txt) after the running B3 Stage 4 seed and the remaining curie chain.
- 9 Oct: AWS takes the large link datasets (tgbl-review-v2, tgbl-coin-v2) with a sparse variant of race_link (HANDOFF).
- 9 Oct: B5-N opened (tgbn-trade); race_affinity v1 queued first at the seed-8 boundary (chain r19: trade, wiki
  heuristics, race_link smoke and dev, then the theory clock-count test and R1 confirmations).

## tgbl-review-v2 on AWS (AWS owner; development, validation only)

- **Data (verified 9 Oct, official loader):** archive sha256 `01336972…1dafb` from the py-tgb URL; 4,873,540 events,
  352,637 nodes, 3,413,837 / 730,784 / 728,919 train/val/test; **not bipartite** (298,589 nodes appear as both source and
  destination); 100 negatives per positive (v2). Files under `data/tgb_aws/root/tgbl_review/` (py-tgb prefixes its package
  directory to `root`, so drivers pass `--root ../../../data/tgb_aws/root`).
- **Pickle safety:** the negative-sample files are pickles; `pickletools` shows only `numpy.dtype`,
  `numpy.core.numeric._frombuffer` and `numpy.core.multiarray.scalar`. `experiments/tgb/safe_tgb.py` loads them with an
  allow-list unpickler and patches py-tgb's `load_pkl` (`safe_tgb.patch_tgb()`); use it on every host.
- **Leaderboard (verified 9 Oct):** GraphMixer **0.521 ± 0.015** test (0.428 val), CTAN 0.405, TNCN 0.377, TGAT 0.355,
  TGN 0.349 (0.313 val), NAT 0.341, DyGFormer 0.224.
- **Protocol check** (`experiments/tgb/heuristics_review.py`, validation only, official Evaluator agreement asserted on
  the first 50 queries): on the first 2,000 validation queries, 30-day decayed popularity alone reaches MRR 0.29 and
  pair-recency + popularity 0.296 (TGN validation 0.313). Full-validation run in progress.
- **Design consequence:** race_link v1's dense 1,000×1,000 co-visitation and full-destination softmax do not scale to
  352K nodes; the AWS variant scores the official candidate set (1 + 100) at evaluation and trains a sampled race
  (positive against sampled historical and random competitors), with pair state held sparsely.
- **Full-validation protocol check (9 Oct, 730,784 queries, 100 negatives each, Evaluator agreement asserted):**
  pair recency + 30-day decayed popularity **0.344** MRR (30-day popularity alone 0.340, 7-day 0.317, 365-day 0.303,
  EdgeBank 0.023). A training-free decayed state already exceeds TNCN (0.325), TGAT (0.324) and TGN (0.313) on validation;
  GraphMixer leads at 0.428. Reviews rarely repeat a pair (EdgeBank 0.02), so the signal is time-decayed destination
  popularity and collaborative identity, which race_link_review learns on top of the state.
- 9 Oct 11:03: **protocol reproduced.** `heuristics.py` on the official loader/negatives/Evaluator (999 negatives per
  query): EdgeBank-∞ test MRR **0.4947**, the value TGB publishes for EdgeBank-∞ on tgbl-wiki (0.495). Our crude
  decayed-recency grid (validation-selected 1-hour pair decay, no popularity) 0.756 val / 0.729 test; adding popularity
  on a linear scale hurt repeat pairs (scale mismatch), so heuristics are not pursued further. Query types on validation:
  repeat pair 85.7%, new pair of a known source 9.2%, new source 5.1%; recency alone ranks repeat pairs at 0.885 and new
  pairs at chance (0.002).
- 9 Oct 11:04: race_link v1 smoke (5K train events, 1K val queries, 2 epochs, 6,665 parameters): val MRR 0.763 (repeat
  0.874, new pair of known source 0.125, new source 0.047). Full v1 development fit running.
- 9 Oct 11:20: **race_link v2** written from that diagnosis (versioned file `race_link_v2.py`): ordinal recency rank of
  the candidate among the source's pages, the pair's share of the source's activity, last-destination flag, and decayed
  page-to-page transitions T[last(s), c] read by the source's last page (a sequential key read; scores new pairs).
  Feature contract checked on a hand example. Queued after v1 (curie_b5_racelink_v2_dev_s0_20261009T1120Z).
- 9 Oct: tgbn-trade v1 job failed at load (float-typed node ids used as indices; fixed); rerun queued as
  curie_b5_trade_affinity_dev_s0_v2_20261009T1105Z. Label-pointer replay confirmed: val label years 2009–2012, test
  2013–2015.
- 9 Oct 11:35: race_link v1 full development fit plateaus at **val MRR 0.773** (epochs 0–5: .772 → .773; repeat pairs .893,
  new pairs of known sources .07, new sources .06). The readout adds little beyond epoch 0: the features limit it.
- 9 Oct 11:45: **race_link v3** (versioned `race_link_v3.py`) = v2 + destination bias and source–destination identity
  inner product (collaborative evidence; AWS found identities decisive on tgbl-review). Queued after v2.
- 9 Oct 11:46: **tgbn-trade race_affinity v1 development fit (seed 0, validation only): val NDCG@10 0.8747** vs persistent
  forecast 0.8604 in our replay (leaderboard PF val 0.860: protocol fidelity) and NAVIS val 0.860; every validation label
  year above (2009–2012: .873 / .876 / .871 / .879); 2,107 parameters, 29 features, best epoch 95 of 200, 4 minutes on
  one CPU thread. First-round development result; test untouched. **Sealed protocol started** with the configuration
  frozen as run (cross-entropy race loss, defaults): seeds 0–2 with `--score-test`
  (curie_b5_trade_affinity_sealed_s{0,1,2}_20261009T1210Z), win rule as pre-registered above.
- 9 Oct 11:42: race_link v1 (tgbl-wiki) final: best val MRR 0.7729 (epoch 5). v2 running.
- 9 Oct 12:30: race_link v2 final: best **val MRR 0.776** (v1 0.773): repeat pairs .896, new pairs of known sources .074,
  new sources .069. Recency rank and page transitions add little under the batch-stale state.
- 9 Oct 12:45: **protocol diagnosis — batch-stale state.** v1–v3 (and `heuristics.py`, EdgeBank's TGB convention) score a
  batch of 200 queries from the state before the batch. The leaderboard's TPNet / DyGFormer / GraphMixer entries use the
  DyGLib_TGB evaluation, whose neighbour sampler is built on the full data and returns every interaction with time
  strictly below the query's (`find_neighbors_before`, `np.searchsorted`; `full_neighbor_sampler` in
  `train_link_prediction.py`), i.e. per-event causal state including earlier events of the same batch. Measured on the
  edge list: **5.4% of validation queries (test 5.4%) repeat a pair seen only earlier in their batch**, which our v1–v3
  scored as new pairs (~.07 MRR), and the next destination equals the source's last destination for **80.3% of queries
  per event vs 70.0% batch-stale** (test 77.9% vs 66.4%). Per-event causal state is the comparable convention and is
  native to the family (one O(n_dst) state update per event).
- 9 Oct 13:00: **race_link v4** (`race_link_v4.py`): v3 with per-event causal state (strict t < t_query; ties excluded),
  for training and evaluation; co-visitation/transition decay by a lazily rescaled reference time. Contract: state equals
  a brute-force exact computation to 2e-7 relative error (v1–v3's batch-level decay over-decayed the newest
  contributions by up to one batch span: 0.5% / 2% relative, documented). Queued: identities d = 16 and d = 0; the
  batch-stale v3 run was withdrawn before it started.
- 9 Oct 13:10: transparency note for the tgbn-trade sealed seeds: since the seed-0 development run, `race_affinity.py`
  gained two inert options (`--loss`, default `ce`; `--drop`, default `none`) and the integer-index fix; the sealed runs
  use the defaults, i.e. the configuration as developed. Their source hash differs from the development run's for that
  reason only. Ablations (`--drop trend`, `--drop reverse`, validation only) are queued after the sealed seeds (theory
  note 159, prediction 3).

