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
| TPNet | 0.827 ± 0.001 (validation 0.842 ± 0.001) |
| Heuristic (LocalGlobal) | 0.821 (validation 0.842) |
| HyperEvent | 0.810 |
| DyGFormer | 0.798 (validation 0.816) |
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
- 9 Oct 12:55: **tgbn-trade sealed seed 0: TEST NDCG@10 0.8683** (2013 / 2014 / 2015: .893 / .843 / .868); validation
  0.8747 (selected epoch 95); persistent forecast in our replay 0.8541 on test (leaderboard 0.855: protocol fidelity on
  test too). Above NAVIS (0.863) on this seed; the pre-registered verdict needs seeds 1 and 2 (queued).
- 9 Oct 13:45: **race_link v4 (per-event causal state, identities d = 16), development run in progress:** validation MRR
  0.8515 / 0.8492 / 0.8452 after epochs 0 / 1 / 2 (selection keeps the best); repeat pairs now 90.8% of queries (in-batch
  repeats visible) at 0.931 MRR. **The leaderboard's validation column: TPNet 0.842 ± 0.001, LocalGlobal heuristic 0.842,
  DyGFormer 0.816.** Correction: earlier external text compared our validation MRR with TPNet's *test* 0.827; replaced
  by the like-for-like validation comparison (0.776 vs 0.842 for the completed v2) until v4 completes.
- 9 Oct 13:40: **tgbl-wiki sealed protocol, fixed before either v4 development run completes.** Model: `race_link_v4.py`;
  identities d = 16 or d = 0, whichever has the higher best validation MRR in its seed-0 development run (tie within
  0.002: d = 0, the smaller model). Configuration otherwise as developed (hidden 64, lr 3e-3, batch 200), `--epochs 3`
  (the development optimum was epoch 0 with a slow decline after it), validation selection over those epochs, seeds 0,
  1, 2, each with `--score-test` (test scored once per seed). Win rule unchanged: mean test MRR over the three seeds
  above TPNet's 0.827 with every seed above 0.827. Reported either way.
- 9 Oct 14:05: race_link v4 d = 16 development run complete: **best validation MRR 0.8515** (epoch 0; then .849, .845, .840,
  .833, .8xx: the identities overfit the training period), 171,627 parameters (identity tables 9,227×16 + 1,000×16 +
  1,000), 6 epochs in 70 minutes. The d = 0 run follows; the sealed variant is chosen by the pre-registered rule above.


## B5-N large datasets on AWS (genre, reddit, token)

- **Checks before any fit (9 Oct):** (1) destination id = class index 0..C−1 and users numbered from C, for genre, reddit and
  token (py-tgb `pre_process.load_edgelist_datetime`, `_sr`, `_token`; the label dictionaries use the same name→index map).
  (2) tgbn-genre labels are daily (1,579 label times, 23/25 h gaps from DST), sum to 1 and summarise the following days:
  L1 distance of a label to the user's next 7 days of edges 1.18 vs 1.40 to the previous 7 days; no simple window
  reproduces them exactly (TGB's own weighting), so features use only strictly earlier edges. (3) TGB's official loop reveals
  each node's label after its label time fires and is scored; the official Persistent Forecast predicts that last revealed
  label (examples/nodeproppred/tgbn-genre/persistant_forecast.py), so revealed past labels are legitimate inputs.
- **Driver `race_affinity_v2.py` (streaming):** TGB's batch-of-200 loop with one label pointer; per-user decayed edge
  affinity at 1, 3, 7, 30 days (lazy decay), the last three revealed labels and decayed label averages, global shares,
  growth and context; MLP race over classes, online updates at training label times. Memory fits token (≈61K users × 1,001
  classes × 4 scales). Structural check: 27 finite features; persistent forecast through our replay 0.356 NDCG@10 on the
  first 60 training label times of genre. Development fit queued on slot 2 (validation only).
- 9 Oct 14:31: **tgbn-trade sealed seed 1: TEST NDCG@10 0.8674** (.893 / .843 / .866), validation 0.8740 (epoch 111).
  Seeds 0–1 both above NAVIS's 0.863; seed 2 pending.
- 9 Oct 15:42: race_link v4 d = 0 development run complete: **best validation MRR 0.8518** (epochs .8517 / .8518 / .8515
  / .8512 / .8510 / .8505: stable), **7,995 parameters**. Pre-registered selection: d = 0 (0.8518 ≥ 0.8515 − 0.002).
  Sealed seeds 0–2 of v4 d = 0 queued (curie_b5_wiki_sealed_v4_id0_s{0,1,2}_20261009T1340Z).
- 9 Oct 16:18: **tgbn-trade SEALED VERDICT: WIN (pre-registered rule met).** Test NDCG@10 seeds 0 / 1 / 2: **0.8683 /
  0.8674 / 0.8683, mean 0.8680 ± 0.0005**, every seed above NAVIS 0.863 ± 0.001 (ICLR 2026, leaderboard leader);
  persistent forecast 0.855 (our replay 0.854); validation mean 0.8744. Per test year (mean over seeds): 2013 0.893,
  2014 0.843, 2015 0.868. Model: race_affinity v1, 2,107 parameters (NAVIS: 1,280), about 4 minutes of fitting on one CPU
  thread. Evidence level: confirmed leaderboard win (3 seeds, configuration fixed before test, official loader and
  Evaluator, test scored once per seed). Scored locally; leaderboard submission after the patent priority filing.
- 9 Oct 16:57: **tgbl-wiki sealed seed 0 (race_link v4, d = 0, 7,995 parameters): TEST MRR 0.8353**, validation 0.8518;
  above TPNet's 0.827 ± 0.001 (and the LocalGlobal heuristic's 0.821) on this seed. Seeds 1–2 pending; verdict by the
  pre-registered rule.
- 9 Oct 17:32: **measured inference cost** (one CPU thread, plain numpy/PyTorch, untuned): tgbn-trade 0.22 ms per node
  prediction over all 255 destinations (2,107 parameters); tgbl-wiki v4 4.5 ms per query including the per-event state
  update and scoring all 1,000 candidates (220 queries/s; d = 0 and d = 16 equal: the cost is the feature map, mostly the
  1,000 × 1,000 co-visitation product, not the readout).
- 9 Oct 17:36: **tgbn-trade trend ablation** (`--drop trend`, validation, seed 0): 0.8677 vs full 0.8747 and persistent
  forecast 0.8604; the history-shape features carry about half (0.0070 of 0.0143) of the gain over persistent forecast.
  Reverse-flow ablation pending (note 159 prediction 3 graded when both are in).
- 9 Oct 18:15: **tgbl-wiki sealed seed 1: TEST MRR 0.8350** (validation 0.8519). Seeds 0–1 both above TPNet 0.827;
  seed 2 pending.
- 9 Oct 18:20: **reverse-flow ablation** (validation, seed 0): 0.8759 (full 0.8747): reciprocity adds nothing measurable.
  Note 159 prediction 3 graded (partly confirmed: history shape carries half the gain; reverse flow none).
- 9 Oct 18:59: **tgbl-wiki SEALED VERDICT: WIN (pre-registered rule met).** race_link v4 without identities (7,995
  parameters): TEST MRR seeds 0 / 1 / 2 **0.8353 / 0.8350 / 0.8356, mean 0.8353 ± 0.0003**, every seed above TPNet
  0.827 ± 0.001 (leaderboard leader), LocalGlobal 0.821, HyperEvent 0.810, DyGFormer 0.798; validation 0.8518. Evidence
  level: confirmed leaderboard win (3 seeds, variant chosen on validation by the pre-registered rule, official loader,
  negatives and Evaluator, test scored once per seed). Scored locally; leaderboard submission after the patent priority
  filing. Inference: 4.5 ms per query on one CPU thread with the exact state update and all 1,000 candidates scored.



## AWS continuation — strict causal review state (9 Oct, current queue)

The unstarted `race_link_review.py` smoke and six-epoch development fit are withdrawn from admission; queue definitions
remain. The batch-stale visibility failure measured on wiki motivates `race_link_review_v2.py`: queries see every
previous event with time strictly less than their own, including within a minibatch. Equal-time events remain pending
across batch/split boundaries. Sparse pair/destination clocks, separate addresses/values, the sampled race readout and
losing-candidate credit are retained. Initial pair-array allocation is smaller and grows with touched pairs. Training
competitors exclude duplicate identities/the positive; AdamW 1e-4 and clipping 1 bound the initial development update.
These are recorded development choices, not a changed frozen TEST protocol. v2 has no TEST flag.

`aws_b5_review_causal2_20261009T2140Z` on slot 3: causal-prefix/batch/tie contracts -> 3K TRAIN/500 VAL smoke ->
50K TRAIN/5K VAL, two-epoch pilot. Both truncated fits replay all remaining TRAIN events into causal state before
validation. Query-type error analysis (repeat pair/new pair of known source/new source) and train/replay/validation
wall times are saved. A prefix MRR is not compared with full-validation or TEST scores.

Non-fitting tmux controller `aws-b5-causal2` (`continue_review_causal_v2.py`) watches the pilot. Passed source-bound
contracts/smoke, finite pilot metrics, 1.5x measured RSS margin within 6 GB and a 1.75x timing projection within 6 h admit
one full TRAIN/full VALIDATION epoch through the same slot scheduler. Projection uses the verified 3,413,837 TRAIN and
730,784 VAL events; it is a timing screen, never a quality forecast. A missed resource screen stops scaling and records
an engineering decision. Full validation then determines the gap to GraphMixer 0.428 and which query class to improve.
No automatic TEST, seed expansion or leaderboard win claim. Numerical contracts and fitting results remain pending.

**AWS review causal-v2 pilot (10 Oct 05:56 UTC, development, VALIDATION prefix only).** Contract passed; smoke (3K TRAIN / 500 VAL) MRR 0.269; pilot (2 epochs on 50K TRAIN events, all TRAIN replayed into causal state, first 5,000 VAL queries, official negatives/Evaluator) **MRR 0.200**, 11.6M parameters (identity tables for 352,637 nodes), 137 s, 3.25 GB RSS. The controller's resource gates passed and admitted one full TRAIN / full VALIDATION epoch (`aws_b5_review_causal2_full_dev_20261010T055631Z`, projected 54 min). **Diagnosis to carry into the full epoch:** the training-free decayed state (30-day popularity + pair recency) scores 0.29–0.296 on the first 2,000 VAL queries and 0.344 on full VAL, so the learned readout is currently below its own state features. If the full epoch stays below 0.344, the next version scores a learned residual on top of the decayed-popularity/recency log-score (initialised at the heuristic) instead of relearning it through identities, and regularises the 11.6M identity parameters (curie's wiki finding: identities overfit the training period). No TEST.

**AWS review v3 full development epoch (10 Oct 10:55 UTC, VALIDATION, no TEST).** The v2 epoch was quadratic in TRAIN events (per-query rebuild of the seen-destination array; diagnosed with py-spy at 2.17M/3.41M after 3 h 13 min) and was stopped; v3 (`race_link_review_v3.py`, bitwise-identical draws, equivalence smoke exact) trained one epoch on all 3,413,837 TRAIN events in 42 min and scored all 730,784 VAL queries in 143 s: **val MRR 0.2678**, below the training-free pop_30d heuristic (0.340) and GraphMixer (0.428). Query profile: repeat pair 0.558 (1,256), **new pair of a known source 0.247 (663,980, 91%)**, new source 0.475 (65,548). Known sources rank worse than unseen ones: source-specific parameters (11.6M identity weights, one epoch) are the prime suspect, consistent with curie's wiki finding that identities overfit the training period. Diagnosis job `aws_b5_review_ablation_*` (no training) scores the checkpoint without identities, with popularity alone and popularity + network residual; it decides v4.

**AWS review ablation (10 Oct 13:30 UTC, VALIDATION, no training, no TEST; `aws_b5_review_ablation_full_s1_20261010T1255Z`).** The saved v3 checkpoint reproduces 0.26783375 exactly. MRR (all / known-source new pair / new source): full 0.268 / 0.247 / 0.475; network without identities 0.277 / 0.263 / 0.408; network + destination bias 0.279; identities alone 0.148; **30-day popularity alone 0.341 / 0.329 / 0.462**; 7-day popularity 0.323; popularity + trained network 0.313. Identities cost only 0.009 on the whole; the trained network itself ranks below popularity and degrades it when added. **Diagnosis:** training competitors (half uniform over all 352K nodes, half uniform over ever-seen destinations) are trivially separable by seen/recency flags, so the network never learns to rank among popular candidates, which the official negatives require. **Next (v5):** popularity-residual scoring (start at the 0.341 heuristic), no identities, and hard historical negatives drawn from recent destination events (∝ recent popularity) instead of uniformly over seen nodes.

