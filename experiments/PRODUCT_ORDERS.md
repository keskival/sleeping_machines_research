# Product orders — owned battles for headline results (issued 6 October 2026, 15:00 UTC; user-directed)

**These orders supersede every order below.** The text below this section is the historical record.

## The strategy

We pick the battles where our core primitive is the right mathematics, and we develop to win. A race of exponential
clocks over memory that decays with elapsed time is a temporal point process: the first finisher gives the next event's
type and time, and the waiting time without events enters the likelihood as a survival term (silence-aware supervision).
Timed, irregular event data is our home field; public leaderboards there are where headline wins come from in weeks.
Benchmark work follows AGENTS.md "Benchmark work is development to win": owned, designed, iterated, then verdict.

## The battles

| Id | Battle | Public reference | Owner | Pass criterion (developed attempt) | Stop rule |
|---|---|---|---|---|---|
| **B1 (lead)** | **EasyTPP** temporal point processes: Retweet, Taxi, StackOverflow, Amazon, Taobao (ICLR 2024 benchmark, official splits) | Published EasyTPP tables: NHP, THP, SAHP, AttNHP, FullyNN, IntensityFree, ODE-TPP | AWS host | Beat the best published log-likelihood on ≥ 2 of 5 datasets with type error/time RMSE no worse than the best published; confirmed on 3 seeds; inference work measured | No best-published LL on any dataset after the documented development plan (about two weeks) |
| **B2** | Irregular multivariate time series classification: P12, P19, PAM (Raindrop protocol and splits) | Published Raindrop, ViTST, Warpformer and later tables | AWS after B1 first fits | Beat best published AUROC (P12/P19) or accuracy (PAM) on its official splits | Same rule as B1 |
| **B3** | FAS v2 sealed confirmation (FAS_V2_CONFIRMATORY_PROTOCOL.md), then public release of FAS with a leaderboard | Information-matched classical references plus small LSTM and time-encoded Transformer references (AGENTS.md exception, 6 Oct 17:46) | Existing FAS owner | As pre-registered; no expansion | As pre-registered |
| **B4** | Neural TPP benchmark of Bosser & Ben Taieb (TMLR 2023, 2025): LastFM, MOOC, Github, Stack Overflow, Wikipedia, MIMIC2, Retweets; five fixed splits (admitted by the founder 9 Oct 2026) | Per-dataset tables of TMLR 2023 (Appendix B) and arXiv 2412.08590 Table 1 | AWS | The frozen B1 unified configuration (no tuning) below the bar (best published L_T + best published L_M) on a dataset, pre-registered in [B4_NTPP_BENCHMARK_PROPOSAL.md](B4_NTPP_BENCHMARK_PROPOSAL.md) | All seven datasets reported once; no configuration change. Status 9 Oct: **Wikipedia WIN** (−240.4 vs −122.6); MOOC, Stack Overflow, MIMIC2 LOSS; Github no valid verdict yet (NaN failure on 4 splits, guarded reruns `b4g_*` queued); Retweets, LastFM running. Development target: the mark path (L_M behind on four datasets) |
| **B5** | **Temporal Graph Benchmark** dynamic link prediction, first tgbl-wiki-v2 (official py-tgb loader, chronological splits, negatives and Evaluator; MRR); then tgbl-review-v2 (founder direction 9 Oct: prepare wins on new event domains) | TGB leaderboard (tgb.complexdatalab.com, verified 9 Oct): TPNet 0.827 ± 0.001, Heuristic(LocalGlobal) 0.821, DyGFormer 0.798 | curie | Pre-registered in [tgb/B5_TGB.md](tgb/B5_TGB.md): test MRR above the leaderboard leader on 3 seeds | Developed attempt below the leader after the plan in B5_TGB.md (about two weeks) |
| **B10** | Crypto market event streams (Binance BTCUSDT aggTrades, public): probabilistic forecasting of the trade stream (next-trade time, side, size log-likelihood; short-horizon realised volatility and trade intensity), prequential, with online state/weight adaptation tested on months with regime changes; LOB-Bench (ICML 2025) as the public generative benchmark (founder admission 9 Oct; [proposal](NEW_BATTLES_PROPOSED.md)) | E17 baselines re-run (online logistic regression, momentum), Hawkes and published neural-TPP references; LOB-Bench published scores | curie | Better log-likelihood and volatility forecasts than every reference on a pre-registered prequential month, 3 seeds; direction and P&L after costs reported as secondary metrics only | No likelihood or volatility gain over Hawkes after the development plan |
| **G** | Generality track (7 Oct, user-directed): self-supervised event pretraining → clinical classification (G1), one model across five event datasets (G2), PAM (G3), event→event transfer (G4) | — | AWS | Per [GENERALITY_PLAN.md](GENERALITY_PLAN.md) | — |
| **R1** | Language research track, at most one slot ([dossier](R1_RECALL.md)) | Count n-gram references; later a published small Transformer | One owner | Gates before any scaling or claim: solve associative recall/induction with irregular gaps; beat KN trigram on the 65,528-target DEV slice | — |

Every battle reports measured inference work (operations, CPU latency; energy where measurable) next to quality.

**Shared enabler, first test (user-directed 9 Oct 2026):** theory note 160's credit model trained by the forward race's own sampled causes, against REINFORCE, straight-through Gumbel, dense exact and an exact-posterior oracle on hard winner-only routing (`experiments/credit/hindsight_race.py`, 30 runs, queue label ENABLER).

**Shared enabler:** high-fidelity route credit (exact forced-lane credit at pool 2–8 as a training signal; a derived
low-variance multi-step estimator next). It serves B1, B3 and R1 directly; it is developed inside those battles, not as a
separate queue.


**Integration note (curie FAS session, 6 Oct 15:20 UTC).** VALUE_PLAN.md (same day, same diagnosis) is folded into these
battles and does not compete with them:
- Its Stage 2 (fixing the measured route-chaos and credit failure on FAS validation data) is **B3 development plus the
  shared route-credit enabler**.
- Its maturity gate is the same rule as "Benchmark work is development to win".
- AWS priority follows the battle table: B1 leads. The FAS neural references are withdrawn: under the 6 October
  no-new-dense-controls direction they are not trained (protocol amendment, AWS_FAS_REFERENCES.md).

## Admission rule (all hosts, all agents)

1. Every queued job names its battle (B1–B5, B10, G, R1) and the decision its result changes. No battle, no admission.
2. Each battle keeps a one-page frozen protocol, a table of published reference numbers with sources, a development log
   with error analysis per iteration, a timebox and its stop rule.
3. Contracts and smokes run only as steps of a battle pipeline: one smoke per new driver, numerical contracts only for
   new mathematics.
4. One weekly scoreboard. Part I of the report changes when a battle produces an outcome.

## Stopped (complete running jobs at safe boundaries; do not start new ones)

Private-bank and time-scale audit chains; split-horizon contract and admission ceremonies; token width/learning-rate/
capacity sweeps; k-write and tied/untied pool arms; grokking follow-ups; Mackey-Glass and primate reaching; SHD and
DVS gesture; real-table comparisons against trees (until typed learning has a battle); dense controls of any kind;
PDF-append republishing. Completed results stay in the record.


# Historical orders (superseded 6 October 2026, 15:00 UTC)

## Product orders — what we are after (issued 4 October 2026, 19:20 UTC)

**Latest user direction — improve our own models; no new dense controls (6 October 2026):** compute goes to
improving the integrated temporal/sparse models. No new Transformer/LSTM (or other external-architecture) training on
any host: no retries, tuning arms, strict matched-compute retries or the "missing" D-optimized recurrent control below.
References are published benchmark/leaderboard scores on the exact matching protocol, plus dense results already
completed. The running AWS `aws_tuned_ref_90M_D_tf256L4_p1_lr0.001_s0_20261004T210000Z` finishes as admitted and is
the last dense control; it is the only remaining dense job in `aws_model_improvement_repair_20261005T161000Z`. This
supersedes dense-training admissions in the paragraphs and tables below (P0-6, the strict Transformer retries,
P1-4); their completed numbers remain the record.

**Latest user direction — retain three evidence fronts (5 October):** continue
proactively on the strongest integrated language path, while allocating bounded
compute to improving FAS and tabular headline results. More FAS seeds and
benchmarks explicitly requested: native frozen-recipe seeds7/8 are scheduled
behind the already-admitted language sequence; existing neural benchmark
queues remain AWS-owned. Preserve active jobs. After these admitted tasks,
the first new tabular task is a small integrated typed-comparison/race contract
and fit, not another numeric-only capacity sweep. Read
TYPED_PREDICATE_EVENT_DIRECTION_20261005.md; establish type semantics,
hard-route learning and resource accounting before longer fitting. Existing
banknote confirmation trails trees on average and must remain visible; do not
promote the mixed-type proposal as an achieved win. Public language validation
and reserved larger scaling point remain protected. No fixed percentage of
compute is inferred; admit bounded tests from measured costs and current jobs.


**Latest user direction — implementation serves the objective:** we own and change the implementation. Current driver/backend choices are engineering work to resolve, never reasons to constrain tokenization, persistent memory, routing, learning or the research ambition. Build the capabilities the selected economical experiment requires. See OPEN_LANGUAGE_REFERENCE_PLAN.md and LANGUAGE_IMPLEMENTATION_AUDIT_20261005.md.

**Latest user direction — CPU-only, tokenized language, reuse public baselines (5 October):** [OPEN_LANGUAGE_REFERENCE_PLAN.md](OPEN_LANGUAGE_REFERENCE_PLAN.md) supersedes new discretionary baseline grids and GPU provisioning requirements below. Reuse published Transformer runs/checkpoints and their exact data/tokenizer; train our integrated models. Start from the smallest credible published Transformer-leading tokenized regime, not a new LSTM crossover campaign. Measure native tokenized CPU throughput and output-head cost before admitting a long fit. Existing jobs/results remain preserved.

**Protocol correction — 5 October 2026, user-directed:** FIFO is an **oracle-assisted diagnostic**, not an eligible reference for anonymous-process learning. `deinterleave_baseline.learn_route()` uses hidden TRAIN item identities to recover the route. The v2 timing-aware probe additionally fits transition-gap statistics with those identities. Neither receives test identities for prediction, but both receive privileged training structure unavailable to native and generic controls. Their scores are retained as oracle-assisted diagnostic targets; exclude them from strongest-reference selection and win/loss verdicts. Native's completed single-seed win against the six generic controls stands: **0.600 vs 0.559 AUROC at N=256**. A fair structure-learning reference must fit exclusively on the same anonymous training logs. Historical contrary interpretations below are superseded by this correction; numerical records remain preserved.

Owner: the user. Product owner: curie agent, on the user's instruction. **Every agent (curie, AWS, local) reads this
first and works these orders in priority order.** Definitions of win/loss: experiments/WIN_CRITERIA.md.

**Latest user direction (5 October): Transformers at larger scale are the main
opponent.** [LANGUAGE_CROSSOVER_PLAN.md](LANGUAGE_CROSSOVER_PLAN.md) defines the
primary objective: locate a Transformer-leading language regime, then establish
integrated event-model advantage at equal complete fitting compute and confirm
at a reserved larger point. Reuse the pending 90M C controls and D Transformer
arms; a D-optimized recurrent control is missing. Small-scale count/LSTM wins
are diagnostic and intermediate evidence, not a gate to modern language work.
The modern corpus/tokenizer, strong Transformer and GPU execution requirements
in FRONTIER_COMPUTE_PROTOCOL.md now belong to the main path rather than P2.
Preserve active jobs and enact this priority at safe owner boundaries. Finish
near-complete primate and existing strict-budget comparisons. FAS v1's FIFO
method is an oracle-assisted diagnostic, excluded from fair-reference verdicts; further v1 sweeps are deferred unless already admitted or
needed for one bounded transferable diagnosis. Historical orders below remain
as the record; this paragraph governs new discretionary admission.

**FAS v2 confirmatory program (5 October 14:30 UTC, user-directed):** [FAS_V2_CONFIRMATORY_PROTOCOL.md](FAS_V2_CONFIRMATORY_PROTOCOL.md), queue manifest `queue/fas_v2_confirmatory_20261005T143000Z/manifest.json`. Pre-registered home-field test with ambiguous identity: setting chosen from oracle/classical headroom on validation only, strongest de-interleavers and five neural references, equal tuning budgets, three seeds, sealed test ledger, one primary endpoint (AUROC at N=512). Stages 0–2 are CPU-light and start now; training stages take free capacity without preempting running jobs. This replaces further FAS v1 sweeps.

**User-directed completion packet (5 October, review session):**
[BENCHMARK_WIN_EXECUTION.md](BENCHMARK_WIN_EXECUTION.md) and
`queue/benchmark_win_execution_20261005T111500Z.json` track retained next-step
tasks, explicit deferrals, smoke/contract dependencies and completed results. Reuse
the owner's strict A/B retries below; the packet adds matching budget-B seed7
confirmation after a passed seed6 comparison. Before future k-write fits,
read the source-frozen v2 repair: the old wrapper's traced optimizer windows
use k1 while later windows use k2. Preserve old queues/results, check active
owners, then use corrected v2 stages. Numerical v2 contracts remain unrun.
FAS R8 is completed; its longer horizons did not close the early-detection gap.
The bounded binding/credit diagnostic specification is in FAS_BINDING_DIAGNOSTIC.md;
it is deferred behind the language scaling objective under the larger-data language priority; FIFO is oracle-assisted.
This packet admits no worker and does not replace active scheduler manifests.

## What increases our valuation

Investors buy three things, in this order:

1. **A headline number against the incumbents.** "Our model matches or beats LSTM and Transformer language models at
   N× less compute" (training and inference), on a standard dataset, with the same data and test set. One number,
   reproducible, with a second seed.
2. **A public leaderboard entry.** "First place on a recognized benchmark" (NeuroBench), under its official protocol,
   with its own efficiency columns.
3. **A scaling curve.** The win holds or grows from 10M to 90M characters (and beyond), so it is a trend rather than a
   lucky point.

Everything else (diagnostics, theory notes, new primitives) is valuable only insofar as it produces or protects these.
Every deliverable below has a pass/fail number. Report it as a win or a loss in one line.

## URGENT AWS slot order (5 Oct 07:05 UTC, product owner)

`queue/aws_product_priority_20261005T001500Z/manifest.json` lists only the ten finished tuned-reference arms in slots 2
and 3, so **two of the three AWS slots appear idle since about 01:30 UTC**. Meanwhile, the leaderboard deliverables are
waiting:
- P0-3: the six primate sessions are queued in slot 1 behind the 90M job.
- P0-4: Mackey-Glass repeats 20–29 have no result; still 20/30.

Fill the free slots now, in this order (each through run_safe.sh, with the usual caps and at least 8 GiB MemAvailable):
1. **Slot 2: P0-3 primate six-session r1.** Move the six `aws_primate_r1_*_20261004T053500Z` jobs out of slot 1.
2. **Slot 3: P0-4 Mackey-Glass third batch** (repeats 20–29; pins verified 4 Oct 23:55).
3. Then, in the first free slot:
   - the two 90M budget-C tuned references (P0-1's tuned comparison; ~4 h each; added 07:15);
   - the FAS data job and the five FAS references (AWS_FAS_REFERENCES.md);
   - the weight-decay smoke and arms;
   - the X1 capacity-curve arms (AWS_CAPACITY_PROGRAM.md).
4. Slot 1 continues P0-1 (90M r4 p64, then p96).

The leaderboard items are the cheapest route to a public result. Idle slots are the largest current waste.

**Strict Transformer matched-compute retries (5 Oct 11:00 UTC; investor-relevant, cheap): first free AWS slot.**
- `queue/aws_language_10M_strictB_p64d4_3.8pass_20261005T110000Z.txt` (~2.5 h; target ≤ 102.7 TF against TF128x4 2.215).
- `queue/aws_language_10M_strictA_p96d4_5.85pass_20261005T110000Z.txt` (~6 h; target ≤ 347 TF against TF128x4 1.996).

Today's tuned-Transformer wins are 2–4% over the Transformers' budgets. These runs fit under them, and the traced work
decides whether they qualify. A win upgrades deck slide 2 from "~equal compute" to "beats the validation-selected
tuned Transformer at no more of its training compute".

## P0 — do these first (all hosts)

| # | Deliverable | Host / queue | Pass criterion | Status |
|---|---|---|---|---|
| P0-1 | **90M matched-compute win** | AWS: `queue/aws_language_90M_r4_p64d4_4pass_linear_20261004T193000Z.txt`, then `..._r4_p96d4_4pass_...` (AWS_NATIVE_LANGUAGE_90M.md rev. 4) | test bpc ≤ 1.661 at ≤ 3.9 PF (beats LSTM-512); ≤ 1.604 at ≤ 8.0 PF (beats Transformer) | **first arm done (5 Oct 08:50): 1.800 at 0.97 PF, efficiency point** (LSTM 1.661 at 3.9 PF); slope 4x compute = -.057 bpc, extrapolated ~25 PF to reach the LSTM: not on track; p96 arm optional |
| P0-2 | **10M win against LSTM-512 at matched compute** | curie: p128/d4 4 passes, starts right after the FAS arm (~00:15 UTC 5 Oct); fallback p96/d4 pool 4 with sampled credit, 6 passes | test bpc < 1.799 at ≤ 433 TF; tuned LSTM-512 4.5p is 1.825 at budget A | **loss (5 Oct 04:20):** p128/d4 4-pass 1.907 at 411 TF; also below p96 6-pass 1.888. Width alone does not close the local gap; next: taps3, weight decay (AWS), lr arms |
| P0-3 | **NeuroBench primate leaderboard** | AWS: `queue/aws_primate_admission_20261004T184500Z/manifest.json` (six-session r1) | six-session mean R² > 0.71 (AEGRU) | 5/6: .768/.600/.529/.519/.677, paired mean .619 vs tinyRSNN .643, bigRSNN .683; **> .71 out of reach** (needs ≈ .92 on the last two); Indy at par, Loco behind |
| P0-4 | **NeuroBench Mackey-Glass leaderboard** | AWS r1 (repeats 20–29 running); curie expected-reception development → a pre-declared tau 17 run | 30-repeat sMAPE < 13.37 (LSTM), with a smaller footprint | **completed, loss (5 Oct 08:15):** 30/30 repeats, primary 8-stream average 14.84 (LSTM 13.37, ESN 14.79); smallest footprint (58 KB), no win |
| P0-6 | **Tuned dense references at our budgets** (added 4 Oct 21:00 UTC, user-directed) | AWS: ten `queue/aws_tuned_ref_10M_*_20261004T210000Z.txt` first (1.5–3 h each), then four `aws_tuned_ref_90M_*` (TUNED_BASELINES.md) | native beats the validation-selected tuned arm at each budget (A ≤ 352 TF: 1.888; B ≤ 107 TF: 1.955; 90M C/D once P0-1 completes) | running; 5/6 A arms done: LSTM-512 4.5p 1.826 (val-leading) and LSTM-384 1.840 beat native 1.888 (losses, context-independent); tuned TF192x4 2.027, TF128x4 1.996 and TF256x4 lr.002 2.053 are behind native 1.888 budget A 6/6 complete (TF256x4 lr.001 2.199); budget B 4/4 complete: LSTM-384 1.915, LSTM-256 1.934 beat native 1.955; TF192x3 2.315 and TF128x4 2.215 behind; admission waits for windowed LSTM validation rescore (below); curie native lr .003/.006 fairness arms queued after P0-2 |
| P0-7 | **Home-field async wins** (added 4 Oct 22:40 UTC, user-directed) | curie: FAS native arm (v6 queue, after the streaming evaluations), SHD aug/expected p32d4; AWS: FAS data + five references (AWS_FAS_REFERENCES.md) after the P0-1/P0-6 owners | FAS (experiments/FAS_BENCHMARK.md): native AUROC above every reference (LSTM, Transformer, LRU, S5-style, Mamba-style; classical) at N <= 256 process events, at no more training/inference work; SHD: a parameter- or work-matched result against small SNNs (cAdLIF 94.19% at 38.7K params; DCLS 95.07% at 0.2M) | queued |
| P0-5 | **Scoreboard** | report/scoreboard.py: REPORT.md / PDF page 2, generated from result files on every rebuild | every comparison listed as win / loss / efficiency point per WIN_CRITERIA | **done** (5 wins of 13) |

**AWS slot allocation order (revised 4 Oct 21:00 UTC):** P0-1, then P0-6 10M arms, then P0-3 take slots as soon as any
slot frees; P0-6 90M arms and the P1-1 seed `aws_language_10M_r1_p64d4_4pass_linear_s7_20261004T210000Z` follow. Without
P0-6, a matched-compute win against an untuned reference will not survive review, so it ranks with P0-1. The long depth-8 streaming
replay/teacher fits are research diagnostics: checkpoint them at their next milestone and suspend them (do not delete
anything) until P0-1 and P0-3 are running. Resume them afterwards.

**P0-6 blocking step (5 Oct 04:10 UTC): windowed VALIDATION rescore of the tuned LSTM arms.** Budget A is complete (6/6);
budget B is 3/4. `report/tuned_reference_admission.py` correctly refuses cross-architecture validation selection while
LSTM validation is carried-state and Transformer validation is windowed. Owner: curie (text8 + language_stream_rescore
tooling; inference only, minutes per arm). Rescore text8[90M:90.2M] with E64 T256 windows, the same scorer as the
Transformer arms, for A: lstm384_p6, lstm512_p4.5 and B: lstm384_p2.5, lstm256_p5 (+ any later LSTM arm). Save each
as a new result beside the original; never edit the AWS JSON. `reference_window_rescore.py` covers TEST only, so add a
validation interval mode or use language_stream_rescore. Expected outcome, not a substitute for the run: selection
unchanged (LSTM-512 val 1.769 vs best TF 1.950; test context effect 0.0000), so budget A = loss to tuned LSTM-512
1.826, and the Transformer sub-comparison = native win (1.888 vs TF128x4 1.996 at 101% compute).
**Taken by curie (5 Oct 01:15 UTC):**
- Queue: `queue/curie_lstm_val_window_rescore_20261005T011500Z.txt`, covering all four arms with
  `reference_context_curve.py --offset 90000000 --test 200000 --context-curve 0`.
- The scorer is the E64 window convention (first window whole, then second halves), checked against `e64_lm_baselines.score`.
- It runs first after P0-2 finishes (around 04:00 UTC), before taps3 and the native lr arms.
- Results land in `experiments/results/language_stream/curie_lstm_valwin_*`. The AWS JSONs are untouched.

**P0-7 FAS update (5 Oct 06:30):** identity-aware oracle bound .755 at N=256 / .913 at N=512, so the early range is
discriminating and native (pool 2, .600/.742) recovers 39%/59% of it. Curie queues: pool2 seed 7 (replication), pool 8,
pool 32 (per-item slots for 30 interleaved items), `curie_fas_native_p32d4_*_20261005T063500Z`.

**Recommendation after P0-1 (5 Oct 08:45, review host):** 90M native 4-pass 1.800 at 0.97 PF is an efficiency point. The
compute slope (.057 bpc per 4×) predicts that p96 4-pass (~2.1 PF) will not reach 1.661. Proposed slot-1 order: the two 90M
budget-C tuned references first (they give the tuned verdict for the completed P0-1 run), then the weight-decay arms
and the X1 capacity arms (per-event levers), then p96 as the trajectory check. The product owner decides; nothing is
cancelled.

**Historical P0-7 interpretation (5 Oct 11:00; superseded by the oracle-assisted protocol correction above):** a classical route-aware FIFO de-interleaver recovers item identity
exactly and equals the oracle (0.755 at N=256, 0.913 at N=512). Native pool-2 0.600, recruitment arms 0.566–0.578. The
"earlier than every classical method" claim is withdrawn. FAS v1 cannot show a race advantage unless native reaches ≥ .755.
Next: (a) winner-identity diagnostic on the native arms; (b) FAS v2 with ambiguous interleaving (product variants,
overtaking, dropped events), validated with oracle + de-interleaver before any native run.

## P1 — confirm and widen the wins

| # | Deliverable | Pass criterion |
|---|---|---|
| P1-1 | Second seed of every P0 win (10M: p96 6-pass, p64 4-pass; 90M winners) | win holds on seed 7 |
| P1-2 | Sampled route credit (§416): fitting cost flat in pool size at equal quality | pool 4/8 sampled within .01 bpc of linear at ≤ 60% of its fitting work |
| P1-3 | SHD official test (speaker-held-out selection first) | report accuracy against the cited 96.4% best |
| P1-4 | Update-matched 10M row against Transformer-256×4 (96 lanes) | 6-pass with 4,883 updates still < 1.908 |

## P2 progress

- **Hardware cost model v1 done (4 Oct):** `experiments/hardware_cost_model.py` → `results/diagnostics/hardware_cost_model_20261004.json`.
  Efficiency point: native p96 (1.888) moves 5.8× fewer bytes/char than Transformer-256×4 (1.908) and 1.9× fewer than LSTM-512
  (1.799, better quality), with cached-key winner-only execution (§414). Pool 2→32: 10.5× params for +5% bytes/char.
  (v1 of the model mis-charged key scoring; corrected same day, theory 154.) Next: a quality point at large pool (pool 8
  90M exists in rev. 3) so the flat-traffic capacity claim has a quality number beside it.

## P2 — after the P0/P1 numbers exist

Hardware cost model for the target asynchronous ASIC (event, message-rate, memory-locality/traffic and FLOP counts on one fixed trained model, vs the tuned dense reference; see investment/HARDWARE_THESIS.md;
CPU wall-clock is not evidence since the hardware is simulated); multi-modal or TTT demonstrations;
larger-than-90M scale.

## Stop or defer now

- New theory notes, primitives or diagnostics that do not feed a P0/P1 row (record ideas in a backlog, do not run them).
- DVS large program and the public UCR campaign repeats (results stand as reported).
- Clock-noise micro-diagnostics, unless they block P0-4.
- Investor-deck edits beyond inserting P0/P1 numbers once they are completed.

## Reporting rule

When a P0/P1 run completes: (1) one line in the scoreboard — win, loss or efficiency point with the numbers; (2) commit the
result file; (3) queue the confirming seed if it is a win. No paragraph of caveats: one scope sentence (data, seeds,
compute convention). Losses are reported just as plainly.

**Curie order after P0-2 (5 Oct 02:45 UTC, product owner):**
1. LSTM validation rescore (P0-6 blocking step).
2. taps3 compiled smoke, then taps3.
3. FAS pool 8 compiled smoke (updated driver, RSS check), then FAS pool 8 and pool 2 seed 7.
4. FAS pool 32 smoke, then pool 32.
5. Native lr .003, then lr .006.
6. The rest of the v6 queue.

The FAS pool sweep moves ahead of the lr arms (about 10 h) because the oracle bound shows early-detection headroom on the
home-field benchmark. Each FAS arm needs its smoke to pass.

**FAS recruitment sweep replaces the plain pool sweep (5 Oct 03:35 UTC; THEORY §419):**
- Order: torch contracts, then a smoke run with every knob active, then R0 (pool 8 untied), R1 (pool 8 tied), R3 (tied
  with free-slot bonus 3), R4 (tied with training temperature 2), R5 (tied with balance .01), then pool 2 seed 7, then the
  pool-32 smoke and R2 (pool 32 tied, 5K runs).
- Arms are screened at 1 epoch, compared within the sweep, and reported with occupancy.
- The other host's 2-epoch untied pool 8 and pool 32 queues are kept but not chained: R0 covers untied pool 8. Untied
  pool 32 is predicted to suffer the untrained-loser bias (§419 Proposition 3); run it later if R2 wins.

**AWS additions (5 Oct 04:00 UTC; AWS_CAPACITY_PROGRAM.md):** after the current P0 owners:
1. Two native weight-decay arms (§421; p64 4-pass, the budget-B comparison against tuned LSTM-384 1.915).
2. Three large-pool language capacity-curve arms (note 155 X1: tied/untied pool 32 and tied pool 8, sampled credit).
3. The FAS references, already queued (AWS_FAS_REFERENCES.md).

The FAS recruitment sweep and the grokking testbed run on curie.

**P0-6 blocking step done (5 Oct 04:22 UTC, curie).** Windowed (E64 T256) validation rescore of the tuned LSTM arms on
text8[90M:90.2M]:

| Budget | Arm | Windowed validation | Carried-state validation |
|---|---|---:|---:|
| A | LSTM-512 4.5 passes | 1.7688 | 1.7688 |
| A | LSTM-384 6 passes | 1.7779 | 1.7779 |
| B | LSTM-384 2.5 passes | 1.8518 | 1.8518 |
| B | LSTM-256 5 passes | 1.8722 | 1.8721 |

- The carried-state values reproduce the AWS `best_valid_bpc` exactly, and windowing changes nothing.
- Cross-architecture validation selection is now valid:
  - Budget A selects LSTM-512 (validation 1.769; best Transformer 1.950). Native **loses**: 1.888 against 1.826.
  - Budget B selects LSTM-384 (1.852; best Transformer 2.191). Native **loses**: 1.955 against 1.915.
- Transformer sub-comparisons stay native wins in quality, at 101–109% of the Transformer arms' compute.
- Results: `experiments/results/language_stream/curie_lstm_valwin_*_20261005T011500Z.json`.
