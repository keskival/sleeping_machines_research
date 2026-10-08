# B3 development log (FAS v2; owner: curie FAS session)

Protocol: `experiments/FAS_V2_CONFIRMATORY_PROTOCOL.md` (with its amendments). The test set is sealed. Development uses
training and validation only. Selection is by validation-clean NLL; validation AUROC is reported separately.

## Setting and references (Stage 1, validation, no learned model)

The amended rule selects **K = 2 lines, event dropout p = .02, no speed offset**. Prefixes are counted per line, and the
merged prefix is N·K.

| detector (v2 validation, 1,000 + 1,000) | N = 512 per line (merged 1,024) |
|---|---|
| identity oracle, line-aware max-step | .821 |
| best information-matched classical (order3) | .685 |
| gap | .136 |

Full grid: `results/fas/fas_v2_calibration_amended_grid_20261006T1550Z.json` (running).

## Native development plan (budget: 8 configurations on v2 validation)

All configurations share these settings:
- deep race network p32, depth 4, heads 2, pool 8, τmax 1000, linear route credit, segment 128;
- lr .003, 1 epoch, 5,000 training samples (~10.5M process events), seed 6;
- `--train-max-events 2048 --max-events 2200`.

The common training cap C is fixed from the C1 smoke.

| id | configuration | theory | question |
|---|---|---|---|
| C1 | race readout (hidden 64) + gated binding memory U_b = 54 (1.5 × anonymous peak concurrency) + step-class mixture M = 3 | §§433–436 | main candidate |
| C2 | standard native head (native.py, pool 8) | — | our previous architecture on v2 |
| C3 | C1 without step classes (single duration law per slot) | §435.2 | does type–duration coupling pay at p = .02? |
| C4 | C1 with U_b = 36 (exactly the anonymous peak, no headroom) | §434.3 | does binding need spare capacity (toy: α .48 vs .90)? |
| C5 | C1 with additive readout context (per-slot laws computable once per write) | §436.3 | does the low-cost readout match C1? (toy: α .922 vs .934) |
| C6 | C1 + recording-cell likelihood, δ = 1 ms | §439 | round 2: does removing the tie-spike density lift timing and total AUROC? |
| C7–C8 | reserved | | |

Evaluation-only on the selected C1 checkpoint: particle evaluation, L = 1, 4, 16 (§434.1.2). This does not count
toward the budget, because it adds no fitted configuration.

Target: native validation AUROC at merged N = 1,024 above .685 (order3) by more than the seed spread, approaching the
oracle's .821.

## Log

- 6 Oct 17:05 UTC: plan declared. The v2 data (Stage 2) is queued behind the amended calibration grid. Training
  waits for a curie window or an AWS gym slot.
- 6 Oct 17:09 UTC: error analysis ready (`readout_diagnostics.py`, evaluation only, validation identities used only for
  diagnosis). It reports binding purity α (the α of §434.2), item concentration and the time-rescaling fit (§435.3:
  KS of rescaled intervals, tie fraction), plus AUROC split by binding purity. It is queued right after C1 and its
  particle evaluation.
  - Data note: over a whole run, faulty samples carry ~7× more plant-clock ticks, because the faulty line runs far
    longer and the single clock runs to the last event. Process-event counts are equal. Within a prefix this is
    legitimate elapsed-time evidence, already available to the `elapsed` and `tick_count` references.
- 6 Oct 17:39 UTC: **small integrated fit of the binding mechanism** (`binding_toy.py`). Synthetic data: 3 interleaved
  processes, 8-step routes, log-normal steps with CV .15, no dropout. Model: binding memory 6 slots + race readout,
  2 step classes, trained from scratch, 300 steps of 16 samples.
  - Binding purity α rose from chance .27 to .71 with additive slot writes.
  - With the per-dimension overwrite gate it reached .75 (sampled writes) and .81 (argmax writes); validation NLL fell
    from 4.87 to 1.94.
  - The Bayes-greedy ceiling (true routes and laws, `binding_toy_oracle.py`) is α = .911, so 300 steps close ~75% of
    the gap from chance to that ceiling.
  - Gated writes go into C1-family iterations once compared on v2 (§436.2: a slot must be able to hold its process's
    latest state). A longer fit (1,200 steps) is running.
- 6 Oct 17:46 UTC: binding toy, 1,200 steps (gated): validation NLL 1.97 → 1.77, but α plateaus at .78–.80 (sampled)
  and .80–.84 (argmax), against the .911 ceiling. The remaining binding gap is not a step-budget gap. Testing
  hard-EM (argmax) training writes and a larger readout (hidden 64).
- 6 Oct 17:46 UTC: binding toy arms (600 steps, gated writes):
  - hard-EM (argmax) training: α .82 (no gain);
  - **readout hidden 64: α .934 with argmax writes (.88 sampled)**, above the Bayes-greedy ceiling .911, with
    validation NLL 1.73.

  The binding gap was readout capacity; C1 already uses hidden 64. A no-context ablation (§436.3) is running.
- 6 Oct 17:46 UTC: **small neural references reinstated** (user direction; AGENTS.md exception, protocol amendment).
  LSTM and time-encoded Transformer, d ∈ {64, 128} × lr ∈ {3e-4, 1e-3, 3e-3}, at most 3 passes over the same 5,000
  clean samples, validation only. Two smokes and 12 grid queues are prepared for AWS (`aws_fas_v2_ref_*_20261006T1815Z`).

  Sizes:

  | model | parameters | forward MFLOP/event (T ~ 2,100) |
  |---|---|---|
  | C1 | 338K | not traced |
  | LSTM d64 / d128 | 73K / 277K | 0.14 / 0.54 |
  | Transformer d64 / d128 | 110K / 425K | 1.3 / 3.0 |

  The Transformer's per-event cost grows with log length; ours is constant per event.
- 6 Oct 17:47 UTC: no-context ablation (§436.3), hidden 32: argmax α .908 (.88 sampled), validation NLL 1.83, against
  α .81 / NLL 1.94 with context at the same size. This supports the shortcut hypothesis: merged-stream context lets
  slots predict without binding.

  The toy processes are independent; FAS processes interact through shared stations, so C1 keeps the context at hidden
  64. If C1's diagnostics show low binding purity, a no-context or context-gated readout is the first C5–C8 iteration.
- 6 Oct 17:49 UTC: hidden 64 without context: argmax α .856, NLL 1.89, worse than hidden 64 with context (.934, 1.73).
  C1's choice of context at hidden 64 stands. Toy arms are single-seed; differences of a few points are within noise.
  Timestamp correction: entries from 17:05 to 17:47 were first written with times 5–40 minutes too late. They now
  carry the commit times, and the same correction is applied to the reference-exception notes (AGENTS.md, protocol,
  AWS_FAS_REFERENCES.md, PRODUCT_ORDERS.md). Queue tags keep their original suffixes; they are identifiers, not times.
- 6 Oct 18:23 UTC: **measured work of the C1 configuration** (`work_c1size_20261006T1820Z.json`; operation audit, one
  eager training window of 8 × 128 events, and inference on 4 validation runs).

  | model | fit, MFLOP/event | inference, MFLOP/event | convention |
  |---|---|---|---|
  | C1 | 5.08 (+0.037 M special) | 1.69 (+0.019 M special) | measured: forward + backward + optimizer |
  | LSTM d64 / d128 | — | 0.14 / 0.54 | shape estimate, forward |
  | Transformer d64 / d128 | — | 1.3 / 3.0 | shape estimate, forward, mean over T ≈ 2,100 |

  The references' fit work is about 3 × forward plus the optimizer (dense.py's convention). The Transformer's
  per-event cost grows with log length; C1's is constant. C1's implementation still computes every deep-layer slot
  proposal densely, so these figures are an upper bound on what the architecture needs.
- 6 Oct 18:45 UTC (user question: do models assume the number of parallel processes?). No model assumes a process
  count. The binding memory has a *capacity* U.
  - Unused slots go silent (pending probability → 0), and finished processes free their slot: gated writes let a new
    process take it over.
  - Only more *simultaneously active* processes than U forces slot sharing. That is graceful mis-binding, with the loss
    of §434.2, not a failure.

  **Sizing fairness:** C1's U = 40 was first set from the identity-measured peak concurrency (35), which is privileged
  information. The anonymous estimate (`anonymous_concurrency.py`: starts of the earliest type minus ends of the latest
  universally visited type; training logs only) gives mean 33.5 / max 36 against identity 34.2 / 35. The **declared
  sizing rule is U = anonymous peak concurrency on training logs + 10%**, which gives 40, unchanged.

  **Declared secondary stress test** (descriptive, outside the decision rule): score C1, trained at K = 2, on a K = 3
  validation-format set (concurrency ~54 > U), against references trained at K = 2. This measures degradation beyond
  capacity.
- 6 Oct 18:45 UTC: toy at 5% dropout (§435.2):
  - 3 step classes against one duration law: validation NLL 2.046 vs 2.143 at the end, 2.045 vs 2.222 averaged over
    steps 400–600, a gain of 0.10–0.18 nats/event (predicted .1–.2);
  - argmax α .917 vs .892.

  C1's 3 step classes stand. Single seed.
- 6 Oct 18:48 UTC: **capacity toy** (6 concurrent processes, Bayes-greedy ceiling α .886, 600 steps):

  | slots | argmax α | validation NLL |
  |---|---|---|
  | 3 | .31 | 2.19 |
  | 6 | .48 | 2.11 |
  | 9 | **.90** | **1.82** |

  Binding needs *spare* capacity, not just capacity equal to concurrency. Early posterior mis-writes need free slots
  to escape into; without them, shared slots lock in. This is the optionality argument of §§422–424 measured on
  binding. The earlier remark that binding gets harder to learn at higher concurrency is withdrawn: the K = 6 arm with
  6 slots had zero headroom.

  **Revised sizing rule: U = 1.5 × anonymous peak concurrency** (v2: 1.5 × 36 = 54). C1 and C3 now use U = 54. C4
  becomes the no-headroom ablation (U = 36). None of these queues has started. Single-seed toy evidence.
- 6 Oct 19:06 UTC: **particle evaluation on the toy** (6 processes, 9 slots, trained 600 steps; THEORY §434.1.2). Validation
  NLL by particle count:

  | particles | 1 | 4 | 16 | 64 | argmax path |
  |---|---|---|---|---|---|
  | NLL | 1.885 | 1.834 | 1.815 | 1.810 | 1.821 |

  The bound tightens monotonically with diminishing returns, as FIVO predicts. Particles are an inference-compute dial:
  L particles multiply inference work by L (C1: 1.69 MFLOP/event at L = 1). Any L > 1 used for v2 scoring is declared
  before test scoring, and its work is reported at that L. C1's particle evaluation (L = 1, 4, 16) on v2 validation
  sets it.
- 6 Oct 19:13 UTC: **sparse inference** (winner-only deep layers, exactly cached reads; contracts 16/16) and C1 work
  re-measured at U = 54 (`work_c1size_u54_20261006T1930Z.json`):

  | C1 work (MFLOP/event) | value |
  |---|---|
  | fit (forward + backward + optimizer) | 6.10 |
  | inference, dense | 2.04 |
  | inference, sparse | **1.50** |

  The readout now dominates inference: 54 slots × (P·h + h·(M·V + 3M + 1)) ≈ 1.2 MFLOP/event. Next work optimisation:
  make the per-slot laws change only on writes, with the context entering as a shared additive logit term (O(1) per
  event), so per-event readout work reduces to the U·M duration CDFs. This changes the context interaction; it is
  tested as a development configuration, not swapped in silently.

  References (forward, shape estimates, T ≈ 2,100): Transformer d64 / d128 = 1.3 / 3.0; LSTM d64 / d128 = 0.14 / 0.54.
- 6 Oct 19:16 UTC: toy, additive readout context (hidden 64, 3 processes): argmax α .922, validation NLL 1.75. Full
  context gave α .934 / NLL 1.73; no context α .856 / NLL 1.89. This is within single-seed noise of full context, at a
  per-event readout cost of only the duration CDFs once the per-slot laws are cached per write. Declared as C5 (budget
  slot 5).
- 6 Oct 19:22 UTC: **T1 (THEORY §437): learned write race against posterior routing**, binding toy (3 processes, hidden 64,
  gated, 600 steps):
  - posterior routing: α .93, NLL 1.73;
  - learned query/key write race with linear write credit: α .10–.22 (chance .33), NLL 2.25 at best, then diverging
    to 3.6.

  The learned route never binds. Posterior routing turns the same route into inference. This is consistent with the
  credit-blindness audit (§429). Single seed; the learned arm has no recruitment aids (free-slot bias and similar).
- 6 Oct 19:35 UTC: **skip-deep ablation** (§438: binding memory and race readout without the deep learned-route layers):

  | processes | with deep layers (α / NLL) | without (α / NLL) |
  |---|---|---|
  | 3 | .934 / 1.73 | .938 / 1.75 |
  | 6 (9 slots) | .90 / 1.82 | .74 / 2.25 |

  The deep layers' history context matters as concurrency grows. FAS v2 has ~36 items in flight, so C1 keeps them; no
  skip-deep v2 configuration is declared. For §438 this means predictive routing should *upgrade* the deep layers' routes,
  not remove the layers.
- 6 Oct 20:07 UTC: **Stage 1 amended calibration complete** (`fas_v2_calibration_amended_grid_20261006T1550Z.json`).
  **All 12 settings qualify**: oracle .747–.821, gap to the best anonymous classical detector .106–.198. Speed
  offsets give the largest gaps (≥ .156), because pooled classical statistics cannot absorb them. The selection is
  confirmed: K = 2, p = .02, δ = 0 (oracle .821, order3 .685, gap .136). Stage 2 data generation started on completion.
- 6 Oct 20:07 UTC: **§438 predictive-routed deep layer, first form: negative.** 6 processes, 9 binding slots:

  | deep layer | α (sampled / argmax) | validation NLL |
  |---|---|---|
  | predictive layer, 9 slots | .67 / .59 | 2.18 |
  | learned deep layer | .82 / .90 | 1.82 |
  | none | .77 / .74 | 2.25 |

  The learned deep layer's value is cross-process history context. The predictive layer outputs only the written slot's
  process-specific memory, duplicating the binding partition, and its local objective may compete with the top
  objective. Not adopted; C1 is unchanged.
- 6 Oct 20:31 UTC: **Stage 2 data frozen**: `fas_v2_K2_drop0.02_delta0_20261005`, with the manifest published in
  results/fas and the split hashes in the B3 queue manifest.
  - Checks: 2,042–2,100 process events per sample; fault kinds 491/509 (validation) and 1,022/978 (test); seeds
    disjoint across splits; no per-event identity in the anonymous files.
  - The declared K = 3 stress set (`fas_v2_stress_K3_drop0.02_delta0_20261005`, validation only) is also generated.
  - Test splits are sealed.
- 6 Oct 20:39 UTC: **surprise-gated processing** (§437 I4), applied at inference only to the trained 6-process toy:

  | gate (nats) | events skipping deep layers | NLL | α |
  |---|---|---|---|
  | none | 0% | 1.930 | .823 |
  | 0 | 2% | 1.950 | .818 |
  | −0.5 | 9% | 1.991 | .801 |
  | −1 | 20% | 2.126 | .775 |
  | −2 | 46% | 2.529 | .731 |

  Negative in this form: the deep layers carry history context, so skipped events leave gaps, and the model was not
  trained with gaps. Untested: gate-aware training, and highly deterministic streams such as FAS routes. Not adopted
  for v2.
- 6 Oct 21:31 UTC: **classical development targets on the frozen v2 validation set**
  (`fas_v2_classical_val_20261006T2115Z.json`; per-run scores saved for Stage 5 alarm calibration).
  - Best at merged 1,024 (N* = 512 per line): order3 .685, reproducing the Stage 1 calibration exactly; ngram3 .675;
    timed_ngram .670; gap_z .665.
  - Earlier prefixes: merged 512 (256 per line) order3 / timed_ngram .565; merged 256 order3 .543.
  - The gap statistics (quantile, CUSUM, robust z) sit near chance. Interleaving destroys adjacent-gap meaning, as
    expected.
  - C1 target: above .685 at N* by more than the seed spread (identity oracle .821).
- 6 Oct 23:31 UTC: **AWS admitted the v2 data and the reference smokes.**
  - The AWS v2 dataset is byte-identical to curie's (all five split hashes and event counts equal): the generator
    reproduces across hosts.
  - Smokes, 2 windows each:
    - LSTM d128: 276,528 parameters, peak RSS 1.05 GB, ~25K events/s. A full 3-pass run takes ~20 min.
    - Transformer d128 (4 lanes): 425,392 parameters, peak RSS 1.36 GB, ~3K events/s. A full 3-pass run takes ~3 h.
  - Suggested caps: LSTM 2.5 GB, Transformer 3 GB.
- 7 Oct 00:02 UTC: **C1 completed on AWS** (round 1; `aws_fas_v2_dev_C1_20261006T1715Z`; 1 pass over 5,000 samples,
  10.24M events; 1.9 h; peak RSS 4.5 GB; 339,364 parameters).

  v2 validation AUROC by merged prefix (×2 = per line):

  | rule | 256 | 512 | 1,024 (N*) |
  |---|---|---|---|
  | total (primary) | .512 | .539 | **.630** |
  | type | .527 | .557 | .673 |
  | gap | .503 | .515 | .550 |

  Targets at N*: order3 .685, oracle .821. **Status: in development, best .630 vs reference .685.**

  - **Reading:** the signal is almost all in event *order* (the type rule matches the order-only trigram); *timing* is
    weak. The oracle's advantage is item-own durations, which need binding. So binding is the suspect, although all
    54 slots are written.
  - **Measured work:** fit 6.06 MFLOP/event; inference 2.04 dense, 1.50 sparse.
  - **Error analysis queued on AWS**, where the checkpoint is: `aws_fas_v2_dev_C1_diagnostics_20261007T0010Z`
    (binding purity, time-rescaling fit, AUROC by purity) and `aws_fas_v2_dev_C1_particles_20261007T0010Z`.
  - **Round-2 candidates, chosen by those results:**
    - if purity is low: binding aids (no-context readout C-variant, longer training up to the 3-pass cap, more slots);
    - if purity is high but timing weak: readout duration laws (more step classes, type-conditional durations).
- 7 Oct 00:34 UTC: **C2 completed on AWS** (previous native head, pool 8; 0.57 h). v2 validation at merged 1,024:
  total .663, type .688, gap .580. That is above C1 on every rule, while C1's validation NLL is far lower (0.338 vs
  2.655).
  - **Diagnosis:** 36% of merged gaps and 29% of item-own durations are exactly 0 ms. C1's continuous own-duration
    density earns up to ~+10 nats per tie (THEORY §439), so its likelihood rewards ties, not fault-relevant durations.
  - **Fix:** the recording-cell likelihood (`--cell-ms 1`; contracts 17/17: small-cell limit, discrete
    normalisation, bounded ties).
  - **Round 2:** C6 = C1 + cell 1 ms, queued on AWS (`aws_fas_v2_dev_C6_20261007T0045Z`) and as a curie copy
    (first in curie_chain49; only one copy runs). C3–C5, not yet run, also get the cell likelihood, so each ablation
    isolates its own change. The curie copies of C1/C2 are dropped (run on AWS).
  - **Status:** in development. Best so far C2 .663 (total) vs reference order3 .685.
- 7 Oct 01:31 UTC: **per-fault analysis at N*** (v2 validation; from the result files):

  | AUROC | wear-and-tear | retry delay |
  |---|---|---|
  | oracle (line-aware max-step) | .749 | **.890** |
  | order3 | .646 | .723 |
  | gap_z | .658 | .679 |
  | C2 total / type | .626 / .645 | .699 / .730 |
  | C1 total / type | .607 / .629 | .652 / .716 |

  - The oracle's lead is largest on retry delay (+.167 over order3). Retries add waiting inside a specific item's
    bowl-feeder steps, visible only through item-own durations (binding).
  - Both native models score better on the type rule than on the total rule, so their timing likelihood currently adds
    noise. That fits tie-dominated, unbound merged timing.
  - **Round-2 success signals for C6:** retry-delay AUROC rising toward .89, the total rule exceeding the type rule,
    and the wear-and-tear AUROC passing gap_z's .658.
- 7 Oct 05:31 UTC: **AWS admitted round 2** (slot 2, after its queued B1/B2 protocols): C6 → C1 diagnostics → C1
  particles. The AWS owner points to B1's remedy for the same tie pathology (RACE_OF_DELAYED_CLOCKS_20261006.md §3,
  race_tpp_v16.py): hold hazards at their one-cell value below the recording cell, and dequantise target gaps within
  their cell.
  - C6's recording-cell probability F(τ + δ) − F(τ) is the exact discretised likelihood. Dequantised training
    estimates it stochastically (by Jensen, E_U log f(τ + U) ≤ log of the cell average), and the held hazard bounds
    the same spike.
  - Both remove the unbounded tie reward, so C6 is unchanged. If C6 helps, B3 and B1 share one principle in two exact
    forms.
- 7 Oct 09:17 UTC: **scoring theory to practice.**
  - The per-step slowdown GLR (THEORY §440.3) beats mean NLL on toy step slowdowns by +.08 to +.10 AUROC on subtle
    faults and stays near it on gross ones.
  - It is declared a priori as the native primary rule (protocol amendment, before any FAS evaluation).
  - `glr_eval.py` evaluates checkpoints; AWS queues are ready for C1 (now) and C6 (after it completes).
  - Expected on FAS: glr_max exceeds the total rule, with the largest gain on retry delay, where the oracle's lead is
    largest.
- 7 Oct 22:10 UTC: **B3 ownership moves to the curie host session** (user direction; the earlier curie FAS session is not
  returning). **Round 2 results** (AWS, v2 validation, merged N* = 1,024):

  | AUROC at N* | total | type | glr_max (declared primary) |
  |---|---|---|---|
  | C6 = C1 + 1 ms recording cell | **.664** | .682 | .578 |
  | C1 | .621 | .675 | .466 |
  | C2 (round 1) | .663 | .688 | — |

  References: order3 .685, line-aware oracle .821. C1 particles (1 / 4 / 16): .634 / .639 / .636.
  - **The recording cell helps** C1's likelihood rule (.621 → .664) but not past order3.
  - **glr_max fails on FAS** (.578, .466), against its toy evidence. Its "own durations" need correct item attribution,
    which C1/C6 lack (below). The rule was declared a priori for the sealed test; the declaration stands in the record,
    and the primary-rule question is reopened before any test scoring (protocol amendment to follow, disclosed).
  - **Binding is the bottleneck.** C1 diagnostics: binding purity α = 0.30 (median), item concentration 0.12; by
    faulty-run binding at N*: **high purity .735, low purity .511**. Where the model binds, it already beats order3;
    where it does not, it is at chance. More particles do not help (inference over a poor binding model).
  - **Round 3 plan (curie owner):** carry the R1 gate-1 mechanism (experiments/R1_RECALL.md): the predecessor message.
    In R1 binding failed entirely until each written key carried the predecessor's message explicitly (21.6% → 97–99%,
    3 seeds; flat over age and irregular gaps), and local race credit learned it without backpropagation (77%). In FAS
    the in-line predecessor is not the previous merged event, so the read must find it by key: each event writes a key
    from (state, type); the next event queries for its in-line predecessor; the matched predecessor's message enters
    the event's write. Second: sufficient-statistic memories per event-type pair (containment of order3/timed_ngram/
    gap_z, as in the P19 win). Status: **in development, best .664 (C6 total) vs reference .685 (order3)**.
- 7 Oct 22:40 UTC: **Round 3 queued on curie** (validation only). `experiments/fas/race_tpp_fas.py`: the B1 race model
  (race_tpp_v16: all five EasyTPP datasets won; hazards held at their one-cell value below 1 ms and target
  dequantization, the remedy for the 36% zero-ms gaps) on FAS v2, scored with native.py's rules and prefixes. C7 = race
  model; C8 = C7 + keyed predecessor-message read (R1). Substitution rationale: the native heads' binding failure
  (purity .30) is the measured limit; the B1 model retains the race of delayed clocks, exact survival, temporal memory
  and addressed slots, and adds separate keys with sparse writes. Protocol cap: 3 passes over 5,000 clean samples
  (≈21 s per 32-run batch, ≈3 h per arm). Functional test: finite per-position scores, ~50k parameters.
  **Sufficient-statistic memories:** whenever used, the small LSTM/Transformer references get the same statistics in one
  arm, so a win does not rest on features the references lacked.
- 8 Oct 00:15 UTC: **C7 (B1 race model on FAS v2, validation)**: total **.684**, type .693 at N*; wear and tear .645 /
  retry delay .721 (total); validation-clean NLL 1.220 (C6 1.366). Above every native configuration (best total C6 .664)
  and level with order3 (.685; per fault .646 / .723). The per-fault profile matches order3's, so C7 has not yet added
  information the order statistics lack; binding (oracle .821, retry delay .890) remains the gap. C8 (keyed predecessor
  read) running; C9 (queried in-line predecessor) after the P19 complementarity jobs. **Status: in development, best .684
  (C7 total) vs reference .685 (order3).**
