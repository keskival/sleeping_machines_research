# B10 — Crypto market event streams (owner: curie; admitted by the founder 9 Oct 2026)

Goal: show that the race-of-clocks family is the better probabilistic model of a real high-frequency trade stream, the
input to execution, market-making risk and volatility systems. No trading-edge claim is made from this battle unless a
separate cost-inclusive, multi-month protocol is pre-registered later.

## History (E17, September 2026)

Preregistered, 21 BTCUSDT days, prequential: direction accuracy on ≥ 1 bp moves race 0.593 vs online logistic regression
0.583 vs 10 s momentum 0.591; the race decided a third earlier; continual learning gave no benefit (−0.2 [−0.6, +0.1]);
every learner lost money after a 2 bp cost. Details: experiments/FINDINGS.md (E17), drivers e17_market.py, e80/e84.

## Current development evidence (10 October)

One-seed matched chronological DEV comparison, 801 TRAIN windows and 756 VAL windows, 1,024 events/window and 128-event context; all models use the same 1 µs recording-cell score. Poisson completes at **−13.70366 nats/event** (6 parameters, 5.0 s); Hawkes completes at **−9.11712** (80 parameters, 1,615.6 s), a +4.58654 DEV likelihood gain. Native race completes at **−6.88224 nats/event** (18,164 parameters, 5,750.7 s), a +2.23487 DEV likelihood gain over Hawkes. TEST is unscored. These are development references and a first native fit, not a family verdict or complete-work comparison; fitting arithmetic/inference work are unmeasured in this driver. The currently admitted native driver uses the v5 temporal/mark-memory core; the richer v19 mechanisms remain an integration priority if this diagnostic exposes missing coverage.

## Frozen protocol (pre-registered 9 Oct 2026, before any B10 fit)

- **Data:** Binance spot BTCUSDT aggregated trades (data.binance.vision daily files; every file's SHA-256 checked against
  the published `.CHECKSUM`). One aggTrade is one event: timestamp (µs), price, quantity, aggressor side.
  About 1.6M events per day (25 Aug 2026).
- **Marks (K = 6):** aggressor side (buy / sell) × price move relative to the previous trade (up / unchanged / down).
- **Splits (chronological):** train 1–24 Aug 2026; validation 25–31 Aug; **sealed test 1–7 Sep 2026**, scored once per
  model after development is frozen.
- **Primary metric:** test log-likelihood per event (time + mark, nats). **Amended 9 Oct 17:45, before any fit:** each
  day is cut into consecutive windows of 1,024 events; the first 128 events of a window are context only and the other
  896 are scored, each from the events before it in its window; every model is scored on exactly the same windows and
  events (full-day streaming with unbounded history was intractable for the encoder and would favour models with longer
  context). Timestamps are recorded on a 1 µs grid and many consecutive aggTrades
  share a timestamp (one taker order walking the book), so **every model's time term is the interval likelihood of the
  1 µs recording cell containing the gap**, S(g) − S(g + 1 µs), not a density at g (the resolution principle of our
  EasyTPP work; a point density at a zero gap is unbounded and would reward grid exploitation); the mark term is the mark
  law at the cell's midpoint, g + 0.5 µs, for every model (specified 9 Oct 17:55, before any fit).
- **Secondary metrics (reported beside, never replacing, the primary):** (a) 60 s realised-variance forecasts issued every
  minute, QLIKE loss, vs EWMA and HAR-style regressions of past realised variance; (b) direction of the 10 s mid move on
  ≥ 1 bp moves (E17's metric) vs online logistic regression and momentum; (c) P&L after a 2 bp cost of the direction
  signal (reported, not claimed).
- **References (fitted on the same training days, selected on validation):** homogeneous Poisson per mark; multivariate
  Hawkes process with exponential kernels (6 × 6 excitation, two time constants, maximum likelihood); E17's baselines for
  the secondary metrics. Classical, information-matched references; no new neural external architecture is trained.
- **Ours:** the race-of-clocks TPP; development starts from the frozen unified EasyTPP configuration (race_tpp_v19) and
  may add the B5 lessons (exact per-event state, multi-timescale rotation clocks, addressed mark memory).
- **Online learning arm:** the selected model streamed over the test days frozen vs with prequential weight updates (the
  corrected E17 protocol: tracking variants start after development, never during the pilot).
- **Win rule:** mean test log-likelihood per event above Hawkes and Poisson with every one of the 7 test days ahead
  (paired by day), three seeds; secondary realised-variance QLIKE better than EWMA and HAR on at least 5 of 7 days.
  Reported either way.

## Development log

- 9 Oct: battle admitted; data download of 1 Aug – 7 Sep 2026 started (checksums verified per file).
- 9 Oct 17:55: data downloaded (38 days, every SHA-256 verified). Driver `b10_tpp.py` (Poisson, multivariate Hawkes with
  two exponential kernels, race_tpp_v5 race) with the common interval likelihood; contract: the Hawkes closed-form
  compensator and interval likelihood equal brute-force numerical integration to 1e-6 nats on a synthetic sequence with
  tied timestamps; the race terms are finite at zero gaps. Development fits of all three queued after data preparation.
- 9 Oct 18:15: data prepared (38 days; `results/market/b10_data_summary.json`): 185,782–2,497,086 events per day
  (mean 793,629); **21%–56% of consecutive aggTrades share a timestamp** (taker orders sweeping
  the book), which the recording-cell likelihood scores correctly and a density would not; marks are dominated by
  side-consistent moves (buy-up, buy-unchanged, sell-unchanged, sell-down; opposite moves ≈ 0.3–0.4%).


**Current diagnostic pipeline (9 Oct):** once all three admitted development fits complete, `analyze_b10_dev.py` replays their saved weights on the exact validation windows. It requires the saved validation likelihood to reproduce, then reports time/mark terms, gap groups (zero, positive <10 µs, 10 µs–1 ms, 1–100 ms, ≥100 ms), mark/side transitions and each validation day. `check_b10_components.py` first checks parity with the original likelihood and causal independence of past scores from a future mark. Fits and TEST scoring are unchanged. The analysis labels these first-pass development results and determines the next improvement plan; it produces no benchmark win/loss verdict. Completed on 10 October; see the measured diagnosis below.

### Completed development diagnosis (10 October)

Saved-weight replay reproduces the native DEV likelihood within 8.9e-16 on 677,376 targets. Native improves over Hawkes by +2.23487 nats/event: +1.89489 timing and +0.33999 marks. On zero-gap cells (323,867 targets), the total gain is +3.35809. On positive gaps below 10 microseconds (3,907 targets), native loses 1.12682 nats/event: timing loses 1.73764 while marks gain 0.61082. Side-changing events (101,163 targets) gain 1.46137. Evidence: `results/market/curie_b10_diagnostics_20261009T210711Z_analysis.json`; same DEV windows and recording-cell protocol, one seed, TEST unscored.

The next clock-design hypothesis is to separate recorded-zero-cell probability from the conditional positive-gap clock law. This may reduce competition between burst mass and microsecond positive-gap shape; the diagnosis does not establish that competition as the cause. Retain temporal race clocks, deep persistent event state, addressed values, keys where integrated, and silence-aware normalized scoring. Require cell-mass normalization, causal score/gradient contracts, and bounded integrated development comparison before a long fit. Measure added heads and all learning work. The zero cell is a recording category, not proof of a physical atom at exact zero.

### Native online-credit integration (12 October)

Completed first integration uses the full18164parameter saved market predictor, native hard-race critic and future-Adam teaching on the same six-type/1us-cell objective. Aug24TRAIN acquisition2048targets precedes4096Aug25DEV prequential forecasts. Online NLL6.965460 versus frozen credit6.966034, gain+0.000574; ordinary online BPTT6.965179 leads by0.000281. The zero-labelled arm has residual audits (32/128), not genuinely zero boundary credit. Scope: one reused development stream/initialization; no sealedTEST/profit or family verdict.

The exact-anchor revision retains full actor credit and trains only a prospective native adjustment. Its quality gate passes: online6.964593 versus BPTT6.964682 (+0.000089), frozen guidance6.964646 (+0.000052), genuine local-only6.964755 (+0.000162), fully frozen6.985292 (+0.020699). Native critic33232parameters; common acquisition41.459s, reuse online81.364s/BPTT24.158s, original saved-model development5750.687s shared by all arms. Prediction-quality advance, with higher learning cost. Both BPTT and frozen guidance lead online in the second half, so the whole-cohort gain is not uniform over time. Preserve every completed result and frozen protocol. See [exact future protocol](NATIVE_EXACT_FUTURE_PROTOCOL.md), [initial integration protocol](NATIVE_TEST_TIME_PROTOCOL.md), and [shared-history theory](../theory/MARKET_POLICY_CLASSES_AND_ADAPTIVE_CREDIT.md).

Development decision: extend the chronological horizon before repeating acquisitions or claiming regime adaptation. Keep the current complete actor, native critic, exact anchor and genuine local-only/frozen/online-BPTT controls. Report early/late quality and cumulative work together; a small first-cohort gain is not investor evidence for profitability or efficient full-model learning.

### Completed longer chronological stage (12 October)

**Longer real-market online credit (12 October):** Unchanged native prospective guidance on16384Aug25DEV forecasts fails the five-control quality gate. Online NLL5.740795, BPTT5.738457, frozen guidance5.742597, local-only5.738483, fully-frozen5.752864; learned gains BPTT-0.002338/frozen+0.001801nats/event. All first4096forecast losses replay within8.88e-16. Complete online continuation395.636s, BPTT137.554s, plus shared5750.687s original predictor development. One initialization/reused development stream, no sealedTEST, profitability, measured regime shift or hardware claim. The completed4096target small quality win remains scoped to that prefix; the longer result measures its persistence.

| Rule | First4096 | Second4096 | Third4096 | Last4096 |
|---|---:|---:|---:|---:|
| fully_frozen | 6.985292 | 5.729460 | 5.214713 | 5.081991 |
| bptt | 6.964682 | 5.710385 | 5.202207 | 5.076555 |
| zero | 6.964755 | 5.710347 | 5.202198 | 5.076631 |
| frozen | 6.964646 | 5.718637 | 5.210710 | 5.076393 |
| online | 6.964593 | 5.713671 | 5.203230 | 5.081687 |

### Completed optimizer-control diagnosis (12 October)

**Native market optimizer teaching (12 October, completed):** Regularizing next-window utility with 0.05 normalized deviation from the exact Adam step removes the uncontrolled amplification: online multiplier mean 1.007, range 0.977–1.050, versus unregularized mean 2.249. On 4096 prequential Aug25 development forecasts, regularized online NLL 6.964974, ordinary BPTT 6.964566, frozen guidance 6.964389, online field-only 6.965015. Online trails BPTT by 0.000408 and frozen guidance by 0.000585 nats/event; the quality gate fails. Online reuse 85.498s versus BPTT 24.239s; shared acquisition and original predictor fitting are additionally charged. Stability improved; prediction advantage did not. Full actor and native event critic retained, actual Adam parity <=2.23e-16. One initialization, development stream; sealed test untouched. Next diagnosis is whether producer-state-only counterfactual inputs expose the forecast-head responses needed by optimizer control, before further fitting.

| Rule | Regularized NLL | Reuse seconds |
|---|---:|---:|
| fully_frozen | 6.984488 | 5.047 |
| bptt | 6.964566 | 24.239 |
| zero | 6.964651 | 18.576 |
| frozen | 6.964389 | 40.476 |
| field_only | 6.965015 | 86.065 |
| online | 6.964974 | 85.498 |

Source-bound results: `curie_b10_native_optimizer_20261012_v1_fit.json` and `curie_b10_regularized_optimizer_20261012_v1_fit.json`. These separately acquired bundles compare controls within each experiment; absolute cross-experiment NLL differences do not isolate the penalty.


Completed TRAIN-only interface contract `curie_b10_forecast_response_20261012_v1_contract`: clock.bias +0.01 gives persistent-state response 0.0, predictive-clock response 306.605; state/forecast widths 88/80. This confirms a missing explicit response channel, not insufficiency of the complete existing critic information or a quality gain. Source and input hashes preserved.

### Predictive-law input comparison (12 October)

**Predictive-law-conditioned native credit (12 October, completed):** Added factual/counterfactual clock-law inputs to the same native event critic through a zero-initialized162-to16 adapter. On4096Aug25development forecasts the seven-arm quality gate fails: online NLL6.964872, ordinary BPTT6.964484, frozen guidance6.964257, online field-only6.964706, response-masked online6.964757. Online gains BPTT-0.000388/frozen-0.000615/masked-0.000115nats/event. Mean Adam multiplier1.0060; adapter parameter change0.030955. Online reuse88.396s versus BPTT25.184s; shared acquisition43.964s plus original5750.687s fitting charged to every arm. Full exact32-event actor credit retained: information/quality experiment, not backprop replacement or efficiency win. Common acquired state/capacity; masked arm tests deployment-input usefulness, not independent acquisition. One initialization/reused DEV; sealedTEST untouched.

| Rule | NLL | Reuse seconds |
|---|---:|---:|
| fully_frozen | 6.984596 | 5.134 |
| bptt | 6.964484 | 25.184 |
| zero | 6.964562 | 19.241 |
| frozen | 6.964257 | 41.958 |
| field_only | 6.964706 | 88.455 |
| response_masked | 6.964757 | 88.811 |
| online | 6.964872 | 88.396 |


**Native credit replacement budget (12 October, measured):** Same2048TRAIN acquisition and512DEV cost profile. BPTT boundary transport/proposals0.780s; native response/inference2.944s, future teacher4.015s. Even deleting all exact transport/proposal work cannot pay the current response/inference phase; fewer audits alone cannot yield a same-opportunity wall advantage. Local differentiation2.025/2.325s for BPTT/online. Arm wall3.585/10.362s includes setup/verification0.729/0.022s. Shared acquisition43.002s and original5750.687s predictor fitting are additional. Phase residuals<0.0002s; proposal parity passes. Small profile is cost diagnosis, not a new quality confirmation. Prioritize shared-leg credit reuse, sparse event/teacher admission and cheaper native execution before another replacement fit. V1 smoke timing guard stopped; v2 charges omitted setup/verification.

### Bounded credit and exact execution repair (12 October)

**Bounded native market credit (12 October, completed):** One aggregate timed receipt per32-target update plus future teaching every16updates reduces deployment overhead. On4096Aug25DEV forecasts online NLL6.964475, ordinary BPTT6.964428, frozen guidance6.964501, local-only6.964502, input-masked online6.964451. Online gains BPTT-0.000047/frozen+0.000026/masked-0.000024nats/event; seven-arm quality gate fails. Reuse online26.475s/BPTT23.197s, common acquisition23.430s plus original5750.687s predictor fitting. Exact actor credit retained; one reused DEV stream/initialization, TEST untouched. Original512-target optimistic cost gate passed0.373<0.807s. Exact joint parameter/cotangent extraction then reproduces all source/512block scores with0error, reduces BP/learned reuse wall27.1%/22.5%, and tightens the remaining transport/proposal budget to0.071s against added learned work0.425s: replacement budget fails. This is an exact implementation improvement, not a learned-credit efficiency win. Next integration targets inside-history graph work using exact local-clock eligibility plus learned cross-leg credit.

| Rule | NLL | Reuse seconds |
|---|---:|---:|
| fully_frozen | 6.984363 | 5.047 |
| bptt | 6.964428 | 23.197 |
| zero | 6.964502 | 17.469 |
| frozen | 6.964501 | 25.338 |
| field_only | 6.964509 | 26.432 |
| response_masked | 6.964451 | 26.340 |
| online | 6.964475 | 26.475 |


**Integrated market leg eligibility (12 October, contract complete):** Restored full18164-parameter actor scores and numerical forward unchanged through128TRAIN forecasts. Streaming event-local backward plus own traces reproduces addressed mark-memory and clock/mark readout gradients to2.22e-16; every parameter gradient exact at1event. Own eligibility7280floats plus separate7280float historical projection workspace, independent of history length; parameters/gradient buffers/current-frame storage additional. Cross-leg/embedding/layer remote histories omitted: complete gradient maxdifference0.227606 at32events, explicitly requiring learned/exact remote credit. CPU NumPy correctness prototype, no fit/speed/whole-model-exactness claim. Source-bound curie_b10_leg_eligibility_20261012_v1_contract; native family forward retained. Next implementation uses shared producer-leg credit to recover useful remote effects while current local/clock derivatives stay exact.


**Shared-path clock credit (13 October, contract complete):** Four fixed clock directions propagate through both native temporal layers and addressed memory. All1/8/32/128TRAIN forecast scores unchanged; raw gradient projections match full BPTT within1.67e-16. Additional352trace floats plus352previous-sensitivity snapshot floats/operator workspace; existing own/history traces and current-event derivatives additional. Remaining full-gradient maxdifference0.212687 at32events; own memory/readout gradients retained exact. NumPy correctness backend, fixed parameters per window; no learning-quality, Adam-step or speed claim. Completed source-bound curie_b10_projected_clock_20261013_v1_contract. Next integrate parameter-version resets and actual optimizer/learned residual comparisons against repaired exact baseline.


**Streaming market credit updates (13 October, completed):** Common exact2048TRAIN actor/Adam acquisition;4096 and16384Aug25DEV causal score-before-update forecasts,32event windows. At4096 projected6.963384/own6.963408/local6.963889/exactBPTT6.964450/frozen6.984690: five-arm exploratory quality gate PASS; projected gains exact+0.001066/own+0.000024nats/event. At16384 own5.737492/projected5.737697/local5.737748/exact5.738462/frozen5.752528. Own wins over exact by0.000970; projected also wins exact by0.000764 but loses own by0.000205, so projected five-arm gate FAIL. Projected helps versus own in first two quarters, hurts in last two. Longer reuse own79.712s/projected100.640s/exact62.950s; no efficiency win. Every shorter-run block agrees exactly. One restored actor/reusedDEV, sealedTEST untouched. No native learned critic yet: these wins isolate temporal credit transport. Traces reset after actual clippedAdam updates while numerical state persists; one-event smoke actual parameters/moments/scores parity exact. Best developed credit rule here is own-leg, with context-dependent remote admission next; do not promote unconditional four-direction correction over it.


**Native remote-credit admission (13 October, completed):** Native31,753parameter hard-race controller trained on delayed actualAdam utility, no BPTT imitation. On16384DEV forecasts acquired-frozen guidance5.737391 wins quality over own5.737492 by0.000101, fixed projected5.737697 by0.000306, online guidance5.737637 by0.000246. Matched savedBPTT5.738462: gain0.001071. Frozen beats all three random57of512admission schedules; random mean5.737596/sampleSD0.000069, mean gap0.000205. One trained controller/reusedDEV; activity-matched development diagnostic, not independent confirmation. Frozen admissions37/9/7/4; online123/34/56/108, online weight updates hurt. V2 repaired preceding-arm RNG shift and all4096score/admission prefixes exact; failed v1 preserved without accepted score. Selected-value execution then reproduces all16384scores/admissions exactly, frozen candidatevalues8192to4096 while keys8192and stateupdates4096unchanged. Frozen reuse105.382s (previous108.381), online110.601s, source28.284s; saved standalone own79.712s/BPTT62.950s, original5750.687s shared. No efficiency/TEST/profit claim. All currentdiagnostic actor arms compute both endpoint gradients; own/projected native counter fields retain source values if controller unused, so deployment Cactivity0for those arms. Native banks reset at day boundary, learned weights/moments transfer. Strongest current rule acquired-frozen native guidance, completed queuecurie_b10_native_selected_20261013_v1_fit; no running job. Next priorities: useful teacher horizons/calibration, pre-window work admission and deeper/longer-quality resource comparisons.

**Early native requests,13October:** Online5.737455909,own5.737491980,alwaysprojected5.737697356; frozen early5.737476598.257/512online computed banks versus512always; remote matrix subtotal0.884vs1.762GFLOPs,49.8%reduction and5.745million fewer compulsorybank scalar read/writes. Controller/history/teacher/optimizer additional work is excluded from that subtotal; no complete hardware efficiency claim. Same2048TRAINacquisition/16384reusedDEV,TESTsealed. Earlierpostwindow5.737391remains qualityleader. Sourcebound early-request/protocol/hardware results and endpoint contracts are saved. ASICtarget reporting prioritizes arithmetic,logicalaccesses and local synchronization over CPUwall. Next: candidate key-read caching parity and completework accounting.

**Executed arithmetic and bank delivery,13October:** Saved512targetprefix audited acrossTorchandNumPy: own112.939MFLOPs/alwaysremote184.578/frozen131.700/online148.441; exactscores/requests. Actor-onlyacquisition288.973MFLOPs,guided1021.878;732.905extra teaching/preparation cost. Nativecontinuation therefore costs more on thisprefix; deploymentsavings and16384qualitygains remain scoped separately. Arrayopformula coverage complete, specialfunctions/Pythonmath/randominternals unpriced,logicalbytes notphysicaltraffic. Rawactorv1 warmupboundary corrected by independently measuring/subtracting4.802MFLOPs; publishcorrectedactor_boundary result. Versionedlocalbanks acceptown/projectedeligibility packets with3actual8target Adamupdates,0parametererror,129packets each,stalerejected; globalclipping/parameterwindowboundary remain. Next reduce/amortizeguidanceteachingcost; fullylocaloptimizer needs explicitversion-awaretrace rules.

**Compact preparation retained,13October:** All16384forecasts/requestsunchanged; actor/Adammoments0error,critic8.882e-16. Executed2048TRAINprep793.655MFLOPs versus1021.878,22.33%less; guidanceextra504.682MFLOPs versus732.905,31.14%less, actorbaseline288.973unchanged.32auditedremotebanksvs43original,128clockcontexttrace onunauditedwindows; fullfactualactorBP/auditendpointgradientsremain. Before/afterdeclaredarithmeticequalconventions; explicitfrequencyproductscounted; specialfunctions/Pythonmath/randomunitsseparate. Preparationstageexecutionwin atpreservedquality, no totalshortepisode/energyclaim. Sources/resultscompact_acquisition andcompact_preparation.
