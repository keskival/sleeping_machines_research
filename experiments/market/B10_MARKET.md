# B10 — Crypto market event streams (owner: curie; admitted by the founder 9 Oct 2026)

Goal: show that the race-of-clocks family is the better probabilistic model of a real high-frequency trade stream, the
input to execution, market-making risk and volatility systems. No trading-edge claim is made from this battle unless a
separate cost-inclusive, multi-month protocol is pre-registered later.

## History (E17, September 2026)

Preregistered, 21 BTCUSDT days, prequential: direction accuracy on ≥ 1 bp moves race 0.593 vs online logistic regression
0.583 vs 10 s momentum 0.591; the race decided a third earlier; continual learning gave no benefit (−0.2 [−0.6, +0.1]);
every learner lost money after a 2 bp cost. Details: experiments/FINDINGS.md (E17), drivers e17_market.py, e80/e84.

## Frozen protocol (pre-registered 9 Oct 2026, before any B10 fit)

- **Data:** Binance spot BTCUSDT aggregated trades (data.binance.vision daily files; every file's SHA-256 checked against
  the published `.CHECKSUM`). One aggTrade is one event: timestamp (µs), price, quantity, aggressor side.
  About 1.6M events per day (25 Aug 2026).
- **Marks (K = 6):** aggressor side (buy / sell) × price move relative to the previous trade (up / unchanged / down).
- **Splits (chronological):** train 1–24 Aug 2026; validation 25–31 Aug; **sealed test 1–7 Sep 2026**, scored once per
  model after development is frozen.
- **Primary metric:** test log-likelihood per event (time + mark, nats), streamed over each full test day (every event
  predicted from all earlier events of that day). Timestamps are recorded on a 1 µs grid and many consecutive aggTrades
  share a timestamp (one taker order walking the book), so **every model's time term is the interval likelihood of the
  1 µs recording cell containing the gap**, S(g) − S(g + 1 µs), not a density at g (the resolution principle of our
  EasyTPP work; a point density at a zero gap is unbounded and would reward grid exploitation).
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
