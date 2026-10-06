# A race of delayed clocks as a marked point process — construction, exactness and the resolution principle

6 October 2026 · Battle B1 (EasyTPP) · Implementation `experiments/tpp/race_tpp_v{3,5,8,11,12,13}.py` · Evidence
`experiments/B1_EASYTPP.md`

## 1. Construction

After event i, a finite set of independent clocks starts. Clock m has a hazard h_m(τ | H_i) and a mark distribution
p_m(k | H_i[, τ]). The first clock to fire produces event i+1; its mark is drawn from that clock's distribution. Hence

- total intensity λ(τ) = Σ_m h_m(τ), survival S(τ) = Π_m S_m(τ) = exp(−Σ_m Λ_m(τ));
- marked intensity λ_k(τ) = Σ_m h_m(τ) p_m(k | τ);
- per-event log-likelihood log λ_k(τ) + log S(τ) — the scoring rule of the published state of the art.

This is the family's temporal race (THEORY P1, P3) used as the output law itself: the next event is the winner of a
race through learned delays, and its mark is decided by which delay won. Every losing clock enters the likelihood
through its survival factor, so credit to unrealized alternatives (P4) is exact and analytic rather than estimated;
silence until the event is supervised by the same factor.

## 2. Clock families (all exact unless marked)

| Clock | Survival S_m(τ) | Role |
|---|---|---|
| Exponential, weight w | exp(−w r τ) | positive hazard at τ = 0; keeps the race proper |
| Delayed, log-normal, fires with prob. π | (1 − π) + π S_LN(τ) | a learned delay that may stay silent (v3) |
| Logistic window [a, b], edge s, fires with prob. π | (1 − π) + π s[sp(−v) − sp(−u)]/Z | flat delay window with learned edges (v8) |
| State clock (v10+) | exp(−∫ λ_s) — Gauss–Legendre compensator | hazard follows a persistent complex memory evolving through the silence |

Logistic window: u = (τ − a)/s, v = (τ − b)/s, density ∝ σ(u) − σ(v), Z = s[sp(b/s) − sp(a/s)], with all differences
evaluated in stable log form (`log1p(−exp(·))` of softplus ratios). The state clock reads z(τ) = e^{(−r + iω)τ} z_i and
emits λ_s(τ) = softplus(Re-linear(z(τ)) + c_i) with marks softmax(U z(τ) + V h_i): marks change continuously with
elapsed time.

**Why each was needed (measured failures).**
- *Always-firing clocks cannot put most mass late.* Amazon gaps are 31% on [0.010, 0.015] and 69% on [0.70, 0.80]; in a
  race of clocks that surely fire, the earliest almost always wins. Defective clocks raised Amazon DEV LL from −0.229 to
  0.705.
- *Log-normal tiles lose on flat windows* (bound analysis: exact two-box 2.639 vs ours 2.603); hard-edged Kumaraswamy
  windows failed to optimize (edges see gradient only from events inside). Logistic windows train through every event:
  Amazon DEV 0.768, best seeds 0.797 with clustered initialization.
- *Clocks fixed at the last event cannot follow the process through a long silence.* The state clock lifted
  StackOverflow DEV time LL from −0.717 to −0.688 and produced a confirmed TEST win (−2.1444 ± 0.0037 vs S2P2 −2.163).

## 3. The resolution principle

Event times are recorded on a grid (Retweet whole seconds; StackOverflow 2⁻¹³; Taxi seconds in hours; Taobao 10⁻⁴ on
87% of gaps). A continuous-time density can gain arbitrary likelihood by concentrating mass on grid points or at an
exactly-zero gap — a property of the recording, not the process. Smooth intensity references cannot exploit it, and
neither may we. Rule: **a model may not resolve time below the data's recording cell.**

- delayed clocks: σ ≥ max(0.005, cell·e^{−μ}) (spread at the delay ≥ one cell);
- windows: width and edge scale ≥ one cell;
- state clock: rotation frequency ≤ one period per 8 cells, decay ≤ one per cell, and hazard and marks held at their
  one-cell value below one cell (compensator λ(c)·min(τ, c) + ∫_c^τ λ).

Audit (`grid_audit.py`): score DEV as recorded and with every gap dequantized inside its cell, plus a 512-node
compensator for the state clock. A grid-safe model loses ≈ 0 (floored clock model on Retweet: 0.0006). The audit caught
and withdrew three apparent leads: unfloored clocks (Retweet drop 0.71), state oscillation at one-second periods (0.23)
and a zero-gap hazard spike (0.22, entirely at the 4% same-second events). Every reported win passes it.

## 4. Mechanism accounting

Inference per event: the encoder update (temporal memory layers, addressed mark memory) plus the clock head; the state
clock adds O(n_s·d + n_s·K). Taxi uses 1/12 of S2P2's parameters and per-event work; Taobao 0.92×; StackOverflow 1.26×
(a matched-size two-layer model, 29,191 parameters, is in its own protocol). Training cost of losing-clock credit is
included in the exact survival terms — no sampled counterfactuals are needed for the output race.

## 5. Open directions

Amazon's limit is optimization across basins (seed spread ±0.03): restart selection on DEV is the current protocol;
better window parameterization or annealed edges should remove it. A grid-safe state clock on Retweet (v13) is running.
The same race-of-clocks output law is the natural head for B2's event streams and for B3's anonymous process logs.
