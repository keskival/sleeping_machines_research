# A Race of Delayed Clocks: Exact, Compact Marked Point Processes — and the Recording-Resolution Pitfall

PRIVATE DRAFT — not for distribution before the IP/disclosure decision (investment/IP_PROTECTION_PLAN.md).
Status 7 October 2026. Pending cells are marked; no prediction fills them.

## Abstract

We model the next event of a marked temporal point process as the winner of a race between independent clocks
started at the previous event: exponential clocks, delayed clocks that fire with a learned probability, flat delay
windows with learned soft edges, and a state clock whose hazard follows a persistent memory evolving through the silent
interval. Each clock carries its own mark distribution, so the mark depends on elapsed time through which clock is
likely to win. Except for the state clock, every term of the likelihood is closed-form: no Monte Carlo compensator is
needed, and every losing clock receives exact credit through its survival factor. On the EasyTPP benchmark with its
official splits and the per-event log-likelihood protocol of the current state of the art, the model beats the best
published result on Taobao (1.399 ± 0.003 vs 1.318 nats/event), Taxi (0.525 ± 0.001 vs 0.522, with 1/12 of the
leader's parameters and per-event compute; a five-model mixture reaches 0.536 at 0.41× its compute) and StackOverflow
(−2.144 ± 0.004 vs −2.163), five seeds each. We also show that continuous-time models can gain large spurious
likelihood on benchmarks whose timestamps are recorded on a grid — by concentrating density on grid points or at exactly
zero gaps — and give a resolution principle and a dequantization audit that rule this out. All reported results pass
the audit and are unchanged under the benchmark's own Monte Carlo estimator.

## 1. Introduction

- Marked temporal point processes (MTPPs) model when events happen and what they are; applications span healthcare,
  commerce, mobility and online platforms. EasyTPP (Xue et al., ICLR 2024) standardized data and evaluation; S2P2
  (Chang et al., NeurIPS 2025), a deep continuous-time state-space model, leads three of its five datasets.
- Intensity models (NHP, THP, SAHP, AttNHP, S2P2) need a Monte Carlo or quadrature compensator; intensity-free mixtures
  (IFTPP) give exact time densities but time-independent marks.
- Contribution 1 — **race of delayed clocks**: an exact, compact MTPP head with time-dependent marks; clock families
  motivated by measured failures (§3).
- Contribution 2 — **the recording-resolution pitfall** and its remedy (§4).
- Contribution 3 — results on EasyTPP at a fraction of the leader's compute, with estimator parity (§5–6).

## 2. Background and related work

NHP (continuous-time LSTM), THP/SAHP/AttNHP (attention), IFTPP (log-normal mixture), RMTPP, ODE-TPP, S2P2 (stacked
latent linear Hawkes / deep SSM layers with parallel scans), HHP (hypernetwork Hawkes). Competing-risks views of point
processes; cure (defective) models in survival analysis; dequantization in density modelling of discretized data.

## 3. Model

### 3.1 The race

After event i, clocks m = 1..M start with hazards h_m(τ | H_i) and mark distributions p_m(k | H_i [, τ]). The next event is
the first to fire: λ(τ) = Σ_m h_m(τ), S(τ) = Π_m S_m(τ), λ_k(τ) = Σ_m h_m(τ) p_m(k | τ); per-event log-likelihood
log λ_k(τ) + log S(τ).

### 3.2 Clock families

| Clock | Survival | Why (measured) |
|---|---|---|
| Exponential, weight w | exp(−w r τ) | positive hazard at τ = 0; keeps the race proper |
| Delayed log-normal, fires with prob. π | (1 − π) + π S_LN(τ) | Amazon gaps: 31% on [0.010, 0.015], 69% on [0.70, 0.80]; always-firing clocks cannot put most mass late — defective clocks raised DEV LL from −0.229 to 0.705 |
| Logistic window [a, b], edge s, prob. π | (1 − π) + π · s[sp(−v) − sp(−u)] / Z | flat windows train through every event (hard-edged windows did not); Amazon DEV 0.768 |
| State clock | exp(−∫ λ_s), Gauss–Legendre | hazard and marks follow a memory evolving through the silence; StackOverflow DEV time LL −0.717 → −0.688 |

### 3.3 Encoder

Persistent complex-diagonal memories that decay and rotate with the real elapsed time (two layers, 16 modes, width 32),
messages mixing event content with memory, and an addressed per-mark memory (one slot per mark, written only when that
mark occurs). Clock parameters are read from the state after each event.

### 3.4 Training

Adam(W), early selection on DEV; restart selection on DEV where the optimizer has distinct basins (Amazon; 3× training
compute reported).

## 4. The recording-resolution pitfall

Event times in EasyTPP are recorded on grids: Retweet whole seconds, Taxi seconds in hours, StackOverflow 2⁻¹³, Taobao
10⁻⁴ on 87% of gaps. A flexible density can place mass on grid points or at exactly zero gaps and gain likelihood that
reflects the recording rather than the process. Audit: score DEV as recorded and with every gap dequantized uniformly
inside its cell; a grid-safe model loses ≈ 0.

| Retweet model | DEV as recorded | DEV dequantized | Drop |
|---|---|---|---|
| unconstrained delayed clocks | −5.919 | −6.632 | 0.71 |
| state clock with rotation | −6.252 | −6.485 | 0.23 |
| state clock, rate/frequency caps | −6.246 | −6.470 | 0.22 (entirely at the 4% zero gaps) |
| floored delayed clocks (grid-safe) | −6.393 | −6.394 | 0.0006 |

Resolution principle: no clock resolves time below one recording cell — delayed-clock spread at its delay ≥ one cell;
window width and edges ≥ one cell; state-clock frequencies ≤ one period per eight cells, decay ≤ one per cell; hazards
held at their one-cell value below one cell. Smooth intensity models cannot exploit the grid; neither may ours.
Recommendation for benchmark maintainers: report the dequantized score alongside the recorded one.

## 5. Experimental protocol

HuggingFace EasyTPP releases, default splits, statistics as in S2P2/HHP; events 2..N scored; per-event LL = total over
TEST / scored events; time and mark decomposition; five seeds; TEST scored once per seed at the best-DEV checkpoint;
reporting rule (single model and five-seed mixture with compute) fixed before any five-seed TEST aggregate.

## 6. Results

| Dataset | Best published | Ours (5 seeds) | Per-event compute vs S2P2 |
|---|---|---|---|
| Taobao | 1.318 ± 0.017 (IFTPP); S2P2 1.304 | **1.3991 ± 0.0025** (time 2.753, mark −1.353) | 0.92× |
| Taxi | 0.522 ± 0.004 (S2P2) | **0.5250 ± 0.0010** (time 0.733, mark −0.208); mixture **0.5363** | 1/12; mixture 0.41× |
| StackOverflow | −2.163 ± 0.009 (S2P2) | **−2.1444 ± 0.0037** (time −0.643, mark −1.501) | 1.26× |
| Amazon | 0.781 ± 0.011 (S2P2) | *pending (restart-selected protocol: 3/5 seeds 0.7964 ± 0.0013)* | 0.28× |
| Retweet | −6.348 (NHP) | *pending (grid-safe model in development)* | — |

Estimator parity: under EasyTPP's 10-point Monte Carlo compensator the same models score Taxi 0.5244, Taobao 1.3992,
StackOverflow −2.1427 (differences within Monte Carlo spread).

## 7. Ablations (from the development log)

Always-firing → defective delayed clocks (Amazon −0.229 → 0.705); log-normal tiling → logistic windows (0.724 → 0.768);
quantile → clustered window initialization (best seeds 0.797; seed variance remains); state clock on StackOverflow
(−2.173 → −2.143 DEV); width and long decayed training did not help on these small datasets.

## 8. Limitations

Amazon optimization has distinct basins (restart selection); Retweet's grid-safe model trails NHP so far; the encoder
recurrence is sequential (its linear part is scannable); compute counts are analytic per-event multiply-accumulates.

## Reproducibility

`experiments/tpp/race_tpp_v{5,8,11,12,16}.py`, contracts (`check_race_tpp.py`, `check_state_clock*.py`,
`check_cell_held.py`), audit (`grid_audit.py`), estimator parity (`easytpp_estimator_check.py`), work accounting
(`work.py`), scoreboard (`b1_scoreboard.py`), restart selection (`b1_restart_select.py`); all result JSONs under
`experiments/results/tpp/`.
