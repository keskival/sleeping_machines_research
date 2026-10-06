# Battle B1 — EasyTPP temporal point processes

Owner: AWS host session (6 October 2026). Orders: PRODUCT_ORDERS.md. Attempt level: **in development**.

## Frozen protocol (matches the published leaderboard)

- Data: the HuggingFace EasyTPP releases `easytpp/{amazon,retweet,taxi,taobao,stackoverflow}`, default train/dev/test
  splits, used as released (`data/easytpp/`). Statistics match S2P2/HHP Table 7/4.
- Primary metric: per-event total log-likelihood in nats on TEST, as computed by the S2P2 fork of EasyTPP
  (`UCIDataLab/state_space_point_process`, commit c3933240): for every sequence, events 2..N are scored with
  log λ_{k_i}(t_i) minus the compensator ∫ λ over (t_{i-1}, t_i]; no term after the last event; the sum over the test set
  is divided by the number of scored events. Their compensator is a 10-point Monte Carlo estimate; ours is exact
  wherever the model admits it (same quantity, no estimator noise).
- Decomposition: time LL = log λ(t_i) − compensator; mark LL = log λ_{k_i}(t_i) − log λ(t_i) (mark given true time).
- Secondary: next-event time RMSE (expected next time, Mei & Eisner) and next-mark accuracy given the true time.
- Selection: early stopping and model choice on DEV only; TEST is scored once per selected configuration and seed.
  Five seeds for any reported number, matching the references.

## Published references (S2P2, NeurIPS 2025, Table 8; 5 seeds; nats per event)

| Model | Amazon | Retweet | Taxi | Taobao | StackOverflow |
|---|---:|---:|---:|---:|---:|
| RMTPP | −2.136 | −7.098 | 0.346 | 1.003 | −2.480 |
| SAHP | −2.074 | −6.708 | 0.298 | 1.168 | −2.341 |
| THP | −2.096 | −6.659 | 0.372 | 0.790 | −2.338 |
| IFTPP | 0.496 | −10.344 | 0.453 | **1.318** | −2.233 |
| MHP | −2.091 | −6.564 | 0.370 | 0.636 | −2.346 |
| NHP | 0.129 | **−6.348** | 0.514 | 1.157 | −2.241 |
| AttNHP | 0.484 | −6.499 | 0.493 | 1.259 | −2.194 |
| S2P2 | **0.781** | −6.365 | **0.522** | 1.304 | **−2.163** |
| HHP (Boyd et al., 2025) | 0.603 | −6.357 | 0.522 | 1.273 | −2.195 |

Best time LL: Amazon 2.652 (S2P2), Retweet −5.584 (NHP), Taxi 0.735 (IFTPP), Taobao 2.719 (S2P2), StackOverflow −0.641
(S2P2). Best mark LL: Amazon −1.871 (S2P2), Retweet −0.764 (NHP), Taxi −0.211 (S2P2), Taobao −1.391 (IFTPP),
StackOverflow −1.510 (AttNHP). Seed standard deviations are 0.002–0.04 on most cells (Taobao S2P2 0.039).
Model sizes are small: HHP uses 5K–24K parameters on these datasets.

Sources: [S2P2](https://arxiv.org/abs/2412.19634), [HHP](https://arxiv.org/abs/2511.01096),
[EasyTPP](https://arxiv.org/abs/2307.08097).

## Task study (`experiments/tpp/data_study.py`; counting only)

| Dataset | Structure | Simple count model vs best published mark LL |
|---|---|---|
| Amazon | Gaps sharply bimodal: ~10% near 0.011, most in 0.72–0.80 (max 0.8). Next mark in last five 61%. | prev mark + gap bin −2.099 vs −1.871 |
| Retweet | Integer seconds, 4.2% zero gaps, heavy tail (median 40 s, p99 63,000 s); log-gap autocorrelation 0.44 | −0.822 vs −0.764 |
| Taxi | Pickup/dropoff alternate: next mark never equals previous; gap informs the mark | −0.237 vs −0.211 |
| Taobao | Strong repetition: next equals previous 60% | prev 2 marks −1.484 vs −1.391 |
| StackOverflow | 22 badges, weak first-order structure; 0.5% zero gaps | −1.710 vs −1.510 |

S2P2's lead over other models comes mainly from time LL. Time densities on Amazon and Taobao are sharp and multimodal;
zero gaps on Retweet and StackOverflow require positive hazard at zero elapsed time.

## Design rationale: a race of delayed clocks over persistent temporal memory

The family's core primitive is the output distribution itself. After each event, M latent clocks start. Each clock m has a
history-dependent firing-time distribution (exponential, for hazard at zero elapsed time, or log-normal, a learned delay
with learned dispersion) and its own mark distribution. The first clock to fire produces the next event. Then
- the total intensity is the sum of the clock hazards, λ(τ) = Σ_m h_m(τ), and the survival is the product of clock
  survivals, so the log-likelihood is exact and closed-form, with no Monte Carlo compensator;
- the marked intensity is λ_k(τ) = Σ_m h_m(τ) p_m(k): the mark distribution depends on elapsed time through which clock is
  likely to be winning — learned delays and races perform the computation;
- every losing clock receives exact counterfactual credit through the survival term (it is credited for not firing):
  credit to unrealized alternatives is analytic here, and silence until the event is supervised.

History enters through the family's persistent state: stacked layers of complex-diagonal memories that decay and rotate
with the real elapsed time between events, written by small messages that mix the event's content with memory, plus an
addressed mark memory — one slot per mark, written sparsely only when that mark occurs, decaying at learned rates, and
read through separate keys and values. The addressed bank gives every mark a Hawkes-like recency trace (capacity K
slots, one write per event) without forcing it through a small shared state.

What the leading models lack and this design adds: IFTPP has a flexible gap mixture but time-independent marks; S2P2 has
continuous-time linear state but a smooth intensity built from damped oscillations, which struggles with sharp delayed
peaks; neither has addressed per-mark memory.

## Development log

| Date | Iteration | Change | DEV result | Diagnosis / next |
|---|---|---|---|---|
| 6 Oct | contract | race_tpp numerical contract | density integral 1 − 3e−12; exact compensator = quadrature; Σ_k λ_k = λ | PASS |
| 6 Oct | r1 | default race clocks (d32, 16 modes, 2 layers, 2 exp + 4 log-normal clocks, dv4), early stop | Taxi 0.475 (time 0.707, mark −0.232); StackOverflow −2.203 (−0.735, −1.468); Taobao 1.249 (2.697, −1.448) | Overfits after ~25–50 epochs. Mark LL at count-model level on Taxi: time-conditional mark resolution limited to 6 clocks. StackOverflow time LL weakest. |
| 6 Oct | calibration | count models scored on DEV and TEST (`split_calibration.py`) | TEST − DEV offsets, total: Taxi +0.10, Taobao +0.12, Retweet +0.03, Amazon −0.01, StackOverflow −0.01 | DEV numbers are read against TEST references with these offsets; verdicts only from one TEST scoring of the selected config, 5 seeds |
| 6 Oct | r1 Amazon/Retweet | same | Amazon −0.229 (time 1.651, mark −1.880); Retweet −6.403 (−5.636, −0.767) | Amazon time far behind S2P2 (2.652): see gap analysis below |
| 6 Oct | analysis | Amazon gap histogram | 31% uniform on [0.010, 0.015], 69% uniform on [0.70, 0.80]; a two-box density alone is worth ≈ 2.61 nats | A race of always-firing clocks cannot put 69% of mass on the later box: the earliest clock almost always wins |
| 6 Oct | r2 (v3) | delayed clocks fire with learned probability π (silent routes), 8 log-normal clocks | Amazon **0.705** (2.585, −1.880); Taobao **1.281** (2.733, −1.453); Taxi **0.485** (0.712, −0.227); StackOverflow **−2.171** (−0.712, −1.459) | v3 improves all four; v3 is the B1 model. Small datasets peak at 13–35 epochs |
| 6 Oct | g0, g1 | long training, no early stop: no decay (g0); d64 + decay 0.1 (g1) | g0 best 0.475 at epoch 27, DEV 0.32 by epoch 589; g1 DEV 0.18 by epoch 209 and falling | no recovery yet; decay 0.1 at lr 3e−3 is too weak; g2 (decay 1.0) and g3 (asymmetric) pending |
| 6 Oct | r3 (v4) | Taxi regularization: dropout 0.3; weight EMA 0.999; EMA + d64 + decay 1.0, 600 epochs | 0.487; 0.484; 0.453 (train ≈ DEV, no memorization, lower plateau) | Regularization alone does not move Taxi past v3 (0.485); g2 (decay 1.0) also plateaus ≈ 0.453 by epoch 809; g3 (asymmetric private decay) pending |
| 6 Oct | r4 | Amazon, 16 delayed clocks | **0.720** (time 2.602, mark −1.881) | denser clock tiling helps box-shaped gaps; d64 variant and StackOverflow arms queued; Retweet v3 queued |

**Status (6 Oct ~16:30 UTC).** Estimated against published TEST bests using count-model split offsets (not a verdict):
ahead on Taxi and Taobao (time and mark), behind on Amazon time (≈0.06 after r4) and StackOverflow time (≈0.03),
near level on Retweet. Next: finish r4 and Retweet v3; per-dataset small grid as the references did; then one TEST
scoring of each selected configuration with 5 seeds; measure inference work.
| 6 Oct | r4 | wider state (d64, 32 modes) with 16 clocks; StackOverflow 16 clocks | Amazon d64 0.661 (vs 0.720 at d32); StackOverflow 16 clocks −2.178, d64 −2.188 (vs −2.171) | wider states overfit these small datasets; capacity is not the limit |
| 6 Oct | g1–g3 | long training with decay 0.1 / 1.0 / asymmetric private ×10 | best DEV all at early epochs (0.478, 0.460, g3 0.29 by epoch 133 and falling) | on these noisy event datasets long decayed training does not recover past early selection; recipe: v3 + DEV early selection |
| 6 Oct | r2 Retweet (v3) | defective delayed clocks on Retweet | DEV **−6.089** at epoch 20 and improving (time −5.315, mark −0.774), vs v1 −6.403 | ≈0.26 nats ahead of the best published TEST (NHP −6.348) before the +0.036 offset |
| 6 Oct | final Taxi | TEST protocol queued: v3, 8 delayed clocks, dropout 0.3; 5 seeds | pending | Taxi verdict |
| 6 Oct | r5 | dropout 0.3 on Taobao / Amazon (16 clocks) / StackOverflow | pending | last DEV round; then TEST protocol per dataset |
| 6 Oct | audit | grid audit (`grid_audit.py`): DEV scored as recorded and with gaps dequantized inside their recording cell | Retweet v3: −5.919 recorded vs **−6.632 dequantized** (drop 0.71; 37.5% of delayed clocks at the 0.005 dispersion floor); Taxi drop 0.000; Taobao 0.0001; StackOverflow 0.009 | **The Retweet DEV lead was an artifact of integer-second recording and is withdrawn.** Smooth intensity references cannot spike on grid points; ours must not either |
| 6 Oct | v5 | resolution floor: every delayed clock's spread at its delay is at least one recording cell, σ ≥ max(0.005, cell·e^(−μ)); cells from TRAIN: Retweet 1 s, Taxi 1 s (1/3600 h), StackOverflow 2⁻¹³, Taobao 10⁻⁴ (87% of gaps), Amazon none | contract PASS | all final protocols use v5 |
| 6 Oct | r5 | dropout 0.3 | Taobao 1.284 (vs 1.281); Amazon 16 clocks **0.724** (vs 0.720) | selected |
| 6 Oct | final | TEST protocols queued (v5, 5 seeds): Taxi, Taobao, Amazon; Retweet v5 development run | pending | |
| 6 Oct | r7, r8 | StackOverflow under the floor: 4 exp clocks −2.190, 16 clocks −2.187; v6 (time since sequence start) queued for StackOverflow and Amazon | — | the unfloored −2.166 partly exploited the 2⁻¹³ grid; StackOverflow ~0.035 behind |
| 6 Oct | final Taxi s0 | first TEST seed | **TEST 0.5250** (time 0.732, mark −0.207) vs S2P2 0.522 ± 0.004 | count-model split offsets overestimated TEST (predicted ≈0.58); no longer used — only TEST protocols decide |
| 6 Oct | resources | `work.py`: parameters / per-event MACs | S2P2 ÷ ours: Taxi 12.3× / 12.1×, Amazon 4.1× / 4.0×, Retweet 17.6× / 17.5×, Taobao 1.1×, StackOverflow 1.1× | |
| 6 Oct | ensemble (DEV) | predictive mixture of five Taxi members (`ensemble_eval.py`) | members 0.484–0.492, mixture **0.497** | |

**Pre-registered reporting rule (fixed 6 Oct before any 5-seed TEST aggregate was seen).** Each dataset reports (1) the
single-model TEST mean ± sd over its 5 protocol seeds and (2) the uniform predictive mixture of those 5 seeds on TEST,
with per-event inference MACs beside S2P2's. The mixture is a matched-compute (inference) comparison only where 5× our
MACs ≤ S2P2's (Taxi 0.41×, Retweet 0.29×) or within 10% (near-matched); elsewhere (Amazon 1.25×, Taobao 4.5×,
StackOverflow 4.6×) it is reported as a higher-compute result, not a matched win.
| 6 Oct | **final Taxi** | v5, 8 delayed clocks, dropout 0.3, floor 1 s; 5 seeds, TEST once per seed at best DEV | single model **0.5250 ± 0.0010** (time 0.733, mark −0.208) vs S2P2 0.522 ± 0.004; **5-seed mixture 0.5363** (time **0.740**, mark **−0.204**) | **WIN, confirmed (5 seeds).** The mixture beats every published model on total, time (best published 0.735, IFTPP) and mark (best −0.211) LL at 0.41× S2P2's per-event inference MACs; the single model leads on the mean at 1/12 of S2P2's parameters and per-event MACs |
| 6 Oct | StackOverflow selection | v5 base −2.175, 3 layers −2.173, v6 −2.179 (DEV) | selected v5 3 layers | TEST protocol queued |
| 6 Oct | **final Taobao** | v5, 8 delayed clocks, dropout 0.3, floor 10⁻⁴; 5 seeds | single model **1.3991 ± 0.0025** (time **2.753**, mark **−1.353**) vs best published 1.318 ± 0.017 (IFTPP), S2P2 1.304; 5-seed mixture 1.4173 (4.5× S2P2 MACs: higher-compute result) | **WIN, confirmed (5 seeds): +0.081 nats over the best published model**, best time and mark LL, at 0.92× S2P2's per-event MACs (24,362 vs 26,016). Grid audit drop 0.0002 (Taxi 0.0) |
| 6 Oct | **B1 pass criterion met** | best published LL on ≥ 2 of 5 datasets, 5 seeds | Taxi and Taobao | Amazon (v7 windows), StackOverflow and Retweet continue |
| 6 Oct | v7 | Kumaraswamy window clocks (hard edges) on Amazon | DEV 0.677 (vs v5 0.724) | hard window edges receive gradient only from events inside the window; dropped |
| 6 Oct | v8 | **logistic-window delayed clocks**: density ∝ σ((τ−a)/s) − σ((τ−b)/s), closed-form stable survival, learned edge scale; contract density integral 1 − 6e−11 | Amazon 16 delayed + 4 window clocks DEV **0.768** (time **2.645**, mark −1.877) vs v5 0.724 | windows close most of Amazon's time-shape loss (DEV bound analysis: exact two-box 2.639 history-free, 2.661 with history counts) |
| 6 Oct | final StackOverflow (v5) | 3 layers, 8 delayed clocks, dropout 0.3, floor 2⁻¹³; 5 seeds | single model −2.1809 ± 0.0055 (time −0.674, mark −1.506) vs S2P2 −2.163 ± 0.009; **5-seed mixture −2.154** (time −0.666, mark **−1.488**) | single model behind by 0.018 (time); **mixture is a pure-accuracy win at 4.6× S2P2's per-event MACs** (not compute-matched). v8 StackOverflow queued |
| 6 Oct | final Amazon (v5) | 16 clocks, dropout 0.3; seeds 0–2 | 0.7008 ± 0.0263 vs S2P2 0.781 | superseded by v8 (DEV +0.044); v8 TEST protocol next |
| 6 Oct | Retweet r10/r13 | d64 + 16 clocks (v7); v9 cascade age (8 delayed + 4 windows) | best DEV −6.396 / −6.391 (from training logs; both runs stopped by the 3 GB RSS watchdog during the final expected-time quadrature after training) | Retweet plateaus at ≈ −6.39 across v5/v7/v9 and width; remaining time gap to NHP not addressed by capacity, age or windows. Future Retweet runs: 9 GB cap |
| 6 Oct | final Amazon v8 s0 | 16 delayed + 4 window clocks | TEST **0.7682** (time 2.622, mark **−1.854**) vs S2P2 0.781 ± 0.011 | 0.013 behind; mark beats every published model; 3.5× fewer parameters/MACs than S2P2 |
| 6 Oct | StackOverflow v8 | windows, 3 layers | DEV −2.177 (v5 −2.173) | no gain: StackOverflow's time gap is not window shape; no periodicity in its times |
| 6 Oct | v10 | **continuous-time state clock**: a persistent complex memory keeps decaying and rotating through the silent interval and drives a hazard and a mark distribution (24-node Gauss–Legendre compensator; contract: density integral 0.99999 at 24 and 48 nodes) | Retweet DEV −6.252 at epoch 2 | **audit: artifact.** Dequantized DEV −6.485 (drop 0.23; 512-node compensator −6.518): learnable rotation frequencies produced hazard peaks on integer seconds. Run stopped, result withdrawn |
| 6 Oct | v12 | resolution floor extended to dynamics: with a recording cell, rotation frequency ≤ one period per 8 cells and decay rate ≤ one per cell; v11 clustered window initialization included | contract PASS | Retweet v12 queued; every state-clock result is audited (dequantized + 512-node compensator) before it counts |
| 6 Oct | v11 | windows initialized from 1-D k-means of TRAIN log gaps | queued (Amazon seeds 0, 1) | Amazon v8 seeds diverge (TEST 0.768 vs 0.736); quantile init placed a window between the two boxes |
| 6 Oct | v10 StackOverflow | state clock (16 modes), 3 layers, 8 delayed clocks, floor 2⁻¹³ | DEV **−2.143** (time **−0.688**, mark −1.455) vs v5 −2.173 | **audit clean**: dequantized drop 0.0001; 512-node compensator identical. A real continuous-time gain; StackOverflow v12 TEST protocol queued |
| 6 Oct | B3 admission | FAS v2 data, reference smokes, native C1/C2 admitted on slot 2 (user-directed) | — | |
| 6 Oct | **final StackOverflow (v12)** | state clock (16 modes), 3 layers, 8 delayed clocks, dropout 0.3, floor 2⁻¹³; 5 seeds | single model **−2.1444 ± 0.0037** (time −0.643, mark **−1.501**) vs S2P2 −2.163 ± 0.009; 5-seed mixture **−2.1113** (time **−0.633**, mark **−1.478**) | **WIN, confirmed (5 seeds): +0.019 nats over the best published model**, best mark LL, time level with the best; 1.26× S2P2's per-event MACs (36,732 vs 29,216): pure-accuracy win; matched-size (2-layer, 29,191 parameters) DEV check queued. Audit: dequantized drop −0.0002; 512-node compensator scores higher than the protocol's 24 nodes (reported numbers are conservative) |
| 6 Oct | Amazon | v10 state clock DEV 0.741 (1 seed); v8 TEST seeds 0.768 / 0.736 / 0.687 / … | — | Amazon's limit is optimization variance across seeds; v11 clustered-window initialization on 4 seeds running |
