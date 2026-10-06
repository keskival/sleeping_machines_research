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
