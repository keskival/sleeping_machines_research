# R1: making depth scale — residual race stacks

15 October 2026, Curie. User direction: depth is the key to efficient learning and hierarchical representations.

## What the completed deep runs show

P24 race-core member, same 8M acquisition, long-history protocol (8192 targets, 1024 windows): D2 6.160, D8 6.290,
D16 6.372, with cost rising roughly linearly in depth (D16: 84,934 s). The trained D16: route entropy 0.33 nats out of
ln 4 = 1.39 (routes collapsed), effective memory rank about 9–10 of 24 per stage, deep receiver key gradients about
1e-5. Cutting the incoming content at ANY of the 16 stages raises NLL by the identical 1.5595 nats: the stack is a strict
serial chain. In `sparse_counterfactual_layer` the stage output is only the winning receiver's transported value; the
stage input is not carried forward. Every token's information passes 16 hard-routed 48-wide stages in sequence.

## Hypotheses tested at initialization (`results/diagnostics/curie_residual_depth_contracts_20261015_v3c.json`)

| Hypothesis | Measurement | Verdict |
|---|---|---|
| Gradients vanish through the serial chain | Random cotangent on the deepest output only: layer-0 bank gradient is 1.6x (D8) and 2.2x (D16) the deepest layer's | Falsified: stage layer norms keep gradients alive at init |
| Token content is lost through the chain | Held-out linear decoding of the current token (101 classes) from each depth: flat 0.55 (D8) and 0.52–0.53 (D16) serial; residual 0.45 / 0.35 | Falsified at init: the serial chain keeps content; the residual sum adds random stage outputs |

Neither initial signal propagation nor initial content retention explains the depth failure. The trained-model evidence
(route collapse, low memory rank, tiny deep receiver gradients) places the failure in training dynamics, and the
credit-horizon argument (theory/CREDIT_HORIZON_AND_ADDRESSED_MEMORY.md) says no stage of a 16-token-credit stack can
learn long memory. A residual stream changes the learning problem rather than the initial signal: each stage learns a
correction to a stream that already carries the token and lower-stage results, instead of re-encoding all of it.

## Construction

`sleeping_machines/residual_race_episodes.py`: `x <- x + g_d * stage_d(x)` with learned per-depth, per-channel gains
`residual_gain` (D x H*P, initialized 1/sqrt(D)). Races, delays, sparse writes, transport, context and counterfactual
credit are unchanged; the stream keeps the stage output arrival (fast bypass plus delayed correction). Without
`residual_gain` the function is the original composition bit for bit (contract at D2 and D8: features, state and loss
identical). Driver `experiments/residual_depth_token_language.py` (versioned copy; `--residual`, `--residual-init`).

## Decisive test (Curie chain v6, after the episodic pair)

Depth ladder at 1M fresh targets, flagship settings, seed 6, compiled: residual D2, serial D8, residual D8, residual
D16. Scored on the long-history protocol by `analysis/score_depth_ladder.py` (`curie_depth_ladder_1m_decision_20261015_v1`).
Pass: D16res < D8res < D2res and D8res < D8serial.

Second construction, run in the same ladder (D2/D8/D16): **per-stage addressed reads**. After stage d the stream queries
a persistent per-lane store (512 entries) of earlier depth-d stream states with a sparse top-16 race (learned age
clock, document-boundary bias from the observed EOS token, null route) and adds sum_j a_j W_v f_j. Old entries are
detached; read-time credit reaches the query and value maps, clock, bias and null for any lag, and in-window entries
keep their writer gradient. Every stage gets lag-independent credit paths: the path from a loss to any earlier event
has length O(depth) rather than O(lag), the property that lets attention stacks train deep. W_v starts at zero, so the
read stack starts exactly at the residual stack. Contracts (`curie_residual_read_contracts_20261015_v1.json`) pass: no
reads/no residual equals the original driver bit for bit; zero value map equals the residual stack bit for bit;
nonzero gradients for every read parameter across credit windows; exact causality; store growth. Decision fields:
`reads_monotone`, `reads_gain_by_depth`, `depth_gain_d16_vs_d2`.

## Trained ladder: which stages carry signal (`results/diagnostics/curie_depth_usage_probe_d8_20261015_v1.json`)

`analysis/depth_usage_probe.py` freezes each arm, collects the readout input (every stage's output, concatenated) on the
long-history protocol and replaces one stage's slice at a time by its DEV mean (loss increase = that stage's unique value).

| Arm, 1M targets | Long-history | Unique value by stage (nats) |
|---|---|---|
| D2 residual | 6.556 | +0.429, +0.277 |
| D8 serial | 6.683 | +0.763, +0.019, -0.006, -0.004, -0.004, -0.003, -0.004, -0.004 |
| D8 residual | 6.590 | +0.029, +0.025, +0.017, +0.012, +0.005, +0.002, +0.001, -0.001 |

The serial chain is a one-stage model with seven idle stages: 76% of its readout-input energy is stage 0, and keeping
stages 0-3 only scores 6.678 against 6.683 for all eight. Residual composition puts five stages to work and gains 0.093
nats at D8 at equal cost, but stages 5-7 still contribute nothing and D8 stays behind D2. This is the predicted failure of
depth without addressed history: a stage whose credit spans 16 tokens and that reads only the stream below can compute
functions of what lower stages already extracted from the last 16 tokens, so its unique value falls with depth. Per-stage
reads give every stage new information (old events) with lag-independent credit; the read arms test whether that makes
the unique value of deep stages positive and the depth series monotone. The probe runs on every arm with the decision.

## Depth per fresh token in the keyed member (Curie chain v9)

`experiments/r1_deep_keyed_stream.py` trains the plan-B deep keyed member (AWS, `r1_deep_keyed_lm.py`: pre-norm residual
blocks, per-layer keyed race read, full-lane credit, exact parallel temporal memory) in one pass over 4M fresh tokens, L2
and L8. Each update's loss on new lanes is held out, so the comparison is prediction per token without the memorization
that dominates the multi-epoch 1M recipe (keyed L2 there: best epoch 2, train 5.54 vs selection 6.31). Pass: L8 below L2
on the last-quarter prequential loss and on the flagship protocol. Cost on Curie: L2 1.69 s, L8 4.56 s per 4x256-token
update (exact activation checkpointing for L8, 1.37 GB).
