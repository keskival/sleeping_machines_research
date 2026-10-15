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
Pass: D16res < D8res < D2res and D8res < D8serial. If residual composition alone does not make depth monotone, the
next construction adds per-stage addressed episodic reads (each stage queries its own store with read-time credit), so
every stage gets lag-independent credit paths: path length from a loss to any earlier event becomes O(depth) instead of
O(lag), the property that lets attention stacks train deep.
