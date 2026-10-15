# Credit horizon bounds learned memory; addressed reads with read-time credit remove the bound

15 October 2026, Curie. R1. Derivation plus the measurements that test it. Read before redesigning language memory.

## 1. Truncated credit does not reward retention beyond the window

Let a recurrent learner update state S_t = U(S_{t-1}, x_t; θ) and predict ℓ_t = −log p(y_t | S_t; θ). Training with a
state-gradient window W uses

    g_W = Σ_t Σ_{s=t−W+1}^{t} (∂ℓ_t/∂S_t)(Π_{u=s+1}^{t} ∂S_u/∂S_{u−1}) ∂U(S_{s−1}, x_s; θ)/∂θ,

with S_{t−W} treated as a constant. Every term that would reward the write at s < t−W for making S carry information
needed at t is absent. Information older than W survives only as a by-product of features that pay off within W
(slowly decaying modes, frequency statistics). The objective never asks the learner to store, protect or address it.

**Prediction P1.** For a learner trained with window W, the cost of resetting all state every N > W tokens is small
and does not grow with more data, because no gradient ever pays for that reliance.

**Test.** The P24 flagship trains with W = 16. Frozen resets (AWS, three seeds) cost 0.004–0.008 nats at N = 64 and
0.033–0.045 at N = 16 at both 512k and 8M targets. The value of history older than 64 tokens did not grow with 16x
data (0.006 → 0.004). Confirmed.

Persistent numerical state is therefore not persistent useful memory. This also explains two other completed results:
erasing a grown stage's memory changes NLL by 0.0002 (nothing was trained to use it), and raw persistence across
documents loses to EOS reset (state retained without learned addressing is noise to later documents).

## 2. Opportunity measured on held-out data

For log loss, the best achievable improvement from information in the window beyond what the predictor uses is at least
the held-out gain of any causal predictor that adds that information. Mixing the frozen flagship with fixed causal
in-window caches (recency, one-token and two-token induction; weights cross-fitted on alternating windows) gives
**5.978388 → 5.684537 (0.294 nats)**, concentrated on rare tokens (+0.59/+0.84/+1.19 nats for ranks 2k–10k/10k–30k/30k+).
51% of targets already occurred earlier in their 1024-token window. At least 0.294 nats are available from addressed
recall alone, more than three times the gain of the last 4M training targets.

## 3. Read-time credit makes the credit horizon capacity-limited

Store entry j = (k_j = f_j, v_j) once and score it at query t through the current maps:
s_tj = (W_q f_t)·(W_k sg(f_j))/√d − a(t−j) + b·[boundary] + c·[exact match]. Then

    ∂ℓ_t/∂W_k = Σ_{j ∈ top-k(t)} (∂ℓ_t/∂s_tj) (W_q f_t) sg(f_j)^T /√d,

and similarly for W_q, the clock a, the boundary bias b and the match bonuses c, for every stored j regardless of t−j.
The learning signal for addressing at lag L is non-zero for all L up to the store capacity C. The horizon is set by C,
not by W, and no gradient flows through the intervening state evolution.

What read-time credit does NOT provide: it does not train the writer to put lag-L information into f_j when j lies
outside the current chunk. The stored content must already be retrievable. This is the reason for content that exists
by construction: token identity and the predecessor message (the token before j). The keyed predecessor read
(theory/156_predecessor_message_binding.md) and the exact-match bonuses of the episodic store
(sleeping_machines/episodic_pointer_memory.py) both give the read an address that requires no long-range write credit.

**Prediction P2.** A member with an identity-addressed read relies on long history far more than the P24 flagship.

**Test.** The keyed temporal-memory member (4M tokens) scored on the flagship's protocol: resetting every 64 tokens
costs **0.117** nats (flagship 0.004–0.008), every 16 tokens **0.290** (flagship ~0.04). It scores 5.624/5.601 (two seeds)
against the flagship's 5.978 on identical targets. Confirmed.

**Prediction P3.** The keyed member trains on 256-token lanes with full backpropagation (W = 256). By P1 its reliance on
history older than 256 tokens should be small, and training on 1024-token lanes should raise it. Measured now:
0.019 nats for history older than 256 tokens. The queued 16.8M fresh-token 1024-lane fit
(`curie_r1g2_keyed1_16M_fresh_l1024_s0_20261015_v1`) followed by the same reset intervention tests the increase.

**Prediction P4.** Adding the episodic store with read-time credit to the P24 flagship improves it on the scored
protocol by an amount comparable to the fixed-cache opportunity, and raises its history reliance at N = 64.

**Test.** After 2M fresh targets the episodic arm scores **5.633532** against the 5.978388 parent (the same weights
through the native readout alone: 5.973390), beyond the fixed cache's 5.684537, with 6,248 memory parameters. Frozen
resets of state and store cost **0.227** nats at N = 64, 0.080 at N = 256 and 0.397 at N = 16
(`results/diagnostics/curie_history_reliance_episodic_2m_20261015_v1.json`): 30–50x the flagship's reliance on history
older than 64 tokens and about twice the keyed member's. Confirmed. The identical-stream control arm (plain continuation,
same 2M targets) scores 6.107894 and its history value is −0.0008 nats at N = 64 and −0.0038 at N = 256
(`curie_history_reliance_control_2m_20261015_v1.json`): added data without addressed reads yields no long-range use, as P1
predicts. The control also drifts worse than its parent on this stream (eager and compiled repeats agree,
`curie_control_drift_check_20261015_v1.json`), so the memory's quality effect is quoted against the parent (+0.345).

**Learned document boundary.** The store's boundary bias, initialized at 0, learns **−4.39**: entries from before the
latest end-of-document marker are discounted by a factor of about e^4.4 ≈ 80 against same-document entries. The model
learns the soft document reset that the forced-reset policy imposed, while keeping persistence available where the
query match warrants it. This resolves the memory-policy result (forced reset 5.978 vs raw persistence 5.996): persistence
fails without learned addressing and succeeds with it.

## 4. Consequences for the architecture

- Separate three things in every language member: persistent numerical state, the credit window, and the addressing
  path that lets later predictions use earlier events. Only the third turns persistence into capability.
- Depth on top of a window-limited member does not create long-range memory (P1 applies per stage; §5). The grown-stage
  and D8/D16 results are consistent with this.
- The family's mechanisms already include the remedy: sparse addressed writes, separate keys and values, temporal
  decay clocks and races. The keyed member and the episodic store are two implementations; the next integrated model
  should carry identity-addressed reads into the race core rather than scale a window-limited core.

## 5. Depth pays when each stage can reach information the stages below did not use

Let stage d read only the stream below it, x_{d-1}, over its credit window W, plus persistent state trained within W.
Then every stage's output is a function of the same information I_W(t) = (x_{t-W+1..t}, state written under W-credit).
By data processing, the loss of a readout over stages 1..D is bounded below by H(X_{t+1} | I_W(t)) for every D. Depth can
only improve the function approximation of that conditional; it adds no information. Once the lower stages capture the
predictable W-window structure the stream width allows, the unique value of stage d falls towards zero with d, and the
extra stages cost optimization noise (here: hard routes with sampled credit) without paying it back.

An addressed read with read-time credit (§3) changes the bound: stage d's read adds I(X_{t+1}; x_{<t-W} | I_W(t), lower
reads) at a credit path of length O(depth) to any stored event. Stacked reads compose: a read whose query is built from the
result of a lower read implements two-step retrieval (find the earlier occurrence of the current token, then read what
followed it), which no single read and no W-limited stack can learn. Depth then multiplies the information each stage can
fetch instead of re-processing the same window.

**Prediction P5a.** In a W-limited race stack, the unique value of stage d (loss increase when its readout input is
mean-replaced) falls with d to zero, and adding stages does not improve the long-history score.

**Test.** 1M-target ladder (`results/diagnostics/curie_depth_usage_probe_d8_20261015_v1.json`): D8 serial uses stage 0
only (+0.763 nats; stages 2-7 <= 0); D8 residual +0.029, +0.025, +0.017, +0.012, +0.005, +0.002, +0.001, -0.001. Long-history
D2 residual 6.556, D8 residual 6.590, D16 residual 6.690, D8 serial 6.683; D16 residual uses stages 0-2 (+0.032, +0.020,
+0.010) and its stages 4-15 are slightly harmful (`curie_depth_usage_probe_d16_20261015_v1.json`). Confirmed; residual composition delays the fall-off (five working
stages instead of one) without changing the bound.

**Prediction P5b.** Per-stage addressed reads make deep stages' unique value positive and the depth series monotone
(D16 < D8 < D2) on the same budget. Test: the read arms of the ladder (`curie_depth_ladder_1m_decision_20261015_v1`,
`curie_depth_usage_probe_ladder_20261015_v1`). Result: first half confirmed, second half refuted at 1M targets. All eight D8
stages carry positive unique value with reads (+0.106 … +0.004; residual D8 had stages 5-7 idle), and reads improve D2 by
0.084 and D8 by 0.027, but D8 + reads (6.563) stays behind D2 + reads (6.472), and D16 + reads diverged through a
persistent-state feedback loop. New information per stage is necessary for depth; at this budget the deep stack's
stages each add less than a shallow stack gains by spending the same data on two stages.

**Prediction P5c.** A member whose layers all have full-lane credit and a keyed read gains from depth per fresh token.
Test: one-pass 4M deep keyed L2 vs L8 (`curie_r1_deep_keyed_stream_L{2,8}_4M_s0_20261015_v1`): L8 below L2 on the
last-quarter prequential loss and on the flagship protocol. Supporting evidence in the multi-epoch regime (AWS plan-B ladder, same blocks, 1M
tokens, early stopping, 65,528-target slice): L2 5.957, L4 **5.899** (`aws_r1_deep_keyed_L{2,4}_1M_s0_20261015_v3.json`);
depth pays once every layer has full-lane credit and its own addressed read, where the race stack's did not. The race
stack's composition errors and the part-whole construction that follows from the data are in PART_WHOLE_COMPOSITION.md.
