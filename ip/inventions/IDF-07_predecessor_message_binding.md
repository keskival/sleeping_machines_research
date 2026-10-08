# IDF-07 — Keyed event memory whose written keys carry the predecessor's message, and its local three-factor learning

CONFIDENTIAL · Invention disclosure for counsel · Status: **unpublished** (all first commits after the 3 Oct 2026 public
boundary) · Jurisdictions: **EP and US open**, provided no other disclosure occurred.

First commits (UTC): keyed mark memory `48372867` 7 Oct 13:10 (`experiments/tpp/recall_tpp.py`); local race-credit
variant `8b038f4e` 7 Oct 13:26 (`recall_tpp_v2.py`); predecessor message `15306a57` 7 Oct 14:05 (`recall_tpp_v3.py`);
mixed-length training `722cd2a0` 7 Oct 15:05 (`recall_tpp_v5.py`); token-language embodiment `6aeabc1c` 7 Oct 17:37
(`experiments/r1_token_keyed_lm.py`); queried in-line predecessor `c86787be` 7 Oct 22:35 (`experiments/fas/race_tpp_fas_v2.py`);
per-type-pair duration laws in the predecessor score `cb1ffa3b` 8 Oct 07:44 (`race_tpp_fas_v3.py`).

## 1. Technical field and problem

Sequence and event processors that must answer "what followed the last occurrence of this?" (associative recall,
induction) over irregularly timed events or tokens: language models, event-log analytics, monitoring. Binding requires
two relations, a match and a successor. A decaying recurrent state carries the predecessor only inside a time-weighted
sum (measured: linearly decodable at 3.9% against 3.1% chance), so a read whose keys describe only each event's own state
cannot learn binding (21.6% recall, set baseline 14.1%).

## 2. Solution

(a) **Predecessor-message keys.** Each event (token) j is written only to the addressed slot of its own type; the written
key is k_j = W_k[h_j ; e(x_{j−1})], the event's state concatenated with the message of the event before it. Slots decay
with elapsed time. Each event issues a query q_t = W_q h_t; every type's score in the next-event race gains
Σ_{j: x_j = v} decay(t − t_j) q_t·k_j. One read then answers the two-hop relation.
(b) **Local three-factor learning of the read.** The state and messages entering the keyed memory are detached; the key
and query maps learn only from the race's error at the read, (1[k = target] − p_k) for every type (losing types
credited by their probability), multiplied by elapsed-time-decayed slot traces. No gradient reaches the network.
(c) **Length robustness:** unit-normalized match with learned sharpness is avoided for extrapolation, or training uses
mixed context lengths, one length per batch.
(d) **Interleaved processes (queried predecessor):** when several processes write into one log, each event attends over
its W most recent events with a learned duration law per (event type, candidate type),
−(log(1 + Δ/s) − μ_ab)² / (2σ_ab²) − log σ_ab, plus type compatibility; the soft predecessor's message and soft own
duration enter the event's input. This is de-interleaving learned from the merged likelihood only, without identities.

## 3. Technical effects (measured; experiments/R1_RECALL.md, experiments/fas/B3_DEVELOPMENT_LOG.md)

- Associative recall with irregular gaps: 97.5 ± 1.7% (3 seeds) vs 21.6% without the predecessor message; flat across
  pair age and elapsed time; mixed-length training 91.6% at twice the longest training length.
- Local three-factor learning only: 77.4 ± 5.4% (3 seeds), no backpropagation into the network (hardware-local learning).
- GPT-2-tokenized FineWeb: 6.009 ± 0.010 nats/token (3 seeds) vs 6.193 without the read and 6.537 for a Kneser–Ney
  trigram at 1M tokens; 5.521 vs 6.100 at 4M tokens.
- FAS v2 (interleaved logs): (d) with per-pair duration laws gives the best early likelihood (1.193 after one pass vs
  1.340 without them); detection result pending.

## 4. Prior art to distinguish

Transformer induction heads (previous-token head + induction head across two attention layers; Elhage et al. 2021,
Olsson et al. 2022); pointer/copy and neural cache models (Grave et al. 2017); linear attention and fast weights; MQAR
studies (Arora et al. 2023). Distinguish: the binding is a single event write into a type-addressed, time-decaying slot
inside a race-of-clocks event model; the read enters the race's mark law; local three-factor learning with time-decayed
traces; per-type-pair duration laws for de-interleaving. Our public material: statistic-valued race memory (generic, IDF-06f)
and key/value separation (IDF-06d).

## 5. Draft claim directions (for counsel)

1. A computer-implemented event/sequence processor writing, for each event, a key formed from the event's state and the
   preceding event's message into a memory slot addressed by the event's type, decaying with elapsed time, and scoring
   next-event types by query–slot match within a race of clocks.
2. Training such a read by a local rule using only the race error at the read and per-slot elapsed-time-decayed traces.
3. De-interleaving merged event logs by attention over recent events with learned per-type-pair duration laws, feeding the
   soft predecessor's message and duration into the event model.
