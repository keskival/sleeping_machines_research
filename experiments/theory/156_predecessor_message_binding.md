# 156. Binding needs the predecessor's message: induction in an event model (§§441–444)

7 October 2026, curie host session. Evidence: experiments/R1_RECALL.md (gate 1 and gate 2), B3_DEVELOPMENT_LOG.md.

## §441 The problem: binding is a two-hop relation

Associative recall and induction ask: "the last time the current event's identity appeared, what followed it?" With
events x_1 … x_t, the answer at time t is x_{j+1} for the j < t with x_j = x_t. That is the composition of two
relations: a *match* (x_j = x_t) and a *successor* (j → j+1). A read whose keys describe only each event's own identity
can express the match but never the successor; a read whose keys describe only position can express the successor but
not the match. One of the two relations must be written into the key when the event is stored.

In a transformer this is the two-layer induction circuit: a previous-token head copies x_{j−1} into position j, and an
induction head then matches the query against those copies (Elhage et al. 2021; Olsson et al. 2022). In an event
model the natural form is a **message from the predecessor**: when event j is written to its addressed slot (the slot
of mark x_j), its key is built from its own state and the message of event j−1. The slot of x_j then answers the query
"which mark followed an occurrence of q?" in one read: score(q, slot v) = q · Σ_{j: x_j = v} decay(t − t_j) k(h_j, x_{j−1}).

## §442 Why a decaying state is not enough

The temporal memory carries the predecessor only inside a decaying sum, h_j = Σ_{i≤j} D(t_j − t_i) W x_i: the
predecessor's contribution is one term among all earlier events, with weights set by elapsed time, not by order. With
irregular gaps the predecessor can be older, in time, than several events that are its neighbours in order, so no fixed
decay isolates it. Linear decodability measures this directly. In the R1 recall task (32 keys, 8 pairs, inter-pair
gaps log-uniform over 0.05–50), the preceding key is decodable from the written state at 3.9% (chance 3.1%) in the
frozen B1 model, and at 20.7% even after the keyed read is trained. The keyed read without the message learned no
binding at all (21.6%, identical to the frozen v5 model, against a set baseline of 14.1%).

## §443 Consequences (measured)

1. **Binding becomes learnable in one read.** With the predecessor message, recall is 97.5 ± 1.7% (3 seeds), flat
   across pair age (99.2–100%) and elapsed time (0.01 to >100 units); the match alone ranks the correct value first in
   99.76% of queries.
2. **Local learning suffices.** The match is a bilinear form between a query and stored keys, and the race supplies its
   own error at the read: for each mark, (1[k = target] − p_k), crediting every losing mark by its probability. With
   keys stored as elapsed-time-decayed traces, the update needs only factors present at the slot and the read (a
   three-factor rule). Measured: 77.4 ± 5.4% recall (3 seeds) with no gradient from the read into the network.
3. **Sharpness must not encode length.** Normalizing keys and queries with a learned sharpness adds ~2 points at the
   training length but costs generalization (16 pairs: 44.6% vs 73.8% without normalization): a learned temperature
   calibrated to the training number of competitors over-sharpens when competitors double. Mixed-length training
   removes the dependence: 91.6% at 32 pairs, twice the longest training length (seed 0).
4. **It carries to language.** The same read over GPT-2-tokenized FineWeb (1M training tokens) scores 6.009 ± 0.010 nats per
   token (3 seeds) on the 65,528-target slice, against KN trigram 6.537 at the same data and 6.100 at 4×. Without the read the same model scores 6.193:
   the read carries 0.175 nats per token of the margin (seed-0 ablation).

## §444 Prediction for interleaved processes (FAS v2)

When K processes write into one log, the in-line predecessor of an event is generally not the previous merged event,
so the message of x_{j−1} is the wrong message. The binding then needs a third hop: the event must first *find* its
in-line predecessor (a keyed match of its type and state against recent events' keys) and take that event's message.
This is race attention over recent events, the anonymous counterpart of FIFO de-interleaving (which learns the route
from hidden identities). The native B3 models' measured binding purity of 0.30, and the split of their faulty-run AUROC
by purity (.735 high vs .511 low), are the signature §442 predicts for a model that must bind through a decaying state.
Testable claim: a model whose writes carry the *queried* predecessor's message raises binding purity and closes most of
the oracle gap (.821 vs order3 .685). Round 3 (B3_DEVELOPMENT_LOG.md) first tests the simpler merged-predecessor
message on the B1 race model; the queried-predecessor read is the next step if purity stays low.
