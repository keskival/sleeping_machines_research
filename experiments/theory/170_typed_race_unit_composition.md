# 170 — The race unit as a typed event transducer, and composition by induction

16 October 2026, Curie. Founder: "Consider what we have already: a super great learnable function for small, asynchronous,
causal data. And it transforms that into what? Dense embeddings? Sequences? And of course dense memories? We need to
understand what our component is as a typed unit before we can compose them … surely we can compose them! Proof by
induction!" Builds on 169 (causal event race; CERL-H §8), 168 (race Fisher geometry), 167 (temporal credit) and
TYPED_SEMANTICS_AND_RACE_COMPOSITION_20261006 §3 (invariance is preserved by induction over events and depth).

## 1. What the unit is good at, stated as a type

Let **E[A]** be the type of *marked event streams* over a finite alphabet A: a causal sequence (t_1, a_1), (t_2, a_2), …
with t_1 < t_2 < … and a_i ∈ A, observed through its own history. Token text is E[V] with unit gaps. EasyTPP, B4 and
market tapes are E[A] with irregular gaps.

The unit that wins our event battles is a race of clocks over persistent addressed memory. Read as a function:

    U_θ :  history in E[A]  →  a law over the next event (which mark, when)

It is an exactly normalised competing-risks law: each candidate cause c has a hazard λ_c(t). The first to fire emits the
event and names its mark. This is the function we learn better than the published references (B1, B4, B5, TGB), on
small, asynchronous, causal data. Our new language member (169) learns it on tokens and beats the dense hybrid by 0.375
nats at a third of the training compute.

## 2. What it currently transforms that into, and why composition broke

Inside every member we built, the unit turns events into **dense vectors**: mark embeddings, a dense temporal-memory
state, and dense residual streams between layers. Whenever we stacked units, we passed those dense vectors upward:
- The race core passed the winner's transported vector up a serial chain, and credit vanished.
- Plan-B passed a residual stream, and became a Transformer/SSM hybrid.
- Part-whole levels passed pooled span summaries, which were blind to arrangement (the arrangement obstruction of
  DEEP_REPRESENTATIONAL_COMPOSITION).

**The interface had the wrong type.** The unit is a superb learner of *E[A] → law*, but we fed the next unit a dense
vector sequence, R^d × time. The next unit then had to be a dense function learner, the thing we are not better at. Each
stack quietly converted the problem into the one dense models own.

## 3. The typed unit: events in, events out

Give the unit an output of the same type as its input. Besides its law over the next input event, a race produces events
of its own:
1. **Attribution events** E[C]. After each input event, the posterior names its cause: which concept unit, which past
   event repeated or continued, or the background route. Emit an event whenever the attributed cause *changes*. A copied
   span, a persistent topic or a running syntactic frame is one event. This is compression.
2. **Surprise events** E[A ∪ {novel}] (optional). Emit an input event upward only when the unit failed to predict it
   (low probability under its law). This is Schmidhuber's history compression (1992) in race form: what a level explains
   it absorbs, and what it cannot explain goes up.
3. **Expectation events**: the winner of the race *before* the next input arrives, with its predicted time. These are
   used top-down, not upward.

So the typed unit is

    U_θ :  E[A]  →  ( law over next A-event ,  E[C] )

with persistent state S: an addressed episodic store over A, dormant concept units with excitation and age clocks, and
clocks. **The output stream is the input type over a new alphabet.** That is type closure, and the inductive step needs
nothing more.

## 4. Composition and the induction

Stack: U_1 consumes E[A_0] (tokens) and emits E[C_1]. U_k consumes E[C_{k-1}] and emits E[C_k]. **Top-down:** U_k's
expectation (its law over the next C_{k-1} event, with a stay/switch hazard for the current one) enters U_{k-1}'s race
as additive log-rate bonuses and as a few top-down-woken candidate causes (`r1_cerl.CERL.race` td hook; `r1_cerl_h.py`).

**Theorem (properties preserved under composition).** Assume each unit satisfies, on its own input stream:
- (P1) exact normalisation of its law;
- (P2) causality: its output events and law at time t depend only on input events before t;
- (P3) type closure: its output is in E[C] for a finite C;
- (P4) compression: its output stream has at most as many events as its input, a fraction ρ_k ≤ 1;
- (P5) local learnability: it is trained by the likelihood of its own input stream, plus the one-hop top-down path.

Then the K-level stack has:
1. **An exactly normalised, causal token law.** *Proof:* top-down terms are added to log-rates inside the lower unit's
   single softmax race. Normaliser and numerator both contain them (contract: 1 + 3e-8). Level-k inputs at token row t
   are cause events that end at or before t − 1, and attribution at row t − 1 uses only tokens up to t. Induct on k.
2. **Monotone attainable likelihood.** The best training likelihood with K levels is at least the best with K − 1 levels.
   *Proof:* top-down gain λ_K = 0 recovers the (K − 1)-level stack exactly (contract: equal to 5e-7), so the K-level
   family contains the (K − 1)-level family. A level can be useless but, at the optimum, never harmful.
3. **Bounded total work under compression.** If each level keeps a fraction ρ ≤ ρ_max < 1 of its input events, total work
   is at most W_1 / (1 − ρ_max), whatever the depth. *Proof:* a geometric series of event counts, each level's per-event
   work bounded by level 1's.
4. **Context reach that grows exponentially with depth.** A level-k memory of N events spans about N / ρ^k tokens at the
   same per-event cost.
5. **Each level's learning problem is the one the unit is best at.** Level k learns a marked event stream E[C_{k−1}] with
   irregular gaps (causes persist for variable durations). That is the TPP problem where the race unit holds published
   wins, not a dense function-fitting problem. *Proof:* type closure (P3); the gaps are the cause durations.

**Base case.** The unit learns E[A] well: B1/B4/B5 wins; on language, CERL beats the dense hybrid (5.4665 vs 5.8414).
**Inductive step.** Level k + 1 is the same unit on an input of the same type, so whatever made level 1 learnable makes
level k + 1 learnable; and items 1–3 make the stack sound, never worse, and bounded in cost. Depth is then limited by the
data's structure (do higher cause streams contain predictable regularity?), not by credit paths, density or cost.

## 5. Where the induction can fail, and the measurements that tell us

The theorem is conditional. Each hypothesis is an empirical property of a trained unit, so each is a measurement.
- **P4 fails: no compression.** The first CERL-H check has ρ ≈ 0.75–0.83: the attributed concept *identity* flips almost
  every token, because four woken units trade wins. The alphabet C is too fine-grained to denote durable causes.
  Repairs, all native and sparse:
  - persistence in attribution: the stay/switch prior from above already favours the current cause; also attribute with
    hysteresis;
  - typed cause identities at the right grain: concept families, and a continuation pointer keeps its identity along the
    span it copies;
  - surprise events instead of every change.
  - *Measure* ρ_k per level, mean event duration and alphabet entropy.
- **Moving targets.** As U_{k−1} learns, the meaning of its causes drifts, and U_k chases them. *Measure* attribution
  stability on fixed text across checkpoints. *Repair* with slower lower-level learning or cause identities anchored to
  the episodic store, since an episodic event's identity never changes.
- **No structure above.** If E[C_k] is unpredictable beyond its unigram law, λ_{k+1} stays near 0 and the level adds
  nothing (item 2 guarantees no harm). *Measure* λ_k, level-k local NLL against its unigram baseline, and the token-loss
  gain from adding level k.
- **Dense leakage.** The interface must stay typed: events with identities and times, never dense vectors passed upward.
  Dense vectors may live *inside* a unit as its private belief state, today a temporal-memory vector. For the hardware
  case that state should also become addressed and sparse (cause excitations with ages, AWS's receiver core). That is a
  separate axis from composition.

## 6. The construction this implies (CERL-H v2) and the test of compositional scaling

1. **Level ≥ 2 units are race TPP units over cause events with real durations.** They predict which cause comes next and
   when the current one ends (the stay/switch hazard is that duration law). This is the B1/B4 machinery applied to the
   cause stream.
2. **Durable cause identities.**
   - Continuation pointers keep one identity along the copied span.
   - Concept attribution uses hysteresis plus the top-down stay prior.
   - Families are offered as a coarser alphabet when the measured ρ stays high.
3. **Optional surprise channel** (unexplained tokens go up) beside the attribution channel.
4. **Everything between levels is typed events**; nothing dense crosses a level.
5. **Per-level local learning**, plus the one-hop top-down path.

**Compositional scaling is demonstrated when:**
- (a) token loss improves with each added level, at total work within W_1 / (1 − ρ);
- (b) the gain per level grows with data, as depth gains do in the dense diagnostic (0.023 at 4M → 0.052 at 16M);
- (c) the gain lands on novel-in-window content, where the single-level race is only level with the dense model (7.87 vs
  8.02), because higher levels supply structure that copying cannot.

The running two-level CERL-H 4M fit measures (a) at ρ ≈ 0.8. The v2 durable-cause repairs target ρ well below 0.5 before
3 levels and 16M.
