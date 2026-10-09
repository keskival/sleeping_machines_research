# 158. What race networks can represent, why they train, and what they assume (§§449–456)

8 October 2026, curie host session. Builds on: race/softmax equivalence (§§96–107), attention and Mamba containment
(notes 08, MAMBA_CONTAINMENT_20261007), the race-of-delayed-clocks construction (RACE_OF_DELAYED_CLOCKS_20261006),
binding (note 156) and de-interleaving (note 157). Evidence cited from the B1/B2/B3/R1 dossiers.

Notation. After event i the model holds state h_i and runs a race of M independent clocks. Clock m has survival S_m(τ)
and hazard h_m(τ) = −d log S_m/dτ; its mark law is p_m(k | h_i). Exponential clock: h_m = w_m e^{r_m}, constant.
Defective delayed (log-normal) clock: fires with probability π_m, then at LogNormal(μ_m, σ_m); S_m = 1 − π_m Φ(z),
z = (log τ − μ_m)/σ_m. Window clock: logistic window on [a, b] with edge scale s (closed-form survival).

## §449 Representational capacity of the race head

**Proposition 1 (cumulative hazard is a sum of monotone soft steps).** The race's cumulative hazard is
H(τ) = Σ_m H_m(τ). An exponential clock contributes a ray w e^r τ. A defective delayed clock contributes
H_m(τ) = −log(1 − π_m Φ(z)), a smooth nondecreasing step from 0 to −log(1 − π_m) centred at e^{μ_m} with log-time
width σ_m; a window clock contributes a smooth ramp over [a, b].

**Theorem 1 (density universality on compact gap ranges).** Let f be any continuous inter-event density on
[τ_min, τ_max] ⊂ (0, ∞) with hazard bounded away from 0 and ∞. For every ε > 0 there is a finite race of exponential and
defective log-normal clocks whose density g satisfies sup |log f − log g| < ε on [τ_min, τ_max].
*Proof sketch.* H_f = −log S_f is continuous and strictly increasing on the interval. In log-time u = log τ, sums of
shifted, scaled normal CDFs with positive weights are dense (uniformly) in continuous nondecreasing functions on compact
intervals (the monotone-function analogue of sigmoid universality; take a fine partition and one step per increment).
A defective clock with small π contributes −log(1 − πΦ) = πΦ + O(π²), so a sum of many small-π clocks realises any
such sum of steps to first order, and an exponential clock supplies the linear component and keeps H unbounded so the
race is a proper distribution. Uniform approximation of H with a smooth step family also approximates its derivative
(the hazard) uniformly when the step widths shrink with the partition, giving uniform log-density error.

**Proposition 2 (time-dependent marks).** The mark law of the race at gap τ is
p(k | τ) = Σ_m h_m(τ) p_m(k) / Σ_m h_m(τ): a hazard-weighted mixture of the clocks' mark laws. With clocks localised in
time (narrow log-normal or window clocks), the weights form a partition of unity over the gap axis, so any continuous
p(k | τ) on a compact gap range is approximable to any accuracy (per-interval mark laws). The type of the next event can
therefore depend on *when* it happens, exactly as in competing-risks models, without a separate time-conditioned decoder.

**Proposition 3 (memory).** The state h is a stack of complex-diagonal linear recurrences discretised by the real elapsed
time (decay e^{−ρΔt}, rotation e^{iωΔt}) followed by position-wise nonlinear maps. Linear diagonal recurrences followed by
nonlinear projections are universal approximators of causal sequence maps with fading memory (Orvieto et al., 2023, for
uniform steps); with elapsed-time discretisation the same holds for fading memory *in time* (memory of an event decays
with the time since it happened, not its position). Attention and selective state spaces are contained exactly (notes 08,
MAMBA_CONTAINMENT).

**Corollary.** For any marked point process whose conditional intensity is a continuous function of a fading-memory
summary of the history, a race-of-clocks network approximates its conditional likelihood arbitrarily well on compact
ranges. The five EasyTPP wins are this capacity used at small size: 20–36K parameters suffice where the published leader
uses 27–300K (S2P2 at its published configurations).

## §450 What capacity is missing: two-hop binding and aliasing

A fading-memory state cannot isolate *one* earlier event among many by identity (note 156 §442): it holds the predecessor
inside a time-weighted sum. Two-hop relations (match, then successor) are representable only if the write already
carries one of the hops. The keyed predecessor read supplies it.

**Proposition 4 (slot aliasing).** A type-addressed slot sums the keys of all occurrences of its type, weighted by
elapsed-time decay. If a key x recurs with different successors y₁ (at age t₁) and y₂ (at age t₂ > t₁), the read prefers
y₁ by the factor e^{−ρ(t₁ − t₂)} per decay channel: it returns a recency-weighted vote, not the successor of a chosen
occurrence. *Prediction:* on recall with repeated keys bound to changing values, accuracy equals the frequency with which
the most recent binding is the asked one, and per-position keys (an O(L) attention read) are needed beyond that.
The token-language model already scores per position and only sums at the output type, so it does not alias the match.

## §451 Trainability: exact likelihood, dense credit for sparse outcomes

1. **No estimator noise in the objective.** The compensator ∫ λ is closed-form for every clock family used (log Φ,
   logistic, linear), so the per-event log-likelihood and its gradient are exact. Neural point-process models that
   integrate a learned intensity by Monte Carlo (NHP, THP, AttNHP) add estimator variance and bias to every step. The
   benchmark's own Monte Carlo scorer reproduces our exact values within its sampling noise (B1 estimator check).
2. **Every clock is taught at every event.** ∂ log L / ∂θ_m = ∂ log h_m(τ)/∂θ_m · 1[m fired] − ∂ H_m(τ)/∂θ_m: the clock
   that fired is pulled toward the observed time; every clock that did *not* fire is taught through its survival, i.e.
   by the silence it predicted. This is counterfactual credit to unrealised alternatives arising from the likelihood
   itself, not from an added estimator; the mark race adds (1[k = observed] − p_k) to every mark, losers included.
3. **Scale-free conditioning.** Delays are parameterised in log-time (μ, log σ), so gaps spanning six decades (seconds to
   weeks) share one well-conditioned parameter space; initialisation from gap quantiles (or clustered gap components)
   places clocks where data lives.

## §452 Trainability: three failure modes, each with a principled fix

| Failure | Mechanism | Fix (and evidence) |
|---|---|---|
| **Grid exploitation** | timestamps on a recording grid allow a density spike: log f(0) → ∞ at ties, an unbounded likelihood | resolution principle: hazards held at their one-cell value below the cell, or the exact cell probability F(τ+δ) − F(τ); audit drop on Retweet 0.224 → 0.004 |
| **Label switching / basins** | clocks are exchangeable; window clocks drifting past every observed gap leave a flat region, so seeds land in different basins (Amazon 0.74 vs 0.80) | anchor each window to a data-derived gap component (bounded start and width); Amazon then 0.8028 ± 0.0007, no restarts |
| **Credit through decay** | the gradient to an old write decays as e^{−ρΔt}; binding through the state is unlearnable when the relevant event is old or buried among others | direct paths: addressed slots and keyed reads with the predecessor message (recall 21.6% → 97.5%); per-pair duration attention for interleaved processes (FAS) |

**Proposition 5 (local learnability of the keyed read).** The keyed bonus is bilinear in the query and the stored keys,
and the race supplies its own error (1[k = y] − p_k) at the read. With keys stored as elapsed-time-decayed sums, the
gradient for W_k at a read is Σ_k (1[k = y] − p_k) q ⊗ e_k, where e_k is the decayed sum of the write inputs to slot k (an
eligibility trace held at the slot). Every factor is local to the slot and the read: a three-factor rule. Measured:
77.4 ± 5.4% recall with no gradient into the network (vs 97.5% with backpropagation).

## §453 Inductive biases, stated as assumptions about the world

1. **Relevance decays with elapsed time, not with position.** Memories decay and rotate by the real Δt. Correct for
   irregular sampling (clinical channels, process logs); a handicap only where position, not time, carries the signal.
2. **The next event is the first of several latent processes (competing risks).** Superposition is native: concurrent
   items, machines, patients' channels, users. This is why the family fits interleaved logs and why de-interleaving is a
   posterior over which clock/process fired (note 157).
3. **Something may never happen.** Defective clocks put mass at "no event"; silence lowers the hazard of what did not
   occur and is supervised (survival terms). Staleness features in B2 are the classification analogue.
4. **Events of a type update their own memory (addressed factorisation).** Writes are sparse by type; slots hold
   sufficient statistics of their type's history (exponential-family bias). Good when many types occur sparsely; the
   P19 result shows the statistics carry most of the signal there, and the temporal memory adds +0.072 AUPRC on top.
5. **Constant work per event, capacity beyond activity.** Per-event cost does not grow with history length; stored
   capacity can grow without raising per-event inference work.
6. **Multiscale timing by construction.** Log-time delays and multi-rate decay cover seconds to weeks without a learned
   positional scale.

**Where these biases should hurt.** Dense synchronous data where every element interacts at every step (images, dense
grids), tasks needing exact arbitrary pairwise content comparison over long contexts beyond the keyed capacity
(Proposition 4), and tables without temporal structure (trees lead on banknote). Large-scale language currently trails
tuned dense models at 90M characters.

## §454 What the evidence says about the theory

- **Theorem 1, numerically** (`experiments/theory/race_universality_check.py`, `results/theory/race_universality_check.{json,png}`):
  maximum-likelihood fits of one exponential + M defective log-normal clocks to three targets on [0.05, 50].
  KL(target ‖ race) for M = 1/2/4/8/16: two separated modes 0.375 / ≈0 / ≈0 / ≈0 / ≈0; truncated Pareto (α = 1.2) 0.283 /
  0.136 / 0.045 / 0.0078 / 0.0039; narrow spike on a broad background 0.051 / 0.0083 / 0.0011 / ≈0 / ≈0 (≈0: within the
  ±0.001 sampling noise). The pointwise log-density error on the two-mode target stays near 0.9 because it is
  concentrated in the inter-mode valley where the target density is nearly zero: the theorem's condition (hazard
  bounded away from 0) is exactly what fails there, while the KL, which weights by the target's own mass, vanishes.
- Theorem 1 at small size: five EasyTPP wins with 20–36K parameters; Retweet at 1/15 and Taxi at 1/12 of the leader's
  parameters and per-event compute.
- §451(1): our exact likelihoods agree with the benchmark's Monte Carlo scorer within its noise on all 25 final models.
- **Prediction 4, measured** (`experiments/theory/compensator_gradient_variance.py`; Taxi checkpoint, 64 TRAIN sequences,
  40 repeats): replacing the exact compensator by the uniform Monte Carlo estimate with J points per interval adds
  relative gradient noise ‖g_MC − g_exact‖/‖g_exact‖ of **59% (J = 1), 18% (J = 10, the EasyTPP scorer's setting) and
  6% (J = 100)**, falling as 1/√J; the mean of the MC gradients is unbiased (residuals at the 40-repeat noise level).
  Exact-compensator training has zero estimator noise and costs one closed-form evaluation per interval instead of J.
- §451(2) and Proposition 2: generated streams reproduce timing and mark transitions (taxi: gap KS 0.014, transition TV
  0.024; naive 0.063 / 0.53).
- §452: each failure mode was measured before its fix and the fix was verified by a pre-registered or audited run.
- §450 / Proposition 5: recall 21.6% → 97.5% with the predecessor message; 77.4% by local credit.
- §453(2): FAS v2, duration-matched predecessor attention gives the first native lead over the best classical detector
  (validation .702 vs .685).
- §453(2) qualified by Prediction 5's refutation: the timing advantage of the race shrinks with the number of superposed
  processes (0.99 → 0.21 nats per event from K = 1 to 8); on superposed streams the value is in binding.

## §455 Predictions to test next

1. **Aliasing (Proposition 4):** recall with keys re-bound to new values: accuracy tracks the recency-weighted vote; an
   O(L) per-event key read removes the gap.
2. **Clock count scaling (Theorem 1):** log-likelihood improves with the number of delayed clocks until the empirical gap
   distribution's components are covered, then saturates; dataset complexity (number of gap components) predicts the
   saturation point.
3. **Time reparameterisation:** replacing elapsed-time decay by position-indexed decay should hurt most on the datasets
   with the widest gap ranges (Taobao, StackOverflow) and least on near-regular ones.
4. **Gradient variance:** at matched model size, exact-compensator training has lower per-step gradient variance than a
   Monte-Carlo-compensator variant of the same model. **Confirmed (§454): 18% relative gradient noise at J = 10, 0 exact.**
5. **Superposition advantage:** on synthetic superpositions of K renewal processes, the race head's advantage over a
   single-intensity model grows with K.
   **Refuted (9 Oct, `experiments/tpp/superposition_test.py`, seed 0):** advantage (full race − single exponential clock with
   the same memory, nats per event) **0.987 / 0.914 / 0.497 / 0.207 for K = 1 / 2 / 4 / 8**: positive at every K but
   shrinking. Reason (Palm–Khintchine): a superposition of many sparse renewal processes approaches a Poisson process; the
   merged gaps become short relative to each process's cycle, the conditional intensity is nearly constant within a gap,
   and a single history-conditioned clock comes close. **Revised statement:** the race's *timing* advantage is largest
   for few concurrent processes with structured timing; on heavily superposed streams its value must come from
   identifying the processes (binding, note 157) and from mark structure, not from the within-gap intensity shape. This
   matches B3, where the gain on interleaved logs came only with predecessor binding.

## §456 Summary

A race network is a universal model of marked point processes on compact gap ranges (Theorem 1, Proposition 2) whose
memory is fading in real time (Proposition 3). It trains on an exact likelihood in which every clock, fired or silent,
receives credit (§451). Its known failure modes are identifiable and have principled fixes (§452). It assumes that
relevance fades with elapsed time, that events are the first of several competing processes, that silence is
informative, and that types update their own memory (§453): assumptions that hold for event data and explain where it
wins, and that predict where dense models should keep an edge.
