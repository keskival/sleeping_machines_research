# 160 — Forward and backward passes as duals: credit assignment is the backward pass's objective

9 October 2026 · Direction from the founder (Tero Keski-Valkama): the usual picture — the forward pass is what matters and
the backward pass serves it, supported by the forward pass only incidentally through stored activations — is misleading.
The two should be duals; credit assignment is the backward pass's own objective, and the forward pass should train the
backward pass to do it well, as the backward pass trains the forward pass (cf. RBMs, whose two directions were identical).
This note shows that our substrate makes the duality exact in two places and turns the proposal into a trainable,
testable construction. Checks: `forward_backward_duality_check.py` (all PASS, max error ≤ 3e−15).

## 1. The backward pass of our memory is the memory itself, time-reversed

Each mode of the temporal memory is z_t = a_t z_{t−1} + b_t with a_t = e^{(−r + iω)Δt_t} and write b_t (gate ⊙ W u_t). For a
real loss L = Σ_t Re(c_t^* z_t) + … the adjoint λ_t = ∂L/∂z_t obeys

  λ_t = c_t + a_{t+1}^* λ_{t+1},

the **same decay, conjugate rotation, run backward in time** (check (1): 4.4e−16 vs autograd). The backward pass is not a
second machine: it is the forward machine played in reverse, as a physical reciprocal system (a delay line carries the echo
back with the same attenuation and the opposite phase advance). The dual direction also exists exactly: forward-mode
eligibility traces S_t = a_t S_{t−1} + ∂b_t/∂θ carry sensitivities forward with the same coefficients (check (2):
2.7e−15), so each layer can apply each error message the moment it arrives, without stored history (reply to the question
"does a layer need all errors at once": no; for diagonal memories, per-message application is exact for fixed weights).

**Consequence for hardware.** On a clockless delay-line substrate the adjoint needs no extra datapath: the same delays,
decays and conjugated rotations, traversed in reverse. Forward-mode traces need only per-parameter accumulators beside
the weights. Either gives learning without a global backward sweep.

## 2. Credit in a race is posterior inference over causes

For clocks with hazards h_i(τ) = e^{θ_i} k_i(τ) and integrated hazards H_i(τ), the log-likelihood of an event at τ is
ℓ = log Σ_j h_j(τ) − Σ_j H_j(τ) and

  ∂ℓ/∂θ_i = ρ_i(τ) − H_i(τ),   ρ_i(τ) = h_i(τ) / Σ_j h_j(τ) = P(clock i caused the event | event at τ)

(check (3): 1.1e−16; with marks ρ_i uses h_i p_i(k)). The backward message to clock i is its **posterior responsibility**
for the observed outcome minus its **exposure** (how long it was at risk without firing). This is credit assignment in the
literal sense: attributing an outcome to its causes; the gradient is that attribution. The race supplies the only global
quantity, Σ_j h_j, as its total rate. At the output, exact credit is therefore local and is an inference result.

## 3. Proposal: the forward pass trains the backward pass (and vice versa)

Inside a deep network the exact posterior over hidden causes is unavailable: today it is computed by backpropagation
(global, synchronous) or approximated by local race credit (77% vs 97.5% recall on R1). The duality suggests a third
route with an explicit objective for the backward pass:

- **Backward network q (credit model):** given an outcome and the forward state, it predicts which hidden routes, clocks
  or writes caused the outcome: a distribution over causes, i.e. credit.
- **The forward pass trains q (sleep phase).** When the forward race is *sampled*, its causes are known: which clock won,
  which route fired, which write was read. Every forward run is a labelled example for q; q is trained to recover the true
  causes (cross-entropy on causes). Its objective — calibrated attribution — is measurable on its own.
- **q trains the forward pass (wake phase).** On real data, q's responsibilities replace backpropagated errors for hidden
  units: each unit updates from its own responsibility minus exposure, as in §2, locally and asynchronously.
- **Symmetry.** With reciprocal parameterization (the same delays and keys traversed in both directions, as in §1) q starts
  as the time-reversed forward network and is refined by the sleep phase: the RBM-like case of identical directions,
  extended to a learned, time-reversed dual.

Relation to known ideas (attributed): the wake-sleep algorithm and Helmholtz machines (recognition model trained on
generative samples), RBMs and Boltzmann machines (symmetric directions), target propagation and synthetic gradients
(learned feedback), feedback alignment, predictive coding and e-prop (local error signals with eligibility traces). The
contribution here is the combination on our substrate: (i) exactness of the time-reversed adjoint for decaying-rotating
memories; (ii) exactness of race credit as posterior responsibility; (iii) supervision of the credit model by the races'
own sampled causes, so the backward pass has a well-posed objective and can be evaluated without the forward task.

## 4. Predictions and tests

1. *(done)* The three identities hold to machine precision.
2. **Credit-model calibration:** on synthetic two-layer race networks, a q trained only on forward samples predicts true
   causes with log-loss near the Bayes posterior, and its responsibilities correlate with exact backpropagated credit.
3. **Learning with q:** a race network trained with q's credit (no backpropagation through hidden layers) closes most of the
   gap between local race credit and backpropagation on R1 recall (77% → toward 97.5%), at lower work per update.
4. **Asynchronous per-event learning:** a B1 race model trained online with forward traces (§1) and per-parameter updates
   (no batch, no BPTT, no global clipping) holds DEV likelihood within a small margin of batched training.
5. **Hardware path:** the adjoint of a trained memory executed by reversing its delays reproduces the gradient (contract on
   the event simulator), so learning needs no separate backward datapath.

A failure of (3) or (4) is evidence about that construction; diagnose credit variance, cause coverage of the sleep samples
and the reciprocity assumption before generalizing.

## Test log

- 9 Oct: `experiments/credit/hindsight_race.py` written. Contract: with the exact posterior, the hindsight router update
  equals the exact marginal-likelihood gradient at depth 1 and 2 (max |diff| 2.4e−7, float32), so the experiment measures
  only the cost of *learning* the posterior from forward samples. Grid queued (5 arms × depth 1, 2 × 3 seeds): dense exact,
  REINFORCE, straight-through Gumbel, hindsight (proposed: one expert per example plus one sleep sample), exact-posterior
  oracle; metrics: held-out marginal log-likelihood against the teacher's (Bayes ceiling), route recovery, expert
  evaluations per example.
- 9 Oct (founder direction: aim for reduced synchrony, reduced FLOPs, and the forward pass training the slow parameters of
  the backward pass): the credit model q *is* the backward pass's slow parameters (learned across examples; per-example
  credit is a cheap read). `hindsight_race_v2.py` adds **hindsight_async** (sleep samples from a stale snapshot of the
  forward model refreshed every 50 steps, consumed through a replay buffer: q trains off the forward lockstep) and
  **hindsight_slow** (q updated at 1/8 of the forward rate), and counts multiply-accumulates per training example for every
  arm (router, experts, credit model, sleep sampling). The v1 addendum was withdrawn before loading; the v2 grid (7 arms ×
  depth 1, 2 × 3 seeds, after a smoke) replaces it. Success criteria stated before results: hindsight within a small margin
  of the oracle and dense on held-out likelihood and route recovery, clearly above REINFORCE at depth 2, at a fraction of
  dense MACs; the async and slow variants close to synchronous hindsight.

## 5. What the forward pass should give the backward pass (founder direction, 9 Oct)

A standard forward pass hands the backward pass stored activations (and optimizer moments, which are running
statistics), and none of it is trained to make credit easier. In our substrate the forward pass can supply: (i) **causes**
(which clock won, which route fired, which write was read: the labels credit needs); (ii) **precomputed sensitivities**
(forward eligibility traces, §1); (iii) **sufficient statistics** that make route credit closed-form (note 59); (iv) **a
prediction of its own credit**, so the backward pass sends only the surprise (actual − predicted) as sparse events: less
traffic, synchrony and work; (v) **confidence** (clock-noise precision, note 151) for precision-weighted credit; (vi) **near
misses** (runner-up clocks and margins) as counterfactual information for the losers.
**Trainable forward messages.** Three parameter sets: forward function θ (trained by credit), credit model ψ (the backward
pass's slow parameters, trained on the forward pass's sampled causes), and a forward message head φ trained by credit
quality alone, so the forward pass learns what to tell the backward pass. Test `hindsight_race_v3.py --arm hindsight_msg`:
4-dimensional learned message; the credit model sees only (message, outcome); queued behind the v2 grid.

## 6. Closing the recursion: one objective both sides climb (founder direction, 9 Oct)

*"If it's just the backward pass mirrored, it's a static universe. It needs to converge towards learning both sides."*

**The mirror is static.** The time-reversed adjoint (§1) has no parameters of its own. It is exact for the current
forward weights and learns nothing across examples: each credit computation starts from scratch. It is the *structure*
of the backward pass (the right architecture and initialization for q, by reciprocity), not its learning.

**Open loops do not converge to a common point.** In sleep-only hindsight, q minimizes KL(p_θ ‖ q) on the model's own
samples and θ follows q's credit on the data. The two losses differ, so this is a pair of games, not one objective
(the known weakness of wake-sleep, Hinton et al. 1995). The router's gradient bias is exactly
E_data[(q − ρ_θ)·∇ log π] ≤ 2·TV_data(q, ρ_θ)·max‖∇ log π‖. That is q's credit error *on the data*, which sleep-only
training reaches only as p_θ approaches the data. The loop helps itself only once it is already right.

**Closed loop: one verdict trains both sides.** For any cause k proposed by q, the forward pass can score the proposal
exactly: p_θ(y, k | x) = π_θ(k|x)·p_θ(y|x,k). The importance weight w_k ∝ p_θ(y, k | x)/q(k | x, y) is the forward pass's
verdict on the backward pass's guess. Trained on the same self-normalized weights (reweighted wake-sleep, Bornschein &
Bengio 2015):

- **Forward (θ):** Σ_s w_s ∇ log p_θ(y, k_s | x). This is an estimate of ∇ log p_θ(y|x), and its router part is Σ_s w_s e_{k_s} − π = ρ̂ − π, i.e. §2's credit.
- **Backward (ψ):** Σ_s w_s ∇ log q(k_s | x, y) (wake), plus the sleep loss on sampled causes as an anchor.

Both are stochastic gradients of the pair (log p_θ(y|x), −KL(ρ_θ ‖ q_ψ)) on the **data**, whose joint fixed point is
q = ρ_θ and ∇_θ log p_θ = 0. The recursion closes as a contraction:

- A better q gives lower-variance, less-biased weights, hence a more exact forward gradient. The self-normalized bias is O(var w / S), and var w = 0 when q = ρ.
- A better forward model moves ρ_θ less per step, so q's target stops drifting.

The credit gap KL(ρ ‖ q) is the measurable state of the backward side. Learning has converged on both sides when the
likelihood plateaus *and* the gap goes to zero. Cost: S proposals per example instead of all K causes (S/K of dense).
Contract (`experiments/credit/check_closed_v5.py`): enumerating every cause once reproduces the exact router gradient
(1.8e−7 at depth 1, 4.2e−7 at depth 2).

**Timescales (refines the "slow backward parameters" direction).** By two-timescale stochastic approximation (Borkar
1997), the joint iteration converges when the follower tracks its moving target faster than the target moves. The credit
model must therefore adapt faster than ρ_θ drifts. It is "slow" relative to a single example: it amortizes credit across
examples, and per-example credit is a cheap read. It is not slow relative to θ. The `hindsight_slow` arm (q at 1/8 of the
forward rate) tests the wrong side of this condition, and the theory predicts its credit gap grows.

**Predictions** (`hindsight_race_v5.py`, 18 runs: closed / hindsight / hybrid × depth 1, 2 × 3 seeds, test credit gap
logged per epoch):

- (a) The closed arm's credit gap falls monotonically to near zero, while sleep-only hindsight keeps a residual gap early in training.
- (b) The closed arm matches the oracle's held-out likelihood and route recovery at depth 2 with S = 2 proposals: 3 expert evaluations per example against 32 for dense.
- (c) Hybrid, with exact posterior supervision on 10% of the data, sits between the two.

A failure of (a) with success of (b) would mean the forward side learns without the backward side converging, i.e. an
open loop that happens to work. That would be evidence against closure as the mechanism.

Test log, 9 Oct: v4 `hindsight_hybrid` (6 runs) and the v5 grid (18 runs) are queued on slot 1 behind the v2/v3 grids.
