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
