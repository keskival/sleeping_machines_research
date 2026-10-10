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
  only the cost of *learning* the posterior from forward samples. Arms (5 × depth 1, 2 × 3 seeds): dense exact,
  REINFORCE, straight-through Gumbel, hindsight (proposed: one expert per example plus one sleep sample), exact-posterior
  oracle; metrics: held-out marginal log-likelihood against the teacher's (Bayes ceiling), route recovery, expert
  evaluations per example.
- 9 Oct (founder direction: aim for reduced synchrony, reduced FLOPs, and the forward pass training the slow parameters of
  the backward pass): the credit model q *is* the backward pass's slow parameters (learned across examples; per-example
  credit is a cheap read). `hindsight_race_v2.py` adds **hindsight_async** (sleep samples from a stale snapshot of the
  forward model refreshed every 50 steps, consumed through a replay buffer: q trains off the forward lockstep) and
  **hindsight_slow** (q updated at 1/8 of the forward rate), and counts multiply-accumulates per training example for every
  arm (router, experts, credit model, sleep sampling). Success criteria stated before results: hindsight within a small margin
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
4-dimensional learned message; the credit model sees only (message, outcome).

## 6. Closing the recursion: a shared verdict for two learning objectives (founder direction, 9 Oct)

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

Their target updates are the marginal-likelihood gradient for θ and the posterior-cross-entropy gradient for ψ on
**data** (the posterior target is detached). At finite S these self-normalized estimators are biased when q differs from
ρ. They are two coupled learning objectives, not the gradient of one established joint scalar potential. A common
fixed point is possible when q represents and reaches the posterior and θ reaches a stationary likelihood; global
convergence is a hypothesis. The feedback mechanism is:

- A better q gives lower-variance, less-biased weights, hence a more exact forward gradient. With adequate proposal support and moment conditions the usual self-normalized bias decreases asymptotically as O(1/S); its constant depends on weight moments and the gradient integrand. When q = ρ all importance weights are equal and the sampled credit is unbiased.
- A better forward model moves ρ_θ less per step, so q's target stops drifting.

The credit gap KL(ρ ‖ q) is the measurable state of the backward side. Learning has converged on both sides when the
likelihood plateaus *and* the gap goes to zero. Expert work: S proposals plus sleep sampling per example instead of all K causes. Total work also includes the credit model, router/candidate discovery and optimizer; S/K is not a total-work ratio.
Contract (`experiments/credit/check_closed_v5.py`): enumerating every cause once reproduces the exact router gradient
(1.8e−7 at depth 1, 4.2e−7 at depth 2).

**Timescales (refines the "slow backward parameters" direction).** Two-timescale stochastic approximation motivates having the follower track its moving target faster than the target moves. Convergence additionally needs stability, step-size and estimator assumptions; the current fixed-rate Adam experiment does not establish those conditions. The credit
model must therefore adapt faster than ρ_θ drifts. It is "slow" relative to a single example: it amortizes credit across
examples, and per-example credit is a cheap read. It is not slow relative to θ. The `hindsight_slow` arm (q at 1/8 of the
forward rate) tests the wrong side of this condition, and the theory predicts its credit gap grows.

**Predictions** (closed vs sleep-only vs dense, test credit gap logged per epoch):

- (a) The closed arm's credit gap falls monotonically to near zero, while sleep-only hindsight keeps a residual gap early in training.
- (b) The closed arm matches the oracle's held-out likelihood and route recovery at depth 2 with S = 2 proposals: 3 expert evaluations per example against 32 for dense.
- (c) Hybrid, with exact posterior supervision on 10% of the data, sits between the two.

A failure of (a) with success of (b) would mean the forward side learns without the backward side converging, i.e. an
open loop that happens to work. That would be evidence against closure as the mechanism.

**Queue state (23:15 UTC).** The v1–v5 grids were withdrawn unstarted because of the unseeded teacher (correction below). Predictions (a) and (b) are tested by the matched, seeded v6 pipeline (§8: dense, closed h64/h8, sleep-only, closed S = 1, with posterior KL/TV). The hybrid (c) and message-head arms wait for a seeded rerun after v6 reports.
**Protocol correction (19:12 UTC): v1–v5 initialize the teacher from an unseeded global torch RNG. The shared `data-seed`
seeds input and cause draws but not the teacher weights, so separate arms do not have a matched teacher. These runs are
individual construction diagnostics; do not use their cross-arm scores as a matched comparison. Preserve their queues
and any completed results. v6 isolates teacher initialization with `data-seed` and records the generated-data digest.
Its held-out toy data is explicitly DEV, not a sealed public benchmark test.

## 7. Prediction 4 built: per-event learning through the memory without BPTT

`experiments/credit/online_race.py` trains a race-of-clocks TPP over one complex-diagonal temporal memory.
- **Learner.** Every memory parameter (write, gate, embedding, gap encoder, decay rates, frequencies) carries a forward
  eligibility trace S_t = a_t S_(t−1) + ∂(a_t z_(t−1) + b_t)/∂θ. When event t+1 arrives:
  - its loss is differentiated locally (head, readout);
  - the memory credit λ_t = ∂ℓ/∂z_t is contracted with the traces;
  - the weights update immediately.
  Nothing is stored, there is no backward sweep, and there is no sequence-level step.
- **Contract (float64, fixed weights).** The summed per-event gradient equals BPTT to 4.4e−16 over all parameters. With
  the traces zeroed it fails (relative difference 1.0, worst parameter Wr), so the contract is sensitive to the traces.
- **Grid** (slot 3; 3 seeds per arm; Taxi DEV log-likelihood per event):
  - BPTT;
  - `online_trace`: per-event, 16 concurrent streams share each update;
  - `online_local`: traces dropped.
- **Pass criterion, stated before results:** `online_trace` within 0.02 nats/event of BPTT, and clearly above
  `online_local`.

**Result (10 Oct 06:15 UTC; Taxi DEV log-likelihood per scored event, nats, higher is better; 3 seeds; 20 passes over
TRAIN for every arm; `experiments/results/credit/online1_*`).**

| Arm | Final DEV LL (SE) | Seeds | Time / mark part | Updates | Wall |
|---|---|---|---|---|---|
| BPTT | 0.4557 (0.0070) | 0.4607 / 0.4645 / 0.4419 | 0.7005 / −0.2448 | 1,760 | 0.9 min |
| **online_trace** | **0.4805 (0.0035)** | 0.4812 / 0.4742 / 0.4862 | 0.7099 / −0.2294 | 63,440 | 3.1 min |
| online_local | 0.4728 (0.0017) | 0.4733 / 0.4696 / 0.4755 | 0.7082 / −0.2354 | 63,440 | 1.3 min |

**PASS** at matched data passes:
- online_trace is 0.025 above BPTT, not merely within 0.02.
- Every trace seed is above the online_local mean (+0.0077).

Two effects separate:
- **Per-event updating** is worth +0.017 over batched BPTT at the same number of data passes. BPTT is still rising at
  epoch 20, so this is sample efficiency per pass, not a converged-quality claim.
- **Exact temporal credit through the memory** (the traces) is worth +0.0077 nats/event on Taxi, on every seed.

Wall time reflects a Python per-event loop, not the method's arithmetic. Next: BPTT at 100 epochs (`online1b_*`, queued)
tests whether converged BPTT ends above the per-event learner.


## 8. Finite-proposal feedback and full learner cost (9 Oct, AWS autonomous continuation)

**Completed mathematical audit:** [estimator_audit.py](../credit/estimator_audit.py), evidence
[aws_estimator_math_audit_20261009T1912Z.json](../results/credit/aws_estimator_math_audit_20261009T1912Z.json).
No fitting was performed. Exact multinomial enumeration tests the estimator used by the `closed` arm.

For two causes with posterior ρ = (0.8, 0.2) and proposal q = (0.2, 0.8):

| Proposals S | Expected first-cause credit | Posterior TV bias |
| --- | --- | --- |
| 1 | 0.200000 | 0.600000 |
| 2 | 0.341176 | 0.458824 |
| 4 | 0.516169 | 0.283831 |
| 8 | 0.666844 | 0.133156 |
| 32 | 0.778042 | 0.021958 |

At S = 1 the normalized weight is always 1. The expected wake update to q's logits is
E_q[e_k − q] = 0: the forward likelihood verdict cannot teach the backward model through this estimator. Sleep can
still train it. At S ≥ 2 the verdict moves q toward ρ in this example. At q = ρ, the expected posterior estimate is
exact at every S; locally on the probability-simplex tangent, the derivative of E[ρ̂] − q is −(1 − 1/S) I.
This is an attracting **mean-field direction for a fixed posterior**; the actual neural update's metric, finite samples,
optimizer and moving forward model determine its dynamics. The audit verifies the two-cause derivative numerically;
it does not prove a global contraction or monotonic KL during coupled training.

The old once-each enumeration contract is valid for a **uniform** proposal. Once-each enumeration followed by division
by a nonuniform q does not reproduce ρ: in the example it yields (0.941176, 0.058824). v6 contracts check every forward
parameter under the valid uniform enumeration, teacher reproducibility, and separation of detached θ/ψ update paths.

**The backward learner must earn its cost.** Under v5's leading-linear MAC convention (three times forward MACs for
differentiated operations, one for sleep generation), including the credit network, two proposals and one sleep draw:

| Depth | Credit hidden width | Dense MACs/example | Closed MACs/example | Closed/dense |
| --- | --- | --- | --- | --- |
| 1 | 64 | 4,224 | 35,424 | 8.386× |
| 2 | 64 | 17,280 | 46,688 | 2.702× |
| 2 | 16 | 17,280 | 9,824 | 0.569× |
| 2 | 8 | 17,280 | 6,368 | 0.369× |

These are **modeled linear work**, not whole-fitting FLOPs, CPU latency or energy. Softmax, sampling, nonlinearities,
biases and Adam work are outside those numbers. Selected indexing of the expert parameter tensor still creates a dense
gradient tensor and invokes dense Adam: sparse expert evaluation does not imply sparse optimizer work. v6 records
optimizer parameter visits, whole-fit/per-example MAC estimates, evaluation work, wall time, RSS and hardware separately.
The depth-2 h8 setting offers a concrete engineering target; its credit fidelity and quality require fitting.

**Next decision:** matched-teacher depth-2 DEV pilots compare dense exact, closed h64/S2, closed h8/S2,
sleep-only h8, and closed h8/S1. The last arm tests whether verdict feedback adds useful learning beyond sleep.
Inspect likelihood, posterior KL/TV, effective sample size, duplicate proposals and total learner cost together.
The diagnostic has hard winner routing and cause credit; it lacks persistent temporal memory and is not an integrated
family benchmark verdict. Successful credit must next be integrated into R1 recall or B1 event learning with those
mechanisms retained. Known method attribution: [Bornschein & Bengio, Reweighted Wake-Sleep](https://arxiv.org/abs/1406.2751)
and [Le et al., Revisiting RWS for Models with Stochastic Control Flow](https://arxiv.org/abs/1805.10469).

Pipeline admitted at 19:16 UTC: `aws_credit6_20261009T191619Z` (slot 1, behind existing work); contract and smoke are prerequisites,
then the five pilots above and `analyze_v6.py`. Automatic result: `experiments/results/credit/aws_credit6_20261009T191619Z_analysis.json`.
The predeclared compact-quality screen is final DEV likelihood within 0.02 nats of dense and lower linear MACs;
passing it selects three-seed confirmation, not a win claim. Numerical contracts and fitting results remain pending.

## 9. Chosen development path: exact pairwise verdicts and corrected cause proposals

**Failure addressed.** Small self-normalized proposal sets teach biased credit; one proposal teaches no expected
wake credit at all. A large recognition MLP spends more linear work than the experts it avoids. The chosen next
construction uses a compact q and an exact *relative* posterior target supplied by two forward causes. This changes
credit estimation, not the inference architecture. Existing v6 queues remain pinned for the comparison.

**Forward trains backward on real data.** Let a_i = p_θ(y, cause i | x) and a_j likewise. The forward pass evaluates only
these two causes. It supplies the exact binary target t_ij = a_j/(a_i+a_j), detached from θ. The credit model predicts
σ(log q_j − log q_i), fitted by binary cross-entropy. This target needs no marginal likelihood, no exact full posterior
and no privileged teacher labels. Each pair gives an informative verdict when its predicted odds are wrong.

For frozen positive pair weights the loss is convex in unconstrained q logits; its Hessian is a weighted graph
Laplacian. With a connected pair graph and positive causes its unique optimum modulo a constant is
log q_i = log a_i + constant, hence q = ρ. The current pair-sampling weights depend on detached proposals and retained
causes, so this fixed-weight convexity result is not a convergence theorem for the adaptive neural learner. It does give
an exact common target for every visited pair. Compact q may lack the capacity to represent it; the DEV credit gap
measures that independently of forward likelihood.

**Backward proposes; forward corrects.** Propose j from g = (1−ε)q + ε/K, ε = 0.1. For retained cause i accept j with

    A(i,j) = min(1, a_j g_i / (a_i g_j)).

This is independence Metropolis–Hastings, not a new sampler. Detailed balance follows from
ρ_i g_j A(i,j) = min(ρ_i g_j, ρ_j g_i) = ρ_j g_i A(j,i). For a fixed forward model and proposal, its invariant law is
ρ even when q is wrong. Positive g makes the finite chain irreducible; self proposals make it aperiodic. If the actual
proposal g equals ρ, one proposal is an independent posterior draw. Otherwise a finite chain started away from ρ is
biased. Neural updates move its target, and persistent chains must track that movement.

At stationarity a sampled cause supplies the complete-data gradient ∇ log a_i, whose expectation is the exact
marginal-likelihood gradient (Fisher's identity). Pairwise q training uses two no-gradient expert scores; forward
training re-evaluates only the accepted cause with gradients. Acceptance decisions and q are detached from that
forward update. There is no new inference-time credit network or altered forward winner rule.

**Completed finite-state evidence:** [pairwise_math.py](../credit/pairwise_math.py), saved in
[aws_pairwise_math_20261009T1958Z.json](../results/credit/aws_pairwise_math_20261009T1958Z.json). Three two-/three-cause
examples give maximum detailed-balance error 1.7e−18 and stationarity error 2.8e−17. For ρ=(.8,.2), g=(.2,.8), TV error
from a g start is .6 initially, .45 after one step, .0601 after eight and .0000603 after 32. Thus correctness of the
stationary distribution does not grant one-step credit fidelity. The pair gradient at the wrong q is (−.6,+.6), giving
useful learning where the one-proposal self-normalized wake gradient is zero.

**Implementation:** [pairwise_race_v7.py](../credit/pairwise_race_v7.py). `pairwise_persistent` retains one cause index per
TRAIN item between visits; `pairwise_reset` restarts from g on every visit. Every visit scores both current and proposed
causes at the current weights, preventing stale likelihood-cache errors. Cause indices take 160,000 bytes for 20,000
items. They are sampler state, not a substitute for the model's persistent temporal memory. No DEV target is used to
fit these states or select epochs. Exploration, duplicate pairs, MH acceptance, actual moves and posterior KL/TV are
reported; high acceptance alone is not evidence of good credit.

**Learning work:** at d=16, K=8, K2=4, C=10, h=8, two no-gradient expert scores plus one differentiated expert,
one differentiated router and one differentiated q cost 4,064 modeled leading-linear MACs per training target versus
17,280 for dense (0.235×). Ten epochs × 20,000 targets model 0.8128 vs 3.456 billion MACs. These use identical units and
target denominators. Nonlinear, bias, sampling and optimizer FLOPs are excluded. Both implementations still allocate
dense expert gradients and run dense Adam; total fitting work and latency must be assessed separately. This is a design
budget, not a measured efficiency win.

**Decision pipeline:** numerical contracts → one smoke → persistent and reset depth-2 DEV pilots (seed 0, ten epochs)
→ automatic comparison against the existing matched v6 dense and compact-closed pilots. Source-bound dependencies
prevent fitting or comparison after a failed contract/smoke. Final-epoch likelihood within .02 nats of dense and
posterior KL ≤ .02 selects three-seed confirmation. Failure calls for posterior-gap, pair-coverage and chain-tracking
diagnosis, not a family loss. No new external-architecture reference is trained.

**Integration target and retained mechanisms.** R1's existing keyed temporal memory and predecessor message remain the
candidate for integrated testing; preserve decay/rotation, irregular elapsed time, separate keys/values, sparse writes,
losing-route credit and complete learning accounting. Its existing marginal over output clock causes can support pair
posterior odds, but those output clocks already have cheap exact responsibilities. The useful target is hidden route/
write credit. A hidden cause must expose a valid causal complete-data score; a nonlinear weighted read cannot silently
be replaced by a mixture likelihood. Derive that interface and its memory/eligibility credit contracts before changing
R1. The present synthetic diagnostic establishes neither hidden temporal-write credit nor integration.

**Attribution.** [Hastings (1970)](https://academic.oup.com/biomet/article-abstract/57/1/97/284580) supplies the sampler;
[Naesseth et al., Markovian Score Climbing (2020)](https://arxiv.org/abs/2003.10374) is the close precedent for learning an
inclusive-posterior proposal with Markovian samples. The construction here adds forward pair-odds supervision and
charges the full forward/backward learner within this race-credit diagnostic. No general novelty or convergence claim
is attached to that combination.

Active continuation: `aws_pairwise7_20261009T195916Z` on gym slot 1 (contract → smoke → two pilots → comparison),
analysis `experiments/results/credit/aws_pairwise7_20261009T195916Z_analysis.json`. The non-training `continue_pairwise_v7.py` controller watches in tmux
`reciprocal-credit7`. Passing BOTH predeclared pilot gates admits learner seeds 1–2 of pairwise-persistent and the same
family's dense reference, with seed 0 reused; a final analysis requires all three seeds to pass the development gates.
A failed pipeline or gate records a development decision and stops automatic scaling. Fits remain CPU-only under
source-bound `run_safe.sh` jobs. Three seeds vary learner initialization on one fixed teacher; they do not test a broad
teacher distribution or establish a public benchmark win. Numerical torch contracts and fitting results are pending.


## 10. Hidden-write credit: causal history scores, not output-clock relabelling

**B1/R1 integration derivation, 9 Oct 20:55 UTC.** The failure addressed is using a local output responsibility or
prefix-only route score as credit for a hidden write that changes future memory. The proposed interface retains sparse
writes, persistent temporal state, separate keys/values, elapsed-time computation and counterfactual learning. This is
an interface specification for a stochastic hidden-route member; the current R1 keyed read is not silently replaced.

Let observations be x_1:T (marks and gaps), hidden writes c_1:T, and causal state s_t = F_theta(s_(t-1), x_t, c_t).
Prediction precedes the write: event density f_theta(x_t | s_(t-1)), then route prior
pi_theta(c_t | s_(t-1), x_t). A valid complete-data log score is

    A_theta(c) = sum_t [log f_theta(x_t | s_(t-1)(c_<t))
                        + log pi_theta(c_t | s_(t-1)(c_<t), x_t)].

Each f includes mark density and the integrated hazard for the observed gap. A censored observation window additionally
includes its terminal survival; the tiny audit has an observed final query, no terminal censoring. State updates are
deterministic conditional on the routes. Observed targets enter the posterior credit model during training, never the
causal forward router. For positive differentiable scores on fixed finite route support:

    grad log sum_c exp(A_theta(c)) = E_posterior[grad A_theta(c)].

**The complete-data gradient still needs memory credit.** For a held-fixed route history, eligibility
S_t = (partial F / partial s) S_(t-1) + partial F / partial theta carries parameter sensitivity. Every event-density and
route-prior score uses its direct parameter derivative plus its state derivative contracted with S_(t-1). For a
state-independent diagonal decay/write this reduces to the reciprocal trace in §1. State-dependent gates or routes
add their state Jacobian; dropping it is an approximation. Posterior inference replaces the discrete-route summation,
not the continuous sensitivity calculation. Fixed-weight equality does not grant exactness after online weight changes.

**A pair verdict over histories is legal, but has a suffix cost.** Replacing c_u while retaining later route indices
requires replay from the state immediately before u, recomputing every affected later state, event density AND route
prior. Prefix terms cancel in A(c') - A(c). Old suffix likelihood caches cannot be reused. Pairwise BCE then uses
sigmoid(A(c') - A(c)); MH correction for a block proposal uses the actual reverse/forward proposal probabilities.
The independence formula in §9 applies only to independence proposals over the chosen history space. Charge replay
length, state snapshots, proposal scoring, continuous sensitivities and optimizer work; two proposed histories need
not cost just two expert evaluations. A bounded suffix is exact only with a specified conditional boundary or a proved
finite dependency; temporal decay alone does not give exact finite support.

**Existing R1 read semantics.** `recall_tpp_v4.py` makes deterministic addressed writes and adds query/key scores to a
mark race. A q-weighted nonlinear read does not generally equal a mixture over sampled reads. There is no missing
latent-write posterior to infer in that deterministic operation. Two legal next paths are (a) learn an approximate
continuous adjoint for the existing operation and validate against its exact gradients, or (b) specify a stochastic
hidden-write member with the complete-data score above and compare it against the existing R1 model on DEV. Path (b)
changes forward semantics and requires the architectural comparison before any long fit. No substitution is made here.

**Completed non-fitting evidence:** `experiments/credit/hidden_write_history_math.py` (renamed after a cross-host filename collision; identical source bytes), source-bound result
`experiments/results/credit/aws_hidden_write_math_20261009T2055Z.json`. Two binary sparse writes, a decaying persistent
scalar state, state-dependent route priors, marked exponential clocks and a future query enumerate four histories at
three parameter settings. Posterior-mean complete-data gradients match finite differences of marginal log likelihood
with maximum error 5.8e-11; memory eligibility score errors are at most 5.4e-11. Prefix-only odds fail all three cases
and even reverse the sign of the full odds. This is a mathematical contract for this tiny model, not fitted recall,
full R1 integration, a nonlinear-memory contract or an efficiency claim.

**Next capability decision:** first let the existing matched v6/v7 DEV pilots settle compact-credit fidelity. Before an
R1 hidden-write fit, implement a tiny stochastic addressed key/value memory with causal suffix replay; enumerate its
route histories to validate pair targets, full parameter gradients and elapsed-time/silence terms. Retain the current
R1 keyed/predecessor model as the same-family reference and measure complete replay/learning work. This closes the
semantic gap without mistaking inexpensive output-clock attribution for deep memory learning.

## 11. Unshared directions that train each other (founder direction, 9 Oct)

*"[The two passes] don't need to share weights. But these two directions of passes should train each other."*

**Where the familiar schemes stand.** An RBM uses one weight matrix in both directions. Backpropagation reuses the
forward weights transposed, which on physical hardware means moving weights to a second datapath. The wake–sleep
algorithm (Helmholtz machine) has separate recognition and generative weights. Each set is trained only on samples from
the other, so the two sets have no common fixed point (§6). Feedback alignment uses separate, fixed random feedback:
unshared, but static.

**The construction: two parameter sets with one fixed point, each set trained by the other.**

- **Credit model.** The credit model ψ (§§3, 6) is separate from the forward model θ.
  - The forward model's verdicts and sampled causes train ψ.
  - ψ's credit trains θ.
  - Their joint fixed point is q_ψ = ρ_θ, the exact posterior.
- **Learned feedback weights.** In the backward path, every matrix whose transpose would carry error gets its own
  feedback matrix B.
  - B receives exactly the update its forward partner W receives: the local product of input and output activity,
    which both ends of the connection observe. Both also receive the same decoupled decay.
  - Then W − B evolves as (1 − η·wd)(W − B). Duality B = W is the attracting fixed point, learned rather than assumed
    (Kolen & Pollack 1994; Akrout et al. 2019 for deep networks).
  - Once aligned, the learned backward pass delivers exact credit. While misaligned, it delivers credit that the
    forward weights themselves align to, as in feedback alignment, so learning proceeds during convergence.
- **Forward traces in the memory.** In the per-event trace learner (§7), the memory's credit runs forward through the
  forward weights in the forward direction, so no transpose is needed there. With learned feedback in the
  readout and clock head, the whole learner is local:
  - no weight transport;
  - no stored history;
  - no global backward sweep.
  On a delay-line substrate, the backward path is a separate physical structure that learns to become the
  time-reversed dual of the forward one.

**Contracts** (`experiments/credit/online_race_v2.py`, float64):
- Feedback initialized at B = W stays identical to W, and every parameter matches the exact-transpose learner after 9
  per-event AdamW updates (difference 0.0; weights moved 8.9e−3).
- The trace gradient equals BPTT (4.4e−16).

**Test (slot 3, Taxi DEV, 3 seeds)** compares three feedback arms, all with identical decay on the four readout/head
matrices:
- `online_kp`: mutual training;
- `online_fa`: static random feedback;
- `online_trace`: exact transposes.

The cosine between each W and its B is logged every epoch.

**Predictions stated before results:**
- `online_kp` cosines rise toward 1 within the first epochs.
- `online_kp` DEV log-likelihood lands within 0.02 nats/event of `online_trace`.
- `online_fa` ends lower than both, with cosines that rise only partially, because the forward weights must adapt to a
  static backward path.

A mutual arm no better than the static one would mean the backward direction's learning adds nothing at this scale.

**Result (10 Oct 07:10 UTC; Taxi DEV log-likelihood per scored event, higher is better; 3 seeds; identical decay on the
four readout/head matrices in every arm; `experiments/results/credit/online2_*`):**

| Backward matrices | Final DEV LL (SE) | Seeds | Final cosine(W, B): R / P / H / Mk |
|---|---|---|---|
| shared, exact transposes (`online_trace`) | 0.4736 (0.0027) | 0.4735 / 0.4689 / 0.4784 | — |
| **unshared, trained by the forward's local updates (`online_kp`)** | **0.4715 (0.0013)** | 0.4698 / 0.4705 / 0.4741 | 0.99998 / 0.99998 / 1.0000 / 1.0000 |
| unshared, fixed random (`online_fa`) | 0.1699 (0.0862) | 0.0392 / 0.3326 / 0.1381 | 0.21 / 0.28 / 0.05 / 0.01 |

**PASS** on all three predictions:
- **Mutual training learns duality.** The readout cosine on seed 0 rises from 0.40 to 0.82 to 0.98 to 1.00 over the
  first epochs.
- **It matches exact credit.** Mutual feedback is −0.0021 against exact transposes, within the 0.02 criterion.
- **Static unshared feedback fails.** It is 0.30 nats/event worse, and the forward weights barely align to it (cosines
  0.01–0.28).

The two directions need not share weights, but they must train each other. Learned that way, the backward path becomes
the exact dual and learning matches backpropagation. Left static, it does not, and learning fails in this recurrent
event model.

Combined with §7 (memory credit runs forward through traces), the learner is fully local at this scale:
- no weight transport;
- no stored history;
- no backward sweep.

It reaches 0.4715, against 0.4557 for BPTT at the same 20 data passes. Decay on the readout/head (wd = 0.1, needed for
the mutual contraction) costs the exact learner 0.007 relative to §7's run without it (0.4736 vs 0.4805).

## 12. Exact per-event learning across depth (9 Oct)

One diagonal memory layer has diagonal traces: each mode depends only on its own parameters, so the trace costs no more
than the parameters. Stacking breaks this. Layer-1 parameters reach the layer-2 state through every past layer-2 input,
so exact forward credit must carry S_2 = ∂z_2/∂P for every lower parameter P:

  S_1,t = a_1,t S_1,t−1 + local_1,t
  ∂u_2,t/∂P = ∂u_t/∂P + R_1 S_1,t (+ readout terms)
  S_2,t = a_2,t S_2,t−1 + own_2,t + J_2,t ∂u_2,t/∂P,

where J_2 = ∂b_2/∂u_2 is the layer-2 write Jacobian. When an event arrives, its loss is differentiated locally with
respect to (z_2, u_2) and the head. The memory-path gradient is λ_2·S_2 + μ·∂u_2/∂P.
- **Cost.** Each event and stream costs O(n_2·|P|) for the trace update and storage. For Taxi (n = 16, |P| = 4,704) that
  is 4·16·4,704 = 301k trace floats per stream. This price of exactness grows with depth × lower parameters, not with
  sequence length. BPTT's stored history grows with sequence length.
- **Truncation.** The layer-local alternative (as in e-prop) keeps no history of S_2 with respect to lower parameters, so
  lower layers receive credit only through the current input.

**Contract** (`experiments/credit/online_deep.py --contract`, float64, fixed weights, two layers):
- online_deep equals BPTT on every parameter: maximum relative difference 9.6e−16.
- The layer-local truncation differs by 112–124% on lower-layer parameters, so the contract separates the two.

**Test.** Slot 3, after the one-layer grids: BPTT, online_deep, online_trunc and online_local, Taxi DEV, 3 seeds each.

**Predictions:**
- online_deep is within 0.02 nats/event of BPTT.
- online_trunc lands between online_deep and online_local. The size of that gap measures how much learning depends on
  cross-layer temporal credit.

If truncation costs nothing here, the cheap layer-local learner is enough at this scale. If it costs a lot, exact depth
traces (or a learned approximation of them, §§3, 11) are the path to per-event learning of our deep stacks.
