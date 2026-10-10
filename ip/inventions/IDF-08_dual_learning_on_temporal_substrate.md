# IDF-08 — Forward and backward passes as trainable duals: learning on a temporal-memory substrate without a global backward sweep

CONFIDENTIAL · Invention disclosure for counsel · Status: **unpublished** (all first commits after the 3 Oct 2026 public
boundary) · Jurisdictions: **EP and US open**, provided no other disclosure occurred. Inventor: Tero Keski-Valkama
(founder direction for every element; see `ip/INVENTORSHIP_AND_AI.md`).

First commits (UTC, 9 Oct 2026):
- duality theory, adjoint and trace identities `e6a1de7a` 18:22 (`experiments/theory/160_forward_backward_duality_and_credit.md`, `forward_backward_duality_check.py`);
- credit model trained by sampled causes `a91a40be` 18:26 (`experiments/credit/hindsight_race.py`);
- asynchronous and slow credit models `469a40c7` 18:27;
- trainable forward message head `90f8a6e7` 18:29;
- closed loop with one shared verdict `e132c67c` 18:36 (`hindsight_race_v5.py`, `check_closed_v5.py`);
- per-event exact traces `477af31d` 18:45 (`online_race.py`);
- matched compact credit audit `d0f22cf0` 19:17;
- pairwise verdicts with Metropolis–Hastings-corrected proposals `70124760` 20:02 (`pairwise_race_v7.py`, `pairwise_math.py`);
- hidden-write history scores `4150f907` 20:56;
- unshared backward matrices trained by the forward updates `d52f3f98` 21:44 (`online_race_v2.py`);
- exact per-event traces across depth `5e773d53` 23:44 (`online_deep.py`).

## 1. Technical field and problem

Training of recurrent and event-driven networks, including on neuromorphic, analog or delay-line hardware.
Backpropagation through time has three costs:
- It stores the full history.
- It runs a global backward sweep, synchronized across the whole network.
- It reuses the forward weights transposed, which on physical hardware means moving weights to a second datapath
  ("weight transport").

Stored activations are the only thing the forward pass gives the backward pass, and nothing in the backward pass learns.
Hard (winner-only) routing has no gradient at all; its usual estimators (REINFORCE, straight-through) are noisy or need
every alternative evaluated.

## 2. Solution (elements, combinable)

(a) **Time-reversed adjoint by the same physical structure.** For a memory whose modes evolve as
z_t = a_t z_{t−1} + b_t, with a_t = e^{(−r+iω)Δt}, the adjoint obeys λ_t = c_t + a*_{t+1} λ_{t+1}. This is the same
decay with conjugate rotation, run backward. The backward pass is executed by traversing the same delays and decays in
reverse, so no separate backward datapath is needed.

(b) **Credit as posterior inference in a race.** For racing clocks, ∂ℓ/∂θ_i = ρ_i − H_i: posterior responsibility minus
exposure. A *credit model* q (the backward pass's own slow parameters) predicts causes of an outcome. It is trained on
causes the forward race itself sampled (labelled for free). Its responsibilities replace backpropagated errors.
Variants:
- q trained asynchronously from a stale forward snapshot through a replay buffer;
- q trained at a reduced rate.

(c) **Closed loop: one verdict trains both directions.**
- q proposes S causes per real outcome.
- The forward pass evaluates only those, giving weights w_s ∝ p(y, k_s | x)/q(k_s | x, y).
- The same self-normalized weights train the forward model (router and the selected experts) and q (towards the
  weighted proposals), plus the sleep anchor.
- The joint fixed point is q = exact posterior and θ = maximum marginal likelihood. The credit gap KL(ρ‖q) is the
  convergence monitor.
- Pairwise variant: two forward-scored causes give an exact relative target a_j/(a_i+a_j) for q's odds (convex for fixed
  pair weights; unique optimum q = ρ). Proposals are corrected by independence Metropolis–Hastings, so the invariant law
  is exact even when q is wrong.

(d) **The forward pass learns what to tell the backward pass.** A forward message head φ emits a small learned message
per event. q sees only (message, outcome), and φ is trained solely by credit quality, so the forward computation is
shaped to make credit assignment easy. Further forward-to-backward messages are claimed:
- the sampled causes;
- precomputed sensitivities (traces);
- the forward pass's own predicted credit, so only the surprise travels backward as sparse events;
- confidence;
- near misses (runner-up clocks).

(e) **Exact per-event learning by forward eligibility traces through decaying-rotating memories, including across
depth.**
- Each memory parameter carries S_t = a_t S_{t−1} + ∂(a_t z_{t−1} + b_t)/∂θ. On each event, the local loss gradient
  with respect to the state is contracted with the traces and the weights update immediately. There is no stored
  history and no backward sweep.
- For stacked memories, the upper layer carries traces with respect to every lower parameter:
  S_2,t = a_2 S_2,t−1 + own_2 + J_2 ∂u_2/∂P, with ∂u_2/∂P = ∂u/∂P + R_1 S_1,t. Cost per event is
  O(n_upper × |P_lower|), independent of sequence length.
- Optional layer-local truncation (no upper-layer history with respect to lower parameters) as a cheaper mode.

(f) **Unshared, mutually trained directions (no weight transport).**
- Each matrix whose transpose would carry error gets its own feedback matrix B.
- B receives the identical locally computed update that its forward partner W receives (pre-/post-activity product),
  with the same decoupled decay. W − B then contracts by (1 − η·wd) per step, so duality is learned rather than assumed.
- Combined with (e), the whole learner is local. On a delay-line substrate, the backward path is a separate physical
  structure that learns to become the time-reversed dual of the forward one.

## 3. Technical effects (measured contracts; learning tests queued, `experiments/theory/160_*.md` §§6–12)

| Contract | Result |
|---|---|
| Time-reversed adjoint vs autograd | 4.4e−16 |
| Forward traces vs autograd | 2.7e−15 |
| Race credit = responsibility − exposure | 1.1e−16 |
| Posterior-credit router update vs exact marginal gradient | 2.4e−7 (float32) |
| Closed-loop estimator, uniform enumeration, vs exact gradient | ≤ 4.2e−7 |
| Pairwise MH: detailed balance / stationarity | 1.7e−18 / 2.8e−17 |
| Per-event trace learner vs BPTT, one memory layer (fixed weights) | 4.4e−16 (fails at 1.0 with traces removed) |
| Per-event trace learner vs BPTT, two memory layers | 9.6e−16 (layer-local truncation differs by 112–124%) |
| Mutually trained feedback started at B = W reproduces the exact learner (9 per-event AdamW updates) | difference 0.0 |

Measured learning (Taxi DEV log-likelihood per event, 3 seeds, 20 passes over TRAIN; theory note 160 §§7, 11):
- per-event trace learner (e) 0.4805 vs BPTT 0.4557; the traces add +0.0077 over per-event learning without them, on every seed;
- unshared feedback trained by the forward updates (f) 0.4715 vs shared exact transposes 0.4736, with alignment cosine
  1.000 on every feedback matrix; fixed random feedback 0.1699 (cosines 0.01–0.28).

Modeled learning work at depth 2 for the pairwise credit learner: 4,064 vs 17,280 leading-linear MACs per target
(0.235× dense). Measured fitting quality, credit gap and wall time are pending (queues listed in `experiments/HANDOFF.md`).

## 4. Closest prior art (external; to be searched by counsel)

- Wake–sleep and Helmholtz machines: separate recognition and generative weights, but no shared objective.
- Reweighted wake–sleep (Bornschein & Bengio 2015); Le et al. 2019.
- RBMs: shared weights in both directions.
- Feedback alignment (Lillicrap et al. 2016): fixed random feedback.
- Kolen & Pollack 1994; Akrout et al. 2019: weight mirroring and mutual updates in feedforward networks.
- Synthetic gradients (Jaderberg et al. 2017).
- RTRL (Williams & Zipser 1989); e-prop (Bellec et al. 2020).
- Learned optimizers (Andrychowicz et al. 2016).

Candidates for novelty are the combinations specific to our substrate:
- exact adjoint and trace duality for decaying-rotating event memories executed by the same delays;
- credit supervised by the races' own sampled causes, with the closed-loop shared verdict;
- the trainable forward message head for the credit model;
- exact cross-depth forward traces for stacked diagonal temporal memories;
- mutual feedback learning combined with forward traces as a fully local learner on a delay-line substrate.

## 5. Claim sketch (for counsel)

1. A method of training an event-driven network comprising decaying-rotating memory modes, wherein parameter updates are
   computed upon each event from a locally available error with respect to the memory state and forward eligibility
   traces propagated with the same decay and rotation coefficients as the memory, without storing the event history.
2. The method of 1 for stacked memories, wherein an upper memory carries traces with respect to parameters of lower
   memories and of the inter-layer readout.
3. A method in which a credit model, distinct from the forward network, is trained on causes sampled by the forward
   network's own races and provides per-cause responsibilities used to update the forward network.
4. The method of 3, wherein the forward network scores causes proposed by the credit model and the resulting
   importance weights update both the forward network and the credit model.
5. The method of 3, wherein the forward network emits a learned message trained by the credit model's loss.
6. A learning system in which feedback matrices distinct from the forward matrices receive the forward matrices' local
   updates and decay, combined with the traces of claim 1, implemented on a delay-line or analog substrate in which the
   backward path is a separate physical structure.
