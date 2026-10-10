# 164 — Sparse, forward-trained, delayed credit: what to condition on, what to share, when to send

10 October 2026, AWS session, on founder direction: advance the formal theory of sparse, forward-assisted /
forward-trained error crediting; define what the optimal credit functions are conditioned on, whether they share
parameters with the forward pass, how asynchrony helps, and how credit itself computes with time delays.
Numerical witnesses: [`credit/sftc_math.py`](../credit/sftc_math.py) →
[`aws_sftc_math_20261010.json`](../results/credit/aws_sftc_math_20261010.json) (all pass; synthetic, no fitting).
Builds on note 160 (§1 reciprocal adjoint, §2 race posterior, §§7, 11, 12 measured learners) and notes 162–163
(predicted credit with unbiased audits). This note adds five results; it does not restate those.

## Setting and definition

Forward state s_t = F_θ(s_{t−1}, x_t, Δt_t); for the temporal memory, mode m evolves z_{t,m} = a_{t,m} z_{t−1,m} + b_{t,m}
with a_{t,m} = exp((−r_m + iω_m)Δt_t). Loss L = Σ_t ℓ_t(s_t). Write c_t = ∂ℓ_t/∂s_t (local cotangent), e_t = ∂F_t/∂θ
(local parameter sensitivity, known to the forward pass at t), λ_t = dL/ds_t (adjoint). Then g = ∇_θ L = Σ_t ⟨λ_t, e_t⟩.

**Definition (SFDC learner).** A credit scheme assigns increments Δ_{t,β} to parameter blocks β, each computed by a
function with its own parameters ψ from an information set I_{t,β}, and applied at learning time t + d_{t,β}. It is
- *forward-assisted* if I contains forward-computed statistics beyond raw activations (eligibilities, transport
  factors, sampled causes, exposures, margins, predicted credit);
- *forward-trained* if ψ minimizes a credit objective whose targets are produced by forward evaluations (sampled race
  causes, pair verdicts, forward replays, TD targets from later forward states) and not by a backward sweep;
- *sparse* if, per event, only a data-dependent subset of (t, β) receives a nonzero message;
- *delayed* if d_{t,β} > 0 is allowed, chosen per block.

## Result 1 — The optimal credit function predicts a state cotangent, conditioned on transport

Among I-measurable estimators of g, mean-squared error is minimized by E[g | I] (orthogonal projection), and enlarging
I never increases it (Rao–Blackwell). Because e_t is known exactly to the forward pass at t,

    E[g | I] = Σ_t ⟨E[λ_t | I_t], e_t⟩   whenever e_t is I_t-measurable.

So **the credit function needs to predict only λ_t, a vector of state dimension, never a parameter-sized object**; the
forward pass supplies the contraction. Unrolling the adjoint over a window of H events,

    λ_t = Σ_{j=0}^{H−1} T_{t,j}^* c_{t+j} + T_{t,H}^* λ_{t+H},     T_{t,j} = a_{t+1} ⋯ a_{t+j} (per mode),

where T is the forward memory's own transport over the **observed gaps**. Hence the optimal credit function is
conditioned on exactly four things:
1. the forward state s_t (a sufficient statistic of the past for the future, by the Markov property of F);
2. the realized local cotangents c_{t:t+H} that have arrived (outcomes);
3. the transport T_{t,·} over the elapsed gaps (decays and phase rotations: time itself);
4. a prediction of the tail E[λ_{t+H} | s_{t+H}, …], the only part that must be learned.

**Witness C1b.** A credit predictor that learns fixed weights over the raw outcome window, without the transport,
errs by 0.583 (mean |λ| 1.579) on a fast mode; sharing the transport exactly errs by 0.0002, about 2,700× smaller. On a
slow mode (r = 0.05) the shared-transport error 1.764 is the unpredictable tail beyond the window, which item 4 carries;
learned weights err by 2.985. The credit function must see elapsed time.

## Result 2 — The adjoint is a Bellman equation whose discount is the memory's own elapsed-time transport

λ_t = c_t + a_{t+1}^* λ_{t+1} (per mode; general F: J_{t+1}^* λ_{t+1}). This is a policy-evaluation equation with
reward c_t and a **complex, gap-dependent discount** a_{t+1}^* = exp((−r − iω)Δt_{t+1}). Consequences:
- The Bellman operator is a sup-norm contraction with modulus ρ_m = sup_t |a_{t,m}| = sup_t e^{−r_m Δt_t} (witness C1:
  observed per-sweep ratio 0.99986 ≤ 0.99999; fixed point = exact adjoint, error 0.0).
- **Learned credit is temporal-difference learning of the adjoint:** train V_ψ(s) ≈ E[λ | s] on n-step targets
  Σ_{j<H} T^* c + T_{·,H}^* V_ψ(s_{t+H}), with all transport exact (Result 1). H = T reproduces BPTT; H = 0 is a pure
  synthetic gradient. The targets come from later forward states: the forward pass trains the credit model.
- **Credit horizon is per mode and known in advance.** Truncating transport after H events errs by at most
  max|c| · ρ_m^{H+1}/(1−ρ_m) (witness C2, all modes and H ∈ {2, 8, 32}). For a target ε,
  H_m(ε) = log(ε(1−ρ_m)/max|c|)/log ρ_m: fast modes need a few events, slow modes need long horizons or a learned tail.
  The forward pass's decay rates schedule the backward pass's work.

Precedent: synthetic gradients with bootstrapping (Jaderberg et al. 2017), TD(λ) (Sutton 1988), RTRL/e-prop for forward
sensitivities. New here: the discount is the substrate's physical transport, so it is exact, shared, gap-dependent and
complex (phase-rotating credit), and it fixes per-mode horizons from forward parameters.

## Result 3 — The optimal credit delay for each mode

Waiting d before applying credit to mode m lets more outcomes arrive (information deficit falls) but the weights move
(staleness grows). For mode decay r the transported-information deficit scales as C e^{−2rd}; staleness of a step
computed at θ_{t} and applied at θ_{t+d} scales as K(ηd)² for Lipschitz gradients and step size η. Minimizing the sum,

    d*_m = W(2 r_m² C / (K η²)) / (2 r_m)        (W = Lambert W).

Witness C5 (C = K = 1, η = 0.01): d* = 40.1, 28.6, 12.7, 6.6, 3.9, 1.6 for r = 0.02 … 3, matching grid search to 10⁻³.
**Slow modes should receive credit later, fast modes sooner.** A synchronous learner applies one delay to all modes and is
therefore suboptimal for every mode but one; asynchrony with per-mode delays is the optimum, not a concession. On a
delay-line substrate the credit for mode m travels on a line whose length is set by r_m.

## Result 4 — What to share with the forward pass, what to learn unshared, what to keep separate

(a) **No causal forward function can be the credit function.** The forward map at t is F_t-measurable; λ_t depends on
c_{t+1:T}, hence is not, unless the future loss is deterministic given the past. A credit function must have inputs the
forward pass does not (future outcomes, or their prediction).
(b) **Transport must be shared exactly** (r, ω and the observed gaps). It is the forward parameters, conjugated (note 160
§1); learning it instead costs error (Result 1, witness C1b) and parameters, and re-learning it would chase θ.
(c) **Local Jacobians may be unshared but must be trained by the forward updates.** Feedback matrices B receiving the
forward partner's local update contract to B = W (note 160 §11; measured cosine 0.99998, −0.0021 nats vs exact
transposes, while fixed random feedback fails by 0.30).
(d) **The tail predictor V_ψ and the cause posterior q_ψ are separate parameters,** reading the forward state s_t
(shared representation, not shared weights), trained by forward-produced targets: TD targets (Result 2), sampled race
causes and pair verdicts (note 160 §§6, 9).
So the credit system is **shared transport + learned dual Jacobians + separate forward-trained predictors**.

## Result 5 — Sparse credit with bounded or zero bias, and why asynchrony supplies the labels

**Deterministic send-on-delta.** Each block accumulates its pending credit contribution and emits only when
|accumulated| > κ, carrying the remainder forward. The applied gradient differs from exact by at most κ per coordinate at
any time, not growing with sequence length (witness C4: κ = 3 sends 528 of 1,200 messages with error 0.45 ≤ 3; κ = 10
sends 157 with 4.9 ≤ 10).
**Stochastic send-on-delta (unbiased).** Emit with probability p = min(1, |A|/κ) the value A/p, else carry nothing:
E = A exactly and Var ≤ κ|A|. Combined with Result 1's predicted cotangent, only the residual λ − λ̂ is sent, as in notes
162–163; their audit allocation p ∝ ‖residual‖/√cost is the Neyman optimum for this budget.
**Packets commute.** For fixed weights, credit packets transported to their writes and applied in any order or delay sum
exactly to the BPTT gradient (witness C3: 20,100 shuffled packets, error 9.4 × 10⁻¹⁵). Asynchronous credit costs only
staleness, which Result 3 prices.
**The race provides causes cheaply, with one caveat.** For clocks with hazards h_i, P(clock i fires first | first
event at τ) = h_i(τ)/Σ_j h_j(τ) = ρ_i(τ) (note 160 §2). (i) When the model *generates* (sleep), the winner is an exact
posterior sample of the cause of the event it produced: free labels for a forward-trained credit model, under the
model's own distribution. (ii) On *data*, the output posterior ρ is analytic and cheap; an internal race that selects a
hidden route is a sample from its causal **prior** π(k | past), and the later outcome supplies its posterior weight
p(y | k)/Σ_j π_j p(y | j). Turning those prior samples into posterior credit is the self-normalized/MH correction of
note 160 §§6, 9; it is not free.

## Implementation consequence for the integrated models (B1 event TPP, R1 keyed memory)

Per layer and mode: the forward pass emits e_t (or its eligibility trace) and the transport a_t; credit is the
transport-exact n-step adjoint over H_m(ε) events plus V_ψ(s_{t+H}) for slow modes; mode credit is applied after d*_m;
blocks send by stochastic send-on-delta; readout/head feedback is unshared and mutually trained; hidden route credit
uses the race's own sampled causes and pair verdicts. Retained: temporal computation, sparse addressed writes, separate
keys and values, counterfactual credit to losing routes, silence (exposure terms appear in c_t through the survival
integral).

**Next decisive tests** (B1 Taxi DEV, the existing online-learning harness, three seeds; TEST sealed):
1. *Horizon schedule:* per-mode H_m(ε) truncation vs exact traces vs uniform H: likelihood vs credit work.
2. *TD tail:* H = 4 transport-exact + learned V_ψ vs H = 4 truncation vs full traces; prediction: V_ψ recovers most of
   the slow-mode gap at a fraction of the trace storage.
3. *Per-mode delays:* d*_m vs a single delay at equal staleness budget.
4. *Stochastic send-on-delta:* messages vs likelihood at κ ∈ {0, small, large}; unbiased vs deterministic.
Each reports complete fitting work (trace storage, credit-model training, messages), not per-update work alone.

## Novelty and disclosure

The components have precedents (Rao–Blackwell, TD learning, synthetic gradients, RTRL/e-prop, Kolen–Pollack feedback,
wake–sleep and reweighted wake–sleep, send-on-delta sampling, gradient sparsification). The combination specified
here — credit conditioned on the substrate's exact elapsed-time transport, TD with a complex gap-dependent discount,
per-mode horizons and delays derived from forward decay rates, race-sampled cause labels and unbiased event-triggered
credit messages — was not found in the literature reviewed for notes 160–163; a formal prior-art search is pending.
This material is post-boundary and unpublished (ip/ disclosure register); do not disclose before the priority filing.

# Part II — Analytic theory: credit as an information and estimation problem

The results above have numerical witnesses; what follows is derived. Throughout, the per-mode credit model is the
backward recursion λ_t = c_t + a_{t+1}^* λ_{t+1} with local cotangents c_t modelled as independent, zero-mean, circular
complex Gaussian with variance σ² (Gaussian is the maximum-entropy, hardest source under squared error, so every rate
below is an upper bound for any other source with the same second moments), and future gaps drawn from a stationary
process independent of the past. |a_t| = e^{−rΔt_t}; with constant gap Δ, ρ = e^{−rΔ}.

## Theorem A — Elapsed time delivers credit information at rate 2r nats per unit time

Let τ be the time elapsed after event t and R(τ) = Var(λ_t | s_t, c_{t:t+d(τ)}) the residual credit variance once the
outcomes up to τ have arrived. Then, in expectation over future gaps,

    R(τ) = e^{−2rτ} R(0)·(S_d/S_0),   E[ I(λ_t ; outcomes in (t, t+τ] | s_t, gaps) ] = 2rτ nats per complex mode.

*Proof.* Condition on the gap path. λ_t − E[λ_t | ·] = Σ_{j>d} (Π_{i≤j} a_i^*) c_{t+j}, with variance
σ² Σ_{j>d} Π_{i≤j}|a_i|² = e^{−2rτ} · σ² S_d, where S_d = Σ_{j>d} Π_{d<i≤j}|a_i|² and Π_{i≤d}|a_i|² = e^{−2rτ}.
Likewise R(0) = σ² S_0. For circular complex Gaussians the mutual information is the log ratio of conditional variances:
I = 2rτ + log(S_0/S_d). With constant gaps S_0 = S_d, so I = 2rτ exactly. With stationary gaps independent of the past,
S_0 and S_d have the same law, so E[log S_0 − log S_d] = 0 and E[I] = 2rτ exactly (whenever E|log S| < ∞). ∎

*Reading.* Asynchrony is not merely tolerated: **waiting is an information source with a known, constant rate** set by
the mode's own decay. A fast mode learns its credit almost immediately; a slow mode keeps receiving credit-relevant
information for a time 1/r. Credit for mode m is complete to within a fraction ε of its variance after τ_m(ε) =
ln(1/ε)/(2r_m), independent of event counts, a purely temporal schedule.

## Theorem B — Optimal delay (closed form, with the information deficit derived)

Applying credit after delay τ costs the residual variance R(τ) = R(0)e^{−2rτ} plus a staleness term from the changed
weights. For an L-smooth loss and step η, the expected loss change of a step taken with gradient computed at θ_{t} but
applied at θ_{t+τ} differs from the fresh one by at most L‖θ_{t+τ} − θ_t‖·‖ĝ‖ ≤ L G² η (τ/Δ); squared-error staleness of the
applied gradient is then O((ητ)²). **Assumption (model, not derived):** staleness cost K(ητ)². Minimizing
R(0)e^{−2rτ} + K(ητ)²:

    τ* = W(2r²R(0)/(Kη²)) / (2r),   R(0) = σ²ρ²/(1−ρ²) (constant gap).

*Proof.* Set the derivative to zero: 2rR(0)e^{−2rτ} = 2Kη²τ; substitute x = 2rτ: x eˣ = 2r²R(0)/(Kη²). ∎
Note 1: R(0) itself grows as r → 0 (≈ σ²/(2rΔ)), so the argument of W is ≈ rσ²/(Kη²Δ). For large r, τ* ≈ ln(·)/(2r)
falls with r; for small r, W(x) ≈ x gives τ* → σ²/(2Kη²Δ), a plateau set by the staleness price. τ* is nonincreasing in r:
fast modes take credit sooner, and all slow modes share the same staleness-limited delay. Note 2: the formula prices delay per mode;
a single global synchronization point cannot be optimal for more than one decay rate.

## Theorem C — Predictive coding of credit: the shared transport is worth log 1/(1−ρ²) nats per event and mode

Suppose the credit stream for a mode is sent to its parameter site at distortion D per event. Coding λ_t directly needs
rate R_dir = log(σ_λ²/D) nats per complex mode with σ_λ² = σ²/(1−ρ²) (stationary). If the site already holds the
transport a and the previous credit λ_{t+1}, only the innovation c_t is new and R_pred = log(σ²/D). The saving is

    R_dir − R_pred = log(1/(1−ρ²)) ≈ log(1/(2rΔ)) nats  for rΔ ≪ 1.

*Proof.* Rate–distortion of a circular complex Gaussian of variance v at MSE D is log(v/D) for D < v; λ_t given
(a, λ_{t+1}) has variance σ². With the decoder holding only its own reconstruction of λ_{t+1} (closed-loop predictive
coding), the saving tends to this value in the high-resolution limit D ≪ σ². ∎
**The slower the mode, the more a shared transport saves.** This is the information-theoretic reason, behind Result 1's
measured 2,700× error ratio, for sharing r, ω and gaps with the backward path rather than learning them.

## Theorem D — Credit messages are a Wyner–Ziv problem; the backward path needs neither forward weights nor state

Model one learning event as: the outcome side (encoder) observes λ; the parameter site (decoder) holds side information
Y from its own forward pass (state s_t, eligibility e_t, transport, its own prediction λ̂). The minimal rate for
distortion D is the Wyner–Ziv rate R_WZ(D) ≥ R_{λ|Y}(D). For jointly Gaussian (λ, Y) under squared error there is **no
rate loss**: R_WZ(D) = R_{λ|Y}(D) = log(Var(λ|Y)/D) (Wyner–Ziv 1976; Wyner 1978), and the saving over sending without
side information is exactly I(λ; Y).
Consequences:
1. **The value of a forward statistic for credit is its mutual information with the adjoint**, I(λ; Y) nats per
   event. This is the formal answer to "what information from forward propagation helps": rank candidate forward
   statistics (eligibility, causes, exposure, margins, predicted credit, confidence) by I(λ; Y | already sent).
2. **Locality without loss.** The encoder need not see Y: the backward path can compress credit without access to the
   forward weights or state, and each parameter site decodes with its own forward side information at no rate penalty
   in the Gaussian case. Separate, unshared backward parameters are therefore not a handicap in communication terms;
   what must be shared is only what both ends observe (the transport, Theorem C).
3. For non-Gaussian credit the Wyner–Ziv loss is bounded (Zamir 1996: at most 0.5 bit per real dimension under MSE),
   so the conclusion holds up to that constant.

## Theorem E — Sparsity is the rate–distortion optimum: weighted reverse water-filling

Let the residual credit after forward prediction have independent components u_i with variances v_i =
Var(λ_i − λ̂_i | Y) and let a credit error in component i cost w_i = ‖e_i‖² in parameter space (the eligibility norm: a cotangent error δ_i
produces gradient error δ_i e_i; for independent zero-mean errors cross terms vanish in expectation). With an MMSE
reconstruction ĝ (error orthogonal to ĝ), a step −ηĝ on an L-smooth loss changes it by −η(‖g‖² − Σ_i w_i D_i) +
(Lη²/2)E‖ĝ‖² in expectation, so credit distortion costs η Σ_i w_i D_i of first-order progress. Minimizing Σ_i w_i D_i under a total rate budget R gives

    D_i = min(θ/w_i, v_i),   rate_i = log(v_i w_i/θ)₊ ,   with θ set by Σ_i rate_i = R.

*Proof.* Lagrangian of Σ w_i D_i + μ Σ log(v_i/D_i) per component; stationarity gives w_i D_i = μ unless D_i hits v_i
(component not sent). ∎
**Components with v_i w_i ≤ θ receive zero bits: no message at all.** Sparse credit is the optimum, and the rule for which
coordinates are silent is explicit: small residual uncertainty after forward prediction (v_i) or small eligibility
(w_i). Both quantities are forward-computable, so the forward pass decides where credit is sent. The send-on-delta
threshold of Result 5 implements this rule online with κ_i ≈ √(θ/w_i).

## Theorem F — The cost of single-cause (race-sampled) credit is the missing information, bounded by collision entropy

For a race with posterior responsibilities ρ over K causes and complete-data scores v_k = ∇_θ log p(y, k), Fisher's
identity gives the exact credit ḡ = Σ_k ρ_k v_k. Using the race's own sampled first finisher k ~ ρ instead is unbiased
with covariance equal to the **missing-information matrix** (Louis 1982), Cov = Σ_k ρ_k (v_k − ḡ)(v_k − ḡ)ᵀ, and if
‖v_k‖ ≤ G,

    tr Cov = ½ Σ_{k,l} ρ_k ρ_l ‖v_k − v_l‖² ≤ 2G² (1 − Σ_k ρ_k²) = 2G² (1 − e^{−H₂(ρ)}),

with H₂ the collision (Rényi-2) entropy of the posterior, in nats. *Proof.* Expand the double sum, bound each squared
distance by (2G)², and use Σ_{k≠l} ρ_k ρ_l = 1 − Σ ρ_k². ∎
**When the race is decisive, sparse single-cause credit is almost exact; its variance is governed by how uncertain the
cause is, not by K.** Averaging S race samples divides this by S; the M-sample pair/MH constructions of note 160 reduce it
further. Combined with Theorem A, a forward pass that waits τ before attributing also sharpens ρ (more outcomes), so the
collision entropy and hence the sparse-credit variance fall with elapsed time for temporally extended causes.

## Theorem G — Learning the credit model from generated race labels: sample complexity

In generation (sleep) each produced event carries an exact posterior sample of its cause under the model (Result 5 (i)).
For a well-specified regular parametric credit model q_ψ with p parameters fit by maximum likelihood on N such labels,
E[KL(ρ_θ ‖ q_ψ̂)] = p/(2N) + o(1/N) (classical asymptotics of the MLE's expected KL). By Pinsker and Jensen,
E[TV] ≤ √(p/(4N)), so the router-gradient bias of note 160 §6 (≤ 2·TV·max‖∇ log π‖) is at most √(p/N)·max‖∇ log π‖
asymptotically. **Credit quality improves with the number of model-generated events alone, with no backward sweep.**
Caveat (note 160 §6): this fits the model's posterior ρ_θ; the posterior under the data distribution additionally needs
the wake-side verdicts (pairs/MH on data), whose bias vanishes as p_θ approaches the data.

## What Part II establishes for implementation

- Condition credit on the forward state, the elapsed-time transport and arrived outcomes; learn only the tail (A, C, D).
- Share transport; keep the backward path's compression and predictors unshared and local (C, D).
- Allocate credit messages by weighted reverse water-filling on forward-computable v_i and ‖e_i‖² (E); silence is optimal
  below the water level.
- Delay each mode's credit by τ*(r) (A, B); synchronous credit is suboptimal for all but one decay rate.
- Use race-sampled causes for hidden credit where the posterior is sharp, more samples where H₂(ρ) is large (F, G).
Open derivations: the joint optimum of delay and rate (both trade against staleness); non-stationary gap processes
(Theorem A becomes a conditional statement on the gap path); the coupled two-timescale convergence of θ and ψ.
