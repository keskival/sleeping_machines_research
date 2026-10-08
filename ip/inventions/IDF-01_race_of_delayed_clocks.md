# IDF-01 — Event prediction by a race of heterogeneous delayed clocks with per-clock mark laws

CONFIDENTIAL · Invention disclosure for counsel · Status: **unpublished** (first commit 6 Oct 2026 14:36 UTC `79444569`,
after the public boundary) · Jurisdictions: **EP and US**

## 1. Technical field

Computer-implemented forecasting of timestamped, typed event streams produced by technical systems (computer and
network logs, industrial and IoT sensors, transaction processing, physiological monitors), and devices that raise alerts
or schedule actions from such forecasts.

## 2. Problem

Neural marked temporal point process models either (a) parameterize an intensity and estimate the compensator ∫λ by
Monte Carlo sampling at every event (computational cost and estimator noise in training and serving), or (b) are
intensity-free mixtures that give an exact time density but predict the event type independently of the time at which
the event occurs. Both struggle with sharply multimodal and box-shaped inter-event gap distributions, and leading models
(S2P2, NeurIPS 2025) use 125K–300K parameters and comparable multiply-accumulates per event.

## 3. Solution

After each event i an encoder updates a persistent state h_i. A finite set of independent clocks starts; clock m has a
hazard h_m(τ | h_i) and its own mark distribution p_m(k | h_i[, τ]). The next event is produced by the first clock to
fire; its type is drawn from the winning clock's mark law. Consequently

- survival S(τ) = Π_m S_m(τ); total intensity λ(τ) = Σ_m h_m(τ);
- marked intensity λ_k(τ) = Σ_m h_m(τ) p_m(k | τ), so the predicted type depends on elapsed time through which clock is
  likely to win;
- per-event log-likelihood log λ_k(τ) + log S(τ) in **closed form** for all clock families below (no sampling); every
  losing clock is credited exactly through its survival factor.

Clock families (each introduced to fix a measured failure):

| Family | Survival S_m(τ) | Effect |
|---|---|---|
| Exponential | exp(−w r τ) | positive hazard at τ = 0; keeps the race proper |
| **Defective delayed clock** (IDF-01b): log-normal delay that fires with learned probability π | (1 − π) + π S_LN(τ) | lets most mass sit late without being pre-empted by early clocks (in a race of clocks that surely fire, the earliest almost always wins) |
| **Logistic-window clock** (IDF-01c): density ∝ σ((τ−a)/s) − σ((τ−b)/s), fires w.p. π | (1 − π) + π s[sp(−v) − sp(−u)]/Z, Z = s[sp(b/s) − sp(a/s)], computed in stable log form | flat delay windows with learned soft edges; edges receive gradient from every event, unlike hard-edged windows |
| State clock | see IDF-02 | hazard follows memory evolving through silence |

**Anchored windows** (IDF-01d): window starts and widths are initialized from components of the training gap
distribution (clustering) and constrained to stay near them: log a = log a₀ + 0.3·tanh(ρ_a), width = exp(log w₀ +
0.5·tanh(ρ_w)). Unanchored windows drifted beyond every observed gap in some training runs (a low-likelihood basin);
anchoring removed the basin.

**Single configuration with data-derived rules** (IDF-01e, 8 Oct 2026): one model configuration serves heterogeneous
event streams; two rules read from the training data replace per-dataset tuning: (i) window clocks are enabled only when
the histogram of log inter-event gaps separates into ≥ 2 components of ≥ 5% mass each (detected by empty-bin separation);
otherwise only exponential, delayed log-normal and state clocks are used, because windows on a single-humped distribution
create a weaker timing optimum (measured: StackOverflow seed 4 −2.1785 with windows vs −2.1406 without); (ii) the
recording resolution of the timestamps sets the cell of the resolution principle (IDF-02). Under a pre-registered protocol
this single configuration is ahead of the best published results on all five EasyTPP datasets (24 of 25 seeds), at
0.07×–1.03× S2P2's per-event compute.

## 4. Embodiments

- Encoder: stacked complex-diagonal temporal memories that decay and rotate with real elapsed time
  (z ← e^{(−r+iω)Δt} z + g ⊙ W u), plus an addressed mark memory; clocks' parameters are linear/softplus read-outs of h_i.
- Inference: next-event time distribution, most likely next type at any horizon, probability of no event before a
  deadline (S(τ)), alarm when the predicted hazard of a critical type exceeds a threshold.
- Training: maximize the exact log-likelihood; sampling-free.
- Hardware: clocks as physical delay elements racing on an event fabric; the first arrival selects the output.

## 5. Technical effects (measured; EasyTPP official splits, 5 seeds, sealed test)

| Dataset | Ours (nats/event, ↑) | Best published | Per-event multiply-accumulates, ours vs S2P2 |
|---|---|---|---|
| Taobao | 1.3991 ± 0.0025 | 1.318 (IFTPP) | 24,362 vs 26,016 |
| Taxi | 0.5250 ± 0.0010 | 0.522 (S2P2) | 20,708 vs 249,856 (1/12) |
| StackOverflow | −2.1444 ± 0.0037 | −2.163 (S2P2) | 36,732 vs 29,216 (matched-size model −2.1525 at 29,660) |
| Retweet | −6.3262 ± 0.0009 | −6.348 (NHP) | 19,850 vs 297,600 (1/15) |
| Amazon | 0.8028 ± 0.0007 | 0.781 (S2P2) | 35,936 vs 125,568 (0.29×) |

Exact likelihood removes the Monte Carlo compensator (the benchmark's own estimator agrees with ours within its sampling
noise). Evidence: `experiments/B1_EASYTPP.md`, `experiments/results/tpp/b1_final_*`, `experiments/tpp/work.py`.

## 6. Prior art to distinguish (non-exhaustive; counsel to search)

Multivariate Hawkes and neural Hawkes processes (type-specific intensities are a race of exponential clocks); intensity-
free log-normal mixtures (Shchur et al. 2020); competing-risks and cure-rate ("defective") survival models in statistics;
our own public theory note 09 (competing class hazards for classification) and `e44_tpp.py` (Hawkes TPP). The candidate
novelty is the combination: heterogeneous clock families with **per-clock mark laws** giving time-dependent marks in
closed form, **defective delayed and logistic-window clocks** inside a neural event model, and **anchoring** of window
parameters to data-derived gap components.

## 7. Draft claim concepts (for counsel; not legal claims)

1. A computer-implemented method of predicting events of a technical event stream, comprising: maintaining a state
   updated at each received event; after an event, instantiating a plurality of clock models each defining a survival
   function of elapsed time and a distribution over event types conditioned on the state; computing a probability of the
   next event's time and type as the event of a race in which the earliest-firing clock determines the time and its own
   type distribution determines the type, the likelihood being evaluated in closed form as the product of the clocks'
   survival functions and the sum of hazard-weighted type probabilities; and outputting an alert, schedule or control
   signal based on the predicted probability.
2. …wherein at least one clock fires only with a learned probability π, its survival being (1 − π) + π S(τ).
3. …wherein at least one clock has a density proportional to the difference of two logistic functions of elapsed time
   with learned edges and scale, its survival computed in closed form.
4. …wherein window positions and widths are initialized from components of the observed gap distribution and constrained
   to bounded ranges around them.
4a. …wherein window clocks are instantiated only if the distribution of observed inter-event gaps separates into at
   least two components, determined automatically from the training data.
5. …training by maximizing the closed-form likelihood without sampling the compensator.
6. …executed on event-driven hardware in which clocks are delay elements and the first arrival selects the output.
7. System and computer-readable-medium claims mirroring 1–6; application claims (IT/network monitoring, industrial
   sensors, transaction streams, physiological monitoring devices).

## 8. Inventors and contributions

Tero Keski-Valkama (founder statement: significant contribution to the conception of this invention); AI coding agents
assisted with implementation and experiments. Contribution record: `../INVENTORSHIP_AND_AI.md`.
