# 157. De-interleaving superposed processes by duration-matched attention (§§445–448)

8 October 2026, curie host session. Context: FAS v2 (B3), theory note 156 §444, B3_DEVELOPMENT_LOG.md (C9 diagnostic).

## §445 The merged log as a superposition with latent predecessors

A FAS v2 log merges K production lines; each line runs N items through a fixed route of steps a → b → c … Event j has
type x_j and time t_j, and (hidden) a predecessor π(j): the previous event of the same item, or none for an item's first
event. Given the predecessor, the gap Δ_j = t_j − t_{π(j)} is the item's own step duration, drawn from a law f_{ab} for the
transition a = x_{π(j)} → b = x_j (plus slowdowns under a fault). The merged-log likelihood marginalizes over predecessor
assignments that respect the route and use each predecessor at most once:

  p(x, t) = Σ_π Π_j [ r(x_{π(j)} → x_j) · f_{x_{π(j)} x_j}(t_j − t_{π(j)}) · S_{others}(t_j) ] ,

where r is the route law and S the survival of every still-pending item between events. Exact marginalization is a
matching problem (#P-hard in general); the identity oracle knows π.

## §446 Attention as an amortized posterior over the predecessor

For one event j the posterior over its predecessor among the last W events, ignoring the at-most-once constraint, is

  q_j(i) ∝ r(x_i → x_j) · f_{x_i x_j}(t_j − t_i) · [i is still pending] ,

so its log-score is log r(x_i → x_j) + log f_{x_i x_j}(Δ_ij) + log pending_i. C10's predecessor score is exactly this form
with learned parts: the type-pair term q(x_j)·k(x_i) learns log r, and the per-pair duration law
−(log(1 + Δ/s) − μ_ab)² / (2σ_ab²) − log σ_ab is a log-normal-type log f_{ab} on the log-gap. Training the merged likelihood
with the soft predecessor's message and duration as inputs is amortized expectation–maximization: the attention is the
E-step, the race-of-clocks model the M-step. The pending term is not modelled explicitly yet; recency enters through the
window and the shared gap term.

**Why C9 had to fail.** C9's timing term was one linear function of (log Δ, log² Δ, zero flag) shared by all type pairs.
Candidates for an event's predecessor are mostly events of the *same* type (the same step for other items), so r is
identical across them and only f can separate them; and f_{ab} differs by transition (conveyor steps of seconds vs
assembly steps of minutes). A single shared function of Δ cannot place its maximum at μ_ab for every (a, b) at once: it
reduces to a recency preference. Measured: C9 put 4.0% of its attention mass on the true item predecessor (uniform 3.1%)
and 11.5% on the previous merged event.

## §447 Identifiability: when timing can bind

Consider an event of type b and two same-type candidates i, i′ of type a, with ages Δ < Δ′. The duration law ranks them
correctly in expectation when the true age is closer to μ_ab in units of σ_ab than the competitor's age. With step
durations at 1% timing noise (FAS: σ_ab ≈ 0.01 μ_ab), two candidates are separable whenever their ages differ by more
than a few σ_ab, i.e. unless two items reached step a within about 3% of a step duration of each other. Hence:

- **Prediction 1 (binding):** with learned laws, attention mass on the true item predecessor should rise far above the
  uniform 3.1% (W = 32) or 1.6% (W = 64) toward the fraction of events whose predecessor is time-separable; the remaining
  confusions concentrate on near-simultaneous same-step items (ties, 36% zero-ms merged gaps partly from such pairs).
- **Prediction 2 (dropped events):** with drop rate p = 2%, the true predecessor is missing for ~2% of events; the posterior
  then picks the two-step-back event only if the law includes skip transitions (f_{ac} as a convolution). Without them,
  attention on those events is diffuse; the effect on AUROC is bounded by ~p.
- **Prediction 3 (detection):** when binding is correct, own durations become observable, so slowdown faults move the
  per-pair duration likelihood directly; the per-fault profile should depart from order3's (rank correlation below the
  .85 measured for C7–C9) and the retry-delay AUROC should rise toward the oracle's .890.

## §448 What remains for exact binding

Two terms of §445 are still absent: the **at-most-once** constraint (each predecessor serves one successor; a soft version
subtracts the attention mass a candidate has already received, a Sinkhorn-like normalization over the window) and the
**pending** factor (an item whose next event has already occurred is no longer a candidate). Both are local, causal and
cheap: they need, per candidate, the running sum of attention it has received. They are the next step if the C10
diagnostic shows high but imperfect purity with confusions among same-step items.
