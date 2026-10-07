# Selective state-space models (Mamba) as an exact member of the family

7 October 2026 · Contract `experiments/theory/mamba_containment_check.py` (float64, evaluation only)

## Statement

A selective state-space layer — Mamba's selective scan, h_t = exp(Δ_t A) h_{t−1} + Δ_t B_t x_t, y_t = C_t h_t + D x_t with
Δ_t = softplus(W_Δ x_t + b), B_t = W_B x_t, C_t = W_C x_t — is exactly the family member in which:

1. every token is an event delivered after a **learned, content-dependent delay** δ_t = Δ_t (Mamba's selectivity is time
   warping: its own derivation reads Δ as a discretization step);
2. persistent memories **decay with the real elapsed time** between arrivals, z ← exp((−r + iω)(τ_t − τ_{t−1})) z, with
   r = −A and ω = 0;
3. the arriving message **writes a content vector** δ_t B_t x_t into every memory (a dense write);
4. a **content-addressed read** with key C_t returns Re⟨C_t, z⟩ plus a skip term;

and there are no races and no hard routes. The equations map one-to-one, with the same parameter count and the same
computation; Mamba's convolution, gates, projections and normalization are local dense operations the family admits.
One scalar decay per head (Mamba-2 / SSD) is the special case r[d, n] = r[d].

## Numerical contract

| Check | Result |
|---|---|
| Mamba-1 selective scan vs the family member (64 steps, 6 channels, 8 states) | max \|Δy\| = 3.8e−13 |
| Mamba-2 scalar-per-head decay vs the family member | max \|Δy\| = 4.0e−13 |
| Rotating member (ω ≠ 0) vs best real-mode Mamba fit at the same state size | relative residual 1.4e−2 |

The last row is an optimization residual on one random instance, not a proof; structurally, real diagonal decays produce
only non-oscillating responses, while the family's memories also rotate with elapsed time.

## What this does and does not claim

- **Claims:** Transformers (through races: an exponential race samples exactly from a softmax; delay-coded aggregation
  reproduces softmax attention over delivered keys) and selective SSMs (through content-dependent delays over decaying
  memory) are both exact special cases of the family.
- **Does not claim:** that our current implementation is faster or learns better than Mamba at scale. Mamba's
  hardware-efficient parallel scan and proven language scaling are real advantages; our recurrence is a Python loop whose
  linear part is scannable.
- **What lies outside both** — temporal races deciding what happens and when, sparse addressed writes (capacity beyond
  activity), counterfactual credit for hard routes, delays deciding which messages meet, silence-aware supervision — is
  where the measured gains come from: silent delayed clocks (EasyTPP Amazon), statistic-valued addressed slots (P19),
  route credit (text8), and wins over S2P2, a continuous-time SSM for events, on three of five EasyTPP datasets.
