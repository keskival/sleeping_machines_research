# IDF-02 — State clock over persistent memory, and resolution-safe continuous-time event models

CONFIDENTIAL · Invention disclosure for counsel · Status: **unpublished** (state clock 6 Oct 20:11 UTC `c6169b89`;
audit 6 Oct 16:57 `38969520`; cell-held hazards 6 Oct 23:38 `8709beb0` and 7 Oct 03:47 `5871aa8f`; target-only
dequantization 7 Oct 02:42 `d5ccb83f`) · Jurisdictions: **EP and US**

## 1. Problems

**(a) Silence.** Clocks parameterized once at the last event cannot follow a process whose state keeps evolving during a
long interval without events.

**(b) Recording resolution.** Event times from logging and sensing equipment are quantized (whole seconds, 10⁻⁴ units,
2⁻¹³ units). A flexible continuous-time density can obtain arbitrarily large, spurious likelihood by concentrating mass on
grid points or exactly at zero gaps. The gain is a property of the recording equipment, not of the process; it misleads
model selection and degrades real forecasts. We observed spurious gains of 0.71, 0.23 and 0.22 nats/event on Retweet
before the fix.

## 2. Solutions

**(a) State clock (IDF-02a).** A clock whose hazard and mark law are read from a persistent complex memory that evolves
continuously through the silence: z(τ) = e^{(−r + iω)τ} z_i; λ_s(τ) = softplus(Re-linear(z(τ)) + c_i); marks
softmax(U z(τ) + V h_i). The compensator ∫λ_s is computed by Gauss–Legendre quadrature (deterministic, a fixed number of
nodes), checked against a 400k-point reference to 1e−11.

**(b) Resolution principle (IDF-02b).** Given the recording cell c (the smallest time resolution of the source):
- every clock's hazard and mark law are held at their value at τ = c on [0, c):
  log S(τ) = −h(c)·min(τ, c) + [log S₀(max(τ, c)) − log S₀(c)], so no density can sharpen inside one cell, and an event
  at exactly τ = 0 still has positive hazard;
- delayed clocks: dispersion σ ≥ max(0.005, c·e^{−μ}); windows: width and edge scale ≥ c;
- state clock dynamics caps: rotation ≤ one period per 8 cells, decay ≤ one e-fold per cell.

**(c) Target-only dequantization (IDF-02c).** During training, only the scored target gap is replaced by a uniform draw
within its recording cell (zero gaps within the first half-cell); the history fed to the encoder stays as recorded, so
training and deployment inputs match. (Dequantizing whole histories created an input mismatch, DEV −6.70; dequantizing
targets with zero hazard at τ = 0 failed, DEV −6.61; holding hazards below the cell fixed both.)

**(d) Dequantization audit (IDF-02d).** A validation procedure: score a model on held-out data as recorded and with every
gap dequantized uniformly within its cell; a drop beyond a baseline flags resolution exploitation. Applied before any
model selection or reporting.

## 3. Technical effects

- StackOverflow: the state clock moved DEV time log-likelihood −0.717 → −0.688 and gave a confirmed TEST win
  (−2.1444 ± 0.0037 vs −2.163).
- Retweet (whole-second timestamps): resolution-held clocks with target-only dequantization reached −6.3262 ± 0.0009 vs
  −6.348 (best published) with an audit residual of 0.0038 nats/event (baseline 0.0006), at 1/15 of S2P2's per-event
  work.
- Reliability: the audit withdrew three spurious leads that would otherwise have been deployed or reported.
Evidence: `experiments/B1_EASYTPP.md` (development log), `experiments/tpp/check_cell_held.py`, `check_state_clock.py`.

## 4. Prior art to distinguish

Dequantization for discrete data in density modelling (uniform/variational dequantization in normalizing flows);
continuous-time state-space and neural ODE intensities; numerical quadrature of compensators. Candidates: holding hazards
and mark laws constant below the recording resolution with the stated survival form; target-only dequantization with
recorded histories; dynamics caps tied to the recording cell; the audit as a selection gate.

## 5. Draft claim concepts (for counsel)

1. A method of training and/or operating an event model on timestamps recorded at a resolution c, wherein each hazard and
   type distribution is held at its value at elapsed time c for elapsed times below c, the survival being computed as
   −h(c)·min(τ, c) + [log S₀(max(τ, c)) − log S₀(c)] in log form.
2. …wherein during training the target inter-event time is replaced by a random value within its recording cell while
   the input history is kept as recorded.
3. …wherein a clock's hazard is computed from a memory state evolving in continuous time from the last event, the
   cumulative hazard being evaluated by fixed-node Gaussian quadrature, with rotation and decay rates bounded by the
   recording resolution.
4. A method of validating a continuous-time event model, comprising scoring held-out events as recorded and with
   inter-event times dequantized within the recording cell, and rejecting models whose score drop exceeds a threshold.
5. Systems/media; monitoring applications as in IDF-01.
