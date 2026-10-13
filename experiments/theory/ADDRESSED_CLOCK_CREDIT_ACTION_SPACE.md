# Addressed clock credit: information and actuation must match

B10,13October2026. The native critic now reads individual mode theta/g/m/v but its retained action is still four uniform clock corrections. Better input identifiability does not remove a restricted output action space.

Let c in R64 add to the two clock-gradient legs of16modes in two layers. The retained action is c=Aa, a in R4, where each column of A is uniform1/sqrt16 within one layer/leg and zero elsewhere. A^T A=I. For actual clippedAdam plus refreshed producer state and eight future targets, define q=partial L/partial c at the issued cut. Then

    q_parallel = A A^T q
    q_omitted = q - q_parallel
    partial L/partial a = A^T q.

The last identity is a chain rule, not an approximation. Under a Euclidean gradient-action radius r, unrestricted optimal first-order loss reduction is r||q||; retained reduction is r||A^T q||. Parameter-step or forecast-curvature trust metrics require their corresponding metric projection, not this Euclidean formula. The actual optimizer's nonlinear global clip is included in measured q; full actor learning still has its ordinary gradients. Rank4 restricts only the extra learned correction, not the whole actor update.

## Completed native cut diagnostic

Source-bound curie_b10_clock_action_rank_20261013_v1: three real TRAIN learner cuts, same actor/Adam and observed state. Four-action derivative agrees with projected64-coordinate derivative within5.43e-20. Actual functional loss parity≤3.55e-11. Omitted squared clock-demand fractions0.937550/0.980824/0.962290. Actual small1e-5steps in the inaccessible negative-demand direction reduce future loss by1.05e-9/2.92e-8/9.77e-9. The first tiny difference is near the region where nonlinear/roundoff effects matter; no precision-optimal or practical gain claim is drawn from it. These are three local eight-target demand witnesses, not population coverage,64parameter total-learning dominance, whole-learner demand, delayed24target policy value or benchmark quality.

This explains a concrete expressive limitation; it does not prove the rank restriction caused every recent failed fit. A learned critic must acquire and generalize these demands before the larger action space is useful.

## Next integrated design

Retain current actor, temporal computation, hard races, private addressed critic banks, typed mode receipts, separate keys/values, actual actor clip/Adam and the exact null-parent contract. Add a zero-initialized bounded per-mode residual to the retained parent action. Route credit to selected clock targets; candidate target embeddings identify layer/mode/leg and local operator state. Shared temporal functions can transfer across target contexts, while address identity preserves mode differences.

First compare with fixed target support before learning admission. A sparse support's local rank can stay small while its union of possible addressed supports spans all64coordinates; four uniform columns cannot do so. Sparse updates alone do not eliminate candidate scoring or own-gradient/optimizer work. Charge discovery, losing-route teachers, message payloads, receipts, decoder/optimizer work and globalclip contraction separately. Availability64 and selected activity are distinct counters.

Numerical gate: zero residual preserves exact parent predictions/RNG; single addressed correction matches realAdam and future evaluation; pooled old actions reproduce old chain-rule derivatives. Integrated fit gate: same restored actor and source/data budget, paired future tapes, actual forecast utility, prefix/full prequential trajectories. Existing parent results retained until improved. Three-cut sensitivity is a design reason, never a predicted win.

## Implemented fixed-support member and conditional geometry

AddressedClockCredit uses64zero-initialized output rows with a frozen retained parent and the same native private-bank receipt path. For causal support S(x) of size4, the bounded chart is a residual0.5tanh(z) on S, zero elsewhere, multiplied by parent scale/4. Two successive decisions share the phase, so source learning at every other decision covers all16modes. Its support union spans64coordinates. A uniform-sphere direction in the active four-dimensional chart gives the conditional ball-smoothed gradient on that face; it does not estimate all64demand coordinates from one receipt. Native internal hard races remain learned; target support is fixed, not learned discovery.

The parent and own gradients still act on every clock coordinate. All64readout values and32receipt events are computed. Selected residual coordinates4 is not a claim of four-coordinate total actor fitting or clocklessAdam. Baseline/core behavior remains fixed under zero residual; added1020head parameters and all output work are charged.

The smaller active support also reduces the residual's energy. For identical latent four-vector and parent leg scales s, the uniform old0.5chart has four times the residual L2norm of the new one-mode-per-leg chart. A uniform0.125chart matches the new chart's L2norm: sixteen entries(s_i*0.125*tanh(z_i)/4) versus one entry(s_i*0.5*tanh(z_i)/4). Therefore recent score improvement toward the parent cannot be attributed to increased action rank without an amplitude-aware native control. This control is distinct from a dense external baseline. Both actual utility and complete work remain decisive.

At64receipts each mode receives only four current-decision labels;256give16. Reserved larger acquisition holds architecture/seed/radius/learning rate fixed and verifies the original64receipt prefix rather than selecting another small-budget member. It scores the full16384DEV trajectory with fixed critic weights and evolving private memory. None of these counts ensures generalization or removes the forward-teacher cost.
