# Tokenized exposure and learning-rate diagnostic — 5 October

The concrete failure is late development deterioration at constant learning rate 0.003 across the completed 64K two-pass fits. Both P24 seeds select step128, the first-pass boundary. The numeric persistent state and counterfactual write mechanisms remain intact; this diagnosis concerns the fitting recipe.

| Payload | Seed | Selected step | Final − selected DEV NLL | Final − selected TRAIN probe NLL | Logged gradients above clip |
| --- | --- | --- | --- | --- | --- |
| 16 | 6 | 128 | 0.131964 | -0.408231 | 4/4 |
| 24 | 6 | 128 | 0.149692 | -0.520631 | 4/4 |
| 24 | 7 | 128 | 0.228186 | -0.310478 | 4/4 |
| 32 | 6 | 64 | 0.262518 | -0.671713 | 4/4 |

Development-only trajectories. train_nll is a fixed fresh-state TRAIN probe of 1024 targets, not full-training loss. Gradient norms are pre-clipping at four logged checkpoints, not every update. Route entropy is factual race probability entropy; lower entropy does not by itself demonstrate receiver collapse or explain loss.

All four final development losses deteriorate while the fixed TRAIN probe improves. This is consistent with excessive adaptation to repeated data or unstable updates; the measurements do not distinguish those explanations. Logged route entropy decreases, but occupancy remains a separate quantity. Do not infer failure of the model family or discard persistent memory from these curves.

## Bounded next comparison

Keep admitted 256K fit unchanged. If P24 final-minus-selected development NLL exceeds 0.05 again while TRAIN probe improves, compare one lower constant learning rate 0.001 against saved 0.003 at the same data, initialization, seed, two-pass exposure and checkpoint cadence before another scale increase. Preserve all mechanisms. No candidate fit admitted by this diagnostic; complete fitting work must be audited independently.

A lower constant rate changes the magnitude of every optimizer update without changing temporal races, addressed state, keys/values, messages, selected writes, credit windows or actual alternative-write suffix supervision. Clipping bounds the gradient norm but does not bound AdamW parameter updates by the same number: moments and parameter scaling remain involved. A smaller rate may preserve useful first-pass learning into the second pass, or may simply underfit; the completed matched trajectory decides.

Use the existing engine CLI `--lr 0.001`; no architectural substitution or frozen-source edit is needed. Retain the original 0.003 result and its complete two-pass work. Compare selected and final development NLL, TRAIN probe, context/addressed-memory utility and actual full fitting cost. A successful seed6 candidate needs an independent repeat before promotion. Keep any changed-recipe cells separate from the fixed-recipe scaling surface, recording tuning work. The 256K result is required before admitting this comparison.

Read-only producer: `token_learning_trajectory_audit.py`. Receipt: `results/diagnostics/curie_token_learning_trajectory_20261005_v1.json`.
