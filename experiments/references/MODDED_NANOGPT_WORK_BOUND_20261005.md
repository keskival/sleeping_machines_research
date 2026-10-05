# Published Transformer output-head arithmetic lower bound

The pinned 2025-01-26 modded-nanoGPT source and completed training log establish a lower bound without executing or retraining the reference. `reference_head_work_bound.py` verifies the source SHA against the metadata and the explicit projection/backward calls. The JSON receipt preserves both source and producer hashes.

The output uses width 768 and rounds 50,257 vocabulary entries to 50,304 output classes. The FP8 custom forward executes XWᵀ; its backward explicitly executes dY W and dYᵀ X. With two arithmetic FLOPs per multiply-add, each contraction costs 2 × T × 768 × 50,304. Training therefore costs at least 231,800,832 FLOPs per target; evaluation costs at least 77,266,944. Across 695,992,320 fitting targets the head alone costs 161,331,598.841610 GFLOPs.

This is conventional source-derived arithmetic, not an executed complete-reference trace. Attention, MLPs, embeddings, normalization, softcap, loss, scalar operations, casts, optimizer, communication and diagnostics add nonnegative work. `complete_reference_work` stays false. Precision and hardware cost differ from CPU float32 native arithmetic; no timing, memory traffic or energy ratio follows.

The full published model scores 3.2774 NLL on its public validation population. Native pilots have different training exposure and score development targets. Shared work columns expose raw arithmetic with common units and denominators; they do not establish a comparable-quality, matched-training-compute or scaling-law win.

Receipt: `experiments/results/diagnostics/modded_nanogpt_head_work_bound_20261005_v1.json`. Published source: https://github.com/KellerJordan/modded-nanogpt/tree/4ea6b937337a4889b8cfe3f38a93d120048d8f71
