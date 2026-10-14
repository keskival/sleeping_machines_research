# R1: preserve residual scales while developing useful depth

14 October 2026. Engineering prerequisite, no new fit admission.

## Concrete failure and intervention

The native constructor assigns every unit gain0.5/sqrt(depth). Rebuilding depth2 as depth8 halves the residual correction scale of the original stages, before any new stage is considered. This is a configuration coupling, not proof that it caused the observed quality deficit. The old packed backend supports only one uniform gain, preventing an explicit old-stage/new-stage allocation.

`stage_gain_token_core.py` supports one fixed gain per stage. It retains hard races, exact existing clock arithmetic, transport, sparse addressed memory writes, separate keys/values, alternative-credit kernels and all parameter banks. No route or time bypass is introduced. Nonuniform gains reproduce the corresponding existing module core, rather than changing the architecture into a dense carrier. No per-step parameter or matrix arithmetic is added: bank views substitute the declared scalar for each stage. Construction currently deep-copies the source before legacy packing and therefore pays temporary model-sized storage; this is explicit, not a claimed construction-memory optimization.

## Retained contracts and remaining growth work

Small synthetic unit fixtures verify exact sparse features, every stored state tensor, hard routes and RNG, plus every used original parameter gradient to1e-10, for tied and untied banks. Uniform packed bank views and parameter counts agree; new checkpoints preserve nonuniform gains independent of tensor dtype. These tests are implementation evidence, not a benchmark fit or depth-quality gain.

A future native growth model can now retain the parent's gains. It must also retain learned old parameters and optimizer ownership, old clocks and timestamped context semantics. Appending layers still changes event timing and recurrent final-context feedback; this adapter does not solve those problems or preserve a shallow model's prediction after append. Use theory34's old-teacher and optimizer boundaries, with actual native state/clock/context contracts. Do not transfer the market growth helper blindly: the market encoder uses observed elapsed time, while native layers alter delivery times.

No new architecture fit is admitted. Await the active deep/local learning results, select a specific useful-depth repair, contract it in the integrated native model and use a single representative resource smoke before acquisition. Compare the repaired learner with ordinary continued-parent fitting and richerD2 NLL6.099337 on matching targets/history, charging complete fitting and scoring work. No public validation tuning.
