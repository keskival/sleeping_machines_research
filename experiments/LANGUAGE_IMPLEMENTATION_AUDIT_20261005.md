# Implementation audit: engineering work for scalable language learning

5 October 2026 · User-directed · Static source audit plus completed result records.

**The leading language code makes choices that we own and can change.** The findings below are engineering work to resolve, not accepted limits on the architecture or the language program. Implement the required capabilities and evaluate the resulting models; do not narrow the objective to accommodate existing code. The tokenized-language program should test and repair these restrictions while retaining temporal computation, persistent addressed state, separate keys/values, sparse selection and counterfactual learning.

Scope: leading compiled segment-batched native language path, its eager and sampled-credit alternatives, native streaming adapter, streaming result records and the existing full-write replay driver. This is not an exhaustive audit of every family member or a newly executed numerical benchmark. No intent is inferred from a restriction.

## Binding to completed evidence

The saved 90M four-pass result is `results/language_batched/aws_language_batched_90M_r4_p64d4_4pass_linear_l64_lr004_cmp_s6_20261004T193000Z.json`. All seven source hashes recorded in that result match the inspected current files. Its actual configuration is D4/H2/P64/U2, 128-character training segments, 64 lanes, local linear route credit, untied receiver maps and T256 evaluation. Therefore the restrictions below are grounded in an executed configuration, not just unused defaults.

The native credit win (2.506→2.371 bpc), capacity gain (2.371→2.345 at unchanged selected writes), and saved Transformer quality/work wins remain valid within their recorded protocols. These restrictions explain what to test next; they do not erase the successes.

## Current implementation choices and required engineering work

| Finding | Exact implementation/evidence | What is constrained | Repair priority and necessary comparison |
| --- | --- | --- | --- |
| **27-character task hardcoded in the main driver** | `language_batched_benchmark.py`: EYE=27, classes/content_dim=27, cross-entropy reshaped to 27; text8-only loader | Proper word/subword language comparison cannot run through this driver | P0: a new source-bound token-ID driver with the selected public reference tokenizer/data. Preserve the historical driver/results |
| **Token adapter constructs dense one-hot inputs** | `native_stream_language.py::forward_chunk` calls `F.one_hot` then the content linear map; batched code stacks dense marks | Scaling vocabulary increases avoidable allocation and input-projection work | P0: equivalent weight-column gather/embedding lookup; prove predictions, gradients and updates agree with the one-hot operation before replacing it |
| **All state reset at every fitting segment** | `compiled_logits` creates zero memories, arrivals, seen masks and context on every call; driver samples independent 128-character segments | No state or learning history survives that boundary in the leading fit | P0: contiguous document lanes carrying state between credit windows; detach the gradient graph where required without erasing numerical state; compare against the reset recipe |
| **Evaluation also limits history** | Window scorer invokes the reset path at T256 with stride 128 | Leading benchmark cannot demonstrate document-scale memory use | Use the reference's scored targets/history policy. Existing streaming scorer already exists: reuse it rather than invent another. Streaming evaluation alone does not train long memory |
| **Credit is a local linearized surrogate** | `linear_route_credit` and compiled equivalent add softmax-gradient-weighted detached alternative proposals | It does not replay how choosing another receiver changes future persistent writes and downstream loss | P0 learning target: a bounded continuation/write-aware credit comparison. Keep current value credit as successful control. Existing full-write replay machinery is a candidate, not an automatically scalable solution |
| **Losing private value maps receive no direct alternative-value gradient from that teacher** | `proposals.detach()` in local route surrogate; factual value gather differentiates the selected receiver | Rarely selected untied receivers can remain poorly trained, making capacity harder to recruit | Measure exposure and loss versus available slots; compare shared maps or derived recruitment/credit repair. Key/score and factual memory gradients still exist: do not report all loser gradients as absent |
| **Dense proposal work in the leading fitting backend** | `layer_step` computes input/output/control/gate proposals for all U slots before gathering the winner | Increasing stored capacity increases training work even at fixed writes | Reuse `sparse_training.py` winner-plus-sampled-alternative path, audit variance and fit quality on tokenized data. It still scores every key and stacks parameters; do not claim constant total cost |
| **Very small active capacity example** | Actual D4/H2/U2 configuration has 16 receiver slots, selecting eight per position | This run is not a large dormant-bank scaling test | After token and credit repairs, one controlled larger-pool point with exposure, quality and complete work; use saved small-pool model as control |
| **Fixed depth and all heads executed** | `for depth in range(D)` and every head evaluated at every input; per-layer max arrival joins heads | The leading member does not learn optional depth/head scheduling; its sparseness is receiver selection | Treat as declared member semantics. Measure useful depth before one derived schedule comparison; changing this is architectural, not a silent performance patch |
| **Bounded local clocks and score clamp** | Delay lies in (0.001,0.011); scores clamp to ±12; token input stamps increment by 1 | D4's total delay is <0.044, so the previous token completes before the next token at +1. Cross-token readiness waiting is inactive for this member. Clamped scores have zero gradient beyond bounds | Measure score saturation and clock sensitivity; compare one numerically contracted scale/race treatment only if diagnosis shows suppression. Timing still affects within-token transport and memory ages; bounded clocks are not proof that time performs no computation |
| **Shared routing noise across all lanes** | Noise shape is H×U and broadcast over the batch | Marginal race law is correct, but batch exploration is correlated | Measure gradient/exposure variance; compare independent lane noise if it answers the diagnosis. No measured quality penalty is established by source inspection |
| **Final weights rather than best validation checkpoint** | Main driver records final selection and logs a dev curve; E64 can save/select best validation weights | Potentially asymmetric checkpoint-selection opportunity | New tokenized driver must support development-selected weights with full work records. Existing final-weight scores remain untouched; no claim that selecting another checkpoint necessarily improves them |
| **“Passes” are random segment presentation budgets** | Uniform random starts sampled with replacement, no sequential epoch iterator | Pass count is not a guarantee that every training position was visited | Record actual targets/presentations and coverage; use the reference's data policy. Contiguous document lanes can give both coverage and persistent state |

## Evidence that constrains the diagnosis

Completed p96 six-pass streaming rescore: T256 1.888476 bpc, carried-state 1.888924 bpc. Deterministic carried-state routing scores 1.888363. Thus removing evaluation resets alone does not materially improve this trained checkpoint. The concrete experiment is persistent-state TRAINING on tokenized documents with useful longer credit, not another inference-only reset sweep. These are different scored-history policies and slightly different target counts; do not use their tiny differences as an isolated accuracy win.

The factorized first-time implementation uses a zero-forward autograd term with score derivative −T·softmax(s). Its `first.detach()` is deliberate differentiation construction, not evidence that all clock learning has been disabled. Likewise, detached alternative proposals in the local teacher target score credit; blindly removing detach changes the estimator and is not a justified repair.

Score bounds, initialization scales and shared noise are candidates to measure, not confirmed causes of the observed quality gaps. Restriction presence is established statically; a quality penalty requires the integrated comparison.

## Existing alternatives to reuse

- `NativeStreamLanguageModel.forward_chunk(tokens,state)` accepts persistent state; the segment-batched fit does not use that ability. Carry-versus-reset is a driver/backend restriction, not impossibility of state in the family.
- `language_stream_rescore.py` and its completed records already test carried-state inference.
- `sparse_training.py` implements cached key reads and sampled local credit, with existing tests for forward/gradient equivalence and estimator mean. Those contracts are present; they were not rerun in this static audit.
- `native_language_replay_benchmark.py` carries factual state across windows and provides full-write replay credit and validation selection. It hardcodes depth8, bounded data sizes and short credit chunks as a diagnostic protocol. Do not treat those guards as a family requirement or bypass them to scale: prepare a new admitted tokenized experiment.
- Theory §419 already identifies missing continuation/write utility and rare-route exposure. Its conditional mechanisms are hypotheses to test, not guarantees that a proposed bonus or shared map solves natural language.

## Engineering order

1. Proper token IDs and exact normalized prediction with public-reference data/tokenizer; efficient token gathers and CPU head/core/optimizer profiling.
2. Persistent document-lane fitting, separate state persistence from gradient truncation, development-selected checkpoints.
3. One diagnosed routing/learning repair: occupancy/exposure, local-versus-continuation utility or credit horizon. Use a small integrated paired fit, then a reserved larger point.
4. Winner-plus-sampled-alternative learning at useful larger capacity, charging discovery and optimizer work; measure quality and variance.
5. Clock/noise/depth changes only when the measured failure calls for them.

Every change must have a concrete failure, retained-mechanism statement, numerical contract and small integrated fit before scaling. Avoid both preserving a restriction for convenience and removing safeguards indiscriminately. No kernels, training queues or completed measurements were changed by this audit.

## Governing user direction: we own the implementation

The user explicitly rejects implementation restrictions as reasons to constrain the project. Existing code describes the starting point; it does not prescribe the capabilities, data representation, state horizon or learning rule of the next model. Change the implementation to serve the architectural objective. Resource measurements determine how to execute and prioritize the work, not whether current code is allowed to veto the objective. Numerical contracts verify that changes implement the intended mechanisms; they are part of engineering, not a reason to leave missing capabilities unresolved.
