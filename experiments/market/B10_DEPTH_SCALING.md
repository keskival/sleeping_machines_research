# B10: useful temporal depth on asynchronous market events

14 October 2026 · Owner: Curie · User-directed market/depth priority.

## Decision and existing evidence

Test whether eight temporal-memory stages improve the probabilistic trade-stream model over the completed two-stage native model on exactly the same chronological events. This is full-model development, not another small credit-only variant. Existing DEV likelihoods are native−6.882245, Hawkes−9.11712 and Poisson−13.70366 nats/event, higher better. TEST stays sealed. The B10 win rule still requires three seeds, all seven test days ahead and the realised-variance secondary gate; event likelihood alone is not a trading-profit claim.

## Why this member should benefit from depth

Trade bursts, interleaved buy/sell flow and slower activity regimes have different elapsed-time structure. Stacked decay/rotation memories can compose those patterns before the clock race predicts the next event time and mark. The recording-cell likelihood scores tied timestamps properly, and the exact hazard/survival calculation teaches all clock alternatives without a fixed sampled-route audit budget being spread over more layers.

The candidate grows the saved two-layer native checkpoint to eight layers. Each appended stage emits `x + A(layer(x,dt)-x)`, with A initialized to zero. This preserves initial evaluation predictions and old-parameter loss gradients exactly while providing nonzero first-update gradients to the new output maps. The temporal dynamics receive prediction credit after A moves; they are not claimed to receive first-update prediction credit. This addresses destructive depth insertion and allows useful corrections to be learned from a competent parent. It does not guarantee improvements or preserve training dropout RNG, optimizer history or physical work. Adam is restarted for the complete grown model; the parent weights and checkpoint hash are retained. No frozen parent is imposed. The added memories, decay/rotation and clock-race likelihood remain; hidden sparse routing is still absent. The candidate changes the native temporal depth2→8 plus these correction maps. Dimension32,16complex memory modes, six addressed mark slots, one exponential/eight delayed clocks, dropout0.1, learning rate0.003, batch16,30epochs/patience5 and TRAIN window stride20 remain fixed. Same1us interval likelihood, mark law at cell midpoint,128context/896scored targets per1024window. Eight layers are actually evaluated; prediction uses the deepest state and the addressed mark memory, with no earlier-layer output bypass.

This established clock-race member uses full BPTT and all-clock scoring. Hidden layers are not hard sparse routes, and the scalable learned-credit replacement is not integrated. All layers execute per event; mark writes are addressed while all slots decay/read. Keep these boundaries explicit. Success would demonstrate useful depth in this family member, not prove zero-cost dormant capacity or fully local deep learning.

## Pipeline and resources

The current D16 language fit completes unchanged. At that boundary, its quality scoring completes, then a uniquely named B10 eight-layer two-update numerical/resource smoke runs through run_safe. It uses full1024event windows and verifies finite training/DEV, all eight stage gradients and peak RSS. Its two-window DEV score is not used for model selection. The full candidate then fits all801registered TRAIN windows and scores all756registered VAL windows, with no small-smoke quality veto or architecture grid. The coordinator then stops; the unchanged32M language stage and duplicate horizon-only fit are withdrawn by the user-directed queue grooming. AWS owns the complementary acquisition/combined learning-repair path.

OneCPU/job,3GB RSS/12GB virtual cap, at least8GiB host availability and2GiB cgroup headroom. Full-fit timeout derives from measured smoke seconds/target plus validation/setup margin, capped at7days. Per-epoch checkpoints retain current/best weights, Adam, NumPy/Torch RNG, counters and history for exact recovery; source/data/settings must match. Prior results are never overwritten. A market resource failure is recorded and stops this pipeline; it never triggers a withdrawn acquisition.

## Results and next engineering decision

Report development-selected likelihood against both native and Hawkes references, total training targets, parameters, CPU training/full-process time, peak RSS and stage gradient totals. Split selected validation loss into time/mark and tied/positive-gap groups with exact target accounting. Those components decide the next repair; a worse first candidate is not a verdict on the family.

If eight-layer quality improves, confirm the selected recipe with independent seeds before any sealed score. Complete the price/quantity-informed realised-variance evaluation against EWMA/HAR before calling B10 won. Cost-inclusive trading performance requires its own executable protocol; earlier E17 strategies lost after2bp cost. The present primary model forecasts time and six marks, not size, realised volatility or executable fills.

Code:`b10_depth_tpp_v1.py`; coordinator:`scripts/continue_market_depth_v2.py`. Tests preserve the completed two-layer constructor/loss exactly, verify causal prefixes and finite gradients through eight stages, and simulate safe coordinator-only handoff plus rejection of withdrawn language launches on market failure.

## Neutral-initialization learning boundary

The temporal-function analysis in `../theory/TEMPORAL_LAYER_FUNCTION_GEOMETRY_STATISTICS.md` gives an exact XOR witness where a zero output map has zero prediction-error correlation, blocks internal feature credit and remains at a saddle despite available predictive information. Our neutral growth is therefore one initialization to test, not a guarantee that depth learns. The admitted smoke verifies actual finite stage gradients and the full fit measures improvement; neither a synthetic gradient fixture nor memory activity proves population usefulness. If growth stalls, inspect issued TRAIN correction/error alignment and internal credit before choosing a bounded nonzero coupling or changing the temporal function. No initialization grid or extra fit is admitted here.
