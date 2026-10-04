# 154 — Key-scoring traffic floor and cheap keys (proposal, 4 October 2026)

**Status: proposal; nothing run.** It feeds P2 (hardware cost model) and P1-2 (cost flat in pool size).

## Concrete failure

The P2 cost model (`experiments/hardware_cost_model.py`, `results/diagnostics/hardware_cost_model_20261004.json`)
counts per-character reads in winner-only inference of the trained AddressedEventHeads models. Every slot's score is
`q · (key + key_read @ m)` with a dense P×P `key_read` per slot. The score therefore reads H·U·P² weights and all
D·H·U·P stored values on every step. Only the winner's `input`, `output`, `gate`, `control`, `rate` and `frequency` are skippable.
The weight-read fraction is 76% at pool 2, 43% at pool 8 and 30% at pool 32, with an asymptote near 25%. Available
capacity grows with U, but so does per-step traffic: dormant slots are not traffic-free. This is the
"dormant state is not proof of zero key-scoring cost" caveat (AGENTS.md), now quantified.

## Theoretical reason for a change

The score needs a content-dependent key for each slot, but not a full-rank transform of the slot's memory. Options that
retain race attention (the race over scores is unchanged) and keep keys separate from values:

1. **Low-rank key read:** `key_read = A_u B` with a shared `B` (r×P) and per-slot `A_u` (P×r). Compute `B m_u` once per
   write and cache the r-dimensional summary `s_u` with the slot. Scoring reads only `q`, `key_u`, `A_u` and `s_u`
   (O(U·P·r)), and the cache is refreshed only for the written winner. With r ≪ P, the P² term disappears from the
   per-step path.
2. **Write-time key caching:** store `k_u = key_u + key_read_u m_u` at write time. Decay and rotation between writes are
   analytic per channel (rate/frequency), so the read-time key is an elementwise transform of the cached key when
   key_read commutes with them. Otherwise use option 1. Per-step scoring then reads U·P cached values, not U·P² weights.
3. **Sublinear candidate discovery:** a coarse first race over slot groups selects a group, then a fine race runs within it
   (hierarchical race). Scoring traffic becomes O(√U) per head. Credit to unrealized groups uses the existing
   counterfactual route credit at both levels.

## Retained and removed

Retained: temporal races and race attention, sparse addressed writes, separate keys/values, persistent decaying
state, counterfactual route credit, silence-aware supervision. Removed: the full-rank, memory-dependent key transform
applied to every slot every step (options 1 and 2) or the flat race (option 3).

## Implications

Inference: per-step weight traffic approaches shared + (cached keys) + winner-only. At pool 32 that is roughly a 3× reduction
over today's 30% figure. Learning: option 2 changes the gradient path through the key (cached at write time; needs a
contract that the gradient equals the uncached form when key_read commutes with decay). Option 3 adds a second-level
route credit (cost charged).

## Required comparison before any long run

10M, p64/d4, pool 8, 4 passes, seed 6: baseline AddressedEventHeads versus option 1 (r = 16) and option 2.
Report test bpc at T256, whole-fit FLOPs, winner-only inference FLOPs, and the cost model's weights-read fraction
and bytes per character in one table. Promotion gate: within .02 bpc of baseline with ≥ 2× lower bytes per character. A
numerical contract (scores and gradients equal for option 2 where exact) precedes the fit.
