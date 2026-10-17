# Ledger: depth and composition in language (what was tried, what it showed, what follows)

Curated 16 October 2026 (Curie) from the result index (`experiments/index/experiments.csv`). One line per attempt family.
Losses are nats per GPT-2 token on FineWeb development slices unless noted. Protocols changed over time; compare only
within a row group:
- **dev**: early P24 slice, `best_dev_nll`;
- **65k**: 65,528-target slice;
- **flag**: flagship protocol, 8,192 targets in 1,024-token windows.

Update this ledger whenever a language depth/composition result completes.

**Dead ends and broken constructions are registered in [DEAD_ENDS.md](DEAD_ENDS.md); read it first.**

## A. Sparse race core with hard routes and counterfactual credit (P24), 5–6 October
- Token race core, data growth 64k → 256k → 1M tokens: dev 8.03 → 7.74 → 7.25 (`curie_data_growth_tokens_*_p24_*`). Credit
  window 64 vs 16: 8.21 vs 8.03 at 64k. Private banks and gather variants are level (8.02–8.08 at 64k).
- **Showed:** the native sparse race learns tokens, but slowly. Its 16-token credit window never rewards history beyond about
  64 tokens.

## B. Keyed temporal-memory member (addressed token-slot read), 7–9 October
- 1M: 65k 6.019 / 6.000 / 6.007 (3 seeds); without the keyed read 6.194. 4M (multi-epoch): 65k 5.521 / 5.499, flag
  5.624 / 5.601 (`curie_r1g2_keyed1_*`).
- **Showed:** addressed memory is the strongest single lever (+0.175 at 1M). The model was shallow.

## C. Depth on the race core (AWS integrated, protected growth), 12–14 October
- Integrated depth-4 at 512k: dev 7.41. Multi-depth readout 512k/2M/8M: 7.20 / 6.98 / 6.52. Protected depth growth: grown
  6.095 vs parent 6.066 (`aws_protected_depth_*`), so depth *lost*.
- **Showed:** stacking race stages did not compose. Credit and readout, not data, limited depth.

## D. Depth ladder on the race core (residual / serial / per-stage reads), 15 October
- 1M dev: residual+reads d2 / d8 / d16 = 6.998 / 7.049 / 7.851; plain residual d2 / d8 / d16 = 7.081 / 7.103 / 7.150;
  serial d8 = 7.176 (`curie_depth_ladder_*`).
- **Diagnosis** (RESIDUAL_DEPTH_20261015.md):
  - the serial chain passes only the winner's value; content cuts at any of 16 stages cost the same 1.56 nats;
  - routes collapse (D16 route entropy 0.33 of 1.39 nats);
  - deep receiver gradients fall to about 1e-5.
- **Showed:** passing a dense winner vector up a chain is the wrong interface (theory 170 §2).

## E. Episodic memory on the race learner, 15 October
- 2M: scored 5.634 vs control 6.108 (+0.474; +0.345 vs the 5.978 parent; `curie_episodic_memory_2m_*`). Gains concentrate on
  rare tokens.
- **Showed:** explicit pointers to the sequence's own past are worth more than added depth. This seeded CERL's episodic causes.

## F. Dense keyed hybrid ("plan-B"; now a labelled diagnostic control), 15–16 October
- 1M multi-epoch, 65k: L2 / L4 / L8 / L16 = 5.957 / 5.899 / 5.874 / 5.888. Async hierarchy b2l3 / b4l4 = 5.939 / 5.891.
- One pass at 4M, flag: L2 5.865, L8 5.841. Part-whole levels with observed word/clause/sentence boundaries on L2: 5.859.
- One pass at 16M, flag: L4 5.270, L8 5.218. The depth gain grows with data (4→8 layers: 0.052 at 16M vs 2→8: 0.023 at
  4M).
- **Showed:**
  - Depth pays when units pass dense residual vectors, but that is a Transformer/SSM hybrid (founder: not our model).
  - Observed-boundary part-whole summaries add almost nothing (0.006).

## G. Native causal event race (CERL, theory 169), 16 October
- 4M one pass, flag **5.4665** (65k 5.3094) vs dense L8 5.8414, at 3.3× less training compute per token
  (`curie_r1_cerl_4M_s0_20261016_v1`). The race has **2 temporal-memory state layers and 1 race level, with no depth**.
- **Error analysis:** the gain is from copying and continuation (copyable targets 3.16 vs 3.75); on novel tokens it is level
  with dense L8 (7.87 vs 8.02).
- **Showed:** the race of causes is the right unit. Its gap to win on novel content is compositional depth.

## H. Native depth attempts in progress, 16 October
- CERL-H, a 2-level hierarchy of races over the compressed cause stream (theory 169 §8, 170): 300-update check 6.53 vs 6.58;
  top-down gain learned 0 → 0.22; compression only ρ ≈ 0.75–0.83. The full 4M run is on Curie
  (`curie_r1_cerlh_L2lev_4M_s0_20261016_v1`).
- AWS fully native 8-layer local learner (receiver races, local eligibility, no recurrent backbone), 64k tokens: flag 7.047
  (`aws_r1_native_local_stream_64k_s0_20261016_v1`), about 29 tokens/s.

## H1. Residual memory (theory 172), 16–17 October
- 4M one pass, flag: K=2 5.4679, K=4 dense 5.4623, K=8 5.4712 vs CERL 5.4665 (no gain at 4M; 65k concept bank 5.4716, cause messages 5.4754).
- 16M one pass, flag: CERL 5.0634; K=4 dense stack 5.0433; **K=4 with old-slot refinement 4.9901** (selection 5.1906 vs 5.2826, prequential 5.0426 vs 5.1204) but 65,528 slice 5.0874 vs 5.0282 (8,192-token lanes: old corrections accumulate over 32x training history). Dense hybrid L16 5.2034.
- **Showed:** native depth pays once data suffices, and refining old memories is the strongest depth mechanism; corrections need a length-bounded algebra (learned decay, under test).

## H2. What the level-1 cause stream is (measurement, 16 October)
- `curie_cause_stream_typing_cerl4M_20261016_v1`: causes change on 79% of tokens (median duration 1; families and span
  identity do not help). Concept units are syntactic slots (function words, auxiliaries, sentence starters, subword
  continuations); 124 of 1,024 units are used. 4.5 bits per event, 0.6 bits predictable from the previous event.
- **Showed:** attribution events are not abstractions. Durable structure lives in memory, so layers must hand over memories
  as well as events (theory 170 §8, CERL-M).

## I. Synthetic compositional binding (G-HIER), 15–16 October
- Parts recognised 512/512; whole objects at chance (50%, NLL 0.693) even after 32× more training, and training
  arrangements are not fitted either (`investment/composition_scale_evidence.json`).
- **Showed:** binding parts into wholes is the unsolved step. More data or training does not fix it.

## H3. Flagship review fixes, 17 October
- Weight decay is now applied to matrices only (`--wd-groups 1`); concept symmetry breaking and load balance (`--unit-init 0.02 --balance 0.01`); per-target old-slot corrections (`--old-per-target 1`); 1,024-token training lanes. Contracts `curie_review_fix_contracts_20261017_v1`, `curie_rmr_contracts_20261017_v5`.
- AWS 16M arms on K4 residdecay: fixes; fixes + 1,024-token lanes; fixes + per-target corrections. Each compares against Curie's unfixed K4 residdecay run.
- The 4M capacity verdicts (65k/262k banks) are confounded by bank collapse; see DEAD_ENDS.md §3.
- Formal trainability conditions for composing good learners (C1 credit path, C2 sufficient interface, C3 identity at init, C4 each level reads something new, C5 conditioning, C6 alignment of local credit, C7 hard choices, C8 time, C9 stability): [theory 173](../theory/173_composition_trainability.md). Fixed with it: an RMR-2 crash (dense writes), straight-through firing `--fire-st`, fp32 old-slot algebra under bf16. Open: counterfactual write exploration, carried memory across lanes, product-key balance, native level 1.

## H4. Integrated member (theory 173 §5), 17 October
- Write-address race with counterfactual write probes, persistent memory across lanes (`--carry`), product-key balance, refined slot values for writers; contracts `curie_rmr_contracts_20261017_v8`. AWS: `aws_r1_rmr_integrated_K4_16M_s0_20261017_v1` (gate first) against the unfixed K4 residdecay run and the three single-fix arms.

## H5. Native level 1 (theory 173 §8), 17 October
- Sparse addressed slots (4,096; 4 written per token; learned clocks; bounded race read; write probes) replace the linear recurrence: `--level1 slots`. AWS: `aws_r1_rmr_nativeL1_K4_16M_s0_20261017_v1` against the integrated arm (identical except level 1), and `aws_r1_rmr_residdecay_pk65536fix_K4_16M_s0_20261017_v1` (capacity with a usable bank). The e2e smoke `aws_r1_rmr_native_e2e_smoke_20261017_v1` gates both.

## H6. Full native member (theory 173 §9), 17 October
- Firing clocks with carry inside RMR-1, full writer-input credit, `--old-norm` held, GPU driver carry/slots/firing. AWS: `aws_r1_rmr_nativefull_K4_16M_s0_20261017_v1` after the native-L1 arm; the e2e smoke v2 gates all native arms.

## H7. Sparse native member (theory 173 §10), 17 October
- Sparse level reads with read probes; product-key slot level 1 with sparse reads (256 key scores per token instead of 8,192; event-driven execution 12 state rows per token instead of 4,100, measured). AWS: `aws_r1_rmr_nativesparse_K4_16M_s0_20261017_v1` after the full-native arm; e2e smoke v3 and a GPU-driver CPU smoke gate it.

## H7b. Shared old-slot decay at 16M (17 October)
- K4 with one learned decay rate per level: 65,528 slice 5.0128 (no decay 5.0874, CERL 5.0282), flagship 5.0381 (no decay 4.9901), selection 5.2597 (5.1906). Fixes long lanes, costs short context; ahead of CERL on all three protocols. Repair: per-correction clocks (theory 174 P8) in the native flagship.

## H7c. K8 at 16M (AWS, 17 October)
- `aws_r1_rmr_residdecay_K8_16M_s0_20261017_v1`: selection 5.2589, 65,528 slice 5.0143, flagship 5.0385, identical to K4 with the same shared decay (5.2597 / 5.0128 / 5.0381) at 2.1x training work (32.7M vs 15.5M FLOPs/target). Confirms the theory 173 §14 prediction "K = 8 does not beat K = 4 by more than about 0.02 at 16M": useful depth is data-limited at 16M.
- Synthetic hierarchy (`curie_rhm_depth_K124_20261017_v1`, running): K = 2 resolves depth 0.92 against K = 1 0.41 at 4,096 trees (leaf NLL 2.076 vs 2.369), after losing at 1,024 trees (0.27 vs 0.38): the depth crossover falls between 1k and 4k trees, where the hierarchy's sample scale m^L v = 4,096 predicts it.

## H8. Depth versus data (theory 173 §14), 17 October
- Founder: useful depth grows roughly with the logarithm of the data (each level is one more layer of hierarchical
  decomposition), and depth needs both trainability and enough data per level. Both families gain more from depth at
  16M than at 4M (dense L4 → L16 0.067, native K4 0.073; at 4M 0.024 and < 0.01).
- Depth does not explain the whole slope gap: shallow dense gains 0.595 per 4× data vs CERL 0.403. Open candidate:
  usable capacity (CERL used 124 of 1,024 concept units). Tests: envelope comparison, synthetic hierarchy with known
  depth, grown depth with ReZero insertion, 64M K4/K8, fixed-capacity 16M arm.
- Implemented 17 Oct: capacity usage in every milestone, `--grow`, `rhm_depth.py` (Bayes ladder B = 2.60/2.03/1.71/1.55/1.47). Queued: hierarchy grid and fixed CERL 4M/16M (Curie v34); grown K4 16M and fixed CERL 64M (AWS).

## H9. Inductive bias and scaling (theory 173 §15), 17 October
- 4M → 16M: race causes' gain over background 0.437 → 0.405 (CERL), 0.433 → 0.386 (K4); background responsibility 0.24 → 0.32 / 0.36;
  the background route carried most of the improvement. The structured causes are a small-data offset, not yet a slope advantage.
- Pass condition for the fixed CERL pair: units_used and concept share rise with data. Scaling claims need three data sizes.

## H10. Pre-registered outcomes of the native round (17 October, written before the results)
Founder: if none of the current experiments shows a decisive advantage, will they teach enough to continue? The answer for each
outcome is fixed here, before the results arrive.

| result | if it holds | if it fails |
| --- | --- | --- |
| native 16M (s0, s1) vs K4 diagnostic 4.9901 | native becomes the scaling member | read certificates, event/firing rates, concept and clock usage; repair the bundle the ablations name |
| seed spread s0 vs s1 | sets the threshold below which no ablation difference counts | — |
| leave-one-bundle-out (6) | bundles whose removal costs more than the seed spread are carrying | a bundle whose removal helps is broken inside the member: repair it at its cause (DEAD_ENDS rule: no verdict from isolated tests) |
| native slope 4M → 16M vs dense 0.60 and CERL 0.40 per 4x | steeper than CERL: run native 64M even if 16M loses | flat: capacity and depth growth are the levers (CERL-fix pair, grown arm) |
| native learning rate (1e-3, 3e-3, 6e-3 at 4M) | best rate fixed for 16M/64M | a loss at the default rate is not an architecture verdict |
| Transformer ladder 4M (four rungs) | native ahead at matched training FLOPs: first external win at our budget | behind: compare slopes (Transformer 16M rungs) before any verdict |
| synthetic hierarchy K = 1/2/4, 1k-64k trees | depth trainable; resolved depth grows with samples | K > 1 never ahead: depth credit is broken, repair before more depth at scale |
| route-tree certificates | small excluded / uncertain mass: trained model near the full route tree | large: the hard model is far from the tree; threshold races and dual-race credit (theory 175) are next |
| cross-lane credit pair | carried and online scores improve: persistent memory needs credit | no change: memory is not yet used across lanes; check concept/slot usage |

Queued at 20:20 so they arrive with the rest: native 64M, a third native seed at 16M, Transformer 16M (two shapes) and 64M
(one shape) at lr 3e-3. Still not covered: threshold races, dual-race write credit and a TD critic (theory 175 §6,
not implemented); more seeds wherever a decision rests on a difference near the seed spread.

## What has not been tested yet (the open route, theory 170)
1. **A typed event interface between units.** Units exchange attributed-cause and surprise *events*, never dense vectors.
   CERL-H v1 does this, but with non-durable cause identities (ρ ≈ 0.8).
2. **Durable cause identities:** continuation pointers keep one identity per copied span; hysteresis in attribution; concept
   families. Target ρ < 0.5.
3. **Higher levels as race TPP units with real durations**, predicting which cause comes next and when the current one ends.
4. **Surprise channel:** unexplained tokens go up (history compression).
5. **Three or more levels and 16M tokens**, testing gain per level against data.
6. **Concept capacity at constant work** (product-key bank 4k–262k; ordered on AWS, not yet run).
7. **Native sparse state inside units** (cause excitations; AWS receiver core) in place of the temporal-memory vector.
8. **Carried memory across lanes** in training and scoring.

## AWS reconciliation, 16 October

- Completed older D16/8M race-core development: best DEV NLL 6.817296 (`aws_native_deep_d16_8m_20261013_s6_v1`); receiver-local clipping D8/8M: 6.910948 (`aws_local_clip_receiver_d8_8m_20261014_s6_v1`). These earlier protocols differ from CERL's flagship protocol; do not rank across them. They provide no reason to repeat the same depth construction.
- Calibrated/global clipping comparison after 2M additional targets: 6.226238 versus 6.231193 on the paired 8,192-target protocol, incremental work ratio 1.0041. This is a small locality-engineering point, not the language quality breakthrough.
- Current-source CERL/RMR admission gates pass on AWS. Ordered race prediction, residual-memory K2/K8, product-key capacity and messages take priority over further local-learner extensions. Current source-bound runs finish unchanged.
- [Cross-host synthesis and winning path](../theory/CURIE_SYNTHESIS_AND_WINNING_PATH_20261016.md).

## Composition credit contract, 16 October

The native historical producer replay previously stopped at saved child views. New bounded actual-child replay reconnects lower composers to delayed upper loss, preserving version/role identity and factual computation. Conditional-DAG gradient, work-cap and acquired optimizer/recovery validation is queued, not completed. No architecture-quality verdict or extra acquisition follows. See [contract](../theory/COMPOSITION_LEARNING_CONTRACT.md).

## Empirical composition-credit decision, 16 October

Analytical audit exposes first-order route-loss bias and distinguishes it from exact selected-mixture and soft-read credit. A new inverse-inclusion sampled path carries content VJPs through full available recorded abstraction depth at <=depth-1 extra reads/root. After analytical and acquired-learner gates, two complete native learners reuse one frozen acquired checkpoint for the same65536additionalTRAINtargets and8192causal onlineDEVtargets; canonical frozen8192/1024evaluation is separate. This tests content-credit coverage, not an exact historical route teacher. Gates and comparison are queued; no predictive advantage claimed. [Audit/protocol](../theory/COMPOSITION_CREDIT_ANALYTICAL_AUDIT.md).
