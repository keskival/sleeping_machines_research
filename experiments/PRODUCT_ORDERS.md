**Update 17 Oct 17:40 UTC (Curie; founder audit: "Are we actually using all the things we have designed and proven?").**
- Every R1 flagship queue file now carries `# Role: flagship` and must pass `scripts/check_native_coverage.py` (all implemented native mechanisms on; designed-but-unintegrated mechanisms printed as declared gaps). Curie chain v36 runs the gate before the native run.
- Training now uses calibrated bank-local radial clipping (`--clip local`, the adopted policy in CALIBRATED_LOCAL_CLIPPING_ADOPTION.md) instead of global clipping. Contracts v19.
- AWS, first free R1 slots (gate first): `aws_r1_rmr_nativeev_K4_4M_s0_20261017_v1` (native slope point), then `aws_r1_rmr_nativeev_grow_K4_16M_s0_20261017_v1` (grown depth on the native member). These supersede `nativefull` (missing five mechanisms) and the dense-read grown arm. `integrated`, `residdecay_fix`, `aws_r1_rmr_grow_K4_16M` and `aws_r1_cerl_fix_64M` are diagnostics and run only after the native arms.
- AWS owns the forward-trained, reciprocal and future-credit learners: port the strongest current form onto `r1_rmr.py`'s native member as a paired arm against ordinary learning on the same member, instead of developing it on separate cores. These are the declared learning gaps the coverage gate prints.

**Update 17 Oct 17:10 UTC (Curie; founder: "You switched off all our unfair advantages and started tuning a version of ReZero Transformers"; the async spine must scale with depth and composition).**
- The flagship R1 member is the full native member with the composing event spine (`--events 1`, theory 173 §17): sparse slot level 1 with product keys, sparse level reads with probes, asynchronous firing, pointer events between levels (each level's read-race winner; the next level races over the lower pointer and the composed pointer with straight-through credit; event arrival drives firing; top-level events enter the readout race), write race, per-target corrections, carry, capacity fixes. Contracts `curie_rmr_contracts_20261017_v18`.
- RMR runs with dense reads and no firing or events (K4 resid/decay, the `integrated` and `residdecay_fix` rungs) are diagnostics only and are never fielded as the flagship (DEAD_ENDS).
- Curie chain v36 runs `curie_r1_rmr_nativeev_K4_16M_s0_20261017_v1` (e2e smoke v2, RSS gate, then the run) right after the running K4 decay job, before everything else.
- AWS: first free R1 slot runs `aws_r1_rmr_nativefull_K4_16M_s0_20261017_v1` (gate first); the grown-depth and 64M orders follow; the dense-read rungs only after the native arms.

**Update 17 Oct 14:15 UTC (Curie; founder: "Be guided by the results ... and fix all our weaknesses observed"; theory 173 §14).**
The results say: native CERL beats the dense diagnostic by a wide margin at matched compute, and depth pays more as data grow, but the native curve gains less per 4x data (CERL 0.40 vs dense 0.60) and no R1 training result has landed since 05:50 UTC. The slope is now the main question. AWS, when a slot frees (running source-bound jobs finish unchanged), run R1 in this order, each gate first:
1. `aws_r1_rmr_residdecay_fix_K4_16M_s0_20261017_v1`: capacity fixes in the depth member, against Curie's unfixed K4 decay run (done about 19:30 UTC).
2. `aws_r1_rmr_integrated_K4_16M_s0_20261017_v1`.
3. `aws_r1_rmr_grow_K4_16M_s0_20261017_v1` (new): depth that grows with data (levels enter at 2M, 4M, 8M), against item 1.
4. `aws_r1_cerl_fix_64M_s0_20261017_v1` (new): the 64M slope point of the fixed CERL curve (about 20 h).
5. Native L1, full native (unchanged).
Curie (chain v34) runs, after the K4 decay run: the synthetic-hierarchy depth test (`experiments/rhm_depth.py`: exact Bayes ladder, resolved composition depth for K = 1, 2, 4 at 1k-64k trees), then the fixed CERL pair at 4M and 16M (does usable capacity steepen the slope?), then the sparse native member and the PTB resume. Every CERL/RMR milestone now reports concept-unit usage (`capacity`: units used, unit perplexity, top 1% share); unfixed CERL used 124 of 1,024 units. Contracts: `curie_rmr_contracts_20261017_v16`, `curie_cerl_contracts_20261017_v14`, `curie_review_fix_contracts_20261017_v6`, `curie_rhm_depth_contracts_20261017_v1`.

**Update 17 Oct 18:40 (Curie; founder: "Fix all. We need to win!").**
- Curie runs `curie_r1_rmr_nativesparse_K4_16M_s0_20261017_v1` (chain v33) after its K4 decay run: e2e smoke, RSS gate below 2.4 GB, then the run, then the PTB resume. **AWS: skip `aws_r1_rmr_nativesparse_K4_16M_s0_20261017_v1`**; use that slot for the integrated, native-L1 and full-native rungs, which attribute the sparse result.
- Resume fix: `--product-keys` was missing from the behaviour-preserving defaults, so your running `aws_r1_rmr_residdecay_K8_16M_s0_20261017_v1` and Curie's K4 run would not have resumed after a stop under current source. Fixed and verified against both checkpoints.
- The region index now supports product-key regions (exact separable π, 2√C scores) and balanced filing (`--index-pk`, `--index-balance`); contracts v15. The indexed arm stays held.

**Update 17 Oct 18:00 (Curie; seventh review round, theory 173 §12).**
- Indexed history discovery is implemented and contracted. `aws_r1_rmr_nativeindexed_K4_16M_s0_20261017_v1` is **HELD**; its release condition is in its header.
- Your writer-input variant `experiments/r1_rmr_writer_input_credit.py` duplicates the fix now in `r1_rmr.slot_values`. That fix is exact, O(T d) memory, needs no checkpoint recomputation, and matches the one-hot reference in values and gradients. Please use `r1_rmr.py` for the ladder and retire the variant.
- The ladder order is unchanged.

**Update 17 Oct 17:15 (Curie; sixth review round).** To get the decisive ladder results sooner:
- Run order: e2e v3 → integrated → native L1 → full native → sparse native → fixes → 65k capacity.
- Withdrawn as superseded: `aws_r1_rmr_residdecay_pertargetfix_K4_16M_s0_20261017_v1` and `aws_r1_rmr_residdecay_lane1024fix_K4_16M_s0_20261017_v1`; their effects are covered by fixes → integrated.
- The GPU-driver CPU smoke can run in any free slot.
- Every result now carries `active_work` (modelled event-driven work per target, with measured firing rates).
- Curie admits no new mechanism until the ladder reports (theory 173 §11).

**Update 17 Oct 16:30 (Curie; fifth review round, theory 173 §10).** The e2e gate is now `aws_r1_rmr_native_e2e_smoke_20261017_v3`, which supersedes v1 and v2 (do not run those). Add `aws_r1_gpu_driver_cpu_e2e_smoke_20261017_v1` (the GPU driver end to end on CPU) and `aws_r1_rmr_nativesparse_K4_16M_s0_20261017_v1` (gate, then the run) after the full-native arm. The sparse arm uses sparse level reads with read probes and a product-key slot level 1 (256 key scores per token instead of 8,192; event-driven execution touches 12 state rows per token instead of 4,100). Order: e2e v3 → integrated → native L1 → full native → sparse native → fixes → 65k capacity → per-target; run the GPU-driver smoke in any free slot.

**Update 17 Oct 15:30 (Curie; fourth review round, theory 173 §9).** Replace the e2e smoke with `aws_r1_rmr_native_e2e_smoke_20261017_v2`, which also exercises firing clocks; v1 is superseded, so do not run it. Add `aws_r1_rmr_nativefull_K4_16M_s0_20261017_v1` (gate, then the run) after the native-L1 arm: identical to it plus firing clocks, so the model is native end to end, with sparse level-1 updates and asynchronous levels. Also fixed in response to AWS's review: per-target writers now receive full first-order input credit through refined slot values, exact and cheap (contract: equals the one-hot reference in values and gradients). The GPU driver now supports carry, slots and firing. Your P2 point (decay alone does not bound accumulation as β → 0) is ready as `--old-norm 1` (evidence-weighted old corrections, exact algebra), but it is not queued: it waits for the K4 decay run's long-lane result.

**Native level 1 and capacity at 16M, 17 Oct 14:30 (Curie; theory 173 §6).** AWS run order:
1. `aws_r1_rmr_native_e2e_smoke_20261017_v1`: a tiny end-to-end run through final scoring (carried protocols, inference FLOPs). It gates the integrated and native arms, because Curie cannot run it beside its job.
2. `aws_r1_rmr_integrated_K4_16M_s0_20261017_v1` (gate, then the run).
3. `aws_r1_rmr_nativeL1_K4_16M_s0_20261017_v1` (gate, then the run): identical, but level 1 is native sparse addressed slots (4,096 slots, 4 written per token, learned clocks, a bounded evidence-weighted read) instead of the linear recurrence.
4. `aws_r1_rmr_residdecay_fix_K4_16M_s0_20261017_v1`.
5. `aws_r1_rmr_residdecay_pk65536fix_K4_16M_s0_20261017_v1`: capacity at 16M with a usable bank.
6. `aws_r1_rmr_residdecay_pertargetfix_K4_16M_s0_20261017_v1`.

`aws_r1_rmr_residdecay_lane1024fix_K4_16M_s0_20261017_v1` is deprioritised: carried memory addresses the same length mismatch. Use current source; the latest contracts are `curie_rmr_contracts_20261017_v9` and `curie_native_slot_memory_contracts_20261017_v1`.

**AWS: publish one checkpoint, 17 Oct 13:10 (Curie).** The 16M error analyses (where native depth gains: novel vs copyable tokens) need `experiments/results/token_language/aws_r1_cerl_16M_s0_20261016_v1.pt`. Only its JSON was published, so both Curie analyses failed with FileNotFoundError. Please commit the `.pt` (Git LFS); Curie reruns the analyses after the K4 decay run.

**Integrated native depth member, 17 Oct 13:00 (Curie; theory 173: trainability conditions for composing good learners).** Run `aws_r1_rmr_integrated_K4_16M_s0_20261017_v1_gate` (16 updates, RSS check, below 2.9 GB) and then `aws_r1_rmr_integrated_K4_16M_s0_20261017_v1` (~14 h). It combines all repairs: review fixes, per-target old-slot corrections, a separate write-address race, counterfactual write probes and persistent memory across lanes (`--carry 256`; final scoring reports reset and carried lanes). Contracts: `curie_rmr_contracts_20261017_v8`, `curie_review_fix_contracts_20261017_v2`, `curie_cerl_contracts_20261017_v10`. Priority: the same level as the three fix arms above; if only two slots are free, run this one and the fixes arm first. Resume checks now accept the new flags at their defaults, so running jobs resume unchanged after a sync. Fixed GPU bug: `ParallelTemporalMemory` built its kernel on the CPU (`torch.arange` without a device), so the GPU driver would have failed on CUDA.

**Flagship review fixes, 17 Oct 11:00 (Curie; founder: check the flagship for anything that cannot work well).** Three defects in the native drivers' learning setup are repaired behind flags (defaults reproduce earlier runs; contract `curie_review_fix_contracts_20261017_v1`): weight decay hit learned clock slopes, decay rates, ReZero gates, biases and the dormant concept tables (`--wd-groups 1`); zero-initialised concept vectors left the bank as identical laws (124 of 1,024 units used; `--unit-init 0.02`); nothing pushed traffic onto dormant capacity (`--balance 0.01`, Switch load balance on the auxiliary objective only). Run in the next two free slots: `aws_r1_rmr_residdecay_fix_K4_16M_s0_20261017_v1` (fixes, 256-token lanes, ~8.5 h) and `aws_r1_rmr_residdecay_lane1024fix_K4_16M_s0_20261017_v1` (fixes plus 1,024-token training lanes, ~15.5 h, 2.0 GB). Both compare against Curie's unfixed `curie_r1_rmr_residdecay_K4_16M_s0_20261017_v1` (same seed and data). Third arm (17 Oct 12:00): `aws_r1_rmr_residdecay_pertargetfix_K4_16M_s0_20261017_v1`, fixes plus per-target old-slot corrections (`--old-per-target 1`, contracts `curie_rmr_contracts_20261017_v5`, ~10 h, 1.2 GB). They take priority over further 4M arms. AWS-owned drivers (`r1_rmr2_stable_stream.py`, the native local learner) should switch new runs to `param_groups(model, wd, True)` from `r1_cerl.py`. Read `experiments/index/DEAD_ENDS.md` before queueing anything else; the 4M capacity verdicts are confounded by the bank collapse.

**Execution repair, 16 October 21:28 UTC.** Use the gated compact current-slot execution for the ordered 16M depth fits: `aws_r1_rmr_compact_K4_16M_s0_20261016_v2` in slot 6 and `aws_r1_rmr_compact_K8_16M_s0_20261016_v2` in slot 5. These preserve the existing model, training data and protocol while eliminating zero old-write storage. They require `aws_r1_rmr_compact_gate_20261016_v1` first; original unstarted 16M queue definitions are superseded. Recover the completed K8 4M checkpoint evaluation without refitting. Active source-bound jobs finish unchanged.

**AWS: run these six R1 native-depth jobs now, one per slot (Curie order 16 Oct 19:00; founder: "Delegate heavier experiments to the AWS agent, it has larger compute available").** Ready one-job queues with full admission declarations are in `experiments/queue/r1/<tag>/<tag>.txt`; drivers `experiments/r1_cerl.py` and `experiments/r1_rmr.py` (contracts passed: `curie_cerl_contracts_20261016_v9`, `curie_rmr_contracts_20261016_v1`, `experiments/rmr_variant_contracts.py`). One thread each, RSS < 1.5 GB, exact resume every 256 updates:
1. `aws_r1_cerl_16M_s0_20261016_v1` — native data scaling (vs dense L8 16M 5.2183; Curie no longer runs it).
2. `aws_r1_rmr_dense_K2_4M_s0_20261016_v1` and 3. `aws_r1_rmr_dense_K8_4M_s0_20261016_v1` — depth ladder of the residual-memory stack (Curie runs K=4 variants).
4. `aws_r1_cerl_pk65536_4M_s0_20261016_v1` and 5. `aws_r1_cerl_pk262144_4M_s0_20261016_v1` — capacity beyond activity (supersedes the earlier 4k–262k list).
6. `aws_r1_cerl_msg_4M_s0_20261016_v1` — cause messages.
**Priority update 21:40 (Curie):** at 4M every native depth/capacity extension stayed within 0.01 of CERL (K=2 5.4679, 65k bank 5.4716 vs 5.4665), as the dense diagnostic predicts (depth paid 0.023 at 4M, 0.052 at 16M). Run next, before the remaining 4M jobs: `aws_r1_rmr_dense_K4_16M_s0_20261016_v1` and `aws_r1_rmr_dense_K8_16M_s0_20261016_v1` (queues ready; ~8.5 h and ~12 h). Curie runs `curie_r1_rmr_resid_K4_16M_s0_20261016_v1`.
**Depth result and next AWS job, 17 Oct 08:10 (Curie):** at 16M the old-slot residual memory (K=4) scores flagship **4.9901** vs CERL 5.0634 and dense L16 5.2034 (selection 5.1906 vs 5.2826), but its 65,528 slice (8,192-token lanes) regresses (5.0874 vs 5.0282) because old-slot corrections accumulate over 32x the training history. Repair: learned decay of old corrections (`--old-decay`, contracts `curie_rmr_contracts_20261017_v3`). Run next: `aws_r1_rmr_residdecay_K8_16M_s0_20261017_v1` (queue ready). Curie runs the K=4 repair.

**Published language benchmarks, 17 Oct 00:30 (Curie):** in the next free slots after the 16M depth runs, the race model on PTB and WikiText-2 (validation only; test stays sealed): `aws_ptb_cerl_d256_do4_s0_20261017_v1` and `aws_wt2_cerl_d256_do3_s0_20261017_v1`, each followed by its `_valid_score_` job (frozen and causal online adaptation, the dynamic-evaluation column). **Skip `aws_ptb_cerl_d256_do3_s0_20261017_v1`:** Curie is already running that arm (`curie_ptb_cerl_d256_s0_20261016_v1`; validation perplexity 152.9 / 120.8 / 107.2 after epochs 1–3). Targets (Melis et al. 2020, validation): PTB 51.4 static / 44.9 dynamic evaluation; WT2 57.3 / 40.2.
7–8. (next free slots) `aws_r1_rmr2_act0_K4_4M_s0_20261016_v1` and `aws_r1_rmr2_act01_K4_4M_s0_20261016_v1` — RMR-2: firing clocks from lower-level events and standing-slot pooling (`experiments/r1_rmr2.py`, contracts `curie_rmr2_contracts_20261016_v1`), activity cost 0 and 0.01.
Why: the native race member beat the dense hybrid (5.4665 vs 5.8414 at 3.3x less training compute per token); depth through residual memory (theory 172) and capacity at constant work are the two scaling axes to prove next. Publish each completed JSON; Curie analyses and integrates. These take priority over further R1 local-learner extensions until they complete.

**Immediate native learning repair,16October.** The4Mcontinuation now runs `aws_r1_native_producer_credit_4m_s0_20261016_v2` in slot4 from exact73728checkpoint,no corpus refit. Liveaudit finds finiteupdates/no queryclamp or gate saturation;static dependency audit identifies missing future-use credit into historical joint producers. Bounded local producer replay adds that gradient while preserving forward predictions and all parameterobjects/Adammoments/state. Two contracts plus acquired-model64targetgate/32discardedresumeaudit pass;full replacement also verifies exact recovery at73760. Originalsource4Mjob explicitly superseded,checkpoint/logs preserved;unsavedtail<=4096targets recorded. v1replacement stopped before training on missinginitialcheckpoint;v2 corrected atomicsave. Newrecords storeproducerinputs/frozenwritecoefficients;no recurrentBPTT/recursivechildreplay. Read-only checkpoint health monitor in tmux `native_learning_health_20261016`,status `.git/aws_r1_native_producer_credit_4m_s0_20261016_v2.learning_health.json`. Newcredit increaseswork;refineETAfromlogs. Source-bound modules remain frozen. [Repair](theory/NATIVE_PRODUCER_CREDIT_REPAIR.md).

**Acquired native hierarchy continuation,16October.** The64Knative foundation completes:selection7.930677→6.998141,flagshipDEV7.047292,nativecause-mixture gain0.413618nats versus its background component. No hybrid/public win. One4Mcumulative-target continuation (`aws_r1_queryable_native_4m_s0_20261016_v1`) is superseded at73728by the producer-credit repair above;it reused all45shared parameter tensors,43Adam state banks,receiver eligibility/indexedbank/inboxes and RNGs. New query/joint maps explicitseed16016;five integration contracts plus full-learner migration/recovery contract pass. Initial migrated selection7.004400,not an improvement claim. Source-bound imports including native/query cores,head,migration and fit driver must remain unchanged;existingL16diagnostic finishes. Fixed262K/1M/4Mselection milestones,no TEST/ablation/automaticextension. Projected severalCPUdays,refine from sustained logs. [Protocol](theory/QUERYABLE_NATIVE_ACQUISITION.md).

**Queryable native language integration,16October.** Existing versioned hierarchy bindings and joint role composition are now inside the native language core:upper stages race over lower publications,read up to2distinct versions and deliver read cotangents into the original own-write eligibility. Five contracts pass,including packed/reference parameter-gradient parity and numerical recovery. Real8layer/4.298Mmodel gate:32targets2.726s/747700KiBRSS;512targets across32optimizer boundaries46.878s/786400KiBRSS,PASS. No recurrentBPTT;optimizer synchronous,changing-parameter credit approximate;query discovery scans declared roots and lineage costs are paid. This is numerical integration,not a predictive win. The completed64Kacquisition is now reused by the admitted4Mcontinuation described above. [Integration](theory/QUERYABLE_NATIVE_LANGUAGE_INTEGRATION.md).

**Evidence-led synthesis,16October (founder correction).** [The empirical architecture synthesis](theory/EMPIRICAL_ARCHITECTURE_SYNTHESIS_20261016.md) governs new integration. The completed64Knative language run is a persistent local-learning foundation: versioned lower-memory queries, identity/role participants, joint operators and bound-pool native races already exist in G-HIER but were absent from its core;the new queryable integration now reuses them. Distinguish measured implementation defects from budget/distribution-limited failures:2048composition scenes contain only64abstractTRAIN arrangements, and64Klanguage targets cannot decide a4M/16Mcompetitive ceiling. Existing source-bound jobs finish unchanged;no new architecture/training admission or benchmark claim follows.

**Completed fixed-architecture composition scaling.** After2048additional paired updates over2048unique TRAIN scenes,original-protocol whole-object DEV NLL1.053453→0.693481,accuracy50.00%;saved acquired reference0.686609. Final fresh DEV NLL0.695857,accuracy50.00%. The saved acquired actor retains the better original-protocol whole-object result. Broader TRAIN128 readout DEV NLL0.703057 is a separately fitted endpoint. Same64TRAIN abstract arrangements,32new event/metric-layout tapes,two full pair epochs,unchanged model/Adam/loss/regularization,one development seed;TEST excluded,no ablation/control fitting. Source:`investment/composition_scale_evidence.json`.

**Completed 16M hybrid diagnostic.** The four-layer, 3.55M-parameter dense hybrid scores **5.2699 NLL** on 8,192 FineWeb GPT-2 development targets in 1,024-token windows, and **5.172638** on the separate 65,528-target slice. One pass over 16M tokens, one seed; 16-layer endpoint pending. This is the saved accuracy target for the fully native deep model at that data budget, not a native win or public leaderboard result. Source: `investment/depthdata_l4_evidence.json`.

**Native deep language priority, 16 October (founder).** First beat the saved dense hybrid with a completely native deep member;event-local learning is primary instead of recurrent BPTT. The eight-layer native core,normalized cause/depth head and factored own-write eligibility pass eleven contracts. The 32-target GPT2/FineWeb local-learning gate passes:4.19M parameters,0.919s/594420KiB,finite credit at all depths. Packed native events/gradients match the reference;current gate is0.399s forward,0.400s backward,0.064s local credit,0.043s optimizer. Integrateintegrate carried indexed episodic history,cause messages and asynchronous/versioned updates before competitive acquisition. No SSM/attention/MLP carrier,no quality win. [Design](theory/LOCAL_NATIVE_DEEP_LANGUAGE.md). Existing source-bound trainers finish unchanged.

**Founder direction 16 October 12:30 — build our own models (AGENTS.md).** The plan-B language member is a dense SSM/attention hybrid and is now a labelled diagnostic control only. Withdrawn: dense PTB/WikiText-2 arms (Curie round 1 and the AWS width/regularisation/WT2 orders), the dense 64M L8 run, the AWS d128 L8 16M order, and the dense GPU ladder variants. Running jobs (Curie 16M L8, AWS 16M L4/L16) finish unchanged; Curie's 16M L2 partner completes the registered P5d diagnostic. The language flagship becomes the **causal event race member** ([theory/169_causal_race_language.md](theory/169_causal_race_language.md)): every token emitted by a race among candidate causes (episodic repeat/continue pointers over addressed persistent memory, dormant concept units, a cluster background route), exactly normalised competing risks, sparse addressed state updates, counterfactual credit, native share and active work reported with every loss. PTB/WT2 and the GPU ladder field it once its contracts and CPU fits pass.

**Identity-preserving native composition,16October:** implemented separate child identities within geometric roles and issued-version local teaching.22combined contracts pass;actual paired optimizer gate PASS6.114s/804980KiB with finite credit at all8levels. The64paired-update integrated acquisition completed:issued-view identity40.43%→53.12%,but whole-label DEV NLL0.70757→1.05345 versus saved acquired0.68661. TRAIN affine-readout NLL improves0.67317→0.61452;the class-transfer generalization fails. The short-run failure is diagnostic;the current fixed-architecture data/training protocol supersedes the earlier stop recommendation. Full current participant/pair work is paid;selective partner discovery is missing. No ablations. [Theory](theory/IDENTITY_ROLE_INTERFACE.md).

**Integrated part-identity acquisition,16October:** the previous native part-identity acquisition completed in slot3;no ablation/control acquisition. TRAIN-observable shape signatures supervise native part memory as targets only;whole-label CE teaches arrangement. Fixed loss CE+0.25part CE,8levels/32native stages,existing acquired parameters plus264teacher-head parameters.21combined numerical contracts pass;actual paired optimizer gate passes at5.932s/797668KiB. One64paired-update fit,exact8update checkpoints,TRAIN-only readout,reusedDEV,TEST closed. Native clocks,hard receipt races,counterfactual learning,indexed state/timers/versioned part access retained;ownership/query policy explicit;recursive-likelihood heads inactive. `part_identity_publication_v1` publishes completed identity/whole-label evidence and updates PDFs;no automatic larger fit. [Theory](theory/INTRINSIC_PART_IDENTITY_LEARNING.md).

**B4 Stack Overflow developed result,16October:** developed attempt below composite target;recorded-data five-split totalNLL11.995598±0.647576SE versus composite11.90. Variant selected by all-five-split DEV mean;TEST once. Source:`experiments/results/tpp_b4/aws_b4x_so_test_20261016_v2.json`. No further fitting automatically admitted.

**B4 Github developed result,16October:** WIN: five-split recorded-data totalNLL-284.465765±54.288843SE versus composite-272.90;best single published model-269.70. Selectedptr by all-five-split validation mean;TEST once. Native pointer histories retained;completed old splits reused and only memory-failed fits replaced. Source:`experiments/results/tpp_b4/aws_b4x_github_test_20261016_v3.json`. No further fitting automatically admitted.

**B4 Wikipedia developed WIN,16October:** recorded-data totalNLL−258.760894±37.733361SE versus the stronger composite−122.62;all5official splits beat that target. Pointer selected by aggregate5-split validation,TEST once. Sources/model checkpoints match;causality and mark-normalization contracts pass. Github and Stack Overflow repairs/scoring are completed;no B4 trainer remains active. [Current evidence](tpp/B4_DEVELOPED_CURRENT.md);[remaining programme](BENCHMARK_CONTINUATION.md).

**G-HIER completed representation diagnosis,16October:** joint local grounding improves calibrated root-geometry DEV MSE0.0401215→0.0262533 versus the frozen encoder(34.6% reduction;constant-reference skills22.91%→49.56%). Same fixed TRAIN ridge.01 linear decoder;encoder features really change. Whole-label NLL0.732158(joint),0.783903(continued),0.686609(frozen acquired);no qualified class transfer. Preserve this geometry gain,do not scale unchanged reconstruction. Root frozen Adam decoder0.203460 was worse than TRAIN constant0.052045,so separate decoder conditioning from information. Exact sparse teacher and centered-innovation native candidate pass15contracts;current actors remain unchanged. Next:stable part identity and nonadditive role binding. [Diagnosis](../experiments/results/hierarchy/aws_g_hier_grounding_diagnosis_20261016_v1.json);[theory](theory/GEOMETRY_INFORMATION_AND_BINDING.md).

**R1 native capacity-scaling ladder (CERL, theory 169; Curie, 16 Oct 14:10). AWS: free one-thread slots, highest R1 priority after the running 16M jobs.** Capacity beyond activity as a scaling law: the causal event race member with a race-woken concept bank of growing size at (nearly) constant active work per token (product-key two-stage wake, exact top-k: contract `curie_cerl_contracts_20261016_v8`). 4M fresh FineWeb tokens, one pass, the plan-B diagnostic's exact lanes and schedule; one job per slot (RSS < 1.5 GB; ~1.5–1.8 s/update, ~2 h each):
`experiments/r1_cerl.py --train-tokens 4194304 --seed 0 --product-keys 1 --units U --tag aws_r1_cerl_pk{U}_4M_s0_20261016_v1` for U = 4096, 16384, 65536, 262144 (Curie runs the flat 1024-unit reference, `curie_r1_cerl_4M_s0_20261016_v1`). Plus the cause-message arm (theory 169 §3: the posterior-attributed cause writes its content into memory at the next event): `--cause-messages 1 --tag aws_r1_cerl_msg_4M_s0_20261016_v1` (flat 1024 units; ~2.3 s/update), paired with Curie's reference. Decision: loss against parameters at fixed active FLOPs — if loss keeps falling with the dormant bank, the family has a scaling axis that dense models pay full compute for; the slope sets the GPU ladder's concept-bank axis. Report flagship/65,528/selection loss, native share by cause, active work per target, train FLOPs.

**R1 published word-level benchmarks (PTB, WikiText-2), 16 October 10:30 (Curie lead; user: train our own models against published benchmarks first, so large Transformers are never retrained).** Targets (Melis et al. ICLR 2020 Table 1, test perplexity static / dynamic evaluation): PTB Mogrifier 24M 50.1 / 44.8, AWD-LSTM 54.4 / 47.7; WikiText-2 Mogrifier 35M 55.1 / 38.6 ([SCALING_SUPREMACY_PROTOCOL.md](SCALING_SUPREMACY_PROTOCOL.md); battle page [PTB_WT2_BATTLE.md](PTB_WT2_BATTLE.md)). Driver `experiments/r1_ptb_native.py` (plan-B deep keyed member for any vocabulary, word-type/input/residual/output dropout; frozen scoring by 512-token windows advanced by 128 or by the exact carried-state stream (`experiments/r1_native_stream_scorer.py`: temporal memory carried through the whole split, keyed reads over the last 512/2048 events; equals whole-lane scoring to 4e-6), chosen on validation, and causal online weight adaptation every 32–64 scored targets, the dynamic-evaluation column; `--mode score --split valid` chooses the online step on validation; test is sealed until one selected model is scored once with `--split test --select-from <valid score tag>`). Contracts pass for both corpora (`curie_ptb_native_contracts_20261016_v4`, `curie_wt2_native_contracts_20261016_v2`). WikiText-2 token files rebuilt from parquet with the standard counts (`scripts/convert_wikitext2_parquet.py`). Temporal-memory kernel chunk 64 (`--tm-chunk`, exact to 1e-6 relative) runs 4.7x faster per layer than chunk 256. **Curie (chain v18):** after 16M L8: smoke → PTB d256 L8 16 epochs (`curie_ptb_native_d256_L8_s0_20261016_v1`, ~8 h) → validation scoring → 16M L2. **AWS, free slots after the 16M depth-data jobs (one smoke per arm first, RSS cap 3 GB):** (1) width arm `--corpus ptb --d 384 --dk 48 --layers 8 --epochs 16`; (2) regularisation arm `--corpus ptb --d 256 --layers 8 --dropout 0.4 --drop-in 0.5 --drop-out 0.5 --drop-emb 0.15 --epochs 24`; (3) WikiText-2 `--corpus wt2 --d 256 --layers 8 --epochs 8`; each followed by `--mode score --split valid --run <tag>`. Report as "in development: best X vs reference Y (validation)".

**R1 matched-compute Transformer reference ladder (AGENTS.md exception, 16 October; activated 16 Oct 12:40 by Curie because no published curve covers our budgets: the record curves start at 5e16 FLOPs, our native frontier spans 1e14–2e15).** Driver `experiments/r1_reference_transformer_stream.py` verified end to end on CPU (training, selection/development/flagship scoring, measured FLOPs, exact resume: interrupted and uninterrupted runs give identical results). **AWS, one slot each, free slots after the depth-data jobs:** at 4,194,304 tokens (native L8 point 1.1e14 FLOPs): `--d 64 --layers 8 --heads 4 --lr 3e-3` and `--lr 1e-3` (LR selection on the selection slice, then fixed), `--d 64 --layers 2`, `--d 128 --layers 4 --heads 4`; at 16,777,216 tokens: the best two 4M shapes at the selected LR (`--train-tokens 16777216`). Tags `aws_r1_reftx_<shape>_<tokens>_s0_20261016_v1`; batch 1 x 1024 (its natural context), one pass, same shard/permutation rule as the native ladder. Decision: first two-frontier comparison on one protocol at matched training FLOPs (development slices; flagship windows); the GPU ladder extends both frontiers to the published range.

**R1 depth must scale: strongest route, updated 16 October 03:45 (Curie lead; user: pursue the route with the strongest case).** Scale depth and data together with the composition that makes depth pay: plan-B blocks (pre-norm residual, per-layer keyed race reads, full-lane credit). Evidence: depth pays per fresh token, one pass over 4M: L8 vs L2 flagship 5.8414 vs 5.8648, last-quarter prequential 5.8095 vs 5.8362, gap widening through the pass (0.008 → 0.028); multi-epoch 1M ladder L2/L4/L8 5.957/5.899/5.874 (65,528 slice), L16 selection 6.198 (best epoch 4: data-limited). Compiled blocks: 1.57x per update at L8, loss agreement 9.5e-7 (`r1_deep_keyed_stream.py --compile-blocks 1`, execution only). Pass (P5d): the 16M L2−L8 gap exceeds the 4M gap (0.023 flagship / 0.027 prequential). **Curie (chain v19, 16 Oct 12:15):** 16M one-pass L8 (running) → PTB round 1 → 16M L2 → P5d analysis → **64M one-pass L8** (`curie_r1_deep_keyed_stream_L8_64M_s0_20261016_v1`, 1.7e15 FLOPs) → L4 resume. **Exact chunk-64 temporal-memory kernel:** run new one-pass jobs through `experiments/r1_deep_keyed_stream_chunked.py ... --tm-chunk 64` (driver unchanged; loss diff 0.0, gradients 5e-6, `curie_tm_chunk_equivalence_20261016_v1`; compiled L2 update 0.36 s vs 0.73 s). **AWS, parallel slots:** 16M one-pass L4 and L16 (`experiments/r1_deep_keyed_stream.py --train-tokens 16777216 --layers N --compile-blocks 1`; L16 with `--checkpoint-blocks 1` instead if compiled RSS exceeds the slot cap) the width x depth arm d=128 L8 at 16M (`--d 128`; decides whether width or data limits the extra layers if the depth gap stops growing); exact resume every 256 updates (the 64M one-pass L8 moved to Curie, chunk-64 wrapper); and **one sealed public-protocol scoring** of the development-selected keyed 4M member (`analysis/public_protocol_score_keyed.py --run curie_r1g2_keyed1_4M_s1_20261008T1840Z --mode public`, ~5 h, RSS < 1 GB at --lanes 2). Program and decision rule: [SCALING_SUPREMACY_PROTOCOL.md](SCALING_SUPREMACY_PROTOCOL.md). Integrate hierarchy levels (part-whole v2: small paired gain, 5.859 vs 5.865 flagship) and the tree read on the deep stack once their decisions pass.

**Completed AWS keyed depth and async comparison,16October:** D2/D4/D8/D16 score5.957197/5.898736/5.874106/5.887896NLL on the65,528-target FineWeb development slice;D8 remains selected. Async base2/levels3 scores5.939234; base4/levels4 scores5.891116. Same1,048,576unique training tokens,one seed,different epoch/work totals;quality comparison,not matched-compute or semantic hierarchy. Source-bound investor packet and evidence brief now contain all completed extensions.

**Current continuation, 15 October:** [Evidence-driven continuation plan](CONTINUATION_PLAN_20261015.md) governs future selection: strongest keyed memory first, useful-depth gains against the strongest shallow model, causal delayed-label price learning, and completed-evidence investor claims. Active source-bound jobs finish unchanged.

**B10-P price prediction and R1 data scaling, user direction 15 October:** predict BTCUSDT prices: return distributions at 10s/60s/300s on exact price cells are B10's primary targets; the six-class tape likelihood is auxiliary. Phase 1 acquisition/selection (native/LSTM/Transformer with identical causal inputs and heads, classical Student-t references), then online price forecasting with delayed labels and a preregistered fresh October day ([B10P_PRICE.md](market/B10P_PRICE.md)). R1: scale the strongest keyed temporal-memory member on fresh tokens; unchanged P24 scaling without addressed memory is superseded.

**Current AWS theory-guided admission — 14 October.** Protected native hierarchy from the strongest acquired D2 language member: insertion containment,64-update integration, actual-schema local calibration and real-update exact recovery pass. Driver/scoring smoke completes; continuedparent2M fit completes NLL6.065562542; protectedD4 completes6.095076744 versus continuedparent6.065562542; parent remains selected. Frozen contribution audit identifies harmful learnedcorrection; no newdepthfit before repair. Same data/order/final DEV endpoint, extra branch/teacher work charged; no parameter sweep. Theory: theory/PROTECTED_NATIVE_LANGUAGE_GROWTH.md and Curie's causal asynchronous synthesis. DEPTH_RESEARCH_DECISIONS.json names the mechanism and decision. Preserve active D8/32M acquisition.

**R1 depth priority, user direction 15 October:** depth must scale. Curie runs the residual/per-stage-read depth ladder before the keyed 16.8M fit and the B10 stage ([RESIDUAL_DEPTH_20261015.md](RESIDUAL_DEPTH_20261015.md)). Pass: monotone improvement D2→D8→D16 on the long-history protocol at matched data, and residual beating serial at D8. A passing arm becomes the deep R1 construction for larger acquisitions on both hosts. **Result (16:55 UTC): failed.** Long-history D2/D8/D16 residual 6.556/6.590/6.690, with reads 6.472/6.563/diverged; residual beats serial at D8 (6.590 vs 6.683). The composition is at fault, not depth: the race stack feeds its top stage into the bottom of the next token, so each token transition is D stages deep. Next: per-stage horizontal recurrence with purely vertical residual depth (RESIDUAL_DEPTH_20261015.md), alongside the keyed-member depth test and plan C.

**R1 strongest fighter, 15 October (Curie):** on the flagship protocol the keyed temporal-memory member scores 5.601/5.624 at 4M tokens against the P24 flagship's 5.978, and uses 15–30x more long-range history. R1 scaling work fields the keyed member (Curie chain v5: 16.8M fresh tokens, 1024-token lanes) and integrates its identity-addressed read into the race core. Further P24-core depth or data scaling without an addressed read is not admitted.

**B10 October confirmation, 15 October:** native 8.078536 beats online LSTM 8.102707 and online Transformer 8.179195 on the predeclared October day (seed 0, 65536 forecasts, all 16 blocks), using 14.65x less scored arithmetic than the cached Transformer. All three seeds are acquired for native/LSTM/Transformer; native leads each reference in each paired fixed-weight DEV fit (means 6.8842/6.8996/6.9686). Next: saved-learner online seed replication with unchanged settings; October is already accessed. Remaining pass criteria: three online seeds and a Hawkes reference on the same day.

**R1 depth result and next decision,15October:** protectedD4 loses to continuedD2 by0.029514201NLL at higher incrementalwork. Do not extend unchanged. Admit one frozen AWSslot5 contribution analysis under [PROTECTED_DEPTH_ATTRIBUTION_20261015.md](PROTECTED_DEPTH_ATTRIBUTION_20261015.md), reusing completed checkpoints; results select correction-learning versus parent-learning-interference repair. No new training grid, publicvalidation or independent-lane fit automatically admitted.

**B10 online priority, founder14October:** chronological online learning is primary. Native0/native1/LSTM0 acquired fits finish for reuse; completedTransformer0 now runs online. Unstarted static repetitions withdrawn and secondaryLSTM2 paused with checkpoint/work retained. Online native/LSTM follow their completed acquisitions automatically through scripts/admit_b10_online_acquired.py. Stronger native future-taught early-request guidance is being integrated on that same actor/stream; source actor and ordinary online forecast parity are gates. Fixed-weight page12 quality lead stays separately scoped; no online win inferred. Septembersealed; no new offline baseline fitting.

# Product orders — owned battles for headline results (issued 6 October 2026, 15:00 UTC; user-directed)

**Apply current explicit directions below in priority order.** Completed evidence stays in result files; superseded orders live in git history.

**Depth-first priority, user direction13October:** winning the Transformer language benchmarks is the immediate objective. Two-layer learning-rule follow-ups do not displace actual deep language acquisition. AWS owns the admitted native depth8/16 uniform-teacher program; Curie owns the distinct depth8/16 allocated-credit implementation and current8M acquisition; its unstarted32M acquisition is withdrawn by14October queue grooming under R1. Finish already-running market evidence, withdraw its waiting query follow-up, then use Curie's single training slot for this deep program. Short smokes are numerical/resource gates only, not quality-selection campaigns. Compare exact GPT2/FineWeb public scoring after development selection; no new Transformer training.

**Learning priority, user direction 10 October:** demonstrate efficient learning as depth and model size grow, with complete work to reach matched quality. Curie owns the compact-credit/depth implementation under B1/R1; preserve AWS's admitted two-layer online3 comparison and completed confirmations. [Execution and evidence gates](DEEP_LEARNING_SCALING.md). One-layer gains per pass are supporting evidence, not the scaling claim.

**PAM decision, user direction 10 October:** retain the original sealed v7 five-split result (accuracy 0.9775 ± 0.0074, F1 0.9803 ± 0.0080). Stop further v8/v9 replacement runs; preserve completed results/checkpoints. Proceed on other fronts before returning to PAM development. No new PAM fitting or TEST scoring is admitted.

**Quality/data priority, user direction13October:** increased compute is allowed when it improves quality at fixed data or reduces the data needed to reach useful quality. Matched compute is a comparison class, not an admission requirement. Report complete work separately and retain resource guards; test actual update utility rather than choosing a credit model on MSE alone.

**Language competitive-progress and queue grooming,14October:** finish running CurieD16/8M. Withdraw unstarted unchangedCurieD8/32M and horizon-onlyD8/8M (AWS also withdrew its combined horizon smoke; no horizon acquisition currently admitted). No replacement long fit automatically admitted. AWS active acquisitions/banked/local fits stay unchanged. Select the next language learner from completed quality/error/fullwork evidence against richerD2 NLL6.099337 on matched scored targets, then the pinned public protocol. Retained prototype mathematics does not constitute a training admission.

**Asynchronous market/depth priority, user direction14October:** finish activeD16 language fit, then one prediction-preserving nativeeight-layer B10 growth smoke/fullfit. Coordinator continue_market_depth_v2.py stops afterward, including on recorded market failure. Saved two-layer parent, chronological recording-cell likelihood and measured costs retained. No automatic32M continuation or credit-only grid. [Depth protocol](market/B10_DEPTH_SCALING.md).

**AWS scaling-frontier integration, latest userdirection:** one conditional integrated local/deferred-consequence R1 pipeline via continue_locality_frontier.py. Wait completed globalbank/localclip8M paired quality; if local degradation exceeds0.01nats stop for update-rule diagnosis. Otherwise local-law continuation/horizon endpoint gate→nativeD8continuous/noEOSreset16factual/upto64consequence resource smoke→one8M acquisition→sameprotocol quality/fullwork against bothdeep arms and richerD2. This is the single combined AWS repair, not restoration of withdrawn component queues or automaticlargerbudget ladder. No smallqualityveto, no newdensecontrols/publicvalidationselection. ActiveD16/reserved32M and CurieB10 preserved.

**Calibratedlocal clipping priority, latestuserdirection:** user rejects cap1 as meaningfulcalibration and authorizes progress towardscalingfrontier. AWSslots4/5 own one matchedglobal/calibrated pair from completedbankD8learner: sameparent/moments/state/RNG/freshdata, manualcap/globalproduction numericalchecks+64update resource smoke, then4096updates (~2Madditionaltargets) and same8192DEV/1024history scoring. Calibratedcap fitcost/exposure and bothsmokes chargedseparately. No horizon/reset/depthchange in thispair. Noautomaticbeyondpair; results choose usefullocalupdatepolicy fordepth/consequencework. Existingdepth/reserved32M and CurieB10 preserved.

**AWS spare-slot native repairs, latestexplicituserdirection:** skip Curie-checkpoint jobs; runthree localartifact comparisons inslots1/3/6 sharing the admitted2Mglobal continuation control. Horizon4/factual16; factual64/teacher16; containedfinalstage decoder96 withparentmomentmigration.64update numerical/resourcesmokes→4096update continuations→identicalpairedDEVscoring. No clipping/reset/depthchange mixture, no newexternaltraining/publicselection, no automaticextension. SCALING_REPAIR_COMPARISONS_20261014.md records decisions and work. This supersedes idle-only/prototype-only admission for these three exact arms.

## The strategy

We pick the battles where our core primitive is the right mathematics, and we develop to win. A race of exponential
clocks over memory that decays with elapsed time is a temporal point process: the first finisher gives the next event's
type and time, and the waiting time without events enters the likelihood as a survival term (silence-aware supervision).
Timed, irregular event data is our home field; public leaderboards there are where headline wins come from in weeks.
Benchmark work follows AGENTS.md "Benchmark work is development to win": owned, designed, iterated, then verdict.

## The battles

| Id | Battle | Public reference | Owner | Pass criterion (developed attempt) | Stop rule |
|---|---|---|---|---|---|
| **B1 (lead)** | **EasyTPP** temporal point processes: Retweet, Taxi, StackOverflow, Amazon, Taobao (ICLR 2024 benchmark, official splits) | Published EasyTPP tables: NHP, THP, SAHP, AttNHP, FullyNN, IntensityFree, ODE-TPP | AWS host | Beat the best published log-likelihood on ≥ 2 of 5 datasets with type error/time RMSE no worse than the best published; confirmed on 3 seeds; inference work measured | No best-published LL on any dataset after the documented development plan (about two weeks) |
| **B2** | Irregular multivariate time series classification: P12, P19, PAM (Raindrop protocol and splits) | Published Raindrop, ViTST, Warpformer and later tables | AWS after B1 first fits | Beat best published AUROC (P12/P19) or accuracy (PAM) on its official splits | Same rule as B1 |
| **B3** | FAS v2 sealed confirmation (FAS_V2_CONFIRMATORY_PROTOCOL.md), then public release of FAS with a leaderboard | Information-matched classical references plus small LSTM and time-encoded Transformer references (AGENTS.md exception, 6 Oct 17:46) | Existing FAS owner | As pre-registered; no expansion | As pre-registered |
| **B4** | Neural TPP benchmark of Bosser & Ben Taieb (TMLR 2023, 2025): LastFM, MOOC, Github, Stack Overflow, Wikipedia, MIMIC2, Retweets; five fixed splits (admitted by the founder 9 Oct 2026) | Per-dataset tables of TMLR 2023 (Appendix B) and arXiv 2412.08590 Table 1 | AWS | The frozen B1 unified configuration (no tuning) below the bar (best published L_T + best published L_M) on a dataset, pre-registered in [B4_NTPP_BENCHMARK_PROPOSAL.md](B4_NTPP_BENCHMARK_PROPOSAL.md) | All seven datasets reported once; no configuration change. Status 10 Oct: Wikipedia LOSS after its guarded split-0 rerun (two test sequences with exploding hazards; marks 27.65 vs 144.79); MOOC, Stack Overflow, MIMIC2, Retweets, LastFM LOSS; Github LOSS by 2.3 (−270.62 vs −272.90; guarded reruns; time ahead); Retweets LOSS (−515.2 vs −538.7; marks level, timing gap); LastFM running. Development target: the mark path (L_M behind on four datasets) |
| **B5** | **Temporal Graph Benchmark** dynamic link prediction, first tgbl-wiki-v2 (official py-tgb loader, chronological splits, negatives and Evaluator; MRR); then tgbl-review-v2 (founder direction 9 Oct: prepare wins on new event domains) | TGB leaderboard (tgb.complexdatalab.com, verified 9 Oct): TPNet 0.827 ± 0.001, Heuristic(LocalGlobal) 0.821, DyGFormer 0.798 | curie | Pre-registered in [tgb/B5_TGB.md](tgb/B5_TGB.md): test MRR above the leaderboard leader on 3 seeds | Developed attempt below the leader after the plan in B5_TGB.md (about two weeks) |
| **B10** | Crypto market event streams (Binance BTCUSDT aggTrades, public): probabilistic forecasting of the trade stream (next-trade time, side, size log-likelihood; short-horizon realised volatility and trade intensity), prequential, with online state/weight adaptation tested on months with regime changes; LOB-Bench (ICML 2025) as the public generative benchmark (founder admission 9 Oct; [proposal](NEW_BATTLES_PROPOSED.md)) | E17 baselines re-run (online logistic regression, momentum), Hawkes and published neural-TPP references; LOB-Bench published scores | curie | Better log-likelihood and volatility forecasts than every reference on a pre-registered prequential month, 3 seeds; direction and P&L after costs reported as secondary metrics only | No likelihood or volatility gain over Hawkes after the development plan |
| **G** | Generality track (7 Oct, user-directed): self-supervised event pretraining → clinical classification (G1), one model across five event datasets (G2), PAM (G3), event→event transfer (G4) | — | AWS | Per [GENERALITY_PLAN.md](GENERALITY_PLAN.md) | — |
| **R1** | Language research track, at most one slot ([dossier](R1_RECALL.md)) | Count n-gram references; later a published small Transformer | One owner | Gates before any scaling or claim: solve associative recall/induction with irregular gaps; beat KN trigram on the 65,528-target DEV slice | — |

Every battle reports measured inference work (operations, CPU latency; energy where measurable) next to quality.

**Shared enabler, first test (user-directed 9 Oct 2026):** theory note 160's credit model trained by the forward race's own sampled causes, against REINFORCE, straight-through Gumbel, dense exact and an exact-posterior oracle on hard winner-only routing (`experiments/credit/hindsight_race.py`, 30 runs, queue label ENABLER).

**Shared enabler:** high-fidelity route credit (exact forced-lane credit at pool 2–8 as a training signal; a derived
low-variance multi-step estimator next). It serves B1, B3 and R1 directly; it is developed inside those battles, not as a
separate queue.


**Integration note (curie FAS session, 6 Oct 15:20 UTC).** VALUE_PLAN.md (same day, same diagnosis) is folded into these
battles and does not compete with them:
- Its Stage 2 (fixing the measured route-chaos and credit failure on FAS validation data) is **B3 development plus the
  shared route-credit enabler**.
- Its maturity gate is the same rule as "Benchmark work is development to win".
- AWS priority follows the battle table: B1 leads. The FAS neural references are withdrawn: under the 6 October
  no-new-dense-controls direction they are not trained (protocol amendment, AWS_FAS_REFERENCES.md).

## Admission rule (all hosts, all agents)

1. Every queued job names its battle (B1–B5, B10, G, R1) and the decision its result changes. No battle, no admission.
2. Each battle keeps a one-page frozen protocol, a table of published reference numbers with sources, a development log
   with error analysis per iteration, a timebox and its stop rule.
3. Contracts and smokes run only as steps of a battle pipeline: one smoke per new driver, numerical contracts only for
   new mathematics.
4. One weekly scoreboard. Part I of the report changes when a battle produces an outcome.

## Stopped (complete running jobs at safe boundaries; do not start new ones)

Private-bank and time-scale audit chains; split-horizon contract and admission ceremonies; token width/learning-rate/
capacity sweeps; k-write and tied/untied pool arms; grokking follow-ups; Mackey-Glass and primate reaching; SHD and
DVS gesture; real-table comparisons against trees (until typed learning has a battle); dense controls of any kind;
PDF-append republishing. Completed results stay in the record.


# Historical orders (superseded 6 October 2026, 15:00 UTC)

## Product orders — what we are after (issued 4 October 2026, 19:20 UTC)

**Latest user direction — improve our own models; no new dense controls (6 October 2026):** compute goes to
improving the integrated temporal/sparse models. No new Transformer/LSTM (or other external-architecture) training on
any host: no retries, tuning arms, strict matched-compute retries or the "missing" D-optimized recurrent control below.
References are published benchmark/leaderboard scores on the exact matching protocol, plus dense results already
completed. The running AWS `aws_tuned_ref_90M_D_tf256L4_p1_lr0.001_s0_20261004T210000Z` finishes as admitted and is
the last dense control; it is the only remaining dense job in `aws_model_improvement_repair_20261005T161000Z`. This
supersedes dense-training admissions in the paragraphs and tables below (P0-6, the strict Transformer retries,
P1-4); their completed numbers remain the record.

**Latest user direction — retain three evidence fronts (5 October):** continue
proactively on the strongest integrated language path, while allocating bounded
compute to improving FAS and tabular headline results. More FAS seeds and
benchmarks explicitly requested: native frozen-recipe seeds7/8 are scheduled
behind the already-admitted language sequence; existing neural benchmark
queues remain AWS-owned. Preserve active jobs. After these admitted tasks,
the first new tabular task is a small integrated typed-comparison/race contract
and fit, not another numeric-only capacity sweep. Read
TYPED_PREDICATE_EVENT_DIRECTION_20261005.md; establish type semantics,
hard-route learning and resource accounting before longer fitting. Existing
banknote confirmation trails trees on average and must remain visible; do not
promote the mixed-type proposal as an achieved win. Public language validation
and reserved larger scaling point remain protected. No fixed percentage of
compute is inferred; admit bounded tests from measured costs and current jobs.


**Latest user direction — implementation serves the objective:** we own and change the implementation. Current driver/backend choices are engineering work to resolve, never reasons to constrain tokenization, persistent memory, routing, learning or the research ambition. Build the capabilities the selected economical experiment requires. See OPEN_LANGUAGE_REFERENCE_PLAN.md and LANGUAGE_IMPLEMENTATION_AUDIT_20261005.md.

**Latest user direction — CPU-only, tokenized language, reuse public baselines (5 October):** [OPEN_LANGUAGE_REFERENCE_PLAN.md](OPEN_LANGUAGE_REFERENCE_PLAN.md) supersedes new discretionary baseline grids and GPU provisioning requirements below. Reuse published Transformer runs/checkpoints and their exact data/tokenizer; train our integrated models. Start from the smallest credible published Transformer-leading tokenized regime, not a new LSTM crossover campaign. Measure native tokenized CPU throughput and output-head cost before admitting a long fit. Existing jobs/results remain preserved.

**Protocol correction — 5 October 2026, user-directed:** FIFO is an **oracle-assisted diagnostic**, not an eligible reference for anonymous-process learning. `deinterleave_baseline.learn_route()` uses hidden TRAIN item identities to recover the route. The v2 timing-aware probe additionally fits transition-gap statistics with those identities. Neither receives test identities for prediction, but both receive privileged training structure unavailable to native and generic controls. Their scores are retained as oracle-assisted diagnostic targets; exclude them from strongest-reference selection and win/loss verdicts. Native's completed single-seed win against the six generic controls stands: **0.600 vs 0.559 AUROC at N=256**. A fair structure-learning reference must fit exclusively on the same anonymous training logs. Historical contrary interpretations below are superseded by this correction; numerical records remain preserved.

Owner: the user. Product owner: curie agent, on the user's instruction. **Every agent (curie, AWS, local) reads this
first and works these orders in priority order.** Definitions of win/loss: experiments/WIN_CRITERIA.md.

**Latest user direction (5 October): Transformers at larger scale are the main
opponent.** [LANGUAGE_CROSSOVER_PLAN.md](LANGUAGE_CROSSOVER_PLAN.md) defines the
primary objective: locate a Transformer-leading language regime, then establish
integrated event-model advantage at equal complete fitting compute and confirm
at a reserved larger point. Reuse the pending 90M C controls and D Transformer
arms; a D-optimized recurrent control is missing. Small-scale count/LSTM wins
are diagnostic and intermediate evidence, not a gate to modern language work.
The modern corpus/tokenizer, strong Transformer and GPU execution requirements
in FRONTIER_COMPUTE_PROTOCOL.md now belong to the main path rather than P2.
Preserve active jobs and enact this priority at safe owner boundaries. Finish
near-complete primate and existing strict-budget comparisons. FAS v1's FIFO
method is an oracle-assisted diagnostic, excluded from fair-reference verdicts; further v1 sweeps are deferred unless already admitted or
needed for one bounded transferable diagnosis. Historical orders below remain
as the record; this paragraph governs new discretionary admission.

**FAS v2 confirmatory program (5 October 14:30 UTC, user-directed):** [FAS_V2_CONFIRMATORY_PROTOCOL.md](FAS_V2_CONFIRMATORY_PROTOCOL.md), queue manifest `queue/fas_v2_confirmatory_20261005T143000Z/manifest.json`. Pre-registered home-field test with ambiguous identity: setting chosen from oracle/classical headroom on validation only, strongest de-interleavers and five neural references, equal tuning budgets, three seeds, sealed test ledger, one primary endpoint (AUROC at N=512). Stages 0–2 are CPU-light and start now; training stages take free capacity without preempting running jobs. This replaces further FAS v1 sweeps.

**User-directed completion packet (5 October, review session):**
[BENCHMARK_WIN_EXECUTION.md](BENCHMARK_WIN_EXECUTION.md) and
`queue/benchmark_win_execution_20261005T111500Z.json` track retained next-step
tasks, explicit deferrals, smoke/contract dependencies and completed results. Reuse
the owner's strict A/B retries below; the packet adds matching budget-B seed7
confirmation after a passed seed6 comparison. Before future k-write fits,
read the source-frozen v2 repair: the old wrapper's traced optimizer windows
use k1 while later windows use k2. Preserve old queues/results, check active
owners, then use corrected v2 stages. Numerical v2 contracts remain unrun.
FAS R8 is completed; its longer horizons did not close the early-detection gap.
The bounded binding/credit diagnostic specification is in FAS_BINDING_DIAGNOSTIC.md;
it is deferred behind the language scaling objective under the larger-data language priority; FIFO is oracle-assisted.
This packet admits no worker and does not replace active scheduler manifests.

## What increases our valuation

Investors buy three things, in this order:

1. **A headline number against the incumbents.** "Our model matches or beats LSTM and Transformer language models at
   N× less compute" (training and inference), on a standard dataset, with the same data and test set. One number,
   reproducible, with a second seed.
2. **A public leaderboard entry.** "First place on a recognized benchmark" (NeuroBench), under its official protocol,
   with its own efficiency columns.
3. **A scaling curve.** The win holds or grows from 10M to 90M characters (and beyond), so it is a trend rather than a
   lucky point.

Everything else (diagnostics, theory notes, new primitives) is valuable only insofar as it produces or protects these.
Every deliverable below has a pass/fail number. Report it as a win or a loss in one line.

## URGENT AWS slot order (5 Oct 07:05 UTC, product owner)

`queue/aws_product_priority_20261005T001500Z/manifest.json` lists only the ten finished tuned-reference arms in slots 2
and 3, so **two of the three AWS slots appear idle since about 01:30 UTC**. Meanwhile, the leaderboard deliverables are
waiting:
- P0-3: the six primate sessions are queued in slot 1 behind the 90M job.
- P0-4: Mackey-Glass repeats 20–29 have no result; still 20/30.

Fill the free slots now, in this order (each through run_safe.sh, with the usual caps and at least 8 GiB MemAvailable):
1. **Slot 2: P0-3 primate six-session r1.** Move the six `aws_primate_r1_*_20261004T053500Z` jobs out of slot 1.
2. **Slot 3: P0-4 Mackey-Glass third batch** (repeats 20–29; pins verified 4 Oct 23:55).
3. Then, in the first free slot:
   - the two 90M budget-C tuned references (P0-1's tuned comparison; ~4 h each; added 07:15);
   - the FAS data job and the five FAS references (AWS_FAS_REFERENCES.md);
   - the weight-decay smoke and arms;
   - the X1 capacity-curve arms (AWS_CAPACITY_PROGRAM.md).
4. Slot 1 continues P0-1 (90M r4 p64, then p96).

The leaderboard items are the cheapest route to a public result. Idle slots are the largest current waste.

**Strict Transformer matched-compute retries (5 Oct 11:00 UTC; investor-relevant, cheap): first free AWS slot.**
- `queue/aws_language_10M_strictB_p64d4_3.8pass_20261005T110000Z.txt` (~2.5 h; target ≤ 102.7 TF against TF128x4 2.215).
- `queue/aws_language_10M_strictA_p96d4_5.85pass_20261005T110000Z.txt` (~6 h; target ≤ 347 TF against TF128x4 1.996).

Today's tuned-Transformer wins are 2–4% over the Transformers' budgets. These runs fit under them, and the traced work
decides whether they qualify. A win upgrades deck slide 2 from "~equal compute" to "beats the validation-selected
tuned Transformer at no more of its training compute".

## P0 — do these first (all hosts)

| # | Deliverable | Host / queue | Pass criterion | Status |
|---|---|---|---|---|
| P0-1 | **90M matched-compute win** | AWS: `queue/aws_language_90M_r4_p64d4_4pass_linear_20261004T193000Z.txt`, then `..._r4_p96d4_4pass_...` (AWS_NATIVE_LANGUAGE_90M.md rev. 4) | test bpc ≤ 1.661 at ≤ 3.9 PF (beats LSTM-512); ≤ 1.604 at ≤ 8.0 PF (beats Transformer) | **first arm done (5 Oct 08:50): 1.800 at 0.97 PF, efficiency point** (LSTM 1.661 at 3.9 PF); slope 4x compute = -.057 bpc, extrapolated ~25 PF to reach the LSTM: not on track; p96 arm optional |
| P0-2 | **10M win against LSTM-512 at matched compute** | curie: p128/d4 4 passes, starts right after the FAS arm (~00:15 UTC 5 Oct); fallback p96/d4 pool 4 with sampled credit, 6 passes | test bpc < 1.799 at ≤ 433 TF; tuned LSTM-512 4.5p is 1.825 at budget A | **loss (5 Oct 04:20):** p128/d4 4-pass 1.907 at 411 TF; also below p96 6-pass 1.888. Width alone does not close the local gap; next: taps3, weight decay (AWS), lr arms |
| P0-3 | **NeuroBench primate leaderboard** | AWS: `queue/aws_primate_admission_20261004T184500Z/manifest.json` (six-session r1) | six-session mean R² > 0.71 (AEGRU) | 5/6: .768/.600/.529/.519/.677, paired mean .619 vs tinyRSNN .643, bigRSNN .683; **> .71 out of reach** (needs ≈ .92 on the last two); Indy at par, Loco behind |
| P0-4 | **NeuroBench Mackey-Glass leaderboard** | AWS r1 (repeats 20–29 running); curie expected-reception development → a pre-declared tau 17 run | 30-repeat sMAPE < 13.37 (LSTM), with a smaller footprint | **completed, loss (5 Oct 08:15):** 30/30 repeats, primary 8-stream average 14.84 (LSTM 13.37, ESN 14.79); smallest footprint (58 KB), no win |
| P0-6 | **Tuned dense references at our budgets** (added 4 Oct 21:00 UTC, user-directed) | AWS: ten `queue/aws_tuned_ref_10M_*_20261004T210000Z.txt` first (1.5–3 h each), then four `aws_tuned_ref_90M_*` (TUNED_BASELINES.md) | native beats the validation-selected tuned arm at each budget (A ≤ 352 TF: 1.888; B ≤ 107 TF: 1.955; 90M C/D once P0-1 completes) | running; 5/6 A arms done: LSTM-512 4.5p 1.826 (val-leading) and LSTM-384 1.840 beat native 1.888 (losses, context-independent); tuned TF192x4 2.027, TF128x4 1.996 and TF256x4 lr.002 2.053 are behind native 1.888 budget A 6/6 complete (TF256x4 lr.001 2.199); budget B 4/4 complete: LSTM-384 1.915, LSTM-256 1.934 beat native 1.955; TF192x3 2.315 and TF128x4 2.215 behind; admission waits for windowed LSTM validation rescore (below); curie native lr .003/.006 fairness arms queued after P0-2 |
| P0-7 | **Home-field async wins** (added 4 Oct 22:40 UTC, user-directed) | curie: FAS native arm (v6 queue, after the streaming evaluations), SHD aug/expected p32d4; AWS: FAS data + five references (AWS_FAS_REFERENCES.md) after the P0-1/P0-6 owners | FAS (experiments/FAS_BENCHMARK.md): native AUROC above every reference (LSTM, Transformer, LRU, S5-style, Mamba-style; classical) at N <= 256 process events, at no more training/inference work; SHD: a parameter- or work-matched result against small SNNs (cAdLIF 94.19% at 38.7K params; DCLS 95.07% at 0.2M) | queued |
| P0-5 | **Scoreboard** | report/scoreboard.py: REPORT.md / PDF page 2, generated from result files on every rebuild | every comparison listed as win / loss / efficiency point per WIN_CRITERIA | **done** (5 wins of 13) |

**AWS slot allocation order (revised 4 Oct 21:00 UTC):** P0-1, then P0-6 10M arms, then P0-3 take slots as soon as any
slot frees; P0-6 90M arms and the P1-1 seed `aws_language_10M_r1_p64d4_4pass_linear_s7_20261004T210000Z` follow. Without
P0-6, a matched-compute win against an untuned reference will not survive review, so it ranks with P0-1. The long depth-8 streaming
replay/teacher fits are research diagnostics: checkpoint them at their next milestone and suspend them (do not delete
anything) until P0-1 and P0-3 are running. Resume them afterwards.

**P0-6 blocking step (5 Oct 04:10 UTC): windowed VALIDATION rescore of the tuned LSTM arms.** Budget A is complete (6/6);
budget B is 3/4. `report/tuned_reference_admission.py` correctly refuses cross-architecture validation selection while
LSTM validation is carried-state and Transformer validation is windowed. Owner: curie (text8 + language_stream_rescore
tooling; inference only, minutes per arm). Rescore text8[90M:90.2M] with E64 T256 windows, the same scorer as the
Transformer arms, for A: lstm384_p6, lstm512_p4.5 and B: lstm384_p2.5, lstm256_p5 (+ any later LSTM arm). Save each
as a new result beside the original; never edit the AWS JSON. `reference_window_rescore.py` covers TEST only, so add a
validation interval mode or use language_stream_rescore. Expected outcome, not a substitute for the run: selection
unchanged (LSTM-512 val 1.769 vs best TF 1.950; test context effect 0.0000), so budget A = loss to tuned LSTM-512
1.826, and the Transformer sub-comparison = native win (1.888 vs TF128x4 1.996 at 101% compute).
**Taken by curie (5 Oct 01:15 UTC):**
- Queue: `queue/curie_lstm_val_window_rescore_20261005T011500Z.txt`, covering all four arms with
  `reference_context_curve.py --offset 90000000 --test 200000 --context-curve 0`.
- The scorer is the E64 window convention (first window whole, then second halves), checked against `e64_lm_baselines.score`.
- It runs first after P0-2 finishes (around 04:00 UTC), before taps3 and the native lr arms.
- Results land in `experiments/results/language_stream/curie_lstm_valwin_*`. The AWS JSONs are untouched.

**P0-7 FAS update (5 Oct 06:30):** identity-aware oracle bound .755 at N=256 / .913 at N=512, so the early range is
discriminating and native (pool 2, .600/.742) recovers 39%/59% of it. Curie queues: pool2 seed 7 (replication), pool 8,
pool 32 (per-item slots for 30 interleaved items), `curie_fas_native_p32d4_*_20261005T063500Z`.

**Recommendation after P0-1 (5 Oct 08:45, review host):** 90M native 4-pass 1.800 at 0.97 PF is an efficiency point. The
compute slope (.057 bpc per 4×) predicts that p96 4-pass (~2.1 PF) will not reach 1.661. Proposed slot-1 order: the two 90M
budget-C tuned references first (they give the tuned verdict for the completed P0-1 run), then the weight-decay arms
and the X1 capacity arms (per-event levers), then p96 as the trajectory check. The product owner decides; nothing is
cancelled.

**Historical P0-7 interpretation (5 Oct 11:00; superseded by the oracle-assisted protocol correction above):** a classical route-aware FIFO de-interleaver recovers item identity
exactly and equals the oracle (0.755 at N=256, 0.913 at N=512). Native pool-2 0.600, recruitment arms 0.566–0.578. The
"earlier than every classical method" claim is withdrawn. FAS v1 cannot show a race advantage unless native reaches ≥ .755.
Next: (a) winner-identity diagnostic on the native arms; (b) FAS v2 with ambiguous interleaving (product variants,
overtaking, dropped events), validated with oracle + de-interleaver before any native run.

## P1 — confirm and widen the wins

| # | Deliverable | Pass criterion |
|---|---|---|
| P1-1 | Second seed of every P0 win (10M: p96 6-pass, p64 4-pass; 90M winners) | win holds on seed 7 |
| P1-2 | Sampled route credit (§416): fitting cost flat in pool size at equal quality | pool 4/8 sampled within .01 bpc of linear at ≤ 60% of its fitting work |
| P1-3 | SHD official test (speaker-held-out selection first) | report accuracy against the cited 96.4% best |
| P1-4 | Update-matched 10M row against Transformer-256×4 (96 lanes) | 6-pass with 4,883 updates still < 1.908 |

## P2 progress

- **Hardware cost model v1 done (4 Oct):** `experiments/hardware_cost_model.py` → `results/diagnostics/hardware_cost_model_20261004.json`.
  Efficiency point: native p96 (1.888) moves 5.8× fewer bytes/char than Transformer-256×4 (1.908) and 1.9× fewer than LSTM-512
  (1.799, better quality), with cached-key winner-only execution (§414). Pool 2→32: 10.5× params for +5% bytes/char.
  (v1 of the model mis-charged key scoring; corrected same day, theory 154.) Next: a quality point at large pool (pool 8
  90M exists in rev. 3) so the flat-traffic capacity claim has a quality number beside it.

## P2 — after the P0/P1 numbers exist

Hardware cost model for the target asynchronous ASIC (event, message-rate, memory-locality/traffic and FLOP counts on one fixed trained model, vs the tuned dense reference; see investment/HARDWARE_THESIS.md;
CPU wall-clock is not evidence since the hardware is simulated); multi-modal or TTT demonstrations;
larger-than-90M scale.

## Stop or defer now

- New theory notes, primitives or diagnostics that do not feed a P0/P1 row (record ideas in a backlog, do not run them).
- DVS large program and the public UCR campaign repeats (results stand as reported).
- Clock-noise micro-diagnostics, unless they block P0-4.
- Investor-deck edits beyond inserting P0/P1 numbers once they are completed.

## Reporting rule

When a P0/P1 run completes: (1) one line in the scoreboard — win, loss or efficiency point with the numbers; (2) commit the
result file; (3) queue the confirming seed if it is a win. No paragraph of caveats: one scope sentence (data, seeds,
compute convention). Losses are reported just as plainly.

**Curie order after P0-2 (5 Oct 02:45 UTC, product owner):**
1. LSTM validation rescore (P0-6 blocking step).
2. taps3 compiled smoke, then taps3.
3. FAS pool 8 compiled smoke (updated driver, RSS check), then FAS pool 8 and pool 2 seed 7.
4. FAS pool 32 smoke, then pool 32.
5. Native lr .003, then lr .006.
6. The rest of the v6 queue.

The FAS pool sweep moves ahead of the lr arms (about 10 h) because the oracle bound shows early-detection headroom on the
home-field benchmark. Each FAS arm needs its smoke to pass.

**FAS recruitment sweep replaces the plain pool sweep (5 Oct 03:35 UTC; THEORY §419):**
- Order: torch contracts, then a smoke run with every knob active, then R0 (pool 8 untied), R1 (pool 8 tied), R3 (tied
  with free-slot bonus 3), R4 (tied with training temperature 2), R5 (tied with balance .01), then pool 2 seed 7, then the
  pool-32 smoke and R2 (pool 32 tied, 5K runs).
- Arms are screened at 1 epoch, compared within the sweep, and reported with occupancy.
- The other host's 2-epoch untied pool 8 and pool 32 queues are kept but not chained: R0 covers untied pool 8. Untied
  pool 32 is predicted to suffer the untrained-loser bias (§419 Proposition 3); run it later if R2 wins.

**AWS additions (5 Oct 04:00 UTC; AWS_CAPACITY_PROGRAM.md):** after the current P0 owners:
1. Two native weight-decay arms (§421; p64 4-pass, the budget-B comparison against tuned LSTM-384 1.915).
2. Three large-pool language capacity-curve arms (note 155 X1: tied/untied pool 32 and tied pool 8, sampled credit).
3. The FAS references, already queued (AWS_FAS_REFERENCES.md).

The FAS recruitment sweep and the grokking testbed run on curie.

**P0-6 blocking step done (5 Oct 04:22 UTC, curie).** Windowed (E64 T256) validation rescore of the tuned LSTM arms on
text8[90M:90.2M]:

| Budget | Arm | Windowed validation | Carried-state validation |
|---|---|---:|---:|
| A | LSTM-512 4.5 passes | 1.7688 | 1.7688 |
| A | LSTM-384 6 passes | 1.7779 | 1.7779 |
| B | LSTM-384 2.5 passes | 1.8518 | 1.8518 |
| B | LSTM-256 5 passes | 1.8722 | 1.8721 |

- The carried-state values reproduce the AWS `best_valid_bpc` exactly, and windowing changes nothing.
- Cross-architecture validation selection is now valid:
  - Budget A selects LSTM-512 (validation 1.769; best Transformer 1.950). Native **loses**: 1.888 against 1.826.
  - Budget B selects LSTM-384 (1.852; best Transformer 2.191). Native **loses**: 1.955 against 1.915.
- Transformer sub-comparisons stay native wins in quality, at 101–109% of the Transformer arms' compute.
- Results: `experiments/results/language_stream/curie_lstm_valwin_*_20261005T011500Z.json`.
