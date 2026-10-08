# Investor proof plan — 4 October 2026

The best way to strengthen our €50M pre-money proposal is to turn a promising
architecture into an asset an investor can verify: repeatable useful quality,
economical execution, and a credible team and ownership path. The AGI ambition
explains the upside; it does not replace those proofs. The €3M raise and €100M
stretch case remain as described in [the valuation rationale](VALUATION_RATIONALE.md).
This is an execution plan, not a prediction of investor acceptance.

## Status update, 6 October 2026

- **Public event benchmark: achieved.** Confirmed EasyTPP wins on Taobao (+0.081 nats/event over the best published
  model, 0.92× the leading model's compute) and Taxi (5-seed mixture beats every published model at 0.41× S2P2's
  compute), 5 seeds, sealed test ([dossier](../experiments/B1_EASYTPP.md)).
- **Third dataset won:** StackOverflow (+0.019 nats/event at 1.26× compute, 5 seeds).
- **Fourth dataset won:** Retweet −6.326 ± 0.001 vs −6.348 (NHP), all five seeds ahead, at 1/15 of S2P2's parameters
  and compute.
- **One configuration wins all five (8 October):** pre-registered unified protocol, 24 of 25 seeds ahead, 0.07–1.03×
  S2P2's per-event compute.
- **Fifth dataset won, leaderboard complete:** Amazon 0.803 ± 0.001 vs 0.781 (S2P2), all five seeds ahead, at 0.29×
  S2P2's compute.
- **Second domain: achieved on P19.** Sepsis prediction, five official splits: AUROC 0.916 ± 0.022, AUPRC 0.639 ±
  0.039 vs the best published 0.903 / 0.583 (MTM) ([dossier](../experiments/B2_IRREGULAR_TS.md)). P12: level with MTM on AUPRC, behind on AUROC.
- **Third domain: achieved on PAM (8 October).** Wearable activity recognition, five official splits: accuracy
  0.978 ± 0.007, F1 0.980 ± 0.008 vs MTM 0.975 / 0.976, 46,316 parameters; four of five splits ahead.
- **Generality track G:** self-supervised event pretraining → clinical prediction with fewer labels; one model across five
  event datasets ([plan](../experiments/GENERALITY_PLAN.md)).
- **FAS v2:** the sealed setting is selected; development is in progress.
- **Language at 90M:** one near-matched win (Transformer-256×4) and losses to the strongest tuned references.
- **Tokenized language (R1):** both gates met: associative recall across irregular gaps (97.5%, 3 seeds) and the KN trigram
  beaten on the 65,528-target FineWeb slice (6.009 vs 6.537 at 1M tokens, 3 seeds; 5.521 vs 6.100 at 4M). Next: a
  published small Transformer under the same protocol.

## 1. What we have, and what would change the investment decision

| Investor question | Present evidence | Next decision-changing artifact |
| --- | --- | --- |
| Does the distinctive mechanism learn? | Matched timing-only versus value-credit language result: 2.506386 → 2.371491 BPC, with about 0.30% more estimated fitting work; exploratory single seed | Independent seeds with the same data, update budget, evaluation and complete-work accounting; report every arm and uncertainty |
| Can useful capacity exceed selected activity? | 16 → 32 available receivers, eight selected writes; 2.371491 → 2.345157 BPC, with 1.650× fitting work and more untied parameters | A quality–work frontier separating available state, key discovery, writes, deliveries and learning; compare matched budgets and tied/untied capacity explicitly |
| Does it scale beyond the small fit? | Completed 90M-character fit: 1.857306 BPC on the restricted test[95M:96M] window, one seed; width and training data also change | Locked larger-data evaluation against tuned controls on the same full test protocol, followed by replication; no conversion of this restricted score into a public leaderboard claim |
| Will sparse execution save system resources? | Reference numerical contracts and arithmetic estimates; trained FP32 sparse-backend parity remains open | Checkpoint-bound parity, then measured latency, throughput, residency, traffic and energy on a fixed serving workload |
| Is the platform broader than language? | Defined family, causal contracts, earlier costed online adaptation, application hypotheses | One independently useful event workload first; later a bounded codec or bidirectional embodied-transfer proof, with dense/event incumbents |
| Can this become a defensible company? | Relevant founder record and substantial research artifacts; sole founder Tero Keski-Valkama | Verified contribution/employer/license chain, founder availability, complementary team, budget quotes and reproducible evaluator package; retain Karoliina Salminen's research credit |

Evidence sources are the [source-bound architecture inventory](../report/architecture_evidence.md),
[frozen comparison ledger](pitch_deck_benchmarks.csv) and
[current report](../report/sleeping_machines_status.pdf). The earlier online
pilot improved 3.190859 → 3.095738 BPC on 8,191 development targets at 10.736×
processing work. It supports learning, not an economic or native asynchronous
TTT victory. No current result establishes AGI, commercial adoption or a
measured energy advantage.

## 2. Focus my work where it removes the largest uncertainty

**First: reproduce the integrated quality result.** Complete and inspect the
existing owner-managed comparisons before inventing new architectures. Preserve
computational delays/races, sparse addressed state, key/value separation and
counterfactual credit together. Build the quality–work frontier from completed
results using one denominator per column. Independent seeds matter more than
a new isolated best score. Preserve the strongest valid result and negative
arms; reject target leakage and unequal-protocol victory claims.

**Second: connect theoretical sparsity to trained execution.** Reuse the
prepared checkpoint-admitted sparse contracts. Validate winners, logits, final
state and cache/version behavior at actual deployment precision. Then measure
the full serving boundary, including discovery, setup, synchronization, memory
movement and persistent residency. Report both steady-state and amortized
cost, hardware, batch size and tail latency. This is the bridge from an
interesting model to an economically meaningful runtime.

**Third: diagnose the memory-credit bottleneck before scaling it.** Local
value credit is already a positive result. Credit for a write's delayed effect
is a different question. Use the existing route-fidelity/replay contracts to
compare proposed credit with causal future loss changes, including time and
state consequences. A larger alternative pool helps only when those
alternatives receive useful utility. Do not spend the next compute budget on
many pool/depth variants before fidelity, conditioning and credit horizons are
understood. Any core change needs theory, numerical contracts and a small
integrated fit before a long run.

**Fourth: prove a second use case and make execution credible.** The existing
official event benchmarks are closer to our asynchronous thesis than a new
AGI-scale project. Finish their declared protocols. Prepare contribution/IP,
staffing and evaluator materials in parallel. Choose an expansion proof only
after the core checks; do not spread limited compute across every opportunity.

Working allocation before new evidence arrives: roughly 45% evaluation and
replication, 30% sparse deployment/accounting, 15% memory-credit diagnosis,
10% investor clarity and expansion protocols. These are priorities for this
work, not a change to the frozen €3M financial budget. Reallocate when a
completed result reveals a concrete bottleneck.

## 3. Reuse the current owners and admission paths

| Workstream | Existing path to inspect and reuse | Admission or promotion condition |
| --- | --- | --- |
| Integrated language | [Curie v6 queue](../experiments/queue/curie_language_batched_v6_20261003T111500Z.txt), [AWS language protocol](../experiments/AWS_NATIVE_LANGUAGE_90M.md), [AWS priorities](../experiments/AWS_NEXT_BATCH.md) | Existing owner order, completed results and seed/control protocol govern; a queue name is not evidence that a job is currently running |
| Public event benchmarks | [Current AWS continuation manifest](../experiments/queue/aws_benchmark_continuation_20261004T150000Z/manifest.json), [protocol](../experiments/AWS_NEUROBENCH.md) and [family benchmark strategy](../experiments/FAMILY_BENCHMARK_STRATEGY.md) | Finish official Mackey–Glass/primate splits and cost conventions; retain every session/seed and failed gate; the older 064000Z waiter is superseded |
| Clock precision | [Frozen clock contracts/pilot manifest](../experiments/queue/native_clock_noise_20261004T065700Z/manifest.json) | Contracts precede tiny integrated fits; immutable pins and memory limits remain intact |
| Trained sparse execution | [p64 contracts](../experiments/queue/local_cached_inference_p64_contracts_20261003T123000Z.txt), [pool4 contracts](../experiments/queue/local_cached_inference_pool4_contracts_20261003T123000Z.txt) | Actual producer checkpoint plus physical-host reservation; missing checkpoints are unresolved, not synthetic proof of deployment parity |
| Delayed route/write utility | [Route-fidelity diagnostic](../experiments/queue/local_language_route_fidelity_20261003T125500Z.txt) | Actual checkpoint, frozen replay admission and owner scheduling; local score agreement does not establish the full sequence gradient |

The latest [shared handoff](../experiments/HANDOFF.md) reports ten guarded
NeuroBench/expected-reception contracts passed, the first official tau17 mix8
batch started, and two source-exact private depth8 replay/teacher fits resumed
under the AWS continuation. Primate data acquisition is also reported running.
These are owner-reported progress states, not completed quality comparisons;
refresh them from actual publication records on continuation. Expected
reception pools conditional current values; hard write-address changes and
future recurrent utility remain separate issues. Do not infer full expected
trajectory learning or a solved depth problem from the contract passes.

This workspace cannot establish physical-host admission from its container
process list or local lock. No training, model runtime or profiling is launched
here. On admitted hosts, use unique one-job queues and `run_safe.sh`, one
training job per host except the explicitly authorized bounded AWS CPU slots.
Keep the RSS watchdog and at least 8 GiB available memory; new dense controls
remain AWS work. Inspect live owner state before every launch. Do not displace
or mutate frozen experiments to accelerate presentation work.

## 4. Gates and decisions, rather than an open-ended experiment catalogue

1. **Technical reproducibility:** preregister at least three independent seeds,
   tuned relevant controls, fixed causal targets and fitting budget; publish
   all outcomes and paired effects/uncertainty. Choose the quality tolerance
   before the final comparison. A tiny uncertain gain does not pass solely
   because its best seed leads.
2. **Deployment economics:** trained sparse contracts pass, then useful quality
   stays within the declared tolerance and total serving cost improves on the
   chosen workload. The deck's proposed ≥20% complete-cost reduction is a
   commercial screen to lock before measurement, not a current result. Report
   latency and memory constraints as well as the average; do not substitute
   estimated FLOPs for measured money or joules.
3. **Independent usability:** deliver a documented reproducible package to an
   authorized evaluator and record integration burden, useful quality and
   willingness to pay. No customer interest exists yet. Preparing materials
   does not authorize outreach or disclosure of the private repository.
4. **Execution readiness:** founder verifies rights, commitment, proposed hires
   and spending assumptions. Private-repository status does not undo earlier
   public disclosure or establish patentability. Legal conclusions need counsel.
   Deliver an ownership/disclosure inventory, counsel review and selective
   priority filings for qualifying technical inventions, with trade-secret and
   prosecution plans. Use the existing €200,000 legal/IP/operations envelope,
   subject to quotes; follow the [IP protection plan](IP_PROTECTION_PLAN.md).
   The founder's earlier employer-owned patents are excluded from company assets.

If a gate fails, preserve the result, identify whether the failure is quality,
credit, execution or protocol, and run the smallest discriminating follow-up.
An advantage may be workload-specific; expand only after establishing that
workload. Do not raise the headline valuation merely because another
application hypothesis has been added.

## 5. The smallest useful AGI-direction proof

Our long-term hypothesis is that a shared persistent model can learn reusable
abstractions across sensing, action, language and reasoning at different
cadences. The differentiator to test is useful cross-domain transfer at a
better complete resource cost, not merely connecting a robot to a language
model. [PaLM-E](https://arxiv.org/abs/2303.03378) and
[RT-2](https://arxiv.org/abs/2307.15818) already establish relevant joint-learning
precedents; neither validates our substrate or proves AGI.

Before an embodied run, define a bounded simulated object world with shared
latent geometry/dynamics, irregular observations, action outcomes and symbolic
queries. Fix query cutoffs and action deadlines. Hold out objects, task
compositions and transfer queries; prevent future-state/label leakage. Compare
the integrated substrate with matched-data/work joint dense or VLA controls,
isolated modules, stopped cross-domain credit and the corresponding unimodal
training controls. Charge adapters, discovery, all optimization, replay and
interaction. Shared parameters alone are not a positive outcome.

Test the two directions separately: motor experience improving held-out
reasoning with cognitive training held constant, and cognitive experience
improving held-out control with motor training held constant. Predeclare the
added-data and equal-total-work comparisons separately. Report retention and
negative transfer after online updates. A reproducible bidirectional result
would strengthen the generalist-learning case; it would still be a limited
transfer result, not an AGI demonstration. This is a future protocol outline,
not a configured queue, trained model or reserved compute job.

## 6. Autonomous follow-through and current deliverables

This iteration adds the AGI opportunity diagram to the main pitch, makes its
hypothesis/first proof explicit, and aligns the proof-gate notes with this plan.
Refresh notes, render under bounds, inspect the slide and pagination, run the
source/evidence checks, and package a new private diligence bundle. Financial
and benchmark ledgers remain unchanged.

On continuation: read the current handoffs and owners' completed publications;
promote only validated results; update the proof table and quality–cost figures;
keep numerical parity and full-cost gaps visible. Execute admitted contracts
before fits and stage the next smallest useful comparison. Escalate spending
only at the declared gates. Work can continue autonomously within these bounds;
financing acceptance, third-party reproduction and measured hardware outcomes
require external events and cannot be promised by this plan.
