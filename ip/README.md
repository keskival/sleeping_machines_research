# Patent-support package — CONFIDENTIAL, prepared for patent counsel

7 October 2026 · Private · Not legal advice, not a filing, not a freedom-to-operate opinion. Everything here is input
for a European/US patent attorney, who decides claim scope, inventorship, jurisdictions and timing.

## What this package contains

| File | Content |
|---|---|
| [DISCLOSURE_REGISTER.md](DISCLOSURE_REGISTER.md) | Every candidate invention with its first commit, public/unpublished status and the jurisdictions still open |
| [INVENTORSHIP_AND_AI.md](INVENTORSHIP_AND_AI.md) | Inventorship position (founder: significant contribution to each invention), AI-assistance disclosure for counsel, contribution-record template, ownership steps |
| [inventions/IDF-01_race_of_delayed_clocks.md](inventions/IDF-01_race_of_delayed_clocks.md) | Event prediction by a race of heterogeneous delayed clocks with per-clock mark laws (EasyTPP 5/5 wins) |
| [inventions/IDF-02_resolution_safe_continuous_time_models.md](inventions/IDF-02_resolution_safe_continuous_time_models.md) | State clock over persistent memory; hazards held below the recording resolution; target-only dequantization; the dequantization audit |
| [inventions/IDF-03_event_native_irregular_series_classifier.md](inventions/IDF-03_event_native_irregular_series_classifier.md) | Addressed per-channel memories with sufficient statistics, written only on measurement (P19 sepsis win; PAM in protocol) |
| [inventions/IDF-04_shared_temporal_core_and_cross_cohort_transfer.md](inventions/IDF-04_shared_temporal_core_and_cross_cohort_transfer.md) | One temporal core serving several event domains; cross-cohort pretraining with a variable map |
| [inventions/IDF-05_selective_state_space_as_delayed_events.md](inventions/IDF-05_selective_state_space_as_delayed_events.md) | Executing selective state-space models (Mamba-class) as content-delayed events over decaying memory |
| [inventions/IDF-07_predecessor_message_binding.md](inventions/IDF-07_predecessor_message_binding.md) | Keyed event memory with predecessor-message keys; local three-factor learning; queried in-line predecessor with per-pair duration laws (unpublished, EP and US open) |
| [inventions/IDF-06_us_grace_period_foundations.md](inventions/IDF-06_us_grace_period_foundations.md) | Foundations already public (September–October 2026): race attention, exponential-race routing, counterfactual route credit, key/value separation, statistic-valued memory, clockless execution. **US-only, with deadlines** |

## The disclosure boundary (founder statement, 7 October 2026)

Everything in the repository up to and including commit **`6278e6b2` / `58ac0f28`** ("Publish guarded public_final
aws_public_PenDigits_selection_20261004T000100Z_final_s6", **3 October 2026 23:59:31 UTC**) is publicly disclosed. Material
first committed after that commit is unpublished as of this package, provided no other disclosure occurred (investor
meetings without NDA, talks, posts, preprints, uploads). **Confirm this with every founder before filing.**

- **Europe (EPC) has absolute novelty and no grace period:** only unpublished subject matter can be claimed, and only if
  it is not obvious over the public material, including our own.
- **United States:** an inventor's own disclosure made one year or less before the effective filing date is not prior
  art (35 U.S.C. §102(b)(1)(A)). Each public item stays fileable in the US until one year after *its own first* public
  appearance; the earliest deadlines fall in September 2027 (register, IDF-06). Disclosures older than one year bar the
  US too (for example the 2021–2022 manifesto: "Emit A after delay x if no B before that").
- Other grace-period jurisdictions (Japan, Korea, Canada and others, each with its own conditions) can be assessed by
  counsel for the IDF-06 subject matter.

## What can be filed where (itemized)

**Europe (EP) and the US: unpublished subject matter, first committed after the boundary**

| ID | Invention | Strength (preliminary, for counsel) |
|---|---|---|
| IDF-01a | Race of heterogeneous clocks with a mark law per clock (time-dependent marks in closed form) | core of the EasyTPP wins; distinguish from Hawkes/competing-risk prior art |
| IDF-01b | Defective delayed clocks (fire with learned probability) in a neural event model | strong candidate |
| IDF-01c | Logistic-window clocks with closed-form stable survival | strong candidate |
| IDF-01d | Windows anchored to data-derived gap components | dependent claim |
| IDF-02a | State clock: hazard/marks from a memory evolving through silence, quadrature compensator | strong candidate |
| IDF-02b | Hazards and mark laws held below the recording resolution; floors and dynamics caps | strong candidate |
| IDF-02c | Target-only dequantization with recorded histories | dependent claim |
| IDF-02d | Dequantization audit of continuous-time models | method claim; validation framing |
| IDF-03a | Irregular-series model with per-channel addressed slots of running statistics, written only on measurement (specific embodiment) | good candidate; must be narrower than public note 59 |
| IDF-03b | Typed threshold comparisons before mixing in a temporal event model | dependent claim |
| IDF-04a/b | Shared temporal core across domains; cross-cohort pretraining with a variable map | weak alone; dependent claims/embodiments |
| IDF-05 | Selective SSMs executed as content-delayed events (implementation, hardware) | inventive-step risk vs public note 08; stronger with measured hardware effects |
| Note 151 | Normalized clock noise and precision credit (committed 4 Oct) | counsel to compare with public notes |

**US only (public before the boundary; one-year grace from each first public appearance)**

| ID | Invention | File in the US before |
|---|---|---|
| IDF-06a | Exponential-race routing with exact softmax selection | 8 Sep 2027 |
| IDF-06b | Counterfactual credit to unrealized routes | 14 Sep 2027 |
| IDF-06c | Race attention / delay-coded softmax attention; clockless execution | 27 Sep 2027 |
| IDF-06d | Separate keys and values in races | 28 Sep 2027 |
| IDF-06e | Capacity beyond activity; silence-aware supervision | 30 Sep 2027 |
| IDF-06f | Statistic-valued race memory (generic form of IDF-03) | 2 Oct 2027 |
| — | Generic forms of IDF-05's motivation (note 08) | 28 Sep 2027 |

**Neither (public more than a year ago or general knowledge):** temporal computing with delays and races as a general
idea (2021–2022 manifesto), sleep-sort-style time-domain sorting, generic multi-task/transfer learning, standard
augmentation and weight averaging.

## Filing logistics

- **No US entity is needed.** Foreign individuals and companies file directly at the USPTO; the Spanish S.L. (or the
  founders, assigning later) can be the applicant, with the human inventors named. A company applicant must be
  represented by a USPTO-registered patent attorney or agent (individual inventors may file a provisional themselves,
  but a practitioner is advisable). Foreign applicants can usually claim small-entity fee reductions; micro-entity
  status has income limits.
- **Check national first-filing rules before filing abroad.** Spain's patent law requires defence-secrecy clearance for
  inventions made in Spain before foreign filing (a first filing at the OEPM, or clearance, depending on the case); counsel
  should confirm the route and timing. If any part of an invention could count as made in the United States, a US
  foreign filing licence may be needed; the AWS compute location is unlikely to matter, but counsel should confirm.
- **One priority date for everything:** a Spanish or EP first filing for IDF-01–05, plus a same-day US provisional
  covering IDF-01–05 and IDF-06, gives Europe its route and the US its date. PCT within 12 months.

## Recommended filing strategy (for counsel's review)

1. **Now, before any further disclosure:** one priority application covering IDF-01, IDF-02 and IDF-03 (the inventions
   behind the six public benchmark wins), with IDF-04 and IDF-05 as additional embodiments if counsel agrees. Options:
   an EP application (or a national first filing in Spain, where the company is being formed) plus a US provisional on
   the same day, so both the EP route and the US date are secured. Twelve months later: PCT and/or national filings.
2. **US provisional(s) for IDF-06** well before the earliest deadline (8 September 2027 for exponential-race routing;
   most others late September 2027), framed around concrete implementations and technical effects.
3. **Hold all external disclosure** of IDF-01–05 subject matter until the priority filing exists: the evidence brief,
   both papers (`report/papers/`), the deck's technical slides and any arXiv/benchmark-leaderboard submission. Investor
   sharing only under NDA, and preferably after filing.
4. **Keep as trade secrets** what is hard to detect in a product and easy to copy once published: training recipes
   (augmentation strengths, EMA, initialization schedules, restart selection), data-specific clock tilings and
   hyperparameters.
5. **Ownership:** assign all rights from each founder/contributor to the company (S.L.) before or at filing; record
   contributors per claim (see INVENTORSHIP_AND_AI.md).

## Patentability framing used throughout (EPO)

AI/ML models are abstract mathematical methods unless they serve a technical purpose or are adapted to a specific
technical implementation (EPO Guidelines G-II 3.3, 3.3.1). Each disclosure therefore states:

- **technical purposes**: monitoring and forecasting of event streams produced by technical systems (computer and
  network logs, industrial machinery and IoT sensors, wearable inertial sensors, bedside physiological monitors),
  generating alerts or control actions;
- **implementation effects, measured**: exact closed-form likelihood without Monte Carlo sampling; per-event compute and
  memory (multiply-accumulates per event, parameters) against the published state of the art; sparse, addressed state
  updates performed only when a channel is measured; robustness to timestamp quantization of recording equipment;
- **hardware embodiments**: event-driven and clockless execution where delays and races are physical.

Medical-method exclusions (EPC Art. 53(c)) apply to diagnostic methods practised on the human body; claims should be to
data-processing methods, systems and devices, not to diagnosis itself. Counsel to frame.

Evidence for every quantitative statement lives in `experiments/B1_EASYTPP.md`, `experiments/B2_IRREGULAR_TS.md`,
`experiments/GENERALITY_PLAN.md` and the result files they cite.
