# EIC Accelerator: eligibility, gap analysis and preparation plan (7 October 2026)

Private founder working document. Rules are taken from the official *EIC Accelerator Guide for Applicants v6.0
(19 November 2025, Work Programme 2026)* and the EIC Accelerator web page, both checked 7 October 2026. Re-check them
before submitting; work programmes change yearly, and the 2027 programme will govern full proposals made in 2027.

## 1. What the EIC Accelerator offers (2026 rules)

| Item | 2026 rule |
|---|---|
| Grant component | Up to **€2.5M**, a lump sum covering at most **70% of eligible costs** of innovation activities at TRL 6–8, completed within 24 months |
| Investment (equity) component | Optional; **€1M–€10M** from the EIC Fund, at most 25% of voting shares, patient capital (7–10 years) |
| **Zero equity ("grant only")** | Possible. Condition: *"evidence that you have sufficient financial means (e.g. revenue flow, existing investors or shareholders) to finance the deployment and scaling up of your innovation"* |
| Blended finance | Grant (TRL 6–8 work) plus EIC Fund equity (remaining and TRL 9 work). One work package cannot mix both |
| Who can apply | A single start-up or SME established in an EU Member State or associated country. **Natural persons may apply** if they intend to set up an SME; *"your company will have to be established prior to signing the Accelerator contract"* |
| Technology readiness | Innovation activities *"currently at TRL 6 or above"*; for primarily technological innovation, *"TRL 5/6 or above is envisaged"* |
| Process | (1) **Short proposal**, any time, batched on the first Tuesday of each month, results in ~4–6 weeks; GO needs ≥ 3 of 4 evaluators. (2) **Full proposal** within 12 months of a GO, at cut-offs; 3 days of free coaching after a GO. (3) **Jury interview** |
| 2026 full-proposal cut-offs | 7 Jan, 4 Mar, 6 May, 8 Jul, 2 Sep, **4 Nov 2026** (5 pm Brussels) |
| Full-proposal threshold | Each criterion ≥ 4/5 and total ≥ 13/15 to be invited; about 2.5× the grant budget is interviewed |
| Criteria | Excellence (novelty, TRL, IP); Impact (market); Level of risk, implementation and need for Union support (team, governance, gender balance, **early investor traction**, need beyond what markets fund alone, risk mitigation) |
| **Submission limit** | After **three unsuccessful submissions at any stage** (short or full), no further EIC Accelerator applications under Horizon Europe. Withdrawing before results does not count |
| Not selected | Seal of Excellence (with consent to share data), which eases national and alternative funding |
| 2026 budget | Open €414M, Challenges €220M. 2026 Challenges are energy, materials, fusion, soils, raw materials and climate adaptation: none fits us, so we apply to **Open** |

## 2. Where we stand against each requirement

| Requirement | Status | Gap and action |
|---|---|---|
| Legal entity (EU SME) | No company yet (deck: terms "subject to company formation") | Allowed at application (natural person). **Incorporate early** anyway, as a Finnish Oy or a Spanish S.L. (§9): the PIC, the cap table, the team criterion, national co-funding and investor traction all go through the company |
| Deep-tech novelty | Strong: a new trainable substrate (temporal races, addressed sparse state, counterfactual credit), exact containment of attention and Mamba, formal theory | Present it as one product-relevant breakthrough, not a research catalogue |
| **TRL 5/6** | Event-native models validated on **real-world public data**: EasyTPP (taxi, e-commerce, Q&A logs) and ICU records (P19). Defensibly TRL 4–5 (validated against real data, in the lab). **No deployment in a relevant operational environment yet (TRL 6)** | **The decisive gap.** Run a pilot on a design partner's own operational event data (industrial or IoT event logs, IT operations or security logs, or clinical time series), with a measured result. Even a short pilot with a letter of intent strengthens TRL 5/6 |
| IP | No company-owned patents; IP plan exists (`investment/IP_PROTECTION_PLAN.md`) | File priority applications on the core constructions **before** any public disclosure; EIC evaluates IP at the short stage. Coordinate with the two private paper drafts, which must wait for the disclosure decision |
| Market and business model | Opportunity analysis exists, aimed broadly at the platform | Choose **one beachhead**: event-native anomaly detection and forecasting on operational logs, sold as models plus a CPU runtime; give pricing and a customer denominator. Keep the platform vision as the long-term upside |
| Team | Sole founder, strong record: lead ML roles; inventor on 35 published US/EP patent documents | The criterion asks for team competences, governance, gender balance and a gap-filling plan. Name 1–3 intended hires or advisers (learning research, runtime engineering, commercial lead) and show them in the video; letters of commitment help |
| Early investor traction | €3M round proposed at €50M pre; no closed investment yet | Any signed term sheet, letter of intent or angel commitment directly supports this criterion **and** grant-only eligibility |
| Need for Union support | Deep-tech, long horizon, compute-heavy validation | State why private markets alone under-fund it: a new architecture class, long validation cycles, and European sovereignty in efficient AI on existing hardware |
| Evidence of results | Public wins: EasyTPP Taxi, Taobao and StackOverflow (matched compute); P19 (AUPRC 0.639 vs 0.583) | Strong; cite the sealed-test protocols and the compute ratios |

**Bottom line.**
- The rules allow applying now as a natural person.
- On TRL the honest position is 4–5. A short proposal sent before a relevant-environment pilot risks a NO GO on
  Excellence, and every NO GO consumes one of the **three** lifetime attempts.
- Submit the short proposal **once a design-partner pilot is under way, or at least signed**. Use the weeks before
  that to incorporate, file priority patents and line up investor and team commitments.

## 3. Zero equity: how to get the grant without giving the EU equity

Choose **grant only** and document *sufficient financial means* to fund deployment and scaling beyond the grant:
- a closed private round (the planned €3M);
- signed commitments;
- revenue (pilot contracts);
- or national funding.

The grant pays 70% of the eligible TRL 6–8 costs; we fund 30% plus everything at TRL 9 and beyond.

Example: a €3.57M, 24-month innovation programme takes the €2.5M maximum, and €1.07M plus later scaling come from the
round. Grant-only and the private round reinforce each other: the round is the evidence of means, and the grant
stretches the round. Blended finance remains an option if a larger total package is wanted; the EIC Fund then takes
at most 25% of voting shares.

## 4. Proposed EIC project (24 months, TRL 5/6 → 8)

Beachhead product: **event-native AI for operational event streams**. It is early fault and anomaly detection and
forecasting on timestamped, interleaved, irregular event data, running on ordinary CPUs at a fraction of the compute of
Transformer-based alternatives.

| Work package | TRL | Content | Indicative cost |
|---|---|---|---|
| WP1 Productise the event-native core | 6→7 | Training/inference runtime (sparse inference, recording-resolution likelihood, calibrated alarms); APIs; reproducible model cards | €0.9M |
| WP2 Pilots in relevant environments | 6→7 | Two or three design-partner pilots (industrial/IoT logs, IT-ops/security logs, clinical time series), each with a pre-registered success metric (detection lead time, false-alarm rate, cost per event) | €1.1M |
| WP3 Validation and certification-grade evaluation | 7→8 | Sealed benchmarks, robustness (capacity, drift, unseen concurrency), privacy and data protection, the AI Act risk file for clinical use | €0.6M |
| WP4 Go-to-market preparation | 7→8 | Pricing, deployment packaging (on-premises CPU), first paid contracts | €0.5M |
| WP5 Management, IP, dissemination | — | Patent prosecution, publication after filing, open benchmark (FAS) | €0.47M |
| **Total eligible** | | | **€3.57M** (grant €2.5M = 70%) |

All costs are planning assumptions to be replaced by the EIC lump-sum budget template.

## 5. Timeline (recommended)

| When | Action |
|---|---|
| Oct 2026 | Incorporate (Oy or S.L., §9); get the company PIC; IP counsel and priority filings on the core constructions; choose the beachhead; approach 3–5 design partners |
| Oct–Nov 2026 | Sign one design-partner pilot or letter of intent; secure written investor interest; name team or adviser commitments; contact the Finnish EIC National Contact Point (Business Finland) for a pre-review |
| Nov–Dec 2026 | Run the pilot's first measurement. Submit the **short proposal** when TRL 5/6 evidence exists (earliest batching: **first Tuesday of December 2026**, or January 2027) |
| Within 12 months of a GO | Full proposal at the first suitable 2027 cut-off, after the free coaching; meanwhile finish the pilot and close the round |
| Interview | ~8–9 weeks after the full-proposal cut-off; jury of entrepreneurs and investors |

## 6. Complementary Finnish funding (check current calls)

- **Business Finland Sprint:** pilot call for small companies' first R&D funding (launched 2026).
- **Business Finland Deep Tech Accelerator:** three €400k phases (two grants, one loan) at 75% of costs, for startups
  under five years old commercialising research. The 2026 call ran 11 May–16 Aug 2026; watch for 2027.
- National funding counts as "financial means" for grant-only and as traction.

## 7. Risks to the application and mitigations

| Risk | Mitigation |
|---|---|
| NO GO on TRL (burns an attempt) | Submit only with relevant-environment pilot evidence; use the NCP pre-review |
| "Research project, not a business" | One beachhead, pricing, design partners, a scaling plan; keep the platform as upside |
| Sole-founder team risk | Named hires or advisers, governance plan, gender-balance plan |
| IP disclosure before filing | Priority filings first; private papers stay private until then |
| Grant-only means questioned | Closed or committed round, or national funding, documented at full-proposal time |

## 8. Prepared materials in this folder

- `SHORT_PROPOSAL_DRAFT.md`: draft answers organised by the three evaluation criteria (12-page limit).
- `PITCH_DECK_10_SLIDES.md`: content for the 10-slide short-proposal deck.
- `VIDEO_SCRIPT_3MIN.md`: the 3-minute video script.

Sources: [EIC Accelerator page](https://eic.ec.europa.eu/eic-funding-opportunities/eic-accelerator_en) ·
[Guide for applicants v6.0, WP2026](https://eic.ec.europa.eu/document/download/9d96fbf3-4d85-4ad0-9483-c77ce348111d_en?filename=EIC+Accelerator+guide+for+applicants_WP26.pdf) ·
[EIC Accelerator FAQ](https://eic.ec.europa.eu/eic-frequently-asked-questions/faqs-eic-accelerator_en) ·
[Business Finland DTA](https://www.businessfinland.fi/en/services/funding/funding-services//deep-tech-accelerator-dta) ·
[Business Finland 2026 funding services](https://www.businessfinland.fi/en/whats-new/news/2026/what-do-business-finlands-funding-services-look-like-at-the-beginning-of-2026/)

## 9. Where to incorporate: Finnish Oy or Spanish S.L. (added 7 October 2026)

The EIC accepts an SME established in **any** EU Member State; both are eligible. The deciding factors are where the
founder and team actually live and work, and the national co-funding.

| Factor | Spanish S.L. | Finnish Oy |
|---|---|---|
| EIC eligibility | Yes | Yes |
| **Seal of Excellence follow-up** | CDTI has run direct-grant calls for Spanish SMEs holding an EIC Accelerator Seal of Excellence (FEDER co-funded; reported up to €2.5M per project; 2025: €22M to 11 SMEs; the 2026 call, €7.5M, was limited to Canary Islands and Valencian companies) | No equivalent Seal-of-Excellence scheme identified (not verified) |
| Early R&D grant | **CDTI NEOTEC 2026**: up to 70% of eligible costs, max €250k (85% / €325k if a PhD is hired); company ≤ 3 years old; **paid-in share capital ≥ €20,000** before applying | **Business Finland Sprint** (2026 pilot, first R&D funding); **Deep Tech Accelerator**: three €400k phases (two grants, one loan) at 75% of costs, startups < 5 years old; 2026 call closed 16 Aug |
| Other national instruments | ENISA participative loans; the Startup Law regime (Ley 28/2022: reduced corporate tax in the first years, better stock-option treatment, certified startup status) — **verify current terms** | Business Finland loans and grants; a strong Nordic seed and VC ecosystem |
| Investor practicalities | Share transfers and capital increases go through a notary; workable, but more friction for VC rounds and option plans | Simple share transfers; familiar to Nordic investors |

**Rule that overrides the table:** incorporate where the company will really be managed and where the founder and team
work. The EIC checks operational capacity. National programmes require a real local establishment. A company managed
from another country risks tax residence and permanent-establishment problems. If the founder lives in Spain, the S.L.
is the natural choice, and CDTI's Seal-of-Excellence funding makes even an unsuccessful EIC full proposal valuable.

Sources: [CDTI Seal of Excellence 2026 call](https://www.cdti.es/en/noticias/cdti-innovacion-75-millones-pymes-canarias-valencia-sello-excelencia-eic-feder-2026) ·
[BOE extract, 18 May 2026](https://www.boe.es/buscar/doc.php?id=BOE-B-2026-16523) ·
[CDTI: €22M to 11 SMEs with the Seal (Dec 2025)](https://www.muypymes.com/2025/12/19/cdti-innovacion-millones-pymes-espanolas-sello-excelencia) ·
[NEOTEC 2026 call](https://www.cdti.es/sites/default/files/2026-04/convocatoria_neotec_2026.pdf) ·
[Seal of Excellence aid up to €2.5M (deducible.es)](https://deducible.es/ayudas-pymes-sello-de-excelencia-2026-hasta-25-millones-de-euros-para-proyectos-de-id-reconocidos-por-el-eic-accelerator/)
