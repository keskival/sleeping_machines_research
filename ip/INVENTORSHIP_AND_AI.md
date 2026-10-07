# Inventorship, AI assistance and ownership — read before filing

7 October 2026 · CONFIDENTIAL · For patent counsel

## The issue

Patent offices require that inventors be natural persons. The EPO (J 8/20, J 9/20 "DABUS") and the US courts (Thaler v.
Vidal, 2022) refused applications naming an AI system as inventor. USPTO guidance on AI-assisted inventions (February
2024) allows patents on AI-assisted inventions only where **a natural person significantly contributed to the conception
of each claim**; merely presenting a problem to an AI system, or recognizing and appreciating its output, is not by itself
enough. An application that names the wrong inventors can be invalid or unenforceable.

## How the inventions in this package arose

The research programme, its theory (temporal races, delays as computation, sparse addressed state, counterfactual credit,
clockless execution) and its strategic direction come from the founders (Tero Keski-Valkama; Karoliina Salminen credited
on research). Since late September 2026 much of the implementation, and several specific technical constructions, were
produced by AI coding agents (Claude) working in this repository under the founders' direction and review. Examples that
need an honest inventorship assessment per claim:

| Construction | Where | How it arose (to be confirmed by the founders) |
|---|---|---|
| Defective delayed clocks, logistic-window clocks, state clock, anchored windows | IDF-01, IDF-02 | Proposed and implemented by an AI agent during the B1 battle, under the founder's direction to own the benchmark with the family's race/delay mechanisms; each followed a measured failure |
| Resolution principle, cell-held hazards, dequantization audit | IDF-02 | Proposed by an AI agent after it diagnosed spurious likelihood on gridded timestamps |
| Statistic-valued addressed channel slots for irregular clinical series | IDF-03 | Implements theory note 59 (statistic-valued race memory, public 2 October 2026); IMTS embodiment by an AI agent |
| Mamba containment | IDF-05 | Derived and checked by an AI agent at the founder's request ("do we algebraically subsume Mamba?") |
| Race attention, exponential-race routing, counterfactual route credit, key/value separation, clockless execution | IDF-06 | Founders' programme; theory notes written with AI assistance; to be assessed |

## What to do

1. **Each founder writes a short contribution statement per invention:** which ideas they conceived (problem framing,
   the decisive insight, choice among alternatives, modifications to AI output), with dates and supporting messages or
   commits. Conversation logs from the agent sessions are evidence of who proposed what; keep them.
2. **Counsel determines inventorship claim by claim.** Where a founder's contribution to a specific construction is not
   significant, counsel can (a) draft claims around the founder-conceived combination (for example the use of the
   family's race output law as the event model of a monitoring system, with the clock families as dependent features),
   or (b) have the founders develop the construction further themselves before filing.
3. **Do not name an AI system as inventor** and do not omit a human contributor. AI tools are not inventors; their use is
   disclosed to counsel, not listed on the application.
4. **Ownership:** written assignment from each human contributor to the company. Check prior employment agreements
   (the founder's earlier patents belong to a former employer; confirm no claim on this work) and the terms of the AI
   tools used (outputs generated for the user).

This is the single largest legal risk to an aggressive filing strategy. It is manageable with documented human
contribution and careful claim drafting, and it should be resolved before the priority filing, not after.
