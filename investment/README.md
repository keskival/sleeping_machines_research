# Private investment materials

Start with the **[19-slide investor pitch](sleeping_machines_pitch_deck_main.pdf)**.
The **[full 36-slide deck](sleeping_machines_pitch_deck.pdf)** adds 17 optional
technical and financial diligence slides. The current discussion draft proposes
a **€3M raise**, with **€50M priced pre-money as the bullish negotiating
case** and €100M as a separate appendix stretch scenario. Neither price is an independent
appraisal or an investor offer. No customer commitments exist yet.

The founder instructed on 3 October 2026 that the repository is now private.
These are local private-review artifacts. Distribution, partner outreach and
future progress publication require a deliberate disclosure decision; building
the files does not authorize those actions. Earlier public disclosures remain
part of IP diligence.

- [Current valuation rationale: ambition, potential and execution](VALUATION_RATIONALE.md)
- [Investor proof plan: priorities, gates and existing execution paths](INVESTOR_PROOF_PLAN.md)
- [IP protection: ownership, patent review and selective filing deliverables](IP_PROTECTION_PLAN.md)
- [Updated investment case](INVESTMENT_CASE.md) and [one-page pitch](PITCH.md)
- [Application opportunities and first proof conditions](../report/model_family_opportunities.md)
- [Editable slide narrative and source registry](PITCH_DECK.json)
- [Slide-by-slide speaker and diligence notes](PITCH_DECK_NOTES.md)
- [Investor-reading review and rationale for the revision](INVESTOR_READING_REVIEW.md)
- [Frozen evidence hashes, derived metrics, budget and financial assumptions](pitch_deck_evidence_20261003.json)
- [Complete same-unit benchmark ledger](pitch_deck_benchmarks.csv)
- [Valuation sensitivity calculations](pitch_deck_financial_sensitivity.csv)
- [Detailed research status report](../report/sleeping_machines_status.pdf)
- [Private diligence bundle: both PDFs, notes, family specification, diagrams, report and completed parents](sleeping_machines_private_diligence_20261004T175000Z.zip)

The 4 October revision leads with the whole model family and its landscape.
Slides 3/4 explain its common temporal/state interface, relationships to existing
families, heterogeneous regions and same-structure learning. The language
fits then illustrate one branch. Slide 5 now diagrams the AGI opportunity:
sensing/action and language/reasoning exchange experience through shared
persistent state and learning. Bidirectional held-out transfer and retention
are its proof requirements. PaLM-E and
RT-2 are attributed joint-learning precedents. The proof plan prioritizes
replication, trained sparse execution and delayed memory credit, with owner
queue links and promotion gates. The report adds a visual family chapter with
the three review axes at every level, compatibility and preservation rules,
and complete illustrative member specifications. A further opportunity slide summarizes irregular ingestion, asynchronous
codecs, memory meta-learning and embodied transfer. The report has a 13-page
family chapter with a proposed codec diagram. Architecture/editorial
coverage is updated; the deck's numerical benchmark/financial ledger remains frozen at
3 October. The protocol appendix separately notes the report's completed
90M result and its restricted single-seed scope; the old pending-status wording
is corrected. Adaptive computation keeps scored queries/fallbacks and full work
explicit. No pending result or hypothetical member is presented as evidence.

The current memo, one-page pitch and deck use the €3M/€50M proposal, with
€100M as a stretch scenario. The prior $10M versions are preserved in the
archive. The updated rationale connects ambitions, potential assets and
execution evidence; application breadth does not establish an appraisal or
a sum of independent markets. Changing the rationale creates no new benchmark evidence.

The deck includes public founder information, AMD's Silo AI acquisition and
LUMI/Poro/Viking work, AMD/Liquid AI collaboration, and potential Intel/Google
strategic fit. Precedents do not establish interest in this project or provide
direct pre-seed valuation comparables. Sole-founder status does not establish
sole research authorship or exclusive IP ownership: preserve Karoliina Salminen's
existing research credit and resolve contributions and employer assignments.

The investor-reading revisions use the same frozen research evidence. Internal
configurations are translated into readable labels and defined in the appendix.
It introduces a specific product hypothesis, annual per-customer economics,
three commercial proof gates and a budget tied to those gates. Conditional exit
arithmetic is retained in the appendix rather than carrying the main pitch.
Previous PDFs, source snapshots and diligence bundles remain in the historical
record; use the current links above for review.

To deliberately refresh the frozen evidence and render with bounded resources:

```bash
python3 scripts/build_pitch_deck.py --prepare
.venv-docker/bin/python scripts/build_pitch_deck.py --notes
.venv-docker/bin/python scripts/build_pitch_deck.py --tag UNIQUE_PUBLICATION_TAG
```

Preparation uses the existing completed evidence snapshot. It does not discover
new experiment results automatically. The renderer uses one CPU thread, nice19,
a 300,000 KiB RSS watchdog, 1,000,000 KiB address-space cap, 120-second timeout
and an 8 GiB available-memory floor. It imports no Torch or NumPy and executes no
model runtime. Source/evidence/output hashes guard concurrent publication; prior
deck versions and immutable publication records are retained. PDF page bounds,
pagination and source links are checked; inspect the preview visually as well.

For an editorial-only change, use `--notes` and a fresh rendering tag; skip
`--prepare` so the research/financial ledger remains untouched. The family
chapter can be refreshed independently with bounded vector rendering:
`.venv-docker/bin/python report/family_report.py --tag UNIQUE_REPORT_TAG`.
It preserves every other report page and archives the prior PDF. Future full
report rebuilds include the same chapter and Markdown entry automatically.
