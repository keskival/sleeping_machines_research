# Patent applications: editable LaTeX for Kile

16 October 2026. Inventor recorded from his express statement: **Tero Keski-Valkama**. AI systems are drafting/engineering support, not named inventors. Applicant, assignee, addresses, priority particulars and signatures have not been invented or filled in.

Open `PF-01.tex` through `PF-12.tex` in Kile and use PDFLaTeX. Each main source is directly editable and contains the description, numbered claims and a separate abstract. Each `PF-xx-drawings.tex` supplies a separate two-sheet black-line drawing set. Keep the shared `sm-patent.cls` and `sm-drawings.sty` beside the sources. **LaTeX is the canonical editable application text.** The older Markdown drafts are source/support records; use the current LaTeX for editing.

```sh
# From the repository root; builds all 24 main/drawing PDFs:
.venv-docker/bin/python ip/latex/build.py
# One family:
.venv-docker/bin/python ip/latex/build.py --family PF-02
# US Letter variant (does not change your source files):
.venv-docker/bin/python ip/latex/build.py --family PF-01 --us
```

In Kile, change `\documentclass[pct]{sm-patent}` to `\documentclass[us]{sm-patent}` for Letter-size technical text. Default A4 text has 30mm top/left,25mm right/bottom margins,12pt Times-style text,1.5 spacing,consecutive centered page numbers and every-fifth-line numbering. Numbered paragraphs restart in each application. Claims and abstract each begin on a new page. Drawings are separate A4 sheets with black vector lines and14pt lettering. These are original lightweight templates rather than an obsolete third-party patent package.

| Family | Kile source | Technical body PDF | Drawing source / PDF |
|---|---|---|---|
| PF-01: Temporal race processing with sparse addressed state and alternative-route learning | [PF-01.tex](PF-01.tex) | [PDF](pdf/PF-01.pdf) | [LaTeX](PF-01-drawings.tex) / [PDF](pdf/PF-01-drawings.pdf) |
| PF-02: Technical-event prediction using heterogeneous defective clocks and clock-specific mark distributions | [PF-02.tex](PF-02.tex) | [PDF](pdf/PF-02.pdf) | [LaTeX](PF-02-drawings.tex) / [PDF](pdf/PF-02-drawings.pdf) |
| PF-03: Recording-resolution-constrained forecasting of quantized event streams | [PF-03.tex](PF-03.tex) | [PDF](pdf/PF-03.pdf) | [LaTeX](PF-03-drawings.tex) / [PDF](pdf/PF-03-drawings.pdf) |
| PF-04: Addressed predecessor binding with persistent event memory and read-time learning | [PF-04.tex](PF-04.tex) | [PDF](pdf/PF-04.pdf) | [LaTeX](PF-04-drawings.tex) / [PDF](pdf/PF-04-drawings.pdf) |
| PF-05: Typed measurement-triggered temporal memory for technical monitoring | [PF-05.tex](PF-05.tex) | [PDF](pdf/PF-05.pdf) | [LaTeX](PF-05-drawings.tex) / [PDF](pdf/PF-05-drawings.pdf) |
| PF-06: Local learning through event-transported eligibility and independently updated feedback | [PF-06.tex](PF-06.tex) | [PDF](pdf/PF-06.pdf) | [LaTeX](PF-06-drawings.tex) / [PDF](pdf/PF-06-drawings.pdf) |
| PF-07: Forward-taught posterior credit for event-routed processors | [PF-07.tex](PF-07.tex) | [PDF](pdf/PF-07.pdf) | [LaTeX](PF-07-drawings.tex) / [PDF](pdf/PF-07-drawings.pdf) |
| PF-08: Learner-aware counterfactual credit from issued optimizer-state worlds | [PF-08.tex](PF-08.tex) | [PDF](pdf/PF-08.pdf) | [LaTeX](PF-08-drawings.tex) / [PDF](pdf/PF-08-drawings.pdf) |
| PF-09: Dependency-validated delivery of local learning updates | [PF-09.tex](PF-09.tex) | [PDF](pdf/PF-09.pdf) | [LaTeX](PF-09-drawings.tex) / [PDF](pdf/PF-09-drawings.pdf) |
| PF-10: Protected temporal hierarchy growth with versioned compositional memory | [PF-10.tex](PF-10.tex) | [PDF](pdf/PF-10.pdf) | [LaTeX](PF-10-drawings.tex) / [PDF](pdf/PF-10-drawings.pdf) |
| PF-11: Delay-mapped execution of selective temporal state with local dispatch | [PF-11.tex](PF-11.tex) | [PDF](pdf/PF-11.pdf) | [LaTeX](PF-11-drawings.tex) / [PDF](pdf/PF-11-drawings.pdf) |
| PF-12: Path-conditioned race-tree output with normalized causal copy selection | [PF-12.tex](PF-12.tex) | [PDF](pdf/PF-12.pdf) | [LaTeX](PF-12-drawings.tex) / [PDF](pdf/PF-12-drawings.pdf) |

`pdf/` contains the current compiled outputs. `build.py` fails on compilation errors or overflowing text boxes. `portfolio.json` records family titles,source support,route notes,claim counts and abstract word counts. [REVIEW_NOTES.md](REVIEW_NOTES.md) is a private companion,not part of the technical specification to upload.

## Submission boundary

The PDFs are technical application documents in filing-oriented layout. They are **not a completed filing package**: the chosen office still requires its application/request or cover sheet,actual applicant and inventor particulars,priority details,fees,and any required declarations and representation. No application has been submitted. Complete the claim-support,prior-disclosure and drawing review before selecting the upload documents. The substantive patentability and human contribution record are not established by template formatting.

PF-01 retains the separately tracked earlier-public US route. Later families have conditional disclosure status; they are not automatically EP-eligible merely because they were committed after the recorded boundary. The 2021–2022 public material remains relevant. See [disclosure register](../DISCLOSURE_REGISTER.md) and [portfolio strategy](../PORTFOLIO_STRATEGY.md).

PCT formatting follows [Rule11](https://www.wipo.int/en/web/pct-system/texts/rules/r11),and abstracts follow [Rule8](https://www.wipo.int/en/web/pct-system/texts/rules/r8). US technical-format references: [MPEP608](https://www.uspto.gov/web/offices/pac/mpep/documents/0600_608_02.htm). Patent Center also has a [DOCX workflow](https://www.uspto.gov/patents/docx);a LaTeX-generated PDF is not a DOCX source and the applicable electronic-filing format/fee rules must be checked for the selected filing. Do not upload this README,the review memo,source hashes,build logs or the complete ZIP as a substitute for the selected application documents.
