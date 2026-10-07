# IDF-04 — One temporal core across event domains; cross-cohort generative pretraining

CONFIDENTIAL · Invention disclosure for counsel · Status: **unpublished** (joint core 7 Oct 04:20 UTC `82a7b39d`;
cross-cohort 7 Oct 20:43 `27541f3a`) · Jurisdictions: EP and US · **Lower priority**: multi-task and transfer learning
are well known; best used as dependent claims or embodiments of IDF-01/03.

## Content

- **Shared core (IDF-04a):** a single continuous-time temporal memory (14,080 parameters) trained jointly on event streams
  from several domains (mobility, shopping, Q&A activity, reviews), each domain keeping its own type embedding, addressed
  type memory and race-of-clocks head. Development result: matches or beats separately trained models on three of four
  domains (Taxi 0.4899 vs 0.4872, Taobao 1.2895 vs 1.284, StackOverflow −2.186 vs −2.177 nats/event, higher is better)
  with about 40% fewer total parameters. The same temporal memory layer (code and size) also serves the clinical
  classifier of IDF-03.
- **Cross-cohort pretraining (IDF-04b):** generative next-measurement pretraining (when, which channels, which values) on
  an unlabeled cohort mapped onto a target cohort's variable layout (24 shared clinical variables), each normalized on its
  own training data, then fine-tuning with few labels. Result pending; same-cohort pretraining gave no robust gain.

## Possible claim angle

A system serving several event-stream sources with one shared temporal memory and per-source race-of-clocks heads,
reducing total model size and enabling deployment of one core across sources; dependent on IDF-01's head.
