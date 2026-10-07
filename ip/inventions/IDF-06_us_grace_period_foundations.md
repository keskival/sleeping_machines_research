# IDF-06 — Foundations already public: US-only candidates under the one-year grace period

CONFIDENTIAL · For counsel · Status: **publicly disclosed in the repository before 3 Oct 2026 23:59:31 UTC** →
**EP closed** (absolute novelty); **US open** until one year after each item's first public appearance (35 U.S.C.
§102(b)(1)(A)); other grace-period countries per counsel.

These are the architectural foundations of the family. They underlie every result, so US coverage of their concrete
implementations has strategic value even without Europe. Claims must target specific constructions first disclosed in
September–October 2026: the general idea of computing with delays and races is in the 2021–2022 public manifesto and in
external literature (time-to-first-spike coding, delay learning, sleep sort) and is not claimable.

| ID | Subject (theory notes) | First public (repo) | File US before | Candidate concrete claims |
|---|---|---|---|---|
| 06a | **Exponential-race routing:** a race of exponential clocks with rates exp(score) selects route i with probability softmax(score)ᵢ exactly (note 08, 151) | 8 Sep 2026 | **8 Sep 2027** | sampling a discrete route/expert/token by racing hardware or software timers instead of computing a softmax and drawing; first arrival selects; ties and precision handled by clock noise (note 151, unpublished: EP possible for 151's specific content) |
| 06b | **Counterfactual credit to unrealized routes** (losing alternatives receive gradient through survival/race terms) | 14 Sep 2026 | **14 Sep 2027** | training method for hard-routed networks crediting losing routes exactly through race survival factors, at a counted extra cost (measured: 2.507 → 2.371 bits per character for 0.3% extra training work) |
| 06c | **Race attention / delay-coded aggregation** reproducing softmax attention over delivered keys; **clockless execution** | 27 Sep 2026 | **27 Sep 2027** | attention computed by delivering values in order of key-dependent delays and aggregating arrivals; event fabric without a global clock; logical key/value read reductions (4LNd → 4Ld bytes for values) |
| 06d | **Separate keys and values in races** (keys decide timing/route, values carry content; joint key/value/clock credit) | 28 Sep 2026 | **28 Sep 2027** | race units with separate key and value paths and credit rules isolating harmful winner changes |
| 06e | **Capacity beyond activity; silence-aware supervision** | 30 Sep 2026 | **30 Sep 2027** | receiver pools larger than the selected activity (measured 2.371 → 2.345 bits per character at equal selected work); losses supervising silence intervals |
| 06f | **Statistic-valued race memory** (values as sufficient statistics; closed-form write credit) | 2 Oct 2026 | **2 Oct 2027** | generic form of IDF-03 |

Recommended: one or more US provisional applications well before **8 September 2027**, ideally together with the
priority filing for IDF-01–03 so that the family's foundations and its benchmark-winning embodiments share a filing
date in the US. Search the public history for each claimed feature's earliest appearance first (dates above come from
text search and may be later than the true first disclosure of a specific feature).

Note 151 (normalized clock noise and precision credit) was first committed 4 Oct 2026 06:51 UTC, after the boundary: its
specific content may still be claimable in Europe; counsel to compare with notes 08 and earlier.
