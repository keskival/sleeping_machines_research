# Disclosure register

7 October 2026 · CONFIDENTIAL · Public boundary: commit `6278e6b2` / `58ac0f28`, 3 October 2026 23:59:31 UTC (founder
statement). "First commit" is the first appearance in this repository's history (`git log --diff-filter=A` or `-S`);
counsel should verify against the GitHub history and any other disclosures.

## Unpublished inventions (EP and US open, if no other disclosure occurred)

| ID | Invention | First commit (UTC) | Evidence | Closest public material (ours) |
|---|---|---|---|---|
| IDF-01a | Race of heterogeneous clocks as a marked event predictor with a mark law per clock | 6 Oct 14:36 `79444569` (`experiments/tpp/race_tpp.py`) | EasyTPP 5/5 wins, B1 dossier | Theory note 09 (competing class hazards, public); e44 Hawkes TPP (public); external: multivariate Hawkes/NHP intensities, competing-risk mixtures |
| IDF-01b | Defective delayed clocks (fire with learned probability π; survival (1−π)+πS(τ)) | 6 Oct 14:56 `07f32f5b` (`race_tpp_v3.py`) | Amazon DEV −0.229 → 0.705 | none found for "defective"/"fires with probability" in the public tree |
| IDF-01c | Logistic-window clocks with closed-form stable survival | 6 Oct 17:55 `3b3bdee6` (`race_tpp_v8.py`) | Amazon DEV 0.724 → 0.768 | none found ("logistic window") |
| IDF-01d | Windows anchored to data-derived gap components (bounded start and width) | 7 Oct 07:32 `79bda7d0` (`race_tpp_v18.py`); clustered init `race_tpp_v11.py` | Amazon TEST 0.8028 ± 0.0007, all seeds, no restarts | "anchored" appears publicly in unrelated contexts (counsel to check) |
| IDF-01e | Single configuration with data-derived rules (windows only for multi-component gap distributions; recording cell as metadata) | 7 Oct 22:42 `abb546b8` (`race_tpp_v19.py`); window rule 8 Oct 06:42 `df4a718d` | unified protocol: all five datasets ahead (24 of 25 seeds) | none |
| IDF-02a | State clock: hazard and marks read from a persistent complex memory evolving through the silence; Gauss–Legendre compensator | 6 Oct 20:11 `c6169b89` (`race_tpp_v10.py`) | StackOverflow win −2.1444 vs −2.163 | "state clock" appears publicly only as a credit-pilot label (REPORT Appendix B, different subject) |
| IDF-02b | Resolution principle: hazards and marks held at their one-cell value below the recording resolution; floors and dynamics caps | 6 Oct 23:38 `8709beb0` (v13); every clock 7 Oct 03:47 `5871aa8f` (v16) | Retweet win with audit residual 0.0038 | none found ("held below", "recording cell") |
| IDF-02c | Target-only dequantization in training (history inputs as recorded) | 7 Oct 02:42 `d5ccb83f` (v15) | Retweet v16 | dequantization is known externally (density models of discretized data); our "target-only" variant is the candidate |
| IDF-02d | Dequantization audit of continuous-time models on gridded timestamps | 6 Oct 16:57 `38969520` (`grid_audit.py`) | withdrew three spurious leads | none found |
| IDF-03a | Event-native irregular-series model: per-channel addressed slots written only at measurement, holding sufficient statistics (count, mean, min, max, first, last, trend, staleness, variance, mean absolute change), decaying with elapsed time | 6 Oct 18:54 `efb1c286` (`race_irts.py`); statistics 7 Oct 00:12 `998d6d7f` (`race_irts_v3.py`) | P19 win AUPRC 0.639 vs 0.583; PAM val 0.981 | **Theory note 59, statistic-valued race memory, public (first 2 Oct 01:27 `statistic-valued`)**: the generic idea is public; the IMTS embodiment is not |
| IDF-03b | Typed comparisons (learned soft thresholds per channel) before neural mixing in a temporal event model | typed semantics note 6 Oct 07:23 `20de60cd` | in all B2 models | "typed comparison" absent from the public tree; typed feature maps exist externally |
| IDF-03c | Ordering latency and measurement rate per channel | 7 Oct 18:23 `7c84fffe` (`race_irts_v8.py`) | no gain on P12 (record honestly) | — |
| IDF-04a | One temporal core trained jointly on several event domains with domain-specific mark embeddings and clock heads | 7 Oct 04:20 `82a7b39d` (`g2_joint.py`) | matches or beats separate models on 3 of 4 | multi-task learning is known externally |
| IDF-04b | Cross-cohort generative pretraining with a variable map | 7 Oct 20:43 `27541f3a` (`race_irts_g1b.py`) | pending | transfer learning is known externally |
| IDF-05 | Selective SSMs executed as content-delayed events over decaying memory with content-addressed write/read; exact equivalence | 7 Oct 04:05 `cfb11282` (`MAMBA_CONTAINMENT_20261007.md`, `mamba_containment_check.py`) | equivalence to 4e−13 | **Public note 08 (28 Sept): "Event selectivity is free … Mamba-style models have to learn"**: the motivation is public; the exact construction is not |
| IDF-07a | Keyed event memory: written keys carry the predecessor's message; type-addressed decaying slots scored inside the race | 7 Oct 13:10 `48372867` (keyed memory); predecessor message 14:05 `15306a57` (`recall_tpp_v3.py`) | recall 97.5% vs 21.6%; FineWeb 6.009 vs KN 6.537 | induction heads, pointer/neural cache (external) |
| IDF-07b | Local three-factor learning of the keyed read (race error × time-decayed slot traces; no gradient into the network) | 7 Oct 13:26 `8b038f4e` (`recall_tpp_v2.py`) | recall 77.4% (3 seeds) | IDF-06b public (generic counterfactual credit) |
| IDF-07c | Mixed-length training / unnormalized match for length-robust recall | 7 Oct 15:05 `722cd2a0` (`recall_tpp_v5.py`) | 91.6% at 2× length | curriculum over lengths is known externally |
| IDF-07d | Queried in-line predecessor with learned per-type-pair duration laws (de-interleaving from the merged likelihood) | 7 Oct 22:35 `c86787be`; duration laws 8 Oct 07:44 `cb1ffa3b` | FAS v2 best early fit; detection pending | FIFO de-interleaving uses identities (oracle) |

## Public inventions (EP closed; US open until one year after first public appearance)

| ID | Subject | First public appearance (repo) | US deadline (no later than) |
|---|---|---|---|
| IDF-06a | Exponential race selecting a route with softmax probability | 8 Sep 2026 (`exponential race`) | **8 Sep 2027** |
| IDF-06b | Counterfactual credit to unrealized routes | 14 Sep 2026 `15c73c0e` (`counterfactual credit`) | **14 Sep 2027** |
| IDF-06c | Clockless execution / delay-coded aggregation = softmax attention ("race attention") | 27 Sep 2026 (`clockless`, `race attention`, `delay-coded` `31d54c99`) | **27 Sep 2027** |
| IDF-06d | Separate keys and values in races; key/value separation | 28 Sep 2026 `8cf5088c` | **28 Sep 2027** |
| IDF-06e | Capacity beyond activity; silence-aware supervision | 30 Sep 2026 `c634929f`, `e8faf5d5` | **30 Sep 2027** |
| IDF-06f | Statistic-valued race memory (generic) | 2 Oct 2026 01:27 | **2 Oct 2027** |

Deadlines are computed from the earliest repository appearance found by text search; the true first disclosure of a
specific claimed feature may be earlier (older notes, the 2021–2022 manifesto, talks). Counsel should search the public
history for each claimed feature, and file well before the earliest date.

Search commands used: `git log --all --reverse -S"<term>" -i`, and `git grep -i -l "<term>" 6278e6b2` for the public tree.
