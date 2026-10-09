# Proposed battles: new domains for wins and generality (curie, 9 October 2026)

Status (9 Oct, 09:40 UTC): **B4 admitted** (founder; AWS owner, queued). **B5 admitted** for preparation and development
(founder direction 9 Oct; curie owner; [tgb/B5_TGB.md](tgb/B5_TGB.md)). The others are **proposed**; each needs admission (PRODUCT_ORDERS.md admission rule: battle name and the decision it changes).
Order = expected value × fit to the family's core mechanisms ÷ cost. Every battle follows the existing rules: published
scores under the exact data, splits and scorer; no training of external architectures by us; development on validation
only; a pre-registered test protocol; first-pass results are labelled as such.

## B5 — Temporal Graph Benchmark (TGB), dynamic link prediction (first: tgbl-wiki-v2)

- **Why it fits:** interactions are timestamped events between entities. The leading methods keep a memory per node
  updated by events (TGN lineage, TPNet, DyGFormer); our addressed slots with elapsed-time decay are that mechanism with
  an exact event model behind it. On tgbl-wiki a recency/frequency heuristic is second on the leaderboard, so time-decayed
  addressed memory carries most of the signal.
- **Leaderboard (verified 9 Oct 2026, tgb.complexdatalab.com):** tgbl-wiki-v2 test MRR TPNet **0.827 ± 0.001**,
  Heuristic(LocalGlobal) 0.821, HyperEvent 0.810, DyGFormer 0.798, NAT 0.749 (all negative destinations per positive edge);
  tgbl-review-v2 GraphMixer 0.521 (100 negatives); tgbl-coin-v2 TPNet 0.832.
- **Protocol:** the official `py-tgb` loaders, chronological splits, negative samples and evaluator (TGB ≥ 0.7.5).
- **Cost:** tgbl-wiki (≈157K events, ≈9K nodes) and tgbl-review are CPU-feasible; coin/flight need more compute.
- **Design rationale:** per-node addressed slots holding time-decayed keys and sufficient statistics of interaction
  history (counts, recency, partner identity), queried by the source node (the keyed predecessor read, note 156); score
  = race/softmax over candidate destinations, i.e. exactly a ranking metric's object.

## B4 — Bosser & Ben Taieb neural TPP benchmark (proposed by AWS, 8 Oct)

- Frozen one-configuration EasyTPP model on a second public neural-TPP benchmark (LastFM, MOOC, Reddit, Wikipedia,
  MIMIC-II, …). Owner: AWS (see its proposal commit 4c3351b7).

## B6 — Neural Latents Benchmark (NLB'21), MC_Maze (spikes + continuous behaviour)

- **Why it fits:** spikes are asynchronous events; hand kinematics are a dense stream; the metrics (co-smoothing bits per
  spike, velocity R², PSTH R²) reward a latent state driven by events. Public leaderboard on EvalAI (challenge 1256).
- **References:** NDT, AutoLFADS, STNDT lineage lead (paper: arXiv 2109.04463). **Exact current leaderboard values to be
  verified on EvalAI before any claim** (the page is browser-rendered; not machine-readable from curie).
- **Cost:** MC_Maze small/medium/large are CPU-feasible (DANDI downloads).

## B7 — Opportunity activity/gesture challenge (dense IMUs + sparse ambient/object sensors)

- **Why it fits:** the clearest small benchmark mixing dense body-worn inertial streams with binary object and ambient
  events (doors, drawers, switches): one substrate for both.
- **References:** challenge protocol and published DeepConvLSTM-lineage results; to verify.
- **Cost:** small; UCI download.

## B8 — HiRID-ICU-Benchmark / YAIB (dense 2-minute vitals + irregular labs, medications)

- **Why it fits:** the clinical successor to P19 on mixed dense and asynchronous data; standard task suite (circulatory
  failure, respiratory failure, mortality, length of stay) with published baselines.
- **Gate:** PhysioNet credentialed access (founder; ~1–2 weeks lead time). **Action for the founder: apply now.**

## B9 — Business-process event logs (BPI Challenges): next activity and remaining time

- **Why it fits:** interleaved cases with timestamps, FAS on real data, and the pilot-kit market (industrial and IT logs).
- **References:** published deep predictive-process-monitoring benchmarks; to verify protocol and splits.

## Later (GPU-scale, after funding)

- Event cameras: Prophesee Gen1 / 1Mpx detection, DSEC (flow, disparity, detection), N-ImageNet, eTraM; event + frame
  fusion tracking (VisEvent, FE108); event + IMU odometry (UZH-FPV).
- Small event-vision parity: DVS128 Gesture, N-CARS; SHD/SSC (learned-delay models lead: our delay thesis).
- Tabular parity: TabArena subset vs tuned boosted trees and TabPFN v2 (needs lifting the "trees" pause).

## Next actions taken (curie)

- B5: `py-tgb` 2.3.0 installed with `python -m pip install --target /workspace/.cache/pylib_tgb py-tgb` (drivers add that
  path; the venv's numpy is used). tgbl-wiki-v2 downloaded (the package stores it under
  `.cache/pylib_tgb/tgb/workspace/data/tgb/`); official loader verified (157,474 events, 9,227 nodes, 110,232 / 23,621 /
  23,621 split, metric MRR). Protocol check (`experiments/tgb/heuristics.py`) and the native model v1
  (`experiments/tgb/race_link.py`, smoke then development fit) queued on curie behind the current chain.
- Open for other hosts: B6 (NLB MC_Maze) and B7 (Opportunity) data and reference verification; B9 protocol survey.
- Founder: apply for PhysioNet credentialed access (HiRID, B8).
