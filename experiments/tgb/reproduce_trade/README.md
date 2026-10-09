# Reproduce the TGB tgbn-trade win

Claim: race_affinity v1 (2,107 parameters) reaches test NDCG@10 **0.8680 ± 0.0005** on the Temporal Graph Benchmark's
tgbn-trade node-affinity task (seeds 0–2: 0.8683, 0.8674, 0.8683), above the leaderboard leader NAVIS (0.863 ± 0.001,
ICLR 2026) and persistent forecast (0.855).

1. `python -m pip install --target .cache/pylib_tgb py-tgb==2.3.0` (or `pip install py-tgb==2.3.0`).
2. `python experiments/tgb/reproduce_trade/reproduce.py` — downloads tgbn-trade through the official loader on first use
   (answer the package's download prompt), verifies the driver and data checksums in `config.json`, trains and scores
   seeds 0–2 (about 4 minutes each on one CPU thread) and compares with the recorded results (tolerance 0.002).

Protocol: official `NodePropPredDataset`/`Evaluator`; TGB's example evaluation loop (batches of 200 edges, one label
pointer; validation label years 2009–2012, test 2013–2015); predictions for a label time use only edges before it.
Development and selection used validation only; the configuration was fixed before the test was scored. Details:
[../B5_TGB.md](../B5_TGB.md), theory: [note 159](../../theory/159_event_graphs_state_freshness_and_affinity.md).
