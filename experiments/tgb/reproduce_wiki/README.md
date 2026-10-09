# Reproduce the TGB tgbl-wiki result

Model: race_link v4 without identities (7,995 parameters): per-event causal state (every query scored from all events
strictly before it), addressed pair clocks with decay and daily/weekly rotation, destination popularity, exact
co-visitation and page transitions, and a race (softmax) over all 1,000 candidate pages.

1. `python -m pip install --target .cache/pylib_tgb py-tgb==2.3.0` (or `pip install py-tgb==2.3.0`).
2. `python experiments/tgb/reproduce_wiki/reproduce.py` — downloads tgbl-wiki through the official loader on first use,
   verifies the driver and data checksums in `config.json`, trains and scores seeds 0–2 (3 epochs each, validation
   selection, test scored once; about 50 minutes per seed on one CPU thread) and compares with the recorded test MRR
   (tolerance 0.002).

Protocol and results: [../B5_TGB.md](../B5_TGB.md); theory: [note 159](../../theory/159_event_graphs_state_freshness_and_affinity.md).
