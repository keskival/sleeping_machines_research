# Reproduce the five EasyTPP wins

```bash
pip install torch numpy
python experiments/tpp/reproduce_easytpp/reproduce.py all --dry-run      # verify drivers and data checksums
python experiments/tpp/reproduce_easytpp/reproduce.py taxi --seeds 0     # one seed, a few minutes on one CPU thread
python experiments/tpp/reproduce_easytpp/reproduce.py all                # all datasets, five seeds each
```

The script checks each frozen driver by sha256, downloads the HuggingFace `easytpp/<dataset>` release and checks every
file by sha256 (the same files the original runs scored: identical test event counts), trains with the exact original
arguments (`config.json`, copied from the original result files), selects on dev, scores test once per seed, and prints
the comparison.

| Dataset | Original (5 seeds) | Best published | Driver |
|---|---|---|---|
| Taxi | 0.5250 ± 0.0010 | 0.522 (S2P2) | race_tpp_v5 |
| Taobao | 1.3991 ± 0.0025 | 1.318 (IFTPP) | race_tpp_v5 |
| StackOverflow | −2.1525 ± 0.0045 | −2.163 (S2P2) | race_tpp_v12 (matched size) |
| Retweet | −6.3262 ± 0.0009 | −6.348 (NHP) | race_tpp_v16 |
| Amazon | 0.8028 ± 0.0007 | 0.781 (S2P2) | race_tpp_v18 |

Metric: total log-likelihood per scored event in nats (time + type; higher is better), events 2..N of every test sequence.
Taxi has already been reproduced on separate hardware (0.5252 ± 0.0007). Expect agreement within seed spread, not bit
equality, across CPUs. Please send result JSONs (`experiments/results/tpp/thirdparty_*`) with `lscpu` and the torch version.
