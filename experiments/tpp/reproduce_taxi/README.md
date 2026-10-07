# Reproduce the EasyTPP Taxi result

One command reruns the Taxi result from scratch: it downloads the public data, trains five seeds and scores the test split.

```bash
pip install torch numpy          # a CPU build of torch is enough
bash experiments/tpp/reproduce_taxi/reproduce.sh          # or: ... reproduce.sh 0   (one seed, ~2-3 min)
```

The script:
- verifies the frozen driver `experiments/tpp/race_tpp_v5.py` by sha256;
- downloads `train/dev/test.json` from the HuggingFace [`easytpp/taxi`](https://huggingface.co/datasets/easytpp/taxi)
  release and checks their sha256;
- trains with model selection on dev only, then scores test once per seed (one CPU thread, < 1 GB RAM);
- writes `experiments/results/tpp/thirdparty_taxi_v5_s<seed>_<stamp>.json` and prints the comparison.

The metric is the total log-likelihood per event in nats (time + mark; higher is better), on the 14,420 scored test
events of the EasyTPP protocol.

| Run | Hardware | Test log-likelihood (5 seeds) |
|---|---|---|
| Original | AWS CPU | 0.5250 ± 0.0010 |
| Independent rerun (same team) | Intel i5-4690 desktop | 0.5252 ± 0.0007 |
| S2P2, best published | (paper) | 0.522 ± 0.004 |

Expect agreement within the seed spread, not bit equality. On the desktop rerun, three seeds equal the AWS values to
≤ 3·10⁻⁹. On the other two, different floating-point arithmetic moved the early-stopping epoch (+0.0001 and +0.0010).
Please send results (the JSON files and `lscpu`/`python -c "import torch; print(torch.__version__)"`) to the authors.
