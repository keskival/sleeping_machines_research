#!/usr/bin/env bash
set -euo pipefail
cd /workspace
while tmux has-session -t curie_data_growth_256k_utility_20261005_v1 2>/dev/null || tmux has-session -t curie_fas_v1_replication_20261005_v1 2>/dev/null; do
  sleep 5
done
recipe_queue=$(python3 experiments/token_lower_lr_admission.py --output experiments/queue/curie_token_lower_lr_256k_admission_20261005_v1.json)
# Parent measured1363s fitting, 500896KiBRSS; identical dimensions/data/exposure.
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=2400 bash experiments/queue/run_safe.sh "$recipe_queue"
