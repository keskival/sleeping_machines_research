#!/usr/bin/env bash
set -euo pipefail
cd /workspace
while tmux has-session -t curie_data_growth_64k_postrepeat_20261005_v1 2>/dev/null || tmux has-session -t curie_data_growth_256k_20261005_v1 2>/dev/null || tmux has-session -t curie_token_message_factors_64k_20261005_v1 2>/dev/null || tmux has-session -t curie_data_growth_256k_utility_20261005_v1 2>/dev/null; do
  sleep 5
done
if [ ! -f experiments/data/fas/fas_v1_20261004/manifest.json ]; then
  MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=1200 bash experiments/queue/run_safe.sh experiments/queue/curie_fas_v1_replication_data_20261005_v1.txt
fi
for fas_seed in 7 8; do
  MEM_CAP_KB=6000000 MEM_CAP_RSS_KB=1800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh "experiments/queue/curie_fas_v1_frozen_p32d4_s${fas_seed}_smoke_20261005_v1.txt"
  MEM_CAP_KB=6000000 MEM_CAP_RSS_KB=1800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=5400 bash experiments/queue/run_safe.sh "experiments/queue/curie_fas_v1_frozen_p32d4_s${fas_seed}_20261005_v1.txt"
done
