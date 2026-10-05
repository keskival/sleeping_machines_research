#!/usr/bin/env bash
set -euo pipefail
cd /workspace
while tmux has-session -t curie_data_growth_256k_20261005_v1 2>/dev/null || tmux has-session -t curie_token_message_factors_64k_20261005_v1 2>/dev/null; do
  sleep 5
done
utility_queue=$(python3 experiments/token_256k_utility_queue.py)
#262KTRAINfeatures~50MiB atP24; one-thread forward sweep plus four2040-target interventions.
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=1200 bash experiments/queue/run_safe.sh "$utility_queue"
