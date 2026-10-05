#!/usr/bin/env bash
set -euo pipefail
cd /workspace
# Reserved larger fit retains its place; never compete for its admission lock.
while tmux has-session -t curie_data_growth_64k_postrepeat_20261005_v1 2>/dev/null || tmux has-session -t curie_data_growth_256k_20261005_v1 2>/dev/null; do
  sleep 5
done
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=300 bash experiments/queue/run_safe.sh experiments/queue/curie_token_message_factors_64k_20261005_v1.txt
