#!/usr/bin/env bash
set -euo pipefail
cd /workspace
# Wait on the named live fit and utility handles; retain their failures/results.
while tmux has-session -t curie_data_growth_64k_p32_20261005_v1 2>/dev/null || tmux has-session -t curie_data_growth_64k_p32_utility_20261005_v1 2>/dev/null; do
  sleep 5
done
repeat_queue=$(python3 experiments/token_capacity_repeat_admission.py --output experiments/queue/curie_data_growth_64k_repeat_admission_20261005_v1.json)
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=1200 bash experiments/queue/run_safe.sh "$repeat_queue"
