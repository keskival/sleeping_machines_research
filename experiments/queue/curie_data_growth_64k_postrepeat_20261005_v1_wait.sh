#!/usr/bin/env bash
set -euo pipefail
cd /workspace
while tmux has-session -t curie_data_growth_64k_repeat_20261005_v1 2>/dev/null; do
  sleep 5
done
utility_queue=$(python3 experiments/token_capacity_postrepeat_gate.py --mode utility)
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=900 bash experiments/queue/run_safe.sh "$utility_queue"
work_queue=$(python3 experiments/token_capacity_postrepeat_gate.py --mode work)
# Instrumented8Kfit437s vs ordinary45s; ordinary64Kfits326-400s.
# Full64Ktrace bound5400s allows measured tracing overhead and adaptive tails.
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=5400 bash experiments/queue/run_safe.sh "$work_queue"
