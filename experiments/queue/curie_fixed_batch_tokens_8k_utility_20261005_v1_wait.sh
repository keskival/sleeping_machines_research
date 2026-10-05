#!/usr/bin/env bash
set -euo pipefail
cd /workspace
while tmux has-session -t curie_fixed_batch_8k_20261005_v1 2>/dev/null; do sleep 5; done
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=240 bash experiments/queue/run_safe.sh experiments/queue/curie_fixed_batch_tokens_8k_utility_20261005_v1.txt
