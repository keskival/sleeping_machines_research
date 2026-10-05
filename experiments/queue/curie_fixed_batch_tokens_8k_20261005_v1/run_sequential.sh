#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=240
for queue in experiments/queue/curie_fixed_batch_tokens_8k_20261005_v1/*.txt; do
  bash experiments/queue/run_safe.sh "$queue"
done
