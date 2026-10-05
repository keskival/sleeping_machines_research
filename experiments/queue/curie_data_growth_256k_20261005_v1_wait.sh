#!/usr/bin/env bash
set -euo pipefail
cd /workspace
while tmux has-session -t curie_data_growth_64k_postrepeat_20261005_v1 2>/dev/null; do
  sleep 5
done
larger_queue=$(python3 experiments/token_256k_admission.py --output experiments/queue/curie_data_growth_256k_admission_20261005_v1.json)
# Four times64Kexposure; ordinary64K326-400s measured; bound2400s.
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=2400 bash experiments/queue/run_safe.sh "$larger_queue"
