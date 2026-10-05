#!/usr/bin/env bash
set -euo pipefail
cd /workspace
while tmux has-session -t curie_fixed_batch_8k_c16_work_20261005_v1 2>/dev/null; do sleep 5; done
python3 experiments/token_64k_admission.py > experiments/queue/curie_data_growth_64k_20261005_v1_admission.json
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=900 bash experiments/queue/run_safe.sh experiments/queue/curie_data_growth_tokens_64k_b64_c16_p16_s6_20261005_v1.txt
