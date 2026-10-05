#!/usr/bin/env bash
set -euo pipefail
cd /workspace
rotation_wait_begin=$SECONDS
while tmux has-session -t curie_fas_v1_replication_retry_20261005_v2 2>/dev/null || tmux has-session -t curie_unified_followup_20261005_v1 2>/dev/null; do
  if (( SECONDS-rotation_wait_begin > 21600 )); then echo 'Deferred: admitted sequence active after6h'; exit 2; fi
  sleep 5
done
rotation_headroom_begin=$SECONDS
while (( $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) < 9504 )); do
  if (( SECONDS-rotation_headroom_begin > 7200 )); then echo 'Deferred: probe headroom unavailable within2h'; exit 2; fi
  sleep 5
done
rotation_status=0
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_rotation_backward_probe_20261005_v1.txt || rotation_status=$?
echo "Rotation probe exit: $rotation_status; no default kernel promoted"
if [ -f experiments/results/token_language/curie_data_growth_tokens_256k_b64_c16_p32_s6_20261005_v1.json ]; then
  MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=1200 bash experiments/queue/run_safe.sh experiments/queue/curie_data_growth_tokens_256k_p32_utility_20261005_v1.txt
fi
