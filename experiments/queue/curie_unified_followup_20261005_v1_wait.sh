#!/usr/bin/env bash
set -euo pipefail
cd /workspace
# Preserve the admitted seed7 -> seed8 FAS sequence before new diagnostics.
followup_begin=$SECONDS
while tmux has-session -t curie_fas_v1_replication_retry_20261005_v2 2>/dev/null; do
  if (( SECONDS-followup_begin > 14400 )); then echo 'Deferred: FAS sequence still occupies host after4h'; exit 2; fi
  sleep 5
done
# Physical-host capacity, including other containers, controls admission.
followup_headroom_begin=$SECONDS
while (( $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) < 9504 )); do
  if (( SECONDS-followup_headroom_begin > 7200 )); then echo 'Deferred: diagnostic admission headroom unavailable within2h'; exit 2; fi
  sleep 5
done
typed_status=0
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_typed_temporal_p8d2_smoke_20261005_v1.txt || typed_status=$?
echo "Typed integration exit: $typed_status; inspect its separate runner log"
# An unrelated typed failure does not discard the fixed language comparison.
followup_headroom_begin=$SECONDS
while (( $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) < 9504 )); do
  if (( SECONDS-followup_headroom_begin > 7200 )); then echo 'Deferred: language admission headroom unavailable within2h'; exit 2; fi
  sleep 5
done
# Reuse the fixed language recipe for the missing capacity/data cell.
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=3600 bash experiments/queue/run_safe.sh experiments/queue/curie_data_growth_tokens_256k_b64_c16_p32_s6_20261005_v1.txt
