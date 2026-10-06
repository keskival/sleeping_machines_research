#!/usr/bin/env bash
set -euo pipefail
cd /workspace
gather_wait_begin=$SECONDS
while tmux has-session -t curie_unified_followup_20261005_v1 2>/dev/null || tmux has-session -t curie_rotation_followup_20261005_v1 2>/dev/null || tmux has-session -t curie_typed_followup_20261006_v1 2>/dev/null; do
  if (( SECONDS-gather_wait_begin > 21600 )); then echo 'Deferred: priority sequence still active after6h'; exit 2; fi
  sleep 5
done
gather_headroom_begin=$SECONDS
while (( $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) < 9504 )); do
  if (( SECONDS-gather_headroom_begin > 7200 )); then echo 'Deferred: gather probe headroom unavailable within2h'; exit 2; fi
  sleep 5
done
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_chunk_gather_probe_20261006_v1.txt
