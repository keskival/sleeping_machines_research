#!/usr/bin/env bash
set -euo pipefail
cd /workspace
while tmux has-session -t curie_token_lower_lr_256k_20261005_v1 2>/dev/null || tmux has-session -t curie_token_phase_timing_20261005_v1 2>/dev/null; do
  sleep 5
done
# Reserve8GiB + 1.8GB job cap +1GiB admission margin; watchdog remains mandatory.
fas_wait_begin=$SECONDS
while (( $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) < 11000 )); do
  if (( SECONDS-fas_wait_begin > 7200 )); then
    echo 'FAS retry deferred: 11000MiB admission headroom unavailable within2h' >&2
    exit 2
  fi
  sleep 5
done
MEM_CAP_KB=6000000 MEM_CAP_RSS_KB=1800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=5400 bash experiments/queue/run_safe.sh experiments/queue/curie_fas_v1_frozen_p32d4_s7_retry_20261005_v2.txt
MEM_CAP_KB=6000000 MEM_CAP_RSS_KB=1800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_fas_v1_frozen_p32d4_s8_smoke_20261005_v1.txt
MEM_CAP_KB=6000000 MEM_CAP_RSS_KB=1800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=5400 bash experiments/queue/run_safe.sh experiments/queue/curie_fas_v1_frozen_p32d4_s8_20261005_v1.txt
